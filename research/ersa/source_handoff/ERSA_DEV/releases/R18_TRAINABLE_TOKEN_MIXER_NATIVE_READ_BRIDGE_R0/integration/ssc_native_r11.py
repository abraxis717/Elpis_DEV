"""ERSA R11: typed C-owned SSC state & C-executed transitions.

External admission registry, transport, contract-object validation, and Python
receipt rendering remain host-provided; no runtime use of SSC ReferenceBank.
All 14 SSC state operations dispatch to bounded native C mutations.
Not a learned attention model, SSC official fixture replay, or production trust.
"""
from __future__ import annotations
import ctypes as C
import math, os, pathlib, subprocess, tempfile, threading
from elpis_ssc import contract_r0 as SC

OPS=tuple(SC.STATE_BANK_METHODS)
# C operation order intentionally differs from the frozen SSC listing.
COPS=('begin_session','observe','read_exact','upsert_exact','read_assoc',
      'propose_relation','transport_and_aggregate','apply_delta','reinforce',
      'decay','evict','step','reset','halt')
CODES=('', 'INVALID_ADDRESS','DUPLICATE_IDENTITY','CAPACITY_EXCEEDED',
       'OP_BUDGET','STEP_BUDGET','UNADMITTED_RELATION','UNKNOWN_PARTICIPANT',
       'ROLE_MISMATCH','ORDINAL_MISMATCH','FLAGS_MISMATCH','PROVENANCE_MISMATCH',
       'STALE_TICKET','UNBOUND_FRAME','UNAUTHORIZED_WRITE','PROTECTED_RECORD',
       'DISCONNECTED_PARTICIPANT','INVALID_VALUE','NONFINITE_VALUE',
       'STALE_PROPOSAL','UNKNOWN_RELATION')
STATS=('OK','REJECTED','NOT_FOUND','PENDING')
POLICIES=('FIFO','LRU','HEBBIAN_ONLY','UTILITY_HEBBIAN')

class Config(C.Structure):
    _fields_=[(x,C.c_int) for x in ('entity_cap','relation_cap','control_cap','window_cap',
       'conv_slots','conv_dim','max_ops','max_steps','policy','proposal_cap')]+[
       ('address_lo',C.c_int64),('address_hi',C.c_int64),('finite_bound',C.c_double),('salient_temp',C.c_double)]
class Request(C.Structure):
    _fields_=[('addr',C.c_int64),('participant',C.c_int64),('tokens_len',C.c_int64),
      ('rid',C.c_int),('pre_code',C.c_int),('authorized',C.c_int),('protected_record',C.c_int),
      ('proposal_id',C.c_uint64),('ticks',C.c_int64),('v',C.c_double*4),
      ('co',C.c_double),('feedback',C.c_double),('tokens',C.POINTER(C.c_int64)),
      ('vectors',C.POINTER(C.c_double)),('proposal_ids',C.POINTER(C.c_uint64)),('n_proposals',C.c_int)]
class Result(C.Structure):
    _fields_=[('status',C.c_int),('code',C.c_int),('number',C.c_int64),('v',C.c_double*4),
       ('v2',C.c_double*4),('scalar',C.c_double),('count',C.c_int),('generation',C.c_uint64)]
class Counters(C.Structure):
    _fields_=[('entities',C.c_int),('relations',C.c_int),('control',C.c_int),('window',C.c_int),
       ('tick',C.c_int64),('logical',C.c_int64),('evictions',C.c_int64),('rejections',C.c_int64),
       ('reinforcements',C.c_int64),('ops_this_tick',C.c_int),('halted',C.c_int),('issued',C.c_int)]
class Entity(C.Structure):
    _fields_=[('used',C.c_int),('pinned',C.c_int),('addr',C.c_int64),('born',C.c_int64),
       ('last_read',C.c_int64),('v',C.c_double*4)]
class Relation(C.Structure):
    _fields_=[('used',C.c_int),('rid',C.c_int),('protected_record',C.c_int),
       ('participant',C.c_int64),('born',C.c_int64),('last_read',C.c_int64),
       ('utility_tick',C.c_int64),('v',C.c_double*4),('utility',C.c_double)]
class Control(C.Structure):
    _fields_=[('op',C.c_int),('status',C.c_int),('code',C.c_int),('tick',C.c_int64)]
class Window(C.Structure):
    _fields_=[('token',C.c_int64),('logical',C.c_int64)]
class R11Hold(RuntimeError):pass

def compile_r11():
    source=pathlib.Path(__file__).resolve().parents[1]/'native/ersa_state_machine_r11.c'
    d=pathlib.Path(tempfile.mkdtemp(prefix='ersa11-native-',dir='/dev/shm'))
    target=d/'libersa11.so'
    result=subprocess.run(['cc','-std=c11','-O2','-pthread','-Wall','-Wextra','-Werror',
         '-fno-fast-math','-fPIC','-shared',str(source),'-lm','-o',str(target)],
         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30,
         env=dict(os.environ,TMPDIR='/dev/shm'))
    if result.returncode:raise R11Hold('C_COMPILE_FAILED:'+result.stdout[-3000:])
    return target

def bind(library):
    l=C.CDLL(str(library))
    def method(n,restype,args):
        f=getattr(l,'ersa11_'+n);f.restype=restype;f.argtypes=args
    P=C.c_void_p;I=C.c_int
    method('new',P,[C.POINTER(Config)]);method('free',None,[P])
    method('run',I,[P,C.c_uint64,I,C.POINTER(Request),C.POINTER(Result)])
    method('generation',C.c_uint64,[P]);method('tick',C.c_int64,[P]);method('has_entity',I,[P,C.c_int64])
    for n,t in [('count',Counters),('entity',Entity),('relation',Relation),('control',Control),('window',Window)]:
        method(n,I,[P,C.POINTER(t)] if n=='count' else [P,I,C.POINTER(t)])
    method('conv',I,[P,I,C.POINTER(C.c_double)])
    return l

class NativeSscAuthorityR11:
    def __init__(self,spec,registry,transport,library,*,policy='UTILITY_HEBBIAN',
                 write_authority='fixture:write:r0',proposal_cap=8192):
        if SC.validate_spec(spec):raise R11Hold('INVALID_SSC_SPEC')
        if policy not in POLICIES:raise R11Hold('INVALID_POLICY')
        if any((type(x) is not int or x<0 or x>65536) for x in (
            spec.max_control_records,spec.kv_window_tokens,spec.max_ops_per_tick,
            spec.max_steps_per_call,spec.conv_state_slots,spec.conv_state_dim)):
            raise R11Hold('RESOURCE_BUDGET_UNSUPPORTED')
        if (spec.conv_state_slots*spec.conv_state_dim>65536 or
            not 1<=spec.max_active_entities<=8192 or not 1<=spec.max_active_relations<=8192 or
            type(proposal_cap) is not int or not 1<=proposal_cap<=8192):
            raise R11Hold('RESOURCE_BUDGET_UNSUPPORTED')
        if type(write_authority) is not str or not write_authority:raise R11Hold('INVALID_AUTHORITY')
        lo,hi=spec.exact_address_namespace
        if not -(1<<63)<=lo<=hi<1<<63:raise R11Hold('INVALID_ADDRESS_NAMESPACE')
        self.spec,self.registry,self.transport=spec,registry,transport
        self.policy_name,self.write_authority=policy,write_authority
        self._lock=threading.RLock()
        self._refs={}
        self._rids=tuple(registry.ids());self._rid_ids={v:i for i,v in enumerate(self._rids)}
        self._lib=bind(library)
        self._cfg=Config(spec.max_active_entities,spec.max_active_relations,spec.max_control_records,
            spec.kv_window_tokens,spec.conv_state_slots,spec.conv_state_dim,
            spec.max_ops_per_tick,spec.max_steps_per_call,POLICIES.index(policy),proposal_cap,
            lo,hi,spec.finite_bound,spec.salient_temperature)
        self._ptr=self._lib.ersa11_new(C.byref(self._cfg))
        if not self._ptr:raise R11Hold('NATIVE_ALLOC_FAIL')
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        with self._lock:
            if self._ptr:self._lib.ersa11_free(self._ptr);self._ptr=None;self._refs.clear()
    def _p(self):
        if not self._ptr:raise R11Hold('CLOSED')
        return self._ptr
    @property
    def generation(self):
        with self._lock:return self._lib.ersa11_generation(self._p())
    def _counters(self):
        c=Counters()
        if self._lib.ersa11_count(self._p(),C.byref(c))!=1:raise R11Hold('BAD_NATIVE_COUNTER')
        return c
    def _proposal_bad(self,p):
        if type(p) is not SC.Proposal:return 'INVALID_VALUE'
        reg=self.registry.get(p.relation_id)
        if reg is None:return 'UNADMITTED_RELATION'
        if p.provenance!=reg.provenance:return 'PROVENANCE_MISMATCH'
        matches=[(role,n) for ent,role,n in reg.participants if ent==p.participant]
        if not matches:return 'UNKNOWN_PARTICIPANT'
        if matches[0][0]!=p.role:return 'ROLE_MISMATCH'
        if matches[0][1]!=p.ordinal:return 'ORDINAL_MISMATCH'
        try: flags=tuple(sorted(p.flags))
        except (ValueError,TypeError):return 'FLAGS_MISMATCH'
        if flags!=tuple(reg.flags):return 'FLAGS_MISMATCH'
        if p.ticket!=reg.ticket or self._lib.ersa11_tick(self._p())>reg.valid_until_tick:return 'STALE_TICKET'
        if (type(p.participant) is not int or p.participant<-(1<<63) or p.participant>=1<<63 or
            not self._lib.ersa11_has_entity(self._p(),p.participant)):
            return 'DISCONNECTED_PARTICIPANT'
        if p.source_frame not in self.transport.list_frames():return 'UNBOUND_FRAME'
        try:SC.vec4(p.value4,self.spec.finite_bound)
        except SC.receipt_invalid as ex:return ex.code
        return None
    def _call(self,name,*args,**kwargs):
        with self._lock:
            if name=='begin_session' and (args or kwargs):raise R11Hold('SESSION_REBIND_REQUIRES_NEW_OWNER')
            if kwargs:
                valid={'step':'action_budget','decay':'ticks'}.get(name)
                if valid is None or set(kwargs)!={valid} or args:raise R11Hold('INVALID_CALL_SHAPE')
                args=(kwargs[valid],)
            q=Request();details='';refs=[];method=COPS.index(name)
            def error(code):q.pre_code=CODES.index(code)
            def fill(values):
                for i,v in enumerate(values):q.v[i]=v
            if name in ('step','decay'):
                val=args[0] if args else 1
                if type(val) is not int or val<0 or val>self.spec.max_steps_per_call or val>=1<<63:
                    error('STEP_BUDGET');q.ticks=0
                else:q.ticks=val
            elif name=='observe':
                (tokens,)=args
                if type(tokens) is not tuple or not tokens or any(type(v) is not int or v<0 or v>=1<<63 for v in tokens):
                    error('INVALID_VALUE')
                elif len(tokens)>self.spec.max_steps_per_call:error('STEP_BUDGET')
                else:
                    q.tokens_len=len(tokens)
                    a=(C.c_int64*len(tokens))(*tokens);refs.append(a);q.tokens=a
            elif name in ('read_exact','upsert_exact'):
                addr=args[0];q.addr=addr if type(addr) is int and -(1<<63)<=addr<1<<63 else 0
                if type(addr) is not int or not self.spec.exact_address_namespace[0]<=addr<=self.spec.exact_address_namespace[1]:
                    error('INVALID_ADDRESS')
                if name=='upsert_exact':
                    _,value,authority=args;q.authorized=int(authority==self.write_authority)
                    # Invalid address takes precedence; unauthorized write takes precedence over invalid vec.
                    if q.authorized:
                        try:fill(SC.vec4(value,self.spec.finite_bound))
                        except SC.receipt_invalid as e:
                            if q.pre_code==0:error(e.code);details=e.detail
            elif name=='read_assoc':
                key=args[0];q.participant=key if type(key) is int and -(1<<63)<=key<1<<63 else -(1<<63)
            elif name in ('propose_relation','apply_delta'):
                (p,)=args;q.proposal_id=id(p)
                # R10 tests stale proposal before applying validation.
                if name=='apply_delta' and self._refs.get(id(p)) is not p:
                    error('STALE_PROPOSAL')
                else:
                    bad=self._proposal_bad(p)
                    if bad:error(bad)
                if type(p) is SC.Proposal:
                    q.rid=self._rid_ids.get(p.relation_id,-1)
                    q.participant=p.participant if type(p.participant) is int and -(1<<63)<=p.participant<1<<63 else 0
                    if name=='apply_delta' and not q.pre_code:
                        try: fill(SC.vec4(self.transport.to_canonical(p.relation_id,p.source_frame,p.value4),self.spec.finite_bound))
                        except (KeyError,PermissionError,ValueError):error('UNBOUND_FRAME')
                        except SC.receipt_invalid as e:error(e.code);details=e.detail
                    if name=='apply_delta':q.protected_record=int(self.registry.get(p.relation_id).protected) if self.registry.get(p.relation_id) else 0
            elif name=='transport_and_aggregate':
                (proposals,)=args
                if not isinstance(proposals,(tuple,list)) or not proposals:
                    error('INVALID_VALUE');q.proposal_id=0
                else:
                    q.proposal_id=id(proposals[0]);vecs=[];ids=[]
                    for p in proposals:
                        ids.append(id(p))
                        if self._refs.get(id(p)) is not p:
                            error('STALE_PROPOSAL');break
                        bad=self._proposal_bad(p)
                        if bad:error(bad);break
                        try:vecs.extend(SC.vec4(self.transport.to_canonical(p.relation_id,p.source_frame,p.value4),self.spec.finite_bound))
                        except (KeyError,PermissionError,ValueError):error('UNBOUND_FRAME');break
                        except SC.receipt_invalid as e:error(e.code);details=e.detail;break
                    if not q.pre_code:
                        q.n_proposals=len(ids)
                        a=(C.c_double*len(vecs))(*vecs);b=(C.c_uint64*len(ids))(*ids)
                        refs.extend((a,b));q.vectors=a;q.proposal_ids=b
            elif name=='reinforce':
                rid,co,feedback=args;q.rid=self._rid_ids.get(rid,-1)
                if any(type(v) not in (int,float) or not math.isfinite(v) or not -1<=v<=1 for v in (co,feedback)):
                    error('INVALID_VALUE')
                else:q.co=co;q.feedback=feedback
            elif name=='evict':q.rid=self._rid_ids.get(args[0],-1)
            elif name not in ('begin_session','reset','halt'):raise R11Hold('UNKNOWN_OPERATION')
            r=Result();version=self._lib.ersa11_generation(self._p())
            rc=self._lib.ersa11_run(self._p(),version,method,C.byref(q),C.byref(r))
            if rc:raise R11Hold('NATIVE_CAS_REJECT_'+str(rc))
            if name in ('begin_session','reset'):self._refs.clear()
            if name=='propose_relation' and r.status==3:self._refs[id(args[0])]=args[0]
            status=STATS[r.status];code=CODES[r.code]
            detail=details
            if name=='observe' and status=='OK':detail=r.number
            elif name=='read_exact' and status=='OK':detail=tuple(r.v)
            elif name=='read_assoc' and status=='OK':detail=(tuple(r.v),r.scalar,r.count)
            elif name=='transport_and_aggregate' and status=='OK':detail=(tuple(r.v),tuple(r.v2))
            elif name=='halt' and status=='OK':detail='REFERENCE_TERMINAL'
            return SC.receipt(status,code,detail)
    def __getattr__(self,name):
        if name in OPS:return lambda *a,**kw:self._call(name,*a,**kw)
        raise AttributeError(name)
    def snapshot(self):
        with self._lock:
            p=self._p(); l=self._lib;c=self._counters();spec=self.spec
            entities=[];relations=[];control=[];conv=[];window=[]
            for i in range(spec.max_active_entities):
                e=Entity();l.ersa11_entity(p,i,C.byref(e))
                if e.used:entities.append((e.addr,tuple(round(x,12) for x in e.v),e.born,e.last_read,bool(e.pinned)))
            for i in range(spec.max_active_relations):
                r=Relation();l.ersa11_relation(p,i,C.byref(r))
                if r.used:
                    rid=self._rids[r.rid];reg=self.registry.get(rid)
                    roles=[(role,n) for ent,role,n in reg.participants if ent==r.participant]
                    role,ordinal=roles[0]
                    relations.append((rid,reg.provenance,r.participant,role,ordinal,
                        tuple(reg.flags),tuple(round(x,12) for x in r.v),rid,
                        round(r.utility,12),r.utility_tick,r.born,r.last_read,bool(r.protected_record)))
            for i in range(c.control):
                t=Control();l.ersa11_control(p,i,C.byref(t))
                control.append((COPS[t.op],t.tick,STATS[t.status],CODES[t.code]))
            for i in range(spec.conv_state_slots):
                slot=[]
                for j in range(spec.conv_state_dim):
                    x=C.c_double();l.ersa11_conv(p,i*spec.conv_state_dim+j,C.byref(x));slot.append(round(x.value,12))
                conv.append(tuple(slot))
            for i in range(c.window):
                w=Window();l.ersa11_window(p,i,C.byref(w));window.append((w.token,w.logical))
            return dict(entities=sorted(entities),relations=sorted(relations),control=control,conv=conv,
                window=tuple(window),tick=c.tick,logical_tokens=c.logical,evictions=c.evictions,
                rejections=c.rejections,reinforcements=c.reinforcements)
    def occupancy(self):
        with self._lock:
            c=self._counters();return dict(entities=c.entities,entity_capacity=self.spec.max_active_entities,
                relations=c.relations,relation_capacity=self.spec.max_active_relations,
                control_records=c.control,window=c.window,window_capacity=self.spec.kv_window_tokens,
                tick=c.tick,logical_tokens=c.logical,evictions=c.evictions,rejections=c.rejections,
                reinforcements=c.reinforcements)
    def policy(self):return self.policy_name
