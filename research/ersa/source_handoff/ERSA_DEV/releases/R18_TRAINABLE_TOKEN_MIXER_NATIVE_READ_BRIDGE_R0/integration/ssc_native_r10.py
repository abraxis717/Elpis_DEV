"""ERSA R10: independently authored SSC host transitions over C-owned RAM.

Authority model:
- A bounded opaque C CAS image is the sole mutable session-state authority.
- Each of SSC's 14 API operations transitions a *temporary* Python value graph.
- Commit is all-or-nothing C-native CAS; images are never persisted.
- This is NOT full C-native operation execution, trained recurrence, or production trust.
- The original SSC reference backend and oracle are never imported for runtime.
"""
from __future__ import annotations
import json, math, threading
from elpis_ssc import contract_r0 as SC
from native.ersa_authority_r10 import NativeAuthority, NativeAuthorityHold

class R10Hold(RuntimeError):pass
POLICIES=('FIFO','LRU','HEBBIAN_ONLY','UTILITY_HEBBIAN')
LIMIT=8192
MAX_PROPOSAL_IDENTITIES=8192

def initial_state(spec):
    return dict(entities={},relations={},control=[],conv=[[0.]*spec.conv_state_dim for _ in range(spec.conv_state_slots)],
                window=[],tick=0,logical_tokens=0,ops_this_tick=0,issued=[],committed=[],
                evictions=0,rejections=0,reinforcements=0,halted=False)

def pack(s):
    return json.dumps(s,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def invalid(code,detail=''):
    return SC.receipt('REJECTED',code,detail)

class NativeSscAuthorityR10:
    def __init__(self,spec,registry,transport,library,*,policy='UTILITY_HEBBIAN',
                 write_authority='fixture:write:r0',max_bytes=16*1024*1024):
        if SC.validate_spec(spec):raise R10Hold('INVALID_SSC_SPEC')
        if policy not in POLICIES:raise R10Hold('INVALID_POLICY')
        if (not 1<=spec.max_active_entities<=LIMIT or not 1<=spec.max_active_relations<=LIMIT or
            spec.max_control_records>65536 or spec.kv_window_tokens>65536 or
            spec.conv_state_slots*spec.conv_state_dim>65536 or
            type(write_authority) is not str or not write_authority):
            raise R10Hold('RESOURCE_BUDGET_UNSUPPORTED')
        self.spec,self.registry,self.transport=spec,registry,transport
        self.policy_name,self.write_authority=policy,write_authority
        self._lock=threading.RLock()
        self._store=NativeAuthority(library,max_bytes)
        self._references={} # identity-liveness anchors ONLY, authority lives in C image
        try:self._store.replace(pack(initial_state(spec)),0)
        except BaseException:self._store.close();raise
    def close(self):
        with self._lock:
            self._store.close();self._references.clear()
    @property
    def generation(self):
        with self._lock:return self._store.generation
    def _load(self):
        blob,version=self._store.read()
        if not blob:raise R10Hold('EMPTY_NATIVE_AUTHORITY')
        return json.loads(blob),version
    def _mutate(self,name,*args):
        with self._lock:
            s,version=self._load()
            if name in ('step','decay'):
                value=args[0] if args else (1 if name=='step' else None)
                if type(value) is not int or not 0<=value<=self.spec.max_steps_per_call:
                    # Reject before SSC budget consumption, but invalidate views.
                    self._store.replace(pack(s),version)
                    return invalid('STEP_BUDGET')
            pending=[]
            receipt=self._transition(s,name,args,pending)
            self._verify(s)
            self._store.replace(pack(s),version)
            if name in ('begin_session','reset'):
                self._references.clear()
            for p in pending:self._references[id(p)]=p
            return receipt
    def __getattr__(self,name):
        if name in SC.STATE_BANK_METHODS:
            return lambda *a,**kw:self._call(name,*a,**kw)
        raise AttributeError(name)
    def _call(self,name,*args,**kwargs):
        with self._lock:
            if name=='begin_session':
                if args or kwargs:
                    raise R10Hold('SESSION_REBIND_REQUIRES_NEW_OWNER')
                return self._mutate(name)
            if kwargs:
                valid={'step':'action_budget','decay':'ticks'}.get(name)
                if valid is None or set(kwargs)!={valid} or args:
                    raise R10Hold('INVALID_CALL_SHAPE')
                args=(kwargs[valid],)
            return self._mutate(name,*args)
    def _budget(self,s):
        if s['ops_this_tick']>=self.spec.max_ops_per_tick:
            s['rejections']+=1
            return invalid('OP_BUDGET')
        s['ops_this_tick']+=1
        return None
    def _control(self,s,name,status,code=''):
        if self.spec.max_control_records:
            s['control'].append([name,s['tick'],status,code])
            if len(s['control'])>self.spec.max_control_records:
                del s['control'][:-self.spec.max_control_records]
    def _bounded_vec(self,values):
        try:return list(SC.vec4(values,self.spec.finite_bound)),None
        except SC.receipt_invalid as e:return None,invalid(e.code,e.detail)
    def _validate_proposal(self,s,p):
        if type(p) is not SC.Proposal:return invalid('INVALID_VALUE')
        reg=self.registry.get(p.relation_id)
        if reg is None:return invalid('UNADMITTED_RELATION')
        if p.provenance!=reg.provenance:return invalid('PROVENANCE_MISMATCH')
        matches=[(role,n) for (ent,role,n) in reg.participants if ent==p.participant]
        if not matches:return invalid('UNKNOWN_PARTICIPANT')
        if matches[0][0]!=p.role:return invalid('ROLE_MISMATCH')
        if matches[0][1]!=p.ordinal:return invalid('ORDINAL_MISMATCH')
        if tuple(sorted(p.flags))!=tuple(reg.flags):return invalid('FLAGS_MISMATCH')
        if p.ticket!=reg.ticket or s['tick']>reg.valid_until_tick:return invalid('STALE_TICKET')
        if str(p.participant) not in s['entities']:return invalid('DISCONNECTED_PARTICIPANT')
        if p.source_frame not in self.transport.list_frames():return invalid('UNBOUND_FRAME')
        _,bad=self._bounded_vec(p.value4)
        return bad
    def _canonical(self,p):
        try:
            vec=self.transport.to_canonical(p.relation_id,p.source_frame,p.value4)
            return self._bounded_vec(vec)[0]
        except (KeyError,PermissionError,ValueError):return None
    def _aged(self,r,tick):
        return r['utility']*math.exp(-3.5e-4*(tick-r['utility_tick']))
    def _victim_relation(self,s):
        pairs=list(s['relations'].items())
        if self.policy_name=='UTILITY_HEBBIAN':
            free=[item for item in pairs if not item[1]['protected']]
            if free:pairs=free
        if not pairs:return None
        if self.policy_name=='FIFO':k=lambda kv:(kv[1]['born'],kv[0])
        elif self.policy_name=='LRU':k=lambda kv:(kv[1]['last_read'],kv[1]['born'],kv[0])
        else:k=lambda kv:(self._aged(kv[1],s['tick']),kv[1]['born'],kv[0])
        return min(pairs,key=k)[0]
    def _verify(self,s):
        spec=self.spec
        if (len(s['entities'])>spec.max_active_entities or len(s['relations'])>spec.max_active_relations or
            len(s['control'])>spec.max_control_records or len(s['window'])>spec.kv_window_tokens or
            len(s['conv'])!=spec.conv_state_slots or any(len(x)!=spec.conv_state_dim for x in s['conv']) or
            s['tick']<0 or s['logical_tokens']<0 or s['ops_this_tick']<0 or
            len(s['issued'])>MAX_PROPOSAL_IDENTITIES or len(s['committed'])>MAX_PROPOSAL_IDENTITIES):
            raise R10Hold('STATE_INVARIANT')
        for r in s['relations'].values():
            if str(r['participant']) not in s['entities']:raise R10Hold('RELATION_DANGLING_PARTICIPANT')
    def _transition(self,s,name,args,pending):
        spec=self.spec
        if name in ('begin_session','reset'):
            s.clear();s.update(initial_state(spec));return SC.receipt('OK')
        if name=='halt':
            s['halted']=True;self._control(s,'halt','OK');return SC.receipt('OK','','REFERENCE_TERMINAL')
        bad=self._budget(s)
        if bad:return bad
        if name in ('step','decay'):
            val=args[0] if args else 1
            s['tick']+=val;s['ops_this_tick']=0
            self._control(s,name,'OK');return SC.receipt('OK')
        if name=='observe':
            (tokens,)=args
            if type(tokens) is not tuple or not tokens or any(type(t) is not int or t<0 for t in tokens):
                return invalid('INVALID_VALUE')
            if len(tokens)>spec.max_steps_per_call:return invalid('STEP_BUDGET')
            for tok in tokens:
                s['logical_tokens']+=1;s['tick']+=1;s['ops_this_tick']=0
                if spec.kv_window_tokens:
                    s['window'].append([tok,s['logical_tokens']])
                    if len(s['window'])>spec.kv_window_tokens:del s['window'][:-spec.kv_window_tokens]
                for i in range(spec.conv_state_slots):
                    phase=((tok+7*i)%13)/13.0
                    s['conv'][i]=s['conv'][i][1:]+[phase]
            self._control(s,name,'OK');return SC.receipt('OK','',s['logical_tokens'])
        if name in ('read_exact','upsert_exact'):
            address=args[0]
            lo,hi=spec.exact_address_namespace
            if type(address) is not int or not lo<=address<=hi:return invalid('INVALID_ADDRESS')
            key=str(address)
            if name=='read_exact':
                if key not in s['entities']:return SC.receipt('NOT_FOUND')
                r=s['entities'][key];r['last_read']=s['tick']
                self._control(s,name,'OK')
                return SC.receipt('OK','',tuple(r['value']))
            _,value,authority=args
            if authority!=self.write_authority:
                s['rejections']+=1;self._control(s,name,'REJECTED','UNAUTHORIZED_WRITE')
                return invalid('UNAUTHORIZED_WRITE')
            vec,bad=self._bounded_vec(value)
            if bad:return bad
            if key in s['entities']:
                r=s['entities'][key];r['value']=vec;r['last_read']=s['tick']
            else:
                if len(s['entities'])>=spec.max_active_entities:
                    ref={str(r['participant']) for r in s['relations'].values()}
                    choices=[(k,r) for k,r in s['entities'].items() if not r['pinned']]
                    if not choices:return invalid('CAPACITY_EXCEEDED')
                    orphans=[(k,r) for k,r in choices if k not in ref]
                    victim=min(orphans or choices,key=lambda it:(it[1]['last_read'],it[1]['born'],int(it[0])))[0]
                    del s['entities'][victim];s['evictions']+=1
                    for rid in [rid for rid,r in s['relations'].items() if str(r['participant'])==victim]:
                        del s['relations'][rid];s['evictions']+=1
                s['entities'][key]=dict(value=vec,born=s['tick'],last_read=s['tick'],pinned=False)
            self._control(s,name,'OK');return SC.receipt('OK')
        if name=='read_assoc':
            (key,)=args
            hits=[(rid,r) for rid,r in s['relations'].items() if r['participant']==key]
            if not hits:return SC.receipt('NOT_FOUND')
            rid,r=min(hits,key=lambda it:(-self._aged(it[1],s['tick']),it[0]))
            score=self._aged(r,s['tick']);r['last_read']=s['tick']
            self._control(s,name,'OK')
            return SC.receipt('OK','',(tuple(r['value4']),score,len(hits)))
        if name=='propose_relation':
            (p,)=args
            bad=self._validate_proposal(s,p)
            if bad:
                s['rejections']+=1;self._control(s,name,'REJECTED',bad.code);return bad
            idstr=str(id(p));
            if idstr not in s['issued']:
                if len(s['issued'])>=MAX_PROPOSAL_IDENTITIES:return invalid('CAPACITY_EXCEEDED')
                s['issued'].append(idstr)
            pending.append(p);self._control(s,name,'PENDING');return SC.receipt('PENDING')
        if name in ('apply_delta','transport_and_aggregate'):
            proposals=(args[0],) if name=='apply_delta' else args[0]
            if name=='transport_and_aggregate' and not isinstance(proposals,(tuple,list)):
                return invalid('INVALID_VALUE')
            transported=[]
            for p in proposals:
                if type(p) is not SC.Proposal or str(id(p)) not in s['issued'] or self._references.get(id(p)) is not p:
                    return invalid('STALE_PROPOSAL')
                if name=='apply_delta' and str(id(p)) in s['committed']:
                    s['rejections']+=1;self._control(s,name,'REJECTED','DUPLICATE_IDENTITY')
                    return invalid('DUPLICATE_IDENTITY')
                bad=self._validate_proposal(s,p)
                if bad:
                    s['rejections']+=1;self._control(s,name,'REJECTED',bad.code);return bad
                cv=self._canonical(p)
                if cv is None:
                    s['rejections']+=1;self._control(s,name,'REJECTED','UNBOUND_FRAME')
                    return invalid('UNBOUND_FRAME')
                transported.append(cv)
            if name=='transport_and_aggregate':
                if not transported:return invalid('INVALID_VALUE','EMPTY_ROUND')
                ex=[spec.salient_temperature*math.sqrt(math.fsum(x*x for x in v)) for v in transported]
                try:
                    maxi=max(ex);w=[math.exp(v-maxi) for v in ex];den=math.fsum(w)
                    salient=tuple(math.fsum((w[i]/den)*v[j] for i,v in enumerate(transported)) for j in range(4))
                except OverflowError:return invalid('NONFINITE_VALUE')
                collective=tuple(math.fsum(v[j] for v in transported)/len(transported) for j in range(4))
                self._control(s,name,'OK');return SC.receipt('OK','',(salient,collective))
            p=proposals[0];rid=p.relation_id;vec=transported[0]
            if rid in s['relations']:
                r=s['relations'][rid];r['value4']=vec;r['last_read']=s['tick']
            else:
                if len(s['relations'])>=spec.max_active_relations:
                    victim=self._victim_relation(s)
                    if victim is None:return invalid('CAPACITY_EXCEEDED')
                    del s['relations'][victim];s['evictions']+=1
                reg=self.registry.get(rid)
                s['relations'][rid]=dict(relation_id=rid,provenance=reg.provenance,
                    participant=p.participant,role=p.role,ordinal=p.ordinal,flags=list(sorted(p.flags)),
                    value4=vec,frame=rid,utility=.020,utility_tick=s['tick'],
                    born=s['tick'],last_read=s['tick'],protected=reg.protected)
            s['committed'].append(str(id(p)))
            self._control(s,name,'OK');return SC.receipt('OK')
        if name=='reinforce':
            rid,co,feedback=args
            for x in (co,feedback):
                if type(x) not in (int,float) or not math.isfinite(x) or not -1<=x<=1:
                    return invalid('INVALID_VALUE')
            r=s['relations'].get(rid)
            if r is None:return SC.receipt('NOT_FOUND')
            s['reinforcements']+=1;r['last_read']=s['tick']
            if self.policy_name in ('HEBBIAN_ONLY','UTILITY_HEBBIAN'):
                v=self._aged(r,s['tick'])+.20*max(co,0.)
                if self.policy_name=='UTILITY_HEBBIAN':
                    v+=.34*feedback
                    if feedback<0:r['protected']=False
                r['utility']=max(0.,min(1.,v));r['utility_tick']=s['tick']
            self._control(s,name,'OK');return SC.receipt('OK')
        if name=='evict':
            (rid,)=args
            if rid not in s['relations']:return SC.receipt('NOT_FOUND')
            if s['relations'][rid]['protected'] and self.policy_name=='UTILITY_HEBBIAN':
                return invalid('PROTECTED_RECORD')
            del s['relations'][rid];s['evictions']+=1
            self._control(s,name,'OK');return SC.receipt('OK')
        raise R10Hold('UNKNOWN_OPERATION')
    def snapshot(self):
        with self._lock:
            s,_=self._load()
            return dict(
                entities=sorted((int(a),tuple(round(x,12) for x in r['value']),r['born'],r['last_read'],r['pinned'])
                    for a,r in s['entities'].items()),
                relations=sorted((r['relation_id'],r['provenance'],r['participant'],r['role'],r['ordinal'],
                    tuple(r['flags']),tuple(round(x,12) for x in r['value4']),r['frame'],
                    round(r['utility'],12),r['utility_tick'],r['born'],r['last_read'],r['protected'])
                    for r in s['relations'].values()),
                control=[tuple(row) for row in s['control']],
                conv=[tuple(round(x,12) for x in slot) for slot in s['conv']],
                window=tuple(tuple(pair) for pair in s['window']),
                tick=s['tick'],logical_tokens=s['logical_tokens'],evictions=s['evictions'],
                rejections=s['rejections'],reinforcements=s['reinforcements'])
    def occupancy(self):
        with self._lock:
            s,_=self._load()
            return dict(entities=len(s['entities']),entity_capacity=self.spec.max_active_entities,
                relations=len(s['relations']),relation_capacity=self.spec.max_active_relations,
                control_records=len(s['control']),window=len(s['window']),
                window_capacity=self.spec.kv_window_tokens,tick=s['tick'],
                logical_tokens=s['logical_tokens'],evictions=s['evictions'],
                rejections=s['rejections'],reinforcements=s['reinforcements'])
    def policy(self):return self.policy_name
    def read_image(self):
        with self._lock:return self._store.read()
