"""C3 bounded single-thread volatile state. No compute/transport authority.

Handles are lifecycle guards, not security capabilities against arbitrary Python.
State rows are fixed-width typed arrays, never returned as mutable aliases.
"""
from array import array
from dataclasses import dataclass
from math import isfinite

MAX_BATCH=1024
MAX_WIDTH=256

@dataclass(frozen=True,slots=True)
class StateHandle:
    owner: object
    epoch: int
    vertex: int
    slot: int
    generation: int

class NodeState:
    __slots__=('index','width','precision','capacity','mode','_owner','_epoch','_count','_values','_zero','_generation','_slot_owner','_vertex_slot','_free')
    def __init__(self,index,width=16,precision='f64',mode='sparse',capacity=4096):
        if type(width) is not int or not 1<=width<=MAX_WIDTH: raise ValueError('WIDTH')
        if precision not in ('f32','f64'): raise ValueError('PRECISION')
        if mode not in ('dense','sparse'): raise ValueError('MODE')
        n=index.vertices
        if type(capacity) is not int or not 1<=capacity<=n or (mode=='dense' and capacity!=n): raise ValueError('CAPACITY')
        self.index=index; self.width=width; self.precision=precision; self.capacity=capacity; self.mode=mode
        self._owner=object(); self._epoch=0; self._count=0
        self._zero=array('f' if precision=='f32' else 'd',[0])*width
        self._values=self._zero*capacity
        self._generation=array('Q',[0])*capacity
        self._slot_owner=array('i',[-1])*capacity
        self._vertex_slot=array('i',[-1])*n if mode=='sparse' else array('i')
        self._free=array('I',range(capacity-1,-1,-1)) if mode=='sparse' else array('I')
    @property
    def resident(self): return self._count
    @property
    def epoch(self): return self._epoch
    def _slot(self,v):
        self.index.vertex_code(v)
        return self._vertex_slot[v] if self.mode=='sparse' else (v if self._slot_owner[v]==v else -1)
    def handle(self,v):
        s=self._slot(v)
        if s<0: raise KeyError('NONRESIDENT')
        return StateHandle(self._owner,self._epoch,v,s,self._generation[s])
    def acquire_address(self,address): return self.acquire_vertex(self.index.resolve(address))
    def acquire_vertex(self,v):
        if self._slot(v)>=0: raise ValueError('ALREADY_RESIDENT')
        if self._count>=self.capacity: raise OverflowError('STATE_CAPACITY')
        s=v if self.mode=='dense' else self._free[-1]
        if self._generation[s]==2**64-1: raise OverflowError('GENERATION_EXHAUSTED')
        if self._slot_owner[s]!=-1: raise RuntimeError('CORRUPT_SLOT')
        if self.mode=='sparse':
            self._free.pop(); self._vertex_slot[v]=s
        self._generation[s]+=1; self._slot_owner[s]=v; self._count+=1
        return self.handle(v)
    def _validate(self,h):
        if type(h) is not StateHandle or h.owner is not self._owner or type(h.epoch) is not int or h.epoch!=self._epoch: raise ValueError('FOREIGN_OR_STALE_HANDLE')
        if type(h.slot) is not int or type(h.generation) is not int or not 0<=h.slot<self.capacity: raise ValueError('HANDLE_FIELDS')
        if self._slot(h.vertex)!=h.slot or self._slot_owner[h.slot]!=h.vertex or self._generation[h.slot]!=h.generation: raise ValueError('STALE_HANDLE')
        return h.slot
    def _row(self,row):
        if type(row) not in (tuple,list) or len(row)!=self.width or any(type(x) not in (float,int) for x in row): raise ValueError('ROW_SHAPE_OR_TYPE')
        try: a=array(self._zero.typecode,row)
        except (OverflowError,TypeError,ValueError) as e: raise ValueError('ROW_CONVERSION') from e
        if not all(isfinite(x) for x in a): raise ValueError('NONFINITE_ROW')
        return a
    def read(self,h):
        s=self._validate(h); a=s*self.width
        return tuple(self._values[a:a+self.width])
    def write(self,h,row): self.commit(((h,row),))
    def commit(self,updates):
        """Validate complete bounded batch before changing any state; no concurrent writers."""
        if type(updates) not in (list,tuple) or not 1<=len(updates)<=MAX_BATCH: raise ValueError('BATCH_SIZE')
        staged=[]; seen=set()
        for item in updates:
            if type(item) not in (tuple,list) or len(item)!=2: raise ValueError('UPDATE_SHAPE')
            h,row=item; s=self._validate(h)
            if s in seen: raise ValueError('DUPLICATE_DESTINATION')
            seen.add(s); staged.append((s,self._row(row)))
        for s,row in staged:
            a=s*self.width; self._values[a:a+self.width]=row
    def release(self,h):
        s=self._validate(h); a=s*self.width
        self._values[a:a+self.width]=self._zero
        self._slot_owner[s]=-1
        if self.mode=='sparse':
            self._vertex_slot[h.vertex]=-1; self._free.append(s)
        self._count-=1
    def reset(self):
        # Zero all reserved slots, not just the current working set.
        # Prepare replacement arrays before publication; reset invalidates every handle.
        values=self._zero*self.capacity
        generation=array('Q',[0])*self.capacity
        owners=array('i',[-1])*self.capacity
        slots=array('i',[-1])*self.index.vertices if self.mode=='sparse' else array('i')
        free=array('I',range(self.capacity-1,-1,-1)) if self.mode=='sparse' else array('I')
        self._values,self._generation,self._slot_owner,self._vertex_slot,self._free=values,generation,owners,slots,free
        self._epoch+=1; self._count=0
    def payload(self):
        values=len(self._values)*self._values.itemsize
        # free array length varies; allocator capacity and Python object costs excluded.
        metadata=sum(len(a)*a.itemsize for a in (self._zero,self._generation,self._slot_owner,self._vertex_slot,self._free))
        return dict(reserved_state_bytes=values,resident_state_bytes=self._count*self.width*self._values.itemsize,logical_metadata_bytes=metadata,logical_total_bytes=values+metadata,resident_nodes=self._count,reserved_slots=self.capacity)
