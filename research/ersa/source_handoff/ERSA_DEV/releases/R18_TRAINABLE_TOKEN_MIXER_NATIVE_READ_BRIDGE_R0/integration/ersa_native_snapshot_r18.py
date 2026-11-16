"""R18: authorized read-only bridge from frozen R11 typed C entity slots.

No read_exact transactions are issued. All C entity snapshots are taken with
R11's pinned owner lock, and generation must remain identical. Host supplies
metadata and per-record admission; this bridge cannot grant rights by itself.
"""
from __future__ import annotations
import ctypes as C
import numpy as np
import torch
from integration.ersa_trainable_layer_r18 import ERSAState
from integration.ssc_native_r11 import Entity


def from_native_owner(owner,episode,allowed,*,dtype=torch.float64):
    if dtype not in (torch.float32,torch.float64):raise ValueError('INVALID_DTYPE')
    n=len(episode.keys)
    if not 1<=n<=8192 or np.shape(episode.keys)!=(n,4) or np.shape(episode.goals)!=(n,4) or np.shape(episode.meta)!=(n,4):
        raise ValueError('INVALID_HOST_METADATA')
    mask=np.asarray(allowed)
    if mask.shape!=(n,) or mask.dtype!=np.bool_:
        raise ValueError('AUTHORITY_MASK_REQUIRED')
    with owner._lock:
        pre=owner.generation
        vals=np.zeros((n,4),dtype=np.float64)
        addr_to_idx={1000+i:i for i in range(n) if mask[i]}
        observed=set()
        for slot in range(owner.spec.max_active_entities):
            e=Entity()
            if owner._lib.ersa11_entity(owner._p(),slot,C.byref(e))!=1:
                raise RuntimeError('NATIVE_SNAPSHOT_REJECT')
            if e.used and e.addr in addr_to_idx:
                idx=addr_to_idx[e.addr]
                if idx in observed:raise RuntimeError('DUPLICATE_NATIVE_ENTITY')
                observed.add(idx)
                vals[idx]=tuple(e.v)
        if len(observed)!=len(addr_to_idx):raise RuntimeError('MISSING_ADMITTED_ENTITY')
        if pre!=owner.generation:raise RuntimeError('STALE_NATIVE_GENERATION')
        def t(a):return torch.as_tensor(np.array(a,copy=True),dtype=dtype).unsqueeze(0)
        return ERSAState(t(episode.keys),t(episode.goals),t(vals),t(episode.meta),
                         torch.tensor(np.asarray(episode.issued,dtype=np.int64),dtype=torch.int64).unsqueeze(0),
                         torch.from_numpy(mask.copy()).unsqueeze(0),tick=64,generation=int(pre))
