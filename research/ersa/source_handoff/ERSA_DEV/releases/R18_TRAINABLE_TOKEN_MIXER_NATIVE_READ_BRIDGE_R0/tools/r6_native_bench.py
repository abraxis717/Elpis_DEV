#!/usr/bin/env python3
"""Only kernel benchmarks; precomputed canonical scores, not full ERSA search."""
import json,random,time,statistics,shutil
from pathlib import Path
import numpy as np
from ersa_native import build_native,NativeOps
from elpis_ersa.geometry_ersa import aggregate_signed

def median_us(fn, n=31):
    a=[]
    for _ in range(n):
        t=time.perf_counter_ns()
        fn()
        a.append((time.perf_counter_ns()-t)/1000.)
    return round(statistics.median(a),2)

def main():
    lib=build_native()
    try:
        ops=NativeOps(lib); rng=np.random.default_rng(7013)
        n=8192;k=8
        s=np.array([round(float(x),12) for x in rng.normal(size=n)],dtype=np.float64)
        ids=np.asarray(rng.permutation(n)+5000,dtype=np.int64)
        active=np.ones(n,dtype=np.uint8)
        v=rng.normal(size=(n,4));w=rng.normal(size=n)
        expected=sorted(range(n),key=lambda i:(-s[i],int(ids[i])))[:k]
        assert ops.rank(s,ids,active,k)==expected
        native=median_us(lambda:ops.rank(s,ids,active,k))
        python=median_us(lambda:sorted(range(n),key=lambda i:(-s[i],int(ids[i])))[:k])
        g_native=median_us(lambda:ops.aggregate(v,w),n=21)
        g_python=median_us(lambda:aggregate_signed([tuple(x) for x in v],list(w)),n=11)
        data={'count':n,'k':k,'rank_native_us':native,'rank_python_us':python,
              'reduce_native_us':g_native,'reduce_python_us':g_python,
              'scope':'KERNEL_ONLY_PRECOMPUTED_SCORES_NOT_END_TO_END_R5',
              'data_write':'NONE'}
        print('R6_NATIVE_KERNEL_BENCH='+json.dumps(data,sort_keys=True,separators=(',',':')))
        return data
    finally: shutil.rmtree(lib.parent)
if __name__=='__main__':main()
