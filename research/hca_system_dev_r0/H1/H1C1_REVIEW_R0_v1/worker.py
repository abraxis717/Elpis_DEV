#!/usr/bin/env python3
"""One process per engine and split, independently sourced H1C0 engine benchmark.
Read-only: no evidence or source writes. Timing excludes input JSON read and H1C0 imports.
"""
import gc
import json
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path


def main():
    if len(sys.argv)!=4:raise RuntimeError('usage: worker.py <verified-h1c0-dir> <split-path> <replay|incremental>')
    package=Path(sys.argv[1]).resolve(strict=True)
    data_file=Path(sys.argv[2]).resolve(strict=True)
    mode=sys.argv[3]
    if mode not in ('replay','incremental'):raise RuntimeError('unsupported mode')
    sys.path.insert(0,str(package))
    import engines
    ds=json.loads(data_file.read_text())
    cases=[s for w in ds['worlds'] for s in w['cases']]
    fn=getattr(engines,mode)
    times=[];work=None
    for _ in range(3):
        gc.collect()
        t=time.perf_counter_ns()
        aggregates={}
        for scene in cases:
            _,w,_=fn(scene,False)
            for category,num in w.items():aggregates[category]=aggregates.get(category,0)+num
        times.append(time.perf_counter_ns()-t)
        if work is None:work=aggregates
        elif work!=aggregates:raise RuntimeError('work nondeterministic')
    gc.collect()
    tracemalloc.start()
    for scene in cases:
        fn(scene,False)
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ru=resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps({'method':mode,'split':ds['split'],'case_count':len(cases),
                      'timings_ns':times,'median_ns':int(statistics.median(times)),
                      'logical_work':work,'traced_peak_bytes':int(peak),
                      'process_ru_maxrss_kib_linux':ru.ru_maxrss,
                      'memory_scope':'tracemalloc engine-evaluation pass: outputs and intermediates, not full process RSS; static input already parsed, shared interpreter excluded',
                      'wall_time_scope':'engine construction, every update/query, all cases; no gold, audit, input parsing, serialization or model training'},sort_keys=True))
if __name__=='__main__':
    main()
