# RuntimeCore live/durable separation: matched before/after timings

PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM. DESCRIPTIVE: single-host observations of the managed HOT lane before and
after the live/durable separation (docs/RUNTIME_CORE.md, "Live and durable authority"). They qualify nothing and
are not evidence of model-level inference throughput. The fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE.

## Identity of what was measured

| | |
|---|---|
| before | `81b738e` (RuntimeCore ABI v3: every managed LEARN publishes continuity; with a store, checkpoints first) |
| after | the Elpis_DEV correction branch (RuntimeCore ABI v4: volatile commits; durability explicit) |
| build | both trees configured identically: gcc 13.3.0, `CMAKE_BUILD_TYPE=Release`, warnings as errors |
| host | Intel Xeon @ 2.80 GHz, 4 vCPUs, Linux 6.18 (shared VM), glibc 2.39, ext4, CPython 3.12.3 |
| driver | the appendix below, run against each tree's own sources, tests helpers and build |

Method: one process per (tree, mode, width); an anchored managed lineage over a standalone K1 state (dim 6); 50
warm-up operations, then 500 individually timed operations through the public `Runtime` facade
(`run_query` / `run_learn`); three rounds per configuration, trees interleaved within each round. Width 1024 LEARN
uses the fixture learning rate scaled by 36/width (both trees alike): at the unscaled rate both trees refuse with
`K1 NONFINITE` after a few hundred LEARNs, identically. The width-1024 QUERY rows were measured with the driver's earlier revision, whose only
difference is the initial weight scale (0.18 instead of 0.18 x sqrt(36/1024)) and the unscaled rate, which a QUERY
does not use; a query's cost does not depend on the weights' values.

## Results (microseconds; median of each round, then the median of the three rounds' p90 / p99)

| width (envelope) | operation | before: per-round medians | after: per-round medians | before p90 / p99 | after p90 / p99 |
|---|---|---|---|---|---|
| 36 (30,376 B) | QUERY | 211, 228, 230 | 80, 81, 76 | 391 / 583 | 116 / 208 |
| 36 | LEARN, no checkpoint store | 1181, 1282, 1149 | 188, 191, 199 | 1708 / 2683 | 243 / 405 |
| 36 | LEARN, checkpoint store attached | 2282, 2121, 2151 | 187, 318, 189 | 2851 / 3539 | 235 / 533 |
| 36 | after only: LEARN + explicit durable publication, waited (no store) | - | 1071, 1187, 1050 | - | 1425 / 3266 |
| 1024 (77,800 B) | QUERY | 459, 466, 453 | 89, 89, 90 | 537 / 707 | 131 / 239 |
| 1024 | LEARN, no checkpoint store | 3807, 3940, 3938 | 2012, 2057, 2156 | 4442 / 5790 | 2762 / 3945 |
| 1024 | LEARN, checkpoint store attached | 6153, 5988, 6222 | 1985, 2022, 2090 | 8387 / 10666 | 2335 / 3101 |
| 1024 | after only: LEARN + explicit durable publication, waited (no store) | - | 3909, 3982, 3762 | - | 4769 / 6407 |

Reading. Before, every managed LEARN paid one continuity record write and `fdatasync` (and, with a store, an
envelope copy, a slot write and a second `fdatasync`), and QUERY and every begin hashed the state. After, a LEARN is
the K1 transaction alone (the width-1024 LEARN is dominated by the K1 schedule itself), with or without a store, and
QUERY hashes nothing. The "LEARN + explicit durable publication, waited" rows show that durability was not made
cheaper: a caller that chooses to wait for an acknowledgment after every LEARN pays about what every LEARN used to
pay. What changed is who decides when to pay it. The rounds include one outlier (after, width 36, store attached,
round 2: 318), consistent with a shared VM.

## Appendix: the driver

The driver predates the split of durable publication into two named kinds (docs/RUNTIME_CORE.md, "Recoverable and
identity-only publication"). Its `publish_durable` ran without a checkpoint store, which is today's
`publish_identity` (identical lane work); it is kept as run. A recoverable publication now also reads its slot back
before continuity names it (one slot-sized `pread` per publication); that path was not part of these measurements.

```python
"""Matched HOT-lane benchmark: the same managed QUERY / LEARN through the Python Runtime facade, on one tree.

Usage: python perf.py TREE MODE WIDTH N    (ELPIS_NATIVE_BUILD must point at TREE's build)
MODE: query | learn | learn-store | learn-publish (HEAD only: an explicit durable publication after each LEARN)
Prints one JSON line: per-operation latency statistics in microseconds (descriptive, single host).
"""
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

tree, mode, width, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, str(tree))
sys.path.insert(0, str(tree / "src"))

import numpy as np  # noqa: E402

from tests.conftest import admit_k1_set, library_id_of, native_authority, require_runtime_library  # noqa: E402
from tests.integration._turn_fixtures import LEARN, RATE, ByteTokens, FixtureMap, admitted  # noqa: E402
from tests.integration.test_codec_ecs_turn import _config  # noqa: E402
from elpis.ECS.k1 import K1Library, K1State  # noqa: E402
from elpis.runtime import Runtime  # noqa: E402
from elpis.runtime.cognition import LearnAuthority, LearnRequest, QueryRequest  # noqa: E402
from elpis.runtime.core import RuntimeLibrary  # noqa: E402
from elpis.runtime.recovery import provision_k1_checkpoint  # noqa: E402

DIM = 6
k1 = K1Library(admit_k1_set()[1].lib)
w = np.random.default_rng(36).normal(0.0, 0.18 * (36 / width) ** 0.5, size=(DIM, width))   # same scale as width 36
# The fixture rate, scaled with width so a wide state stays finite over hundreds of LEARNs (both trees alike).
authority = LEARN if width == 36 else LearnAuthority(RATE * 36 / width, LEARN.grant)
learn = LearnRequest("bench", ByteTokens(), admitted(FixtureMap()), authority)
query = QueryRequest("bench", ByteTokens(), admitted(FixtureMap()))

with tempfile.TemporaryDirectory(dir=os.environ.get("BENCH_TMP")) as tmp:
    tmp = Path(tmp)
    config = _config(tmp / "continuity")
    if mode == "learn-store":
        path = require_runtime_library()
        library = RuntimeLibrary.admit(path.parent, path.name, native_authority(), library_id_of(path))
        provision_k1_checkpoint(library, tmp / "checkpoint", k1.envelope_bytes(DIM, width))
        config = replace(config, k1_checkpoint_dir=tmp / "checkpoint")
    with K1State.create(k1, DIM, width, memoryview(np.ascontiguousarray(w).reshape(-1)), max_rows=8) as state, \
            Runtime(config) as runtime:
        runtime.anchor_cognition(state)
        if mode == "query":
            op = lambda: runtime.run_query(state, query)
        elif mode == "learn-publish":
            def op():
                runtime.run_learn(state, learn)
                assert runtime.publish_durable(state).acknowledged
        else:
            op = lambda: runtime.run_learn(state, learn)
        for _ in range(50):
            op()
        samples = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            op()
            samples.append((time.perf_counter_ns() - t0) / 1000.0)
samples.sort()
print(json.dumps({"mode": mode, "width": width, "n": n, "median_us": round(statistics.median(samples), 1),
                  "p90_us": round(samples[int(0.9 * n)], 1), "p99_us": round(samples[int(0.99 * n)], 1),
                  "mean_us": round(statistics.fmean(samples), 1),
                  "envelope_bytes": k1.envelope_bytes(DIM, width)}))
```
