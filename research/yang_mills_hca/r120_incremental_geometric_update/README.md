# ELPIS_YM_AWR_R120 — incremental geometric updates on the Yang–Mills/AWR substrate

Research component (additive, not production-admitted). It lets admitted observations modify the
**existing** relation states of the R118 sparse AWR substrate in place, with a frozen CGC cubic
operator supplying bounded invariant coefficients, and propagates the change through the frozen
88-parameter HCA on the affected windows only — in one native call.

> The ledger's earlier "R120" is the unrelated checkpoint-selection study in
> `../r120_checkpoint_policy/`. This component does not touch it.

| File | Role |
|---|---|
| `SPEC.md` | update law, gauge argument, event/batch contract, propagation, non-claims |
| `PROTOCOL_R120U.json` | criteria, tolerances, benchmark and utility-study design, frozen before the registered runs |
| `csrc/awr_kernel_r120.c` | fused kernel; `#include`s the pinned R119 kernel and calls its `instance_pass` |
| `native_r120.py` | builds the kernel in volatile scratch (R119 flags + OpenMP), ctypes binding |
| `engine_r120.py` | `IncrementalState` (R118 state + side tables), `IncrementalEngine` |
| `cgc_bridge.py` | CGC weight contract, synthetic weights, IEEE-single reference forward |
| `reference.py` | independent pure-Python oracle of the whole contract |
| `fixtures_r120.py`, `tests/` | fixtures and unit tests |
| `qualify.py` | R120D adversarial qualification |
| `bench.py` | R120E performance measurement |
| `science.py` | R120F exploratory utility study |
| `RESULTS.md`, `evidence/` | measured results and raw evidence |

## Use

```python
from research.yang_mills_hca.r120_incremental_geometric_update import engine_r120 as E

state  = E.IncrementalState(registry, capacity)            # R118 AWRState + side tables
engine = E.IncrementalEngine(state, "54007/STATE8_SIGNED8", controller="R114")
engine.build_cache()                                        # optional, volatile, bounded
report = engine.apply(rel_ids, seq, participants, obs, weight, aux)   # update + propagate
report.disposition, report.applied, report.instances, report.pending_rows
```

`apply(..., propagate=False)` commits the updates and defers propagation
(`COMMITTED_DEFERRED`); `engine.propagate_pending()` runs it. `engine.step(handles)` is the
plain R119-semantics step through the same kernel.

## Reproduce

From the repository root (numpy and gcc; torch only for one optional cross-check):

```bash
python -B -m unittest research.yang_mills_hca.r120_incremental_geometric_update.tests.test_r120_update
```

```bash
python -B -m research.yang_mills_hca.r120_incremental_geometric_update.qualify --out /dev/shm/r120_qualification.json
```

```bash
python -B -m research.yang_mills_hca.r120_incremental_geometric_update.bench --all --out /dev/shm/r120_bench.json
```

```bash
python -B -m research.yang_mills_hca.r120_incremental_geometric_update.science --out /dev/shm/r120_science.json
```

Runners write only under `/dev/shm`; the committed copies are in `evidence/`.
