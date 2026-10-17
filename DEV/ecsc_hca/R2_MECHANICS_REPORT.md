# ECS_C HCA R2 — current-source mechanical closure

**Disposition:** `DEV_MECHANICS_REPRODUCED_WITH_HISTORICAL_HASH_DRIFT`.
**Source:** `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`. **Scientific qualification:** `NOT RUN`.

40/40 saved event outputs reproduce under current source at strict JSON-wire semantics,
including every nontiming prediction, status, source-validity field, refusal and operation count.
The prior local tests report `UNITTEST_AND_CANONICAL_SELECTED_PASS`.

Three files (`r2_run.py, r2_trajectories.py, test_r2_mechanics.py`) changed since `R2_DEV_RESULTS.json` recorded its
source metadata. The original file remains immutable. This report and
`R2_CURRENT_SOURCE_CLOSURE.json` constitute an additional current-source
reproduction record, **not a retroactive validation of the historical source state**.

| Method | Completed/40 | Instrumented logical work on completed calls | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| B0 | 38/40 | 9834 | 130 | 0 | 10 |
| B1 | 38/40 | 16898 | 124 | 0 | 16 |
| B0I | 38/40 | 13288 | 130 | 0 | 10 |
| B2_cold | 38/40 | 365176 | 120 | 0 | 20 |
| B2S | 36/40 | 253844 | 128 | 0 | 10 |

B0I is the preferred **research candidate** for a WARM exact-context adapter.
B2S demonstrates bounded activation retention under supported source changes,
not a better contextual-retrieval result. Its extra scope refusals must remain
visible. Unknown/invalid sources never become factual authority through salience.

No HOT integration, no HCA science, no training, no durable store, no PR or merge
is authorized by this closure. The R3 prospective protocol is still a draft.
