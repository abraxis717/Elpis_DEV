# HCA_REVISION_R0: does retained local HCA dynamics help re-establish a readout after an ECS_C source revision?

Classification: **DEV PROTOCOL, NOT QUAL.** Predeclared before any measurement and committed before the bench
runs. HCA stays research-only and its readout stays `SALIENCE_ONLY`. B3 (learned HCA) is **not authorized** and is not
trained here. The published H1A–H1D0 record (`research/hca_system_dev_r0`) is not reopened or relabelled.

## 1. Why this experiment, and why only this one

H1D0 closed with `NO_HCA_ADVANTAGE` on a synthetic source-evolution task that a TRAIN-only symbolic rule solved. This
experiment does not build another such task. It asks a narrower question at the one interface that now exists in
code, ECS_C's admitted relational view → HCA:

> When an ECS_C source revision changes a small part of the live relational view, can any native HCA comparator
> (B0, B1, B2) re-establish a correct downstream readout at a lower full cost than an equally capable exact
> incremental method, or give a more useful partial readout under a bounded cost?

Its prior is negative, for two reasons that the experiment checks rather than assumes:

* The native HCA graph is immutable (`docs/ECS_C_HCA.md`): every revision is a new graph.
* A B2 activation is owner-bound to one graph's evidence identity and one exact task, so it cannot be carried
  across a revision.

If both hold, retained B2 activation is a cache of one propagation, not memory that survives the change the workload
is about.

## 2. The workload

**Interface.** `ECS_C relational view (admitted evidence) → elpis_ecs_c_rel_export → research adapter → HCA graph +
task → SALIENCE_ONLY readout`. It is one-way: nothing from any HCA output is written to the view, the authority or the
memory.

* **Evidence.** Binary typed relations (type 5, roles 1 → 2) over E entities, admitted into the ECS_C view under the
  bindings of S = 16 sources (synthetic authority rows of the real `elpis_ecs_c_entity` type). Each relation has one
  source; with probability 0.3 it also has a second, distinct source.
* **Downstream readout.** The HCA message law on the live relations, with every factor at weight 900 and four seed
  entities (salience 1000): `score(v) = max over live u→v of floor(900 · score(u) / 1000)`. Its exact value is HCA B0's
  closure on the graph built from the post-event export (the **reference**). A method is *exact* on an event when its
  score for every cell equals the reference.
* **Events.** Each event revises one source (`SOURCE_VERSION`, which retires all of that source's evidence in the
  view). It then re-admits 50% of that source's relations under the new version, and admits fresh relations (drawn
  from a pool and admitted by other sources) until the live count is back to its pre-event size. Every event therefore
  deletes and inserts relations.

**Regimes**

* **R-RANDOM.** E ∈ {64, 128, 256} cells; relations F = min(2E, 448) (HCA limit: 512 factors, 256 cells, degree 32).
  Three seeds per scale, 40 events per seed.
* **R-LOCAL.** A locality sweep. One designated source supports exactly 8 relations next to the seeds. A growing
  unrelated body, F_u ∈ {32, 64, 128, 256, 440}, is admitted by other sources and reachable from the seeds. The 40
  events alternately revise the designated source and re-admit its 8 relations, so every event changes exactly
  8 relations whatever F_u is.

## 3. Methods (full cost: everything a method needs after the event, from retained state)

| Method | What it does after the event |
|---|---|
| M-INC (exact incremental) | View sync (dependency-indexed) and the event's admissions. Then the relation delta: `elpis_ecs_c_rel_dependents` of the revised source before the event, re-read after it, plus the fresh admissions. Then an exact decremental/incremental closure update over a retained adjacency: over-delete the scores supported through deleted edges, re-derive them from intact neighbours, propagate the insertions (delete-and-rederive). |
| M-FULL (exact recomputation) | `elpis_ecs_c_rel_rebuild` (all derived state), `elpis_ecs_c_rel_export`, and an exact closure recomputed from scratch over the export. |
| M-B0 | Export, adapter (cells and factors in HCA canonical order), `elpis_hca_graph_create`, `elpis_hca_predict` B0. |
| M-B1 | As M-B0, then `elpis_hca_predict` B1 with steps = 32 (the graph's round limit). |
| M-B2 | As M-B0, then `elpis_hca_b2_start`, k × `elpis_hca_b2_warm`, and one `elpis_hca_b2_observe`. k is the fewest ticks that make the activation equal the reference (found by the evaluation instrument; the instrument's own comparisons are not charged). |
| M-B2-CARRY | Continue the B2 activation retained before the event on the post-event graph's session (`elpis_hca_b2_warm` with the old activation). |
| M-STALE | Keep the previous exact readout (zero cost). This is the baseline any partial readout must beat. |

Costs are reported separately and never pooled:
* **Wall time.** Median of 5 replicate runs of the whole event sequence, one thread, Release build, same process.
* **Logical work.** Each library's own counters: HCA's records, visits, copies and hashed bytes; the view's rows,
  probes, contributions, relations and recomputes; the closure's edge and node visits. The units differ by design and
  are not compared across libraries.
* **Retained bytes.** Each method's state between events: the view's arena, the closure's arrays, HCA's graph and
  session blocks (measured by allocator wrapping).

## 4. Hypotheses and predictions (fixed now)

* **H1 (exact locality).** In R-LOCAL, M-INC's per-event logical work does not grow with F_u. *Predicted: holds.*
* **H2 (HCA locality).** In R-LOCAL, the per-event logical work of M-B0, M-B1 and M-B2 grows with F_u, although
  every event changes 8 relations. *Predicted: holds (every event rebuilds the graph).*
* **H3 (carry-over).** M-B2-CARRY is accepted by the native substrate. *Predicted: refused `INVALID_STATE` on every
  event, because the activation's owner names the old graph.*
* **H4 (bounded-cost partial readout).** There is some tick count k below M-B2's convergence at which B2's k-tick
  readout is more accurate than M-STALE's, at less wall time than M-INC's exact update. Accuracy is the fraction of
  cells with the exact score. *Predicted: no.*
* **Mechanics gates.** M-INC and M-FULL are exact on 100% of events, and B0 equals an independent exact closure.
  Otherwise the result is **MECHANICS HOLD** and no comparison is reported as a finding.

## 5. Disposition rule (fixed now)

* If any mechanics gate fails: `MECHANICS_HOLD`.
* Else if H4 holds in any regime, or any HCA method is exact on every event of a regime at a lower median wall time
  than M-INC: `PROMISING_DEV_OBSERVATION`. That would need its own protocol; it qualifies nothing.
* Else: `NO_HCA_ADVANTAGE_ON_ECSC_REVISION_WORKLOAD`.

Whatever the outcome, B3 remains `NOT_AUTHORIZED`.

## 6. Not claimed

No QUAL, no semantic truth from salience, no general performance guarantee (DEV measurements on this host and
workload only), no statement about HCA architectures other than the native B0/B1/B2. The 4-seed max–min decay closure
is a mechanical downstream readout, not cognition.
