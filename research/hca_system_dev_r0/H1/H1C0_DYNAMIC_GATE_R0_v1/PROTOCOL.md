<!-- Publication adaptation: local host paths redacted; source SHA-256 in manifest. -->
# HCA H1C0 — Prospective multi-query exact-dynamics gate (DEV only)

**Identity:** `elpis.hca.h1c0.dynamic_gate.dev.v1`. Separate from, and does not repair or reopen, H1A/H1B1/H1B2A/H1B2B. This source is intended to be hashed and executed as an isolated research phase under `[LOCAL_HOST_PATH_REDACTED]`.

## Negative authority carried forward

- H1A audit PASS but train-only symbolic induction solved 160/160 DEV; scientifically nondiscriminating.
- H1B1 source-conditioned synthetic task solved by symbolic baseline: 256/256 TRAIN and 128/128 consumed TUNE.
- H1B2A exact online recomputation/incremental controls PASS and scientific B3 HOLD.
- H1B2B typed factor-node seeds 11021=96/128 TUNE and 11029=128/128 TUNE; B3 HOLD persists.
- Prior files and evidence never rewritten. H1C0 verifies their SHA-256 identities and the canonical tracked Git checkout before writing its own new evidence.

## Exact scientific scope

This is a **pre-B3 task stress test**, not HCA science. It investigates the full-cost efficiency of exact source-aware multi-query dynamic typed-factor inference, specifically whether a strong retained incremental exact engine already dominates the simple full-recompute reference. It does **not** establish or assume a performance advantage for HCA/B3. A closed-form exact algorithm is a legitimate upper-bound competitor. The deterministic task is *not* artificially obscured from the exact solver.

Visible rule: a query `(from,to,scope,context)` is true if and only if there is a live three-factor path of relation types `R0 -> R1 -> R2`. Every factor has explicit `from`, `to`, `scope`, `context` roles and a source/epoch binding. It is live only if the latest event of its source is `(VALID, matching epoch)`. Unknown and stale sources are inactive. Every event prefix is independently scored, including the empty prefix. Every query is evaluated at every prefix. No silent positive admission or semantic truth claim is possible: all outputs are task-specific Boolean predictions.

### Prospective datasets

- TRAIN: seed `0xC0102026`, 2 separate generator worlds, 32 cases, 16 paired interventions, 8 queries per case, 4,032 prefix/query decisions.
- TUNE (consumed diagnostic DEV only): seed `0xC0202026`, 1 other world, 16 cases, 8 paired interventions, 8 queries per case, 2,016 prefix/query decisions.
- Total: 6,048 decisions. No hold-out or sealed DEV generator or test material is included in this bundle. Worlds/cases have disjoint randomized identifiers. The public `target_query_id` identifies the paired counterfactual probe but not its label; polarity is randomized across pair presentation positions.
- Paired interventions: source swaps, event-order reversals, epoch rebinding, and structural rewiring with per-relation role marginals preserved. Final primary probe truth must invert within each pair. Both label polarities are present in each kind.
- Independent scratch oracle recomputes all source states and reachable relation paths for every prefix. Gold is stored separately from public inputs. Mutation-adequacy checks include deliberate gold/public/schema/source/identity/role/timing corruption and pair-nuisance attacks with **gold recomputed**.

### Exact baselines and cost accounting

- `replay`: rebuilds source state from event history and rescans live factors for all 8 queries at *every* prefix, i.e. a valid but intentionally nonincremental baseline.
- `incremental`: builds a complete factor-to-source and path-to-factor dependency index once, maintains a nonnegative count of live three-hop witnesses per query, invalidates/revalidates dependent witnesses on status changes, and returns all query answers after every update. Counts precomputation, repeated updates and all queries, with no uncounted free warm-up.
- Independent scratch oracle + both engines MUST agree at every scored prefix; any difference is `MECHANICS_HOLD`, not a model failure.
- Three end-to-end in-process CPU wall-time replicates per engine per split, alternating order, summarize median; includes engine construction and all updates/queries, excludes dataset generation, audit and evidence serialization. During timed runs only, payload-size serialization instrumentation is disabled **equally** for both engines.
- Logical-work category counters and serialized retained-state payload-size proxies are provided. Work categories are not calibrated FLOPs/CPU instructions and must not be directly equated across engines. Replay's small transient payload is not a full-process memory estimate; incremental stored path witnesses may be substantially larger. Process `ru_maxrss` is only a shared descriptive ceiling, not separate engine RSS.
- Zero-answer control included; all intermediate predictions and denominators preserved. The exact incremental engine is **task-specific**, not native B0I/ECS_C. This is not a native-runtime benchmark.

### Predeclared disposition

- On clean verification and full parity: `PASS_H1C0_TASK_EXACT_COMPARATOR_GATE_DEV_ONLY` with science `HOLD_B3_NO_HCA_EFFECT_MEASURED_EXACT_SOLUTION` regardless of whether either exact method is faster. No HCA is run here, so its advantage cannot be inferred.
- Any source/evidence drift or parity violation: `MECHANICS_HOLD`, never scientific nonpass.
- The only possible immediate next step is independent H1C1 observer/red-team review and formal definition of a **fair** resource-limited prediction/repair hypothesis. It must include full setup, incremental index storage, amortization across query counts, hard correctness and invalid-source safety. Do not authorize B3 merely because replay is slower than retained exact inference. No SEALED, QUAL, native HCA wiring, production admission, or external network.

## Nonclaims

This instrument does not demonstrate multi-hop learning, predictive generalization, trained factor-node matching on the new workload, damaged-state recovery, temporal ambiguity handling, late-event reconciliation, native HCA effect, learned recurrent B3, or formal scientific qualification. It does not support promoting HCA into RuntimeCore, ECS_C authority, memory learning, semantic inference or production.
