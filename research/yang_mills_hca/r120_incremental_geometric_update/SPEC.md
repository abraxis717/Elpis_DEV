# ELPIS_YM_AWR_R120 — incremental geometric updates: specification

Status: research component, additive, not production-admitted. Frozen before the registered
qualification, benchmark and utility runs together with `PROTOCOL_R120U.json`.

> **Naming.** The result ledger already uses "R120" for the unrelated checkpoint-selection
> study in `../r120_checkpoint_policy/` (untouched). This phase is `ELPIS_YM_AWR_R120`; its
> component is `r120_incremental_geometric_update/`.

Principle: **update existing geometric representations, not the computational architecture.**

## 1. What exists and what an update may touch

| Layer | Owner | Mutated by an update? |
|---|---|---|
| Relation identity (64-bit ID, CSR incidence, frames `U[r,p]`) | R118 `Registry`, immutable | never |
| Evidence authority (which observations are admitted) | the caller, upstream | never — the engine neither admits nor infers evidence |
| HCA88 operator (88 parameters, six R77/R78 checkpoints) | authenticated, read-only array | never |
| CGC operator (120 parameters, `float32 [3,5,8]`) | engine, read-only array | never |
| Materialization (pool row, bindings, generations) | R118 `AWRState` | never |
| **Learned geometric state** `A[4], B[4]` of a materialized relation | pool row | **yes — in place** |
| Event sequence of that row (`row_seq`), deferred-propagation marks | R120 side tables | yes |

`IncrementalState` is the unmodified R118 `AWRState` plus five preallocated side tables
(`row_gen`, `row_seq`, `row_pending`, `pending_list`, `inst_mark`). No relation, instance,
operator, HCA checkpoint or CGC parameter set is created, replaced or selected by an event.

## 2. Update law

An event carries an observation `o ∈ R^4` expressed in the frame of an incident participant
`p` of relation `r`, a signed evidence weight `w ∈ [-1, 1]` and an auxiliary scalar
`aux ∈ [0, 1]`.

```
v  = w · U[r,p] o                                   transport into the relation frame
x  = (tanh⟨A,v⟩, tanh⟨B,v⟩, tanh⟨A,B⟩, tanh|v−A|², aux)      five invariants, cast to float32
(a, b, c) = κ · tanh(CGC3(x))                       CGC3 frozen, |a|,|b|,|c| ≤ κ = 0.25
A' = clamp4(A + η (a A + b B) + γ v)
B' = clamp4(B + η (c A − a B) + γ ρ v)
```

`clamp4` is the HCA's own radial clamp (norm ≤ 4). Defaults `η = 1`, `γ = 0.25`,
`ρ = 0.63`. `CGC3` is the Q4E5 cubic operator of `research/cgc_lao/reference/src/cgc/lao_q4e5.py`
(`out_o = 0.5 Σ_n (z + z² + z³)`, `z = Σ_d x_d w[o,d,n]`), evaluated in IEEE single precision
in a fixed order.

Why this form.

* **No padding or reshaping.** The three CGC outputs are not written into the 8-dimensional
  state. They are *scalar coefficients* of a linear map acting on the existing vectors `A`, `B`.
* **Gauge covariance.** Under a relation gauge `g ∈ SO(2)×SO(2)`: `A,B → gA, gB` and
  `U[r,p] → g U[r,p]`, hence `v → g v`. Every entry of `x` is an inner product of covariant
  vectors, so `x`, and therefore `(a, b, c)`, are invariant. The update is a linear combination
  of covariant vectors with invariant coefficients, and `clamp4` is radial, so `A', B' → gA', gB'`.
  Under a participant gauge `h`: `U[r,p] → U[r,p] hᵀ` and the participant's observation becomes
  `h o`, so `v` is unchanged. An update that used `o` without `U[r,p]` is *not* covariant; this
  is the registered negative control.
* **A–B coupling.** `(a, b; c, −a)` is the traceless family proposed in the task. It can rotate
  and shear the `(A, B)` pair and trade magnitude between the channels; it cannot scale both up.
* **Evidence enters through `γ v`.** With zero CGC weights the law reduces to transported
  additive injection; CGC modulates the state around it. This keeps the evidence path independent
  of whether the CGC weights are meaningful.
* **Signed evidence and reversal.** `w < 0` flips the transported observation. A flagged
  reversal (`w = −1` on a sign-reversed observation) is therefore identical to the unreversed
  observation. Unflagged contradictions are not detectable by the law itself.
* **Boundedness.** Coefficients are bounded by `κ`, the state by `clamp4`; any finite
  observation gives a finite state of norm ≤ 4 per channel. Non-finite results refuse the batch.
* **Zero and subnormal states.** `A = B = 0`, `o = 0` stays exactly zero (`x = 0`, `CGC3(0) = 0`).
  `A = B = 0` with evidence gives exactly `A' = γ v`, `B' = γρ v`. No flush-to-zero or
  denormals-are-zero mode is used; subnormal inputs follow IEEE arithmetic and are verified
  bitwise against the reference.
* **Compatibility with HCA88 transport.** The updated row is an ordinary relation state. The
  frozen HCA pass that follows reads it through the same links `L = U[d,p] U[s,p]ᵀ` as before.

This is a *mathematically* admissible rule. Whether it is a *useful learning rule* is a separate,
empirical question (§7); with synthetic weights no semantic claim is made.

## 3. Event and batch contract

Event: `(relation_id u64, seq u64, expected_generation u32 | 0, participant index | −1,
observation[4], weight, aux)`.

* `seq` is a per-relation monotonic sequence. An event with `seq ≤` the last applied sequence of
  its relation is a **duplicate**: skipped and counted (idempotent redelivery).
* Two events of one batch with the same `(relation, seq)` and different payloads are a
  **conflict**: the whole batch is refused.
* Events of one relation apply in batch order; events of different relations commute. A batch
  with repeated relations therefore equals the same events delivered one by one, followed by a
  single propagation (coalescing preserves semantics because propagation reads only committed
  rows).
* `expected_generation` (from `IncrementalState.relation_ref`) guards against **stale
  identities**: a row generation is bumped whenever the row is freed, so a reference captured
  before release and re-materialization is refused.
* A participant index of −1 means "no observation" and requires `o = 0`.
* **Atomicity.** Everything is computed in preallocated compact scratch. The pool, sequences and
  the pending list are written only after every check and every propagation budget has passed.
  A refusal leaves all of them, and all scratch marks, bit-identical.

Refusals (first failure decides; per event in batch order): `REFUSED_EVENT_BUDGET`,
`REFUSED_ARGUMENT`, `REFUSED_INVALID_VALUE` (non-finite, `|w| > 1`, `aux ∉ [0,1]`),
`REFUSED_UNKNOWN_RELATION`, `REFUSED_NOT_MATERIALIZED`, `REFUSED_STALE_RELATION`,
`REFUSED_NOT_INCIDENT`, `REFUSED_CONFLICT`, `REFUSED_ROW_BUDGET`, `REFUSED_NONFINITE`,
`REFUSED_INSTANCE_BUDGET`, `REFUSED_LINK_BUDGET`, `REFUSED_KERNEL`, `REFUSED_PENDING_CAPACITY`.

## 4. Selective propagation

The affected windows of a batch are found through the existing R118 binding lists
(`row_head` / `bind_next`): level 1 is every live instance bound to an updated row; each
further level (radius ≤ 2) adds the instances bound to rows introduced by the previous level.
The cost is proportional to the affected bindings; the registry is never scanned.

The selected windows then run `passes` synchronous **Jacobi** passes of the frozen R119
`EXACT_GUARDED` `instance_pass` (the R120 translation unit `#include`s the pinned R119 source and
calls its static function; the R119 file is hash-checked at build time), followed by MEAN (or
EXCLUSIVE) reconciliation, exactly as an R119 step over the same handles. Rows of affected
windows that were not updated take part as ordinary neighbours.

*Deterministic reconciliation.* `reconcile120` produces the same bits as the frozen R119
`reconcile`: ascending summation and one division per coordinate. It differs only in the sorting
algorithm for more than 24 candidates and in allowing the independent rows to run in parallel.

*Exact parallel Jacobi (opt-in, `Limits.threads > 1`).* All windows of a pass read the same input
buffer and write disjoint candidate blocks, so the pass is parallelized with OpenMP without
changing a single bit. Single-thread and parallel timings are always reported separately.

*Window cache (volatile, bounded).* Per instance a contiguous block holds the window's member
lists and frames, so the pass does not chase registry rows. Instances that do not fit stay on the
registry path with identical results. The cache is dropped for an instance when it is released.

*Deferral.* `propagate=False`, or a propagation budget overflow with `defer_on_budget=True`,
commits the updates and records every touched row in the pending list; the call reports
`COMMITTED_DEFERRED` and the number of pending rows. `propagate_pending` executes it later and
equals immediate propagation when no other update intervened. Required propagation is never
discarded silently; a full pending list refuses the batch.

## 5. Complexity and memory

Per call: `O(E log R)` relation lookup (binary search over the sorted ID table) + `O(E)` CGC and
update + `O(affected bindings)` selection + `O(links)` HCA + `O(affected slots)` reconciliation.
No allocation in the kernel. The Python wrapper copies the event arrays into preallocated
buffers and returns a 144-byte report snapshot. Storage is fixed at construction: side tables
17 bytes per row capacity + 1 byte per instance capacity, engine scratch ≈ 1 MB at the default
limits, window cache as configured.

## 6. Correctness criteria (frozen in the protocol)

Bitwise: native law = independent reference; fused update-and-propagate = independent reference
(dictionary model, brute-force window discovery, R118 pure-Python HCA); fused step = R119
`EXACT_GUARDED` step; fused call = reference update followed by an R119 step; serial = parallel;
replay = first run. Covariance ≤ 1e-12 end to end with negative controls ≥ 1e-3. All pins
unchanged: frozen manifest 39/39, six checkpoint hashes, R118 component digest, R119 kernel hash,
CGC reference hash. R93 stays `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.

## 7. What is not claimed

* The CGC weights are **synthetic** (or, in the utility study, picked by random search). The
  repository contains no trained relational CGC. The bridge is qualified mechanically; its
  semantic usefulness is measured, not assumed.
* The latency reference is the R119 step on the same window sets in the same process, not a
  re-run of the R119 registered benchmark.
* Nothing here changes R77/R78, R93, R114–R119, the earlier R120–R122 records, the CGC/LAO
  records or the public RuntimeCore.
