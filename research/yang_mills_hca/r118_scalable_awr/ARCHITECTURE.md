# R118 architecture: sparse AWR substrate over the frozen HCA

## 1. Units and hot path

```
active selection -> sparse gather -> gauge transport -> shared frozen HCA -> controller -> reconciliation -> volatile AWR
   (handles)        (relation rows)   (U_d U_s^T per     (88 params,         (32-coord     (order-independent
                                       window link)       one copy)           per instance)  MEAN | EXCLUSIVE)
```

* **Local arity is preserved.** The frozen operator processes exactly four relation
  slots (`range(4)` in `graph_step_split`, `reshape(..., 4, ...)` in the stacked
  gates) and the controller couples them through one radial scalar. R118 keeps the
  4-slot window as the unit of HCA invocation and composes larger networks from many
  overlapping windows. The decomposition seam is reconciliation (§4); it is qualified
  for correctness and covariance (R118-C) and characterized scientifically (R118-E).
* **Weights are shared**, never copied per instance: one immutable `W[8,10]`, `b[8]`
  per engine, loaded from the authenticated R77 record and pinned by hash.
* **State lives on relations.** A relation has volatile state `(A, B)` only while an
  instance binds it. An instance's 32-coordinate state is the view of its slots.
  Instances with disjoint relation sets are isolated; sharing a relation is the
  explicit coupling.

## 2. Registry (immutable, O(P + R + M))

64-bit participant and relation IDs (sorted arrays; vectorized lookup with refusal of
unknown IDs), CSR relation → members (participant index ascending), per-incidence
frame as two row-major 2×2 blocks (exact for historical frames, which are *not*
exactly `[[c,-s],[s,c]]` after floating-point products), relation index per
incidence, reverse CSR participant → incidences. Contracts enforced at build:
arity ∈ [1, max_arity ≤ 64], participant degree ≤ limit, unique IDs and members,
finite SO(2)×SO(2) frames (orthonormality and det within 1e-12). No dense P×P or R×R
structure exists anywhere. Bytes: `8P + 8R + 8(R+1) + 4M + 64M + 4M + 8(P+1) + 4M`.

Window links are derived, never stored: for dst slot `d`, src slot `s ≠ d`, every
participant `p` shared by both relations (ascending) contributes
`L = U[r_d,p] U[r_s,p]ᵀ`, the exact `link_builder` order.

## 3. Volatile state and addressing

`AWRState` preallocates everything: instance tables (ID, generation, slot → relation,
slot → row, link and membership counts), a relation-state pool of
`max_materialized_relations × 8` float64, a relation → row index (int32 per
registered relation), and an O(bindings) doubly linked reverse index
(`row_head`, `bind_next`, `bind_prev` over binding codes `handle·4 + slot`).
Instance creation is all-or-nothing (bulk, vectorized) with typed refusals:
`DUPLICATE_INSTANCE_ID, EMPTY_WINDOW, UNKNOWN_RELATION,
DUPLICATE_RELATION_IN_WINDOW, BINDING_CAPACITY, STATE_ROW_CAPACITY,
INSTANCE_CAPACITY`. Handles carry generations; stale or dead handles are refused.
Nothing touches the filesystem.

## 4. Step semantics

* **Synchronous (Jacobi) passes**: all active instances read the same pre-pass
  relation states.
* **Reconciliation MEAN**: a relation reached by `m` active (instance, slot)
  bindings receives the mean of its `m` candidates; per coordinate the candidates
  are sorted ascending and summed in that order. The result is a function of the
  multiset, hence bitwise independent of the active-set order and of duplicates in
  the request. **EXCLUSIVE** refuses overlapping active windows.
* **Transactional**: the native call works on a compact copy of the touched rows and
  writes the pool only if every pass succeeded and every candidate is finite.
* **Budgets** (checked before any work): active instances, active relations,
  memberships, links per pass, passes, work units (= passes × (links + bound slots)),
  participant-expansion incidences, engine scratch bytes. Exhaustion returns
  `REFUSED_<BUDGET>` with `exact = False` and leaves the AWR unchanged. Participant
  selection beyond budget is refused, or, only if explicitly allowed, returned as
  `PARTIAL` with the exact omitted count.
* **Native kernel**: C11, `-O2 -ffp-contract=off`, glibc libm, historical summation
  order wherever it is defined, a bit-exact port of CPython's `math.fsum` for the
  controller (so it equals `reference.correct_node` bit for bit), no heap allocation,
  compiled at runtime into volatile scratch and unlinked after loading.
  Optional FTZ/DAZ (`flush_subnormals=True`) is scoped to the call (MXCSR restored)
  and **off by default**; its deviation is measured (RESULTS.md, R118-D).

## 5. Gauge covariance of the composed step

Relation gauge `g_r ∈ SO(2)×SO(2)` acts by `X_r → g_r X_r`, `U[r,p] → g_r U[r,p]`; then
every window link transforms as `L → g_d L g_sᵀ`. (i) The per-slot law uses only
R⁴ norms and dot products of `(A, B, sal, col)` and is linear in those vectors with
invariant coefficients, so it is covariant under any per-node orthogonal transform
(R118-A measured O(4) covariance to 3e-16). (ii) The controller's `s·s` and `s·d` over
32 coordinates are invariant under per-slot orthogonal transforms, so the corrected
output is covariant. (iii) Every MEAN candidate for relation `r` is expressed in
`r`'s own frame, so the mean is covariant. Participant gauge
`U[r,p] → U[r,p] h_pᵀ` leaves every link unchanged. Local covariance does not by
itself prove the composed system covariant; R118-C therefore tests the whole
gather–transport–operator–controller–reconcile path with overlapping windows, plus
negative controls (state-only, frames-only, inverse frames).

## 6. Boundedness (finite state for any overlap structure)

Native HCA: every slot channel is clamped to norm ≤ 4 after each pass.

Controlled step with gains `(α, β)`, α < 1: write `p' = (1+β) p + c·s` with
`c = −α + (α−β)(s·p)/|s|²`, so `|c| ≤ α + |α−β| |p|/|s|`. With `|p_slot| ≤ √32`,
`|p| ≤ √128` and `|s_slot| ≤ |s|`,

    M' ≤ (1+β)√32 + |α−β|√128 + α·M      (M = max relation 8-norm)

and MEAN (a convex combination) preserves it. Fixed points: R104 (0.5, 0.5) → 16.97;
R107 / R114-signed (0.70, 0.45) → 36.77. The test suite checks the one-pass
recursion at every step under heavy overlap. (An earlier per-instance bound of 64
holds only for isolated instances and is not used.)

## 7. Known seams and limits

* Reconciliation changes the dynamics relative to an isolated window: the composed
  teacher itself moves (R118-E). This is semantics, not error, but it is not
  validated against any external ground truth.
* Windows are extracted by a deterministic heuristic (seed + 3 most-sharing
  neighbors). Other window policies are possible; none is claimed better.
* In an input-free, fully dissipative regime, states decay until intermediate
  products underflow; on the exact path the latency then degrades (subnormal
  arithmetic). FTZ/DAZ removes it with measured deviation ≤ 1.7e-307, but it is not
  enabled by default.
* The registry is built in one bulk phase; incremental relation registration is
  not implemented (instances are dynamic, relations are not).
* Single-threaded kernel; measured on one 4-vCPU x86-64 (AVX-512) VM only.
