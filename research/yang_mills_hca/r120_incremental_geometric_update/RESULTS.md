# ELPIS_YM_AWR_R120 — incremental geometric updates: results

Registered runs executed once each, in sequence, after the frozen source commit
`7144ffb19986a88f162bed8d0ff5a912af40bfb1` (base `501be02d…`). Protocol sha256 `8d47e5b4…96da14`,
component digest `bb98a741…5b599d`. Raw evidence: `evidence/` (`INDEX.json` lists the hashes).
Host: Intel Xeon @ 2.10 GHz, 4 vCPU (shared VM), gcc 13.3, Python 3.13.18, numpy 2.5.3.

> The ledger's earlier "R120" (`../r120_checkpoint_policy`) is unrelated and untouched.

## Disposition

**The update contract is qualified and exact; the latency targets are met only with the exact
parallel pass; CGC utility is not established.**

1. In-place, transactional, gauge-covariant updates of existing relation states work and are
   bit-identical to an independent reference. Identity, materialization, HCA88 weights and CGC
   weights are never touched by an update. R118 and R119 are reproduced bit for bit.
2. A complete update-and-propagate call, **single-threaded**, costs the same as the R119
   propagation step alone on the same windows (the update itself is nearly free). The frozen HCA88
   pass is about three quarters of the call and cannot be made faster without changing its
   arithmetic, so the
   preferred 256- and 1,024-window targets are **not met single-threaded** with the R114 controller.
3. With the **exact parallel Jacobi pass (4 threads, bit-identical)** all three preferred targets
   are met: 0.35 / 0.64 / 2.32 ms p95 at 64 / 256 / 1,024 windows on the million-relation fixture.
4. On the synthetic tracking task, frame transport is essential and sufficient for evidence to
   become useful state; the CGC coefficients (synthetic or randomly searched) add no established
   benefit over matched lightweight rules; and propagation through the frozen HCA88 after each
   update *reduces* the tracked information. Negative results, reported as measured.

## R120D — qualification (9/9 checks pass)

| Category | Result |
|---|---|
| Pins | frozen manifest 39/39; six checkpoint records authenticated; R118 digest `770411b1…`, R119 kernel `125e138b…`, CGC reference `04fc09ef…` unchanged |
| CGC operator | native = IEEE-single reference, 0 / 2,000 bit mismatches; vs the frozen torch `FrozenCGC3` max relative deviation 2.0e-7 |
| Update law | native = independent reference, 0 / 2,000 bit mismatches (scales 3.5 … 1e-310, zeros, signed zeros); zero state + zero observation stays exactly zero; zero state + evidence = γv exactly |
| Fused call vs reference | 576 batches in 48 cells (4 topologies incl. high overlap and arity-12 hubs × NATIVE/R104/R114 × normal/1e-150/subnormal/mixed scale, radius 1–2, passes 1–3, cache on/off, threads 1/4): **0 rows not bit-identical**, 0 disposition mismatches, 0 affected-set mismatches |
| R119 regression | fused step = R119 `EXACT_GUARDED` step: 0 / 144 mismatches; fused update-and-propagate = reference update + R119 step: 0 / 144 |
| Identity and memory | 16,000 updates: row, generation, binding and instance tables unchanged; pool and scratch never reallocated; HCA88 and CGC arrays read-only and unchanged; Python net retained 372 bytes / 7 blocks in total (transient peak 1.5 kB per call); kernel allocates nothing |
| Atomicity | 17 cases, every refusal (invalid value, unknown / unmaterialized / stale relation, non-incident participant, conflict, event / instance / row / link budget, bad argument) leaves pool, sequences, pending list, identities and marks bit-identical; stale reference after release + re-materialization refused, fresh one accepted |
| Event semantics | redelivery skipped; in-batch duplicate applied once; order-independent across relations; repeated events on one relation coalesce to one propagation with the same result as one-by-one delivery; deferred = immediate after `propagate_pending`; budget overflow deferred and reported, not dropped; rows outside affected windows bitwise unchanged |
| Geometry | relation-gauge covariance 2.2e-15, participant-gauge invariance 2.1e-15, slot permutation 2.2e-16 (tolerance 1e-12); negative controls break it as required: state-only gauge 0.36, observation-only rotation 0.25, untransported observation 0.90 |
| Stability and replay | 5,000 batches: max channel norm 3.40, replay bit-identical, threads 1 = threads 4 bit-identical; 20,000 propagated updates of one relation with evidence amplitude 1e3: finite, updated row within the clamp |

Unit tests: 15 / 15 pass (`tests/test_r120_update.py`).

## R120E — performance

Complete `apply()` call (numpy event arrays in → update → affected-window selection → HCA88 →
reconciliation → commit), 100 warm-up + 1,000 measured calls per cell, interleaved with the R119
`EXACT_GUARDED` step on the same affected windows. 26,640 bitwise comparisons in the benchmark
itself: **0 mismatches, 0 failed calls**. No FTZ/DAZ.

### Million-relation fixture STRESS1M (1,000,000 relations, 50,000 windows) — ms

| Controller | Threads | Windows | Events | R120 update+propagate p50 / p95 / p99 / max | R119 step only p50 / p95 | Target (p95) |
|---|---|---|---|---|---|---|
| **R114 (primary)** | **1** | 64 | 34 | **0.279 / 0.345 / 0.444 / 0.96** | 0.320 / 0.422 | < 1 — **met** |
| **R114 (primary)** | **1** | 256 | 113 | **1.017 / 1.300 / 1.568 / 4.09** | 1.064 / 1.347 | < 1 — **not met** |
| **R114 (primary)** | **1** | 1,024 | 424 | **4.049 / 5.638 / 5.955 / 6.46** | 4.004 / 5.594 | ≈ 3 — **not met** |
| R114 | 4 | 64 | 34 | 0.284 / 0.349 / 0.388 / 0.50 | 0.332 / 0.407 | met |
| R114 | 4 | 256 | 113 | 0.457 / 0.642 / 1.027 / 3.80 | 1.069 / 1.532 | met |
| R114 | 4 | 1,024 | 424 | 1.716 / 2.317 / 3.779 / 6.93 | 3.879 / 5.403 | met |
| NATIVE | 1 | 64 | 34 | 0.196 / 0.298 / 0.327 / 4.41 | 0.230 / 0.350 | met |
| NATIVE | 1 | 256 | 113 | 0.688 / 0.962 / 1.052 / 2.45 | 0.725 / 1.021 | met |
| NATIVE | 1 | 1,024 | 424 | 2.710 / 3.659 / 4.108 / 4.85 | 2.669 / 3.552 | not met |
| NATIVE | 4 | 64 | 34 | 0.189 / 0.229 / 0.306 / 2.29 | 0.219 / 0.268 | met |
| NATIVE | 4 | 256 | 113 | 0.361 / 0.496 / 2.021 / 5.32 | 0.735 / 1.064 | met |
| NATIVE | 4 | 1,024 | 424 | 1.291 / 1.728 / 3.323 / 12.04 | 2.605 / 3.567 | met |

### High-overlap fixture OVERLAP (4,096 relations each bound by 4 windows on average, up to 196) — ms

| Controller | Threads | Windows | Events | R120 update+propagate p50 / p95 / p99 / max | R119 step only p50 / p95 | Target |
|---|---|---|---|---|---|---|
| R114 | 1 | 64 / 256 / 1,024 | 22 / 72 / 277 | 0.263 / 0.381 / 0.446 / 1.49 · 1.024 / 1.465 / 1.649 / 4.14 · 3.975 / 5.309 / 6.118 / 7.47 | 0.323 / 0.473 · 1.098 / 1.556 · 4.017 / 5.197 | met · not met · not met |
| R114 | 4 | 64 / 256 / 1,024 | same | 0.261 / 0.296 / 0.336 / 0.56 · 0.412 / 0.504 / 0.696 / 1.18 · 1.673 / 2.039 / 4.017 / 9.84 | 0.318 / 0.366 · 1.052 / 1.592 · 4.037 / 5.996 | met · met · met |
| NATIVE | 1 | 64 / 256 / 1,024 | same | 0.184 / 0.301 / 0.410 / 2.31 · 0.684 / 1.071 / 1.634 / 4.61 · 2.708 / 3.980 / 4.401 / 6.71 | 0.235 / 0.412 · 0.773 / 1.171 · 2.773 / 4.151 | met · not met · not met |
| NATIVE | 4 | 64 / 256 / 1,024 | same | 0.189 / 0.294 / 0.335 / 1.18 · 0.328 / 0.433 / 0.579 / 2.52 · 1.212 / 1.510 / 2.631 / 8.28 | 0.239 / 0.379 · 0.777 / 1.158 · 2.782 / 3.979 | met · met · met |

### Reading the numbers

* **Against R119.** Single-threaded, the whole update-and-propagate call costs what the R119
  step costs for propagation alone: +1 % at 1,024 windows, −4 % at 256, −13 % at 64 (p50). The fused propagation-only step is 15–20 % faster than the
  R119 step (3.38 vs 4.00 ms p50 at 1,024, R114); the update and the selection use that margin.
  With 4 threads the complete call is 2.3× faster than the R119 step (p95 2.32 vs 5.40 ms).
  The R119 registered figure (≈ 5 ms p95 at 1,024 uniformly random windows) is not re-measured
  here; the interleaved R119 step on update-affected windows gave 5.4–5.6 ms p95.
* **Where the time goes** (STRESS1M, R114, 1 thread, 1,024 windows, mean per call 4.27 ms):
  transport + HCA88 3.27 ms (77 %), validation + CGC + update 0.47 ms (11 %), reconciliation
  0.24 ms, affected-window selection 0.19 ms, commit 0.09 ms; Python wrapper 0.04 ms.
  The frozen HCA pass costs ≈ 3.2 µs per window with the R114 correction and ≈ 1.8 µs natively
  on this host. That is the single-thread floor of an exact implementation.
* **Update only** (`propagate=False`, STRESS1M, p50 / p95 ms): 1 event 0.009 / 0.015; 16 events
  0.022 / 0.037; 64 events 0.054 / 0.085; 256 events 0.183 / 0.256; 1,024 events 0.825 / 1.046.
  About 1 µs per event at scale, of which the update law with CGC is ≤ 0.4 µs (profiled upper
  bound) and most of the rest is the 64-bit ID lookup in the million-entry table. On OVERLAP
  (small table) 1,024 events take 0.43 / 0.58 ms.
* **Tail.** The parallel pass has a heavier tail than its p95 suggests (p99 up to 3.8 ms, max up
  to 12 ms at 1,024 windows) from thread wake-up and scheduling on a shared 4-vCPU VM.
* **Compiler flags.** `-O3` and `-march=native` gave no reproducible gain in development, so the
  kernel keeps the frozen R119 flags (`-O2 -ffp-contract=off`) plus `-fopenmp`.

### Memory and allocation

| | STRESS1M | OVERLAP |
|---|---|---|
| Registry (R118, unchanged) | 320.6 MB | 1.3 MB |
| State, R118 layout | 24.3 MB | 1.7 MB |
| R120 side tables (17 B per row capacity + 1 B per instance) | 3.45 MB | 0.28 MB |
| Engine scratch (fixed) | 1.0 MB | 1.0 MB |
| Window cache used (optional; 128 MiB reserved, untouched pages unmapped) | 61.6 MB, 50,000 / 50,000 windows | 5.6 MB |
| Growth per update | 0 bytes | 0 bytes |
| Peak RSS of the measurement process | 1,073 MiB | 114 MiB |

Per call the wrapper allocates one 144-byte report snapshot and call temporaries (≈ 1 kB
transient, freed). The qualification measured 372 bytes net over 2,000 calls. The benchmark's
own allocation loop shows ≈ 42 bytes per call, which is the three Python integers it appends to
its bookkeeping lists.

## R120F — utility study (exploratory, synthetic)

Latent tracking on 5 evaluation seeds; mean over the last 50 of 300 batches. Columns: cosine of
the state with the hidden vector · gauge-invariant link readout correlation (downstream).

**Update only (no HCA propagation)**

| Arm | CLEAN | NOISY | REVERSED, flagged | CONTRADICTORY, unflagged | MIXED reliability | CHANGE point |
|---|---|---|---|---|---|---|
| No update | −0.01 · 0.02 | −0.02 · 0.02 | 0.00 · −0.04 | 0.00 · 0.01 | 0.02 · 0.01 | 0.00 · −0.03 |
| Direct, non-geometric | 0.01 · −0.02 | 0.03 · −0.03 | 0.01 · 0.00 | 0.02 · 0.05 | 0.00 · −0.01 | −0.01 · 0.00 |
| Geometric additive (zero CGC) | 0.995 · 0.90 | **0.91 · 0.79** | 0.994 · 0.91 | **0.93 · 0.85** | **0.90 · 0.77** | 0.64 · 0.47 |
| Geometric EMA | **0.996 · 0.99** | 0.76 · 0.65 | **0.996 · 0.99** | 0.69 · 0.58 | 0.74 · 0.60 | **0.99 · 0.99** |
| CGC, synthetic weights | 0.989 · 0.89 | 0.89 · 0.76 | 0.987 · 0.89 | 0.85 · 0.71 | 0.89 · 0.74 | 0.47 · 0.28 |
| CGC, random search | 0.05 · 0.03 | −0.14 · −0.01 | 0.05 · 0.01 | −0.02 · 0.01 | −0.08 · 0.03 | 0.17 · 0.01 |

**Update and propagate (frozen HCA88 on the affected windows after every batch)**

| Arm | CLEAN | NOISY | REVERSED, flagged | CONTRADICTORY, unflagged | MIXED reliability | CHANGE point |
|---|---|---|---|---|---|---|
| HCA only | −0.04 · −0.03 | −0.02 · −0.01 | −0.02 · 0.00 | 0.02 · 0.01 | 0.00 · 0.01 | −0.02 · −0.03 |
| Direct, non-geometric | 0.00 · 0.01 | 0.01 · 0.00 | 0.01 · 0.00 | 0.00 · 0.02 | 0.02 · 0.02 | 0.03 · −0.03 |
| Geometric additive (zero CGC) | 0.64 · 0.35 | 0.44 · 0.18 | 0.64 · 0.33 | 0.37 · 0.13 | 0.46 · 0.18 | 0.66 · 0.33 |
| Geometric EMA | **0.67 · 0.39** | 0.42 · 0.17 | **0.67 · 0.37** | 0.34 · 0.12 | 0.45 · 0.16 | **0.69 · 0.37** |
| CGC, synthetic weights | 0.65 · 0.35 | 0.44 · 0.18 | 0.64 · 0.33 | 0.36 · 0.13 | 0.46 · 0.18 | 0.66 · 0.33 |
| CGC, random search | 0.66 · 0.37 | **0.46 · 0.20** | 0.65 · 0.35 | **0.39 · 0.14** | **0.47 · 0.19** | 0.67 · 0.34 |

Findings.

* **Frame transport is what makes evidence usable.** The untransported update is at chance in
  every regime and every seed (cosine ≈ 0, link sign accuracy ≈ 0.50); the transported update
  reaches cosine 0.90–0.996. Equivariance here is not decoration: it is the difference between
  signal and noise.
* **CGC utility is not established.** With synthetic weights the CGC arm is never better than
  the zero-weight law without propagation and indistinguishable from it with propagation. The
  best of 256 random draws, selected with propagation on the training seeds, is ahead of the
  additive rule in all six regimes with propagation (by about 0.01 correlation, within one
  standard deviation) and ahead of the EMA in three; it does not beat both in every regime, and
  without propagation it destroys the state. The registered rule therefore reports **no claim**.
* **Flagged reversal is handled; unflagged contradiction is not.** Weight −1 on reversed evidence
  reproduces the clean result. Unflagged contradictions cost every arm (EMA correlation
  0.99 → 0.58).
* **Additive injection gets the direction but not the magnitude** (relative error ≈ 3.8: it runs
  into the clamp); the error-correcting rule gets both in clean regimes and adapts to a change
  point, but averages less under noise.
* **Propagation through the frozen HCA88 degrades tracking.** Downstream correlation falls from
  0.90–0.99 to 0.35–0.39 when the HCA runs after every update, and HCA-only drives every state to
  zero norm. The HCA is contractive on these states. Mechanically compatible, semantically not
  aligned with this readout — expected for an operator trained for Yang–Mills relaxation, and a
  reason not to read the fused path as a learning system yet.

## Limitations and things not executed

* **CGC weights are synthetic.** No trained relational CGC exists in the repository. Nothing here
  shows that a trained one would help; the bridge only guarantees that it could be installed
  without touching geometry, identity or the hot path (`replace_cgc_weights`, cold).
* **Single-thread targets not met** at 256 windows (R114) and 1,024 windows (both controllers).
  The parallel result needs 4 cores and has a heavier tail.
* The R119 registered benchmark itself was **not re-executed**; R119 is measured interleaved on
  the update-affected windows.
* The reference-parity cells seed their random batches from Python's salted string hash, so a
  re-run draws different batches (same criterion, same pass condition). Everything else is
  seed-deterministic.
* Radius is limited to 2 and passes to 8 per call by configuration; larger neighbourhoods go
  through deferral and `propagate_pending`.
* One `IncrementalEngine` per state; concurrent writers are not supported.
* The utility study is synthetic, five seeds, exploratory. It is not a prospective scientific
  claim about Yang–Mills behaviour and does not bear on R93, which stays
  `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.
* Not production-admitted. The public RuntimeCore, the frozen host, R118, R119, the earlier
  R120–R122 records and the CGC/LAO records are unchanged.
