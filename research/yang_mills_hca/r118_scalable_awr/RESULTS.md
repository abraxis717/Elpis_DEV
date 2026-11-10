# R118 results

All numbers come from `evidence/*.json` (component digest `770411b1f6788f95…` for A, C, E;
D records are tied to commits below). Hardware: one KVM guest, 4 vCPU Intel Xeon
@ 2.10 GHz (AVX-512), 15 GiB RAM, Linux; Python 3.13.18, numpy 2.5.3, torch
2.15.0+cpu (oracle only), gcc 13.3.0. Single-threaded measurement.

## R118-A — authentic operator reconstruction: **PASS**

| Check | Result |
|---|---|
| Manifest (pinned `b0415964…`) | 39/39 files SHA-256 + size; 26/26 historical ledger entries agree; no unmanifested files |
| Sealed archives | single-member, CRC, canonical JSON, NaN-rejecting, size-bounded reads |
| Checkpoints | 6/6: 88 parameters, `sha256(struct.pack('<88d'))` = archived = pinned = `r64.parameter_pin` |
| Historical R78 `run.sh` protocol from loose published files | `verify_r78.py` 12/12 adversarial rejections; `spot_r78.py` 72/72, log-gain delta 0; **`r78.py run` result archive byte-identical (`7646e113…`)** |
| Full R77 evaluation (576 rows, 8 seeds, 4 graphs, 6 models, H8/H12/H20 + teacher-forced diagnostic) | **bitwise identical** (canonical rows `051fa3bc…`) |
| SO(2)×SO(2) covariance, 96 cases (8 graphs × 2 transports × 6 checkpoints) | H1 8.9e-16; H20 1.7e-16 (native) and 1.7e-16 (R114) |
| O(4) per-node covariance (finding) | 3.1e-16 |
| Participant-gauge invariance / slot-permutation equivariance | 1.7e-16 / 1.1e-16 |
| Negative controls (must exceed 1e-6) | state-only 0.27, links-only 0.79, transposed links 0.27, non-orthogonal scaling 0.19, slot-mismatched gauge 0.44 |
| Instance isolation | cross-instance leak 0.0; batch-vs-single 2.8e-17 (**not bitwise**: torch rounding depends on batch shape) |
| Vectorized controller vs `reference.correct_node` | 4.4e-16 (144 instances) |
| Oracle step, 1,152 homogeneous instances | 3.8 ms p50 |

Reproduction obstacles diagnosed: torch was not installed (installed CPU wheel in an
isolated venv); **no historical torch version is pinned** in the frozen sources (2.15.0
reproduces bitwise here); `r77.run()`/`r78.run()` write their archive into their own
directory, so in-place execution would overwrite frozen evidence — all historical code
runs from a verified volatile copy; plain import would write `__pycache__` into the
frozen tree — modules execute from verified in-memory bytes. The frozen tree is
unchanged (manifest re-verified after every run).

## R118-B — sparse substrate correctness: **PASS**

* Native engine vs the pure-Python simple reference: **bitwise equal** on connected
  sparse, Zipf-heterogeneous, 8-component disconnected and hub-adversarial (arity 16)
  families; NATIVE, R104 and R114 controllers; 1 and 3 passes; overlap reconciliation
  with up to 18 bindings per relation; worst-case window in-degree 192 (four arity-64
  relations with identical members, the kernel's hard limit).
* Capacities 1, 10, 100, 1,000, 10,000, 50,000 instances: fill, refuse N+1, release,
  reuse with new generation.
* Isolation: perturbing one disconnected component leaves every other relation bitwise
  unchanged; inactive and unbound relations are never touched.
* Active-set order, reversal, duplicates: bitwise identical results. Replay: bitwise.
* Budgets (instance, relation, membership, link, work, pass, expansion, scratch):
  refused before work, state and mutation epoch unchanged; partial selection only on
  request, labelled with exact omitted count; dead / stale handles refused; injected
  NaN refused transactionally.
* Boundedness: per-slot recursion `M' ≤ (1+β)√32 + |α−β|√128 + αM` verified at every
  step under overlap (observed 31.3 vs bound 36.77 for R107 gains).
* Binding reverse index verified against a full scan under randomized churn.

## R118-C — geometry and global recurrence: **PASS**

| Check | Result |
|---|---|
| Historical rollouts through the substrate, 96 cases | oracle adapter **bitwise 96/96**; native H1 3.6e-16, H20 1.1e-16 |
| R114-controlled, native vs oracle, H20 | 1.4e-16 |
| Overlapping sparse windows (in-degree ≤ 23, bindings ≤ 8), native vs oracle | 1 pass ≤ 6.7e-15; 10 passes ≤ 3.6e-16; native vs simple 0.0 |
| End-to-end relation-gauge covariance (bindings ≤ 14, MEAN, controllers) | ≤ 8.3e-16 native, ≤ 2.7e-15 oracle |
| Participant-gauge invariance | ≤ 9.6e-16 native, ≤ 7.6e-15 oracle |
| Negative controls (state-only, frames-only, inverse frames) | ≥ 0.31 |
| Hidden channels: 141 unshared/foreign frames, 534 foreign relations, 51 inactive states perturbed | active rows bitwise unchanged |
| 5 passes in one call vs 5 calls | bitwise identical |
| Weights after all runs | 6/6 pins unchanged |

Declared tolerances (fixed before the runs): 1e-12 one pass, 1e-10 multi-pass, 1e-9
covariance, ≥ 1e-6 negative controls.

## R118-D — performance: **HOT_PATH_QUALIFIED for the registered workload, with a disclosed exception**

Protocol `PROTOCOL_R118D.json` (`7ac1a0f7…`) committed in `027dc3c` before any latency
measurement. Primary: R114 on `54007/STATE8_SIGNED8`, 1 pass, MEAN, uniform random
active sets over all instances, 100 warmup + 2,000 measured calls, every call timed.

Fixtures: SMALL 2,000 participants / 8,000 relations / 31,960 memberships / 1,024
instances; INST50K 50,000 / 200,000 / 799,883 / 50,000; STRESS1M 50,000 / 1,000,000 /
3,997,898 / 50,000. Arity 2–6 uniform; Zipf 0.5 participant popularity (STRESS1M
degree p50 58, p99 401, max 9,025); window links p50 8, p99 14–22, max 26; relation
binding overlap max 30 / 411 / 219; float64 throughout.

Run 2 (`448bd73`, hot-path sources byte-identical to the final tree), primary,
p50 / p95 / p99 in ms:

| Registry | A = 64 | A = 256 | A = 1,024 |
|---|---|---|---|
| SMALL | 0.232 / 0.275 / 0.335 | 0.838 / 1.016 / 1.266 | 3.298 / 4.111 / 5.101 |
| INST50K | 0.262 / 0.360 / 0.400 | 0.896 / 1.064 / 1.276 | 3.520 / 3.935 / 5.148 |
| STRESS1M | 0.268 / 0.413 / 0.458 | 0.927 / 1.062 / 1.329 | 3.603 / 4.814 / 5.642 |
| target p95 | 2 | 5 | 20 |

Run 1 (`92080f4`) gave the same disposition (STRESS1M p95 0.412 / 1.340 / 4.701).
Zero failures in all 72,000 measured calls of each run (18,000 primary). Capacity p95 ratios vs SMALL
(rule ≤ 2.0): run 2 INST50K 1.31 / 1.05 / 0.96, STRESS1M 1.50 / 1.05 / 1.17. Parity on the
STRESS1M fixture: native vs simple reference 0.0. Torch oracle at A = 64: p95 73 ms
(run 2), 89 ms (run 1) — at A = 64 on SMALL the native p95 is 265× (run 2) and 322× (run 1) lower.

Memory (run 2): registry 2.5 / 61.8 / 305.8 MiB; AWRState 0.4 / 20.1 / 23.2 MiB; engine
scratch 0.85 MiB; peak RSS 78 / 247 / 1,072 MiB (includes bulk-generation transients).
Python transient peak per call 4.9 KB (A = 64) to 28.4 KB (A = 1,024); the ~1 net live
block per call is the measurement harness's own result list; the native kernel does
no heap allocation. Run 1 measured AWRState at ~800 MiB because the binding reverse
index was a dense rows × 1,024 table — fixed (O(bindings)) before run 2.

Full 50,000-instance sweep (separate, not hot path): p50 193 ms (INST50K), 206 ms
(STRESS1M). Cold: STRESS1M registry construction 7.0 s, window extraction 3.5 s,
instance creation 0.15 s, kernel build 0.3 s.

**Exception (diagnostic `evidence/r118d_subnormal_diagnostic.json`).** Without external
input, repeatedly stepping the same instances is dissipative; states decay
exponentially. Around state magnitudes 1e-143 … 1e-157, intermediate controller
products (`s · (s·d)` with the denominator pinned at 1e-24) become subnormal. The exact
path then reaches **25.2 ms block p95 at A = 1,024 (> 20 ms)** — SMALL R104, all 1,024
instances stepped 4,000 times. The registered secondary configurations on SMALL
show the same effect (NATIVE A = 1,024 p99 18.8 ms, max 37 ms; R104 A = 1,024 p95
12.6 ms), because the protocol measured configurations sequentially on one
evolving state — a protocol weakness, disclosed. Opt-in FTZ/DAZ removes the
spike (block p95 ≤ 4.5 ms) with deviation **exactly 0 until subnormals appear and
≤ 1.7e-307 after**. It is not enabled by default. Consequently the hot path is
qualified for the registered workload but **not** for unbounded input-free decay on
the exact path.

## R118-E — transfer characterization: **DESCRIPTIVE ONLY**

OOD R77 fixtures embedded among k ∈ {0, 2, 6} distractor windows (distractors bind
intermediate relations plus new random relations); fresh seeds (checked disjoint from
224 historical seeds); nothing fitted or selected. References are evaluation-only.
32 paired samples per cell; all six checkpoints; NATIVE and R114.

| Pooled, H20 | NSE vs composed teacher | NSE vs isolated teacher | directional 2(1−cos) | energy ratio | teacher shift |
|---|---|---|---|---|---|
| k0 NATIVE | 0.320 | 0.320 | 0.0007 | 0.88 | 0 |
| k2 NATIVE | 1.563 | 0.396 | 0.0003 | 4.23 | 0.551 |
| k6 NATIVE | 1.576 | 0.485 | 0.0002 | 4.18 | 0.600 |
| k0 R114 | 0.514 | 0.514 | 0.0007 | 0.18 | 0 |
| k2 R114 | 0.387 | 0.773 | 0.0008 | 0.33 | 0.551 |
| k6 R114 | 0.493 | 0.856 | 0.0013 | 0.16 | 0.600 |

* Composed teacher at k = 0 equals the historical teacher rollout bitwise.
* Composition changes the target itself (teacher shift NSE ≈ 0.55–0.60 at H20).
* Error is magnitude, not direction: directional error ≤ 0.0061 in every cell; per-cell
  energy ratios span 0.002–19.1 (worst cell `R77_OOD_D4/k2/NATIVE/54007/STATE8_SIGNED8/H20`,
  NSE 11.3, energy ratio 19.1). Pooled means hide this heterogeneity.
* No new recurrence instability observed: finite-difference amplification at H20
  falls with overlap (k0 0.012–0.023 → k6 0.001–0.004). Observation, not proof.
* Local transition fidelity remains poor; R93 `ENERGY_NORMALIZED_DENOMINATOR_HOLD`
  is unaffected.

## Blockers and next requirements

* Scientific: R93 HOLD unchanged; composed fidelity is magnitude-dominated and
  checkpoint-heterogeneous.
* Engineering: register a protocol (R119) that includes decayed / input-free
  regimes and evaluates FTZ/DAZ as a candidate with its measured error bound; reset
  state per configuration; measure on target hardware; incremental relation
  registration; multi-threaded sweeps.

## Closure (2026-11-10)

Independently re-verified from a fresh clone of the published branch at
`f901cf0e4db7d93945b590d1b19262e3af37aeff` with
`python -B research/yang_mills_hca/r118_scalable_awr/evidence/verify_closure.py --torch --tests`:
**R118_CLOSURE_PASS, 20/20 checks** — component digest `770411b1…df71`, protocol `7ac1a0f7…0fbc`,
kernel source `87b53392…1e54`, frozen manifest 39/39 and ledger 26/26, six checkpoint pins,
R78 result archive byte-identical, R77 576/576 rows bitwise, 33/33 tests, no boundary-policy
violations, frozen tree unchanged. Record: `evidence/CLOSURE.json`. The verifier lives in
`evidence/` so that it cannot change the digest it verifies. R118 sources are frozen from
here on; follow-up numerical work is in `../r119_numerical_hotpath/`.

## Follow-up (R119, 2026-11-10)

The R118-D regime exception (exact path above the A = 1,024 target in input-free decayed regimes) was re-measured under a new registered protocol in `../r119_numerical_hotpath/`. It also fails A = 256 at the decayed peak. It is **closed for the default exact path** by the R119 guarded kernel, which is bit-identical to this kernel in every verification. The R118 sources and evidence above are unchanged.
