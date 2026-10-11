# ECS Binary Observation R0 results: useful and mechanically sound; latency not resolvable here, so INCONCLUSIVE

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `UNQUALIFIED` · `NO_WORLD_REPRESENTATION_CLAIM` · `NO_BFM` · `NO_H-ECS`

| Item | Result |
|---|---|
| Disposition (frozen rule) | **INCONCLUSIVE** |
| Mechanics (M1-M7) | PASS |
| Usefulness (S1, S2a, S2b, S3) | GRAY4 USEFUL; SPROJ USEFUL |
| Transition probe (S4) | GRAY4 and SPROJ pass |
| Performance | INCONCLUSIVE for both: the p99 tail is unresolvable on this host |
| Conditional Bernoulli-transition follow-up | **not justified** (the frozen rule requires `BINARY_OBSERVER_FEASIBLE`) |

The authority is the write-once evidence under `research/ecs_binary_observer_r0/evidence/`. This document summarizes
it.

## 1. Pinned source and authority

| What | Value |
|---|---|
| Mainline | `bec6da7327bc164f8f18f38274223fb89969b89b` (not `fd9c128`) |
| ADR, written before any code | `fb8a043` |
| Preregistration (protocol, laws' calibration rules, gates, perf rule, disposition) | `11a6441` |
| DESIGN evidence | `a9be051` |
| DEV evidence and law-preserving optimization (`DEV_NOTES.md`) | `eecb746` |
| Freeze | `341be8a`: source commit `eecb746`, protocol `e34ae276...`, source `4b3b5cff...`, laws GRAY4 `0bdfe732...` / SPROJ `a46bd94e...` |
| QUAL | 24 held-out worlds, run once |
| Performance QUAL | run once after the freeze |

Host and environment:

* Shared cloud container: Xeon 2.1 GHz, 4 vCPU, Python 3.13, numpy 2.5.3, GCC Release with `-Werror`.
* `/mnt/primesauce` does not exist in this container. The work lives in the repository checkout and session scratch.
  Nothing was written under `/home/joe`.

## 2. What was observed

The observer sees only the caller-owned result of one accepted canonical QUERY, `elpis_ecsg_k1_query_identity`. The
K1 forward map is `f_W(x) = sum_i phi(x . w_i)`, with `phi(z) = 0.5z + 0.5z^2 + 0.5z^3`.

| Field | Content |
|---|---|
| Query rows | the first `min(rows, 64)` rows of X |
| Answers | the first `min(rows, 64)` values of `f_W(X)` |
| Retained-state digest | its first 8 bytes |

Not observed:

* **Pre-activations and neuron activations** exist only transiently in K1's private scratch. Exposing them would need
  an ABI change.
* **W, H and a** are privileged state, used only as laboratory ground truth.
* **The LEARN-path S3 readout**: the observer has no LEARN access.
* **Transformer activations** do not exist, since no qualified codec does. Nothing was simulated in their place.

Observing requires no re-execution, no extra K1 crossing and no device sync. K1 is CPU binary64 and has no device.

## 3. Encoding law and exact format (EBO1, 64 bytes, little-endian)

| Offset | Bytes | Field |
|---|---|---|
| 0 | 4 | magic `EBOR` |
| 4 | 2 | format version 1 |
| 6 | 1 | encoder id |
| 7 | 1 | flags: TRUNCATED when rows > 64, PADDED when rows < 64 |
| 8 | 2 | rows |
| 10 | 1 | dim |
| 11 | 1 | window |
| 12 | 4 | law tag = `SHA-256(law blob)[0..4]` |
| 16 | 8 | state identity: digest prefix. Exact bytes; no geometry. |
| 24 | 8 | input identity: a 64-bit four-lane multiply-rotate hash of the window rows' bit patterns, rows and dim. Exact but non-cryptographic. |
| 32 | 32 | payload |

The two payload laws were calibrated on the 24 DESIGN worlds only and frozen as canonical blobs:

* **GRAY4.** Each window answer gets a level `b = #{k : v >= t_k}` over 15 fixed thresholds (DESIGN quantiles
  k/16). It is stored Gray-coded, 4 bits per row. An exact tie goes up a level, and -0.0 is treated as +0.0.
* **SPROJ.** For each of 256 bits, `bit_k = (s_k >= tau_k)`, where
  `s_k = ((((0 + ±u_a) + ±u_b) + ±u_c) + ±u_d)` and `u = (v - mu) * inv_scale`. The 4 distinct window rows and the
  signs come from a pinned SplitMix64 stream. `tau_k` is the DESIGN median of `s_k`.

Refusals:

* Codes: INVALID, NONFINITE (an answer or query row in the window), SHAPE (dim 1..64, rows 1..4096), VERSION and LAW.
* A refusal writes no record and never touches its inputs.
* Values beyond the window are never read.

Handoff:

* A caller-allocated single-producer/single-consumer ring of 64-byte slots (a power of two, up to 65536).
* `DROP_NEWEST` when full: drops are counted, and the producer never blocks or allocates.
* The record is produced by C (`native/ebo_r0.c`) and reproduced bitwise by the pure-Python reference (`law.py`).

## 4. Tests completed

**Native mechanics test** (`ctest research.test_ebo_r0`, against the real K1 library). It passed under GCC Release
`-Werror`, clang Debug, GCC ASan+UBSan and GCC TSan. It covers:

* law refusals that leave the law unchanged;
* encode refusals that leave the record buffer and inputs unchanged;
* ties and signed zeros;
* the bounded window;
* repeatability;
* OFF/ON non-interference on a K1 QUERY sequence: answers, digest, snapshot bytes and counters identical, and no
  added `forward_calls`;
* zero allocations across QUERY + encode + push + pop, with an interposed allocator that the test first proves is
  live;
* record admission;
* the bounded ring, and FIFO and loss accounting under threads.

**Python** (`tests/research/ecs_binary_observer_r0`, scientific lane). It checks:

* native/reference bitwise parity, with ties, signed zeros and rows from 1 to 4096;
* identical refusal codes;
* record admission;
* K1 non-interference;
* import boundaries: `observer.py` and `law.py` import no `elpis` module, and `src/` never names the lineage;
* the symbol audit;
* the freeze binding.

**Repository suites.** The boundary suite and the default FAST lane pass.

**QUAL mechanics** (`evidence/qual`):

* 2496 records, native == reference bitwise, every record encoded twice identically;
* clean-process re-encoding identical;
* OFF/ON identical in all 24 worlds (answers, digests, snapshots at every state, K1 counters);
* the observer library imports only `memcpy`, `memset` and `__stack_chk_fail` (plus weak runtime symbols);
* `ebo_encode`, `ebo_ring_push` and `ebo_ring_pop` reach no external call except the stack-protector failure path;
* no ECS, runtime or continuity symbol.

## 5. Measured overhead (`evidence/perf_qual`, direct native)

* OFF = `query_identity`. ON = `query_identity` + `ebo_encode` + `ebo_ring_push`.
* K1 at dim 6, width 36.
* 20000 interleaved pairs per row count (4000 at 4096 rows).
* The CPU was pinned on a best-effort basis.
* Medians and p99s in µs. The 95% CIs are block-bootstrap intervals.

| Encoder | rows | median OFF → ON | p99 OFF → ON | Δp99 95% CI | A/A p99 CI | θ | isolated encode+push | / median | verdict |
|---|---|---|---|---|---|---|---|---|---|
| GRAY4 | 1 | 127.5 → 127.5 | 276.6 → 276.3 | [-10.3, 6.7] | [-4.1, 10.5] | 1.5 | 0.026 | 0.02% | INCONCLUSIVE |
| GRAY4 | 16 | 135.5 → 135.4 | 237.7 → 235.0 | [-16.8, 5.8] | [-6.4, 16.4] | 1.5 | 0.13 | 0.10% | INCONCLUSIVE |
| GRAY4 | 64 | 127.5 → 128.1 | 251.9 → 252.9 | [-6.2, 9.8] | [-20.1, 1.4] | 1.5 | 0.48 | 0.38% | INCONCLUSIVE |
| GRAY4 | 256 | 149.8 → 150.5 | 263.0 → 262.5 | [-8.0, 5.7] | [-14.8, 9.5] | 1.5 | 0.49 | 0.33% | INCONCLUSIVE |
| GRAY4 | 4096 | 357.7 → 357.9 | 587.0 → 599.5 | [-3.6, 35.7] | [-4.3, 63.4] | 1.5 | 0.46 | 0.13% | INCONCLUSIVE |
| SPROJ | 1 | 125.6 → 126.1 | 238.1 → 238.3 | [-19.7, 6.4] | [-25.3, 10.2] | 1.5 | 0.43 | 0.34% | INCONCLUSIVE |
| SPROJ | 16 | 126.6 → 127.0 | 218.1 → 219.4 | [-2.5, 6.5] | [-2.6, 9.8] | 1.5 | 0.64 | 0.51% | INCONCLUSIVE |
| SPROJ | 64 | 126.6 → 127.4 | 244.2 → 249.6 | [-3.2, 25.8] | [-11.5, 11.4] | 1.5 | 0.66 | 0.52% | INCONCLUSIVE |
| SPROJ | 256 | 152.7 → 154.2 | 265.8 → 269.5 | [-2.4, 12.1] | [-9.6, 5.8] | 1.5 | 0.77 | 0.51% | INCONCLUSIVE |
| SPROJ | 4096 | 351.1 → 350.8 | 597.8 → 603.6 | [-12.5, 16.3] | [-30.5, 19.7] | 1.5 | 0.78 | 0.22% | INCONCLUSIVE |

* **Throughput** changes by at most about 1%: GRAY4 at 64 rows went from 7207 to 7160 queries/s.
* **Worst case.** Single-call maxima of 1–10 ms appear in both arms. They are host scheduling.
* **Accounting per observation:**
  * 0 K1 crossings added (`forward_calls` = 2 × pairs exactly);
  * 0 allocations;
  * 0 device syncs;
  * 2 copies of 64 bytes (staged record → caller record → ring slot);
  * inputs read in place.

  The drop path costs about 1.2 ns.

**Why INCONCLUSIVE.** The canonical QUERY's p99 tail on this host varies by tens of µs between equal-condition
halves (the A/A column). That is one to two orders of magnitude above the preregistered 1.5 µs ceiling, so the rule
cannot certify the tail. The deterministic cost is well under 1% of the median QUERY, and the median shifts agree
with it. The rule is not reinterpreted: a noisy tail is INCONCLUSIVE, not PASS.

**Disclosed limitation.** The 1% budget is generous only because the canonical QUERY recomputes a SHA-256 over
about 30 KB of retained state on every call. That costs about 125 µs; the bare forward at one row costs 0.07 µs.
Measured against a bare forward, the observer is 0.4× (GRAY4) to 6.5× (SPROJ) the cost at 1 row. At 64 rows it is
0.13× to 0.19×.

## 6. Information preserved and lost (QUAL, 24 worlds × 13 states)

| Probe | GRAY4 | SPROJ | RAW16 (matched 256 bits) | RAW4 | Controls |
|---|---|---|---|---|---|
| S1: neuron-permuted twin `W[:, pi]` | payload identical 100%; digest differs 100% | payload identical 100%; digest differs 100% | — | — | — |
| S2a: transitions with a changed payload | 288/288 | 287/288 | — | — | — |
| S2b: Spearman (Hamming, ‖Δf(X_HELD)‖) | 0.706 | 0.676 | 0.903 (L2) | 0.758 (L2) | DIGEST 0.012 |
| S3: R² of f(X_HELD) from one record | **0.851** | 0.639 | 0.787 | 0.388 | DIGEST −0.045, RANDOM −0.048, SHUFFLED ≤ −0.79, RESET ≤ −0.28, ROWPERM ≤ −0.16 |
| S4: R² of Δf(X_HELD) from consecutive records | **0.886** | 0.659 | 0.872 | 0.503 | all ≤ 0 |
| Matched budget | BINARY_AT_OR_ABOVE_RAW | RAW_SUPERIOR | | | |

**Preserved:**

* **Functional equivalence where identity breaks it.** On DESIGN worlds, 76% of the twin's answers differ bitwise
  from the original (relative difference up to 1.4e-11). Every payload was still identical.
* **Graded similarity between states.** The digest carries none.
* Enough of the forward map on 64 observed rows to predict the state's answers on 64 held-out rows, and the change
  across a LEARN transition, far beyond every control.
* At 256 bits, GRAY4 predicts held-out behavior better than the same budget spent on 16 binary16 answers.

**Lost:**

* **Precision.** GRAY4's midpoint decode has RMSE 0.38 std, and 10% of values fall in the open outer levels.
* **Everything outside the window.** 24% of workload queries had more than 64 rows.
* **Everything about H, a, epoch and history** beyond a 64-bit identity prefix.
* **Neuron identity.** W is not identifiable from `f_W`, by symmetry.

Collisions:

* GRAY4 had no colliding state pairs among 312.
* SPROJ had 1 colliding pair (rate 2.1e-5), the same transition that left its payload unchanged.
* No observed value or projection lay within 1e-9 of a threshold.

Bit entropy:

| Encoder | mean | min | constant bits |
|---|---|---|---|
| GRAY4 | 0.969 | 0.65 | none |
| SPROJ | 0.982 | 0.90 | none |

The record encodes one forward map on caller-chosen rows. It is **not** a world representation, a state
representation or a model of ECS.

## 7. Failures, limitations and blockers

* **Performance blocker (environmental).** The preregistered tail criterion needs a quiet host to resolve 1.5 µs at
  p99, for example isolated cores, a fixed frequency and no co-tenants. This shared container gives A/A spreads of
  tens of µs. Re-running the frozen performance gate on dedicated operator hardware is the only path to a PASS or
  NONPASS. It would be a new, separately recorded run; this evidence is not re-run.
* **DEV-phase repairs, all recorded in `DEV_NOTES.md` with records bitwise unchanged:**
  * the encoder implementation was optimized after the DEV rehearsal showed SPROJ at 0.99% of the median;
  * a test-only loss-accounting defect in the threaded ring test was fixed;
  * the symbol audit now follows local calls.

  No gate, threshold, sample size or law was moved.
* **Synthetic regime.**
  * Cognitive R0 worlds: dim 6, width 36, synthetic teachers.
  * One standing 64-row reference query supplies the science. It is synthetic; a real workload would need its own
    observation design.
  * Decoders are ridge regressions trained on DESIGN.
* **Boundary registrations (outside the lineage):**
  * `tests/boundary/test_mission.py` now pins `elpis_ebo_r0` among the research native targets.
  * `tests/boundary/test_one_ecs.py` lists this lineage's files that name the K1 ABI.
* **Law blobs committed late.** The repository `.gitignore` ignores `*.bin`, so the two frozen law blobs were not in
  commits `a9be051`..`70f3026`. CI failed the freeze-binding test in a clean checkout as a result.
  * They were committed byte-identical in `5427f86`, under a narrow `.gitignore` exception.
  * Their SHA-256 values (GRAY4 `0bdfe732...`, SPROJ `a46bd94e...`) were bound in the DESIGN summary and the freeze
    record from the start, and every phase used these exact files.
* **Raw timing telemetry.** It stays out of Git by convention; its SHA-256 is recorded. It lives only in the session
  container's scratch, which is ephemeral.
* **Not done, by design:**
  * no production wiring;
  * no runtime, K1, ABI, continuity or frozen-object change;
  * no persistence or background consumer;
  * no BFM;
  * no H-ECS.

## 8. Disposition

**INCONCLUSIVE.** Under the frozen rule, mechanics pass and both encoders are USEFUL (GRAY4 the stronger), but no
USEFUL encoder reached a performance PASS. The latency evidence does not contradict feasibility (deterministic cost
≤ 0.52% of the median QUERY, no added crossing, allocation or sync). It also cannot certify the p99 tail on this
host, so it is not reported as PASS.

## 9. Is a conditional Bernoulli-transition experiment justified?

**No, not yet, by the frozen rule.** The follow-up requires `BINARY_OBSERVER_FEASIBLE` and a perf-PASS useful encoder
that passes S4 with non-degenerate bits.

* GRAY4 meets every condition except the performance PASS: S4 R² 0.886, mean bit entropy 0.97, no constant bits.
* The prerequisite is therefore a resolved performance gate on quiet hardware, not more science.
* Any later Bernoulli Flow Model work (arXiv:2610.11362) must keep two things apart:
  * that paper's denoising-time posterior over bits;
  * world-time evolution of these records across ECS transitions.

  A BFM's sampling chain is not a model of how ECS state moves.
* Governing rule, unchanged: a binary observer must not turn a fast inference system into a simulator. Usefulness
  and negligible bounded latency are both necessary, and only the first is established here.
