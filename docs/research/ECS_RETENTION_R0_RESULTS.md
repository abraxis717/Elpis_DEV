# ECS_G Retention R0: qualification results (v1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Laboratory: [`research/ecs_retention_r0`](../../research/ecs_retention_r0/README.md).
Candidates: [`CANDIDATES.md`](../../research/ecs_retention_r0/CANDIDATES.md).
Prior result: [`COGNITION_R0_RESULTS.md`](COGNITION_R0_RESULTS.md) (plain sequential G1 does not retain).

**Disposition: `MECHANICS_PASS` · `PARTIAL_REDUCTION` (OUTCOME_C).** The
selected ECS-internal candidate, C1 (coarse S3-space functional
consolidation), sharply reduces forgetting: A is retained in 18 of 24 QUAL
worlds against 0 of 24 for plain G1, never catastrophically, with a median
retention ratio of 4.2 against 67.5. It fails the two pre-registered
per-world gates that require it everywhere: B is learned in only 10 of 24
worlds and A is retained in 18 of 24, so both A and B hold in 7 of 24. The
rehearsal control (stored experience, not eligible) does no better: both in
9 of 24. **No candidate is eligible for canonical promotion.** Canonical
ECS_G learning is unchanged.

## Question

Can one ECS_G state acquire experience B while retaining experience A, and
what state does that require? Every response is the canonical native forward
map of the evaluated `W`; every G1 step is the canonical native step.

## Why these candidates (pre-registered reasoning)

For the ECS_G kernel `f_W(x) = 1/2 phi(x) . S3(W)` exactly (83 cubic
features for `d = 6`): the learned function lives in `S3` while G1 moves the
216 weights, and `phi` has no locality. B's data carry no information about
where A lives, so any retaining mechanism needs some memory of A. C1 keeps it
in function space (A's input-feature second moments `H` and the
consolidated coarse state, `b`), C2 per microscopic weight (diagonal
importance and anchors). Both read A's experienced inputs, never its
targets.

## Chronology and authority

| step | commit | content |
|---|---|---|
| RET0A.1 | `e444814` | `CANDIDATES.md`, specification (regime, tasks, splits, metrics, DEV rules, gates A-J, dispositions, binding requirements); no code |
| RET0A.2 | `10f39f2` | laboratory and mechanics tests (test worlds only) |
| RET0B | `e35ffdc` | DEV evidence (DEV worlds only) |
| RET0B.1 | `23c841d` | fix of the freeze record (pass-rule digest overwritten), found at freezing, before any QUAL evaluation |
| RET0B.2 | `57f238b` | DEV re-executed under the fixed source; refused unless every DEV result equalled the first record's |
| RET0C | `71eb4e3` | frozen authority |
| RET0D | `f7b6ab6` | QUAL evidence, run once |
| RET0E | `f1251ca`, this commit | performance characterization (PERFORMANCE_ONLY), interpretation |

| item | value |
|---|---|
| base | Runtime R1 `2799f839f11b2a52d882adea91ca3631c5bfcd11` (merged as `61f81e6`, identical tree) |
| spec digest | `a87da8b4d8e39d1b836bfd10da4e513b4aa252497748fc5f93754c34863b00b2` |
| pass-rule digest | `b2b21cc495ae68a01ffbcfc1f87f785042278a3377ec6cd9d925e686478de729` |
| `CANDIDATES.md` SHA-256 | `8fb5df87a202d2571828006c76f82b3cf1f19a14f50e53cfa069eb9a33fa6073` |
| laboratory source digest | `1e217564fd3022735ff86337d04962d2b03bc6f7cdbbf45e9991c9e3b2049160` (commit `23c841d`) |
| DEV records | `1cc6bf68d72f27f54bcf708a24d336b18221f7ecf2456d418610e2d4d1996e3b` (superseded), `d06c89765d85415f502f3be805e2bd9df20fe98f767b8c5898385bb9a98198af` |
| frozen authority digest | `2b4e0618813b2cf5ec77dbcb02957c4ee87b67befa5fb1ed111dd3f6fd4a2547` |
| QUAL record digest | `f8e631163e8280398b8d36df903c6dc0017f521350404bb503d0d6a15efdce68` (file SHA-256 `b7864fdb...`) |
| library | `libelpis_ecsg_math.so` `2374876e3a837761c87bdbc39389a2c5b2e007fcbaa81dd4b39e05deb1e5dc09` (the Runtime R1 binary), gcc 13.3.0 Release `-O3 -DNDEBUG` |
| bound sources | `native.py` `236de9d7...`, `cognition.py` `0cf4d058...`, `ecsg_math.h` `94f3c4e7...`, `ecsg_state.h` `34b173c4...`, `ecsg_executor.h` `67502a5e...`, `ecsg_math.c` `fbfc48e5...`, `ecsg_state.c` `c9887848...`, `ecsg_executor.c` `1d94739c...` |
| numerical profile | numpy 1.26.4, Python 3.11.15, x86_64, IEEE-754 binary64, BLAS/OpenMP single-threaded |

Every record carries the full binding; QUAL refused unless the frozen
specification, laboratory source, bound sources, library, build and profile
held and the tree was clean.

Known record defect (found in RET0E, left as recorded): the QUAL body's
`worlds` field holds the integer 24, not the list of world ids, because the
gate verdict (which carries `worlds: <count>`) is spread after the list in
`run.qual`. The evaluated world ids are the `per_world` keys
(`qual-0000` ... `qual-0023`), which equal the frozen `qual_worlds`; no
measurement, gate or disposition is affected. The evidence is write-once and
was not rewritten.

Numerical-profile limitation (found in RET0E): the recorded profile states
the BLAS/OpenMP thread *environment*, not the thread count OpenBLAS actually
uses. The laboratory pins the environment when its package is imported; that
takes effect only if NumPy has not been loaded yet. Under the QUAL entry point
(`python -m research.ecs_retention_r0.run`) the package loads first and
OpenBLAS ran single-threaded (checked with `openblas_get_num_threads`). In a
process that loaded NumPy earlier (a full pytest session), OpenBLAS keeps its
default thread count while the environment reads 1. Re-running QUAL worlds
in such a process changes only the representability ceiling (LAPACK least
squares), and only in its last one or two bits. Every native result and `W`
digest is unaffected. The evidence tests therefore reproduce QUAL worlds in a
fresh interpreter with the environment fixed at start. A future version
should record the effective thread count. Since CE2.1 the replay test also
reads the effective count in the replay process and requires 1.

### Reproduction contract and portability (CE2.1, after QUAL; the evidence is unchanged)

The v1 binding covers the laboratory source, the eight bound sources, the
library SHA-256, the build, and the numerical profile: NumPy, Python, machine,
float format and the thread *environment*. It does not cover the OpenBLAS
kernel family, the effective thread count, or the CPU. NumPy's bundled
OpenBLAS chooses its kernel per CPU, and on CPUs it does not recognize it
falls back to the generic Prescott kernel. One such CPU is Emerald Rapids
(family 6, model 207), which this VM moved to during the CI correction.

A post-QUAL audit re-ran all 24 QUAL worlds with one kernel of each class
forced: Cooperlake, Haswell and Prescott (`OPENBLAS_CORETYPE`, single
thread). SkylakeX, Zen and Sandybridge were checked on three worlds
(qual-0000, -0013, -0020), and each gave exactly the same results as the other
member of its class:

| kernel class | W digests changed (of 192: 8 per world, 24 worlds) | largest relative change | gates, mechanics, disposition, counts |
|---|---|---|---|
| AVX-512 (Cooperlake, SkylakeX); the recording host | 0 (bitwise) | 0 | identical |
| AVX2+FMA (Haswell, Zen) | 66 | 23% (selected candidate, A->B->C->D stage D, task C); 11% (mismatched ablation, B after) | identical |
| non-FMA (Prescott, Sandybridge; the fallback) | 192 (all) | 23% (same); 15% (mismatched ablation, A after B) | identical |

Under all three classes, these were identical to the record: the gate table,
the mechanics, `PARTIAL_REDUCTION` / `OUTCOME_C`, and for every mechanism the
counts of worlds that retained A, learned B, held both, were catastrophic, or
refused a learn (M0 0/24/0/18/0, C1 18/10/7/0/0, C2 9/11/3/10/6, M1
15/15/9/0/0; ablations removed 0/24/0/18/0, mismatched 4/12/1/16/0, isotropic
7/4/2/6/0). The A->B->C->D median of tasks held at the end (1 for every
mechanism) was also identical. Two descriptive quantities did change:
- the mismatched ablation's median error on A after B: 2.892 recorded, 2.891
  under Haswell, 3.117 under Prescott;
- C1's total number of tasks held at the end, 28 recorded and 27 under both
  other classes (qual-0015).

The medians, ratios and other floats in this report are therefore the
recording host's arithmetic (an AVX-512 OpenBLAS kernel). The conclusions rest
on the gates and counts, which did not move in this audit. The suite keeps the
questions apart:

| question | test | contract |
|---|---|---|
| evidence integrity | `test_evidence.py` (byte pins, digests, binding, chronology from full git history) | always |
| historical replay | `test_evidence.py::test_qual_measurements_reproduce_exactly` (3 worlds) | bitwise; demanded only under the full v1 binding, with OpenBLAS kernel Cooperlake or SkylakeX and an effective single thread, in a fresh interpreter (that kernel condition was established after QUAL and is not in the record); skipped with the reason otherwise |
| current-runtime regression | `test_runtime_regression.py` (all 24 worlds) | never skipped for a host or profile; gates, mechanics, disposition, outcome, every per-mechanism count and the sequence medians must equal the record exactly; no float or `W` digest is compared |

Future experiment versions should record effective runtime properties next
to the environment strings: the OpenBLAS kernel name, the effective thread
count, the CPU model and flags, and NumPy's build configuration.

## DEV (8 DEV worlds) and frozen choices

* Task rule: rehearsal met both criteria in 3/8, 6/8 and 6/8 DEV worlds at
  offsets 0.5, 1.0, 1.5; no offset qualified (`REPLAY_INFEASIBLE_ON_DEV`),
  so the rule chose 1.0. The unconstrained representability ceiling already
  failed A in 2 of 8 DEV worlds.
* Candidate rule at offset 1.0: C1 `lambda` 4 and 2 each reached 6/8; 4 had
  the smaller median joint nmse (0.455): **selected C1 `lambda` = 4**. Best
  C2: `lambda` = 16 (3/8): secondary. M0: 0/8 retained, median ratio 51.3.
* Frozen: offset 1.0, step budget 4000, rate 0.002, the thresholds and gates
  of the specification, 24 QUAL worlds.

## QUAL results (24 QUAL worlds, run once)

Held-out nmse, median (min-max). `W_A` is shared bitwise by every mechanism.

| mechanism | A after A | A after B | B after B | retention ratio | A retained | B learned | both | A catastrophic | extra state |
|---|---|---|---|---|---|---|---|---|---|
| M0 plain G1 | 0.050 (0.001-0.245) | 2.236 (0.862-142.0) | 0.047 (0.004-0.306) | 67.5 (10.7-1193) | 0/24 | 24/24 | 0/24 | 18/24 | 0 |
| **C1 `lambda` 4 (selected)** | 0.050 | **0.224 (0.025-0.971)** | 0.621 (0.050-7.13) | **4.17 (0.97-223)** | **18/24** | 10/24 | 7/24 | **0/24** | 28,552 B (`COGNITIVE_STATE`) |
| C2 `lambda` 16 (secondary) | 0.050 | 1.222 (0.001-70.0) | 0.585 (0.028-59.4) | 12.5 (1.0-286) | 9/24 | 11/24 | 3/24 | 10/24 | 3,456 B (`COGNITIVE_STATE`) |
| M1 rehearsal (control) | 0.050 | 0.394 (0.023-3.18) | 0.360 (0.027-3.28) | 6.99 (1.48-81.9) | 15/24 | 15/24 | 9/24 | 0/24 | 3,584 B (`EXTERNAL_MEMORY_CONTROL`) |
| O ceiling (analysis) | | 0.450 | 0.299 | | | | | | none |

Untaught A 1.422 (0.551-5.970); B before learning B 2.325 (0.724-59.4). C1's
W moves as much as plain G1's (state change 0.86 vs 0.74 of `||W_A||`): it
does not retain by freezing `W`.

### Gates (pre-registered; selected candidate C1 `lambda` 4)

| gate | result | evidence |
|---|---|---|
| A acquisition | PASS | every world LEARNED_A; median 0.050 <= 0.25 |
| B acquisition | **FAIL** | LEARNED_B in 10 of 24 worlds |
| C retention (central) | **FAIL** | RETAINED_A in 18 of 24 worlds |
| D joint state | PASS | one executor created from each final `W` answers A and B; state digest recorded |
| E state causality | PASS | snapshot/restore and reset reproduce responses bitwise in every world; with the consolidation state removed (same engine) A is retained in 0 of 24 worlds and median A after B is 2.236 vs C1's 0.224 |
| F no external answer store | PASS | consolidation receives `(W, inputs)` only; queries read `W` only |
| G determinism | PASS | qual-0000 twice in process and once in a clean process: identical `W_A` (`f37d44e8...`) and `W_AB` (`eaa03d79...`) and metrics |
| H negative baseline | PASS | M0 not retained in 24 of 24, median ratio 67.5 (Cognitive R0: 8/8, 31.6) |
| I control honesty | PASS | M1 reported separately, never selected |
| J capacity accounting | PASS | extra state and compute recorded for every mechanism |

Mechanics: S3 identity within 6.6e-16; the candidate engine without
consolidation reproduces the canonical core bitwise in all 24 worlds; no
baseline refusal; PTE microstates with identical `S3` (difference 0) diverge
after one C1 step (difference >= 3.2e-4): `W` stays authoritative.

### Ablations of the selected candidate

| ablation | A retained | both | median A after B |
|---|---|---|---|
| consolidation state removed (gating, E) | 0/24 | 0/24 | 2.236 (identical to M0) |
| statistics from another region's inputs (task C) | 4/24 | 1/24 | 2.892 |
| isotropic statistics (trace-matched identity) | 7/24 | 2/24 | 0.679 |

Retention follows the declared state, and specifically the memory of **A's
input distribution**: anchoring the coarse function without knowing where A
lives (isotropic) or knowing the wrong region (mismatched) loses most of it.

## Representability (descriptive, post hoc)

The unconstrained least-squares cubic fit to both experiences (the ceiling of
the function class, ignoring that `M` must be positive semidefinite and
realizable by 36 entities) meets both learning criteria in only **8 of 24**
QUAL worlds. In the other 16, no single state of this function class can be
learned on both A and B, whatever the mechanism, so the per-world gates B
and C could not pass everywhere. Within the 8 representable worlds C1
retains A in 7 and holds both in 5; rehearsal holds both in 6. C1 and
rehearsal hold both in largely the same worlds (6 shared; C1 alone in 1,
rehearsal alone in 3). This cross-tabulation was not pre-registered and does
not change the disposition; it locates the failure: mostly the capacity of
the 83-coefficient function class for two independent teachers on
overlapping regions, and secondarily the C1 trade-off at `lambda` = 4, which
favours A over B.

## Longer sequence A -> B -> C -> D (secondary, descriptive)

Median nmse after the last experience (D), per task:

| mechanism | A | B | C | D | tasks held at the end (median) |
|---|---|---|---|---|---|
| M0 | 1.83 | 2.01 | 1.89 | 0.09 | 1 (only D) |
| M1 rehearsal | 0.76 | 0.69 | 0.64 | 1.11 | 1 |
| C1 `lambda` 4 | 0.38 | 0.90 | 0.98 | 1.77 | 1 |
| C2 `lambda` 16 | 0.93 | 1.53 | 1.44 | 1.95 | 1 (33 refused learns: the cumulative explicit penalty diverges) |

With four experiences no mechanism holds more than about one task: plain G1
keeps only the last, cumulative C1 keeps the first and blocks the later
ones, rehearsal spreads a compromise across all four. Four independent cubic
teachers exceed what one 83-coefficient cubic function can represent on
these regions.

## Capacity and cost (PERFORMANCE_ONLY, NO_SCIENTIFIC_CLAIM)

`evidence/perf.json`, DEV world dev-0000, learning B (4000 steps), p50 / p95 /
p99 over 28 samples (canonical executor) or 7 samples (laboratory engine; with
7 samples p95 and p99 are the maximum), one shared VM:

| path | latency |
|---|---|
| canonical executor, M0 (64 rows) | 27.5 / 31.6 / 33.3 ms |
| canonical executor, M1 rehearsal (128 rows) | 53.6 / 56.5 / 56.5 ms |
| laboratory engine without consolidation | 203 / 238 / 238 ms |
| laboratory engine, C1 `lambda` 4 | 1,191 / 1,292 / 1,292 ms |
| laboratory engine, C2 `lambda` 16 | 292 / 314 / 314 ms |

The laboratory engine makes one native call per step plus NumPy work; it is
a research implementation. Natively, C1 would add about 42,500 operations
per step to G1's 82,900 (one `S3` reduction, one 83 x 83 mat-vec, one local
contraction per entity; fusable into the executor's K-step loop, no
historical examples), C2 about 860. State: `W` 1,728 bytes; C1 28,552 bytes
(16.5x `W`); C2 3,456 bytes (2x `W`); rehearsal 3,584 bytes of stored
experience per experience, growing with every experience.

## What this supports, and what it does not

Supported under this frozen synthetic regime:

* Plain sequential G1 still destroys A (0/24 retained, 18/24 catastrophic).
* An ECS-internal consolidation state built only from experienced inputs and
  the model's own coarse state (C1) prevents catastrophic forgetting in every
  world and retains A in 18 of 24, comparable to explicit rehearsal of stored
  experience, without storing any target. Retention is caused by that state
  (removed: 0/24) and specifically by its memory of A's input distribution.
* A purely local, per-weight consolidation (C2) is much weaker (3/24 both),
  consistent with the kernel's function living in the global, entity-coupled
  `S3` coordinates rather than in individual weights.

Not supported:

* **Retention is not solved.** No candidate meets the pre-registered
  retention and acquisition gates; the selected one trades B for A. No
  canonical change follows and no candidate is eligible for promotion.
* The dominant limit observed is representational: two (or four) independent
  cubic teachers on overlapping regions do not fit one 83-coefficient cubic
  function in most worlds. A retention mechanism cannot exceed the capacity
  of the state it protects.
* No language, meaning or general cognition; synthetic task, SEMANTICS=NONE.
* Bitwise reproduction is established for one build, numerical profile and
  AVX-512 OpenBLAS kernel; the conclusions (gates, counts, disposition) held
  under every kernel class audited (Reproduction contract and portability).

## Open questions

* Capacity: whether a larger or locally supported ECS state (wider `d`, more
  coefficients, entity-local response) makes joint representation possible,
  and whether C1-type consolidation then retains in every world. This needs
  a new kernel or regime and a new experiment version.
* The C1 trade-off: `lambda` was chosen on DEV for joint counts; an
  acquisition-aware schedule, or consolidation that decays with confidence,
  is untested.
* State authority of a consolidation state: if anything like C1 were ever
  promoted, `H` and `b` would be persistent cognitive state, part of the
  snapshot, and an FMS-materialized asset beside `W`.
