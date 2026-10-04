# ECS_G Cognitive R0: qualification results (v1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `SEMANTICS=NONE`

Laboratory: [`research/ecs_cognition_r0`](../../research/ecs_cognition_r0/README.md).
Authority: [`docs/COGNITION_R0.md`](../COGNITION_R0.md).

**Disposition: `MECHANICS_PASS` · `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME`.**
Every pre-registered gate (A-G) holds on all 8 QUAL worlds. Sequential
learning of a second experience **interferes** with the first in 8 of 8 QUAL
worlds, catastrophic in 6 of 8 (the error on A after learning B is at least
the error of the untaught state). That is reported, not hidden; it was
pre-registered as measured and not gating.

## What was asked

Can one ECS_G state acquire and retain a bounded input -> response
relationship in its own microscopic state `W` through the qualified G1
recurrence, and answer later queries from that `W`? Every response is
`f_W(x)` by the native forward map of the current state
(`CognitiveCore.query`); every learning step is the G1 recurrence committed
atomically (`CognitiveCore.learn`). No other model, table or cache exists in
the procedure.

## Protocol and authority

| item | value |
|---|---|
| regime | `d=6`, `N=36`, `W0 ~ N(0, 0.18^2)`, rate 0.002, two cubic teachers (4 columns, scale 0.5), regions `N(+-0.5 e0, 0.5^2 I)`, 64 experience / 256 held-out rows |
| DEV | 4 DEV worlds; chose `steps = 4000` by the pre-registered DEV rule |
| freeze | commit `bdb8823aec62c83bebc6aeb0072dca3f4aa1a759` (laboratory source and DEV evidence) |
| spec digest | `d5172e7af41410323b6e3bce8c69b87234d3f05c66975fd441cae88ff2c24670` |
| laboratory source digest | `d107f1470c43b4753e568c7d20fc595d9d951d0c086fa73384f1cae2b08f46a3` |
| DEV evidence digest | `12e3406775d00f11e31c58c2ae56b3469fbab45eb24742b9116106ca20ce0bb6` |
| frozen authority digest | `71fba13cdea64551e5d08818e20a1b89cc8b76f0b2fc172ac317d711d9311eb1` |
| QUAL evidence digest | `9dfe55adab68bf578511a4e0a463aef318281e1f2ac1073ef3c20910cff10b7d` |
| QUAL evidence file SHA-256 | `cd801cc9796dcfdc692b541564735c9281608443b0a6f5bc44fa7376a405d429` |
| numerical profile | numpy 1.26.4, Python 3.11.15, x86_64, IEEE-754 binary64 |

QUAL ran once, on 8 QUAL worlds disjoint from DEV. The evidence file is
write-once; `tests/research/ecs_cognition_r0` re-runs QUAL worlds and requires
bit-identical measurements under the recorded numerical profile.

## Results (8 QUAL worlds)

Held-out `nmse = mean((f_W(x) - y)^2) / var(y)`.

| measurement | median | min | max |
|---|---|---|---|
| A, untaught state | 1.129 | 0.943 | 1.902 |
| A, after learning A | **0.031** | 0.023 | 0.167 |
| A, after then learning B | 1.246 | 0.828 | 3.185 |
| B, after learning B | 0.102 | 0.029 | 0.442 |
| control: permuted A targets, same budget | 1.857 | 1.165 | 8.703 |
| control: untrained fresh state | 1.030 | 0.808 | 2.098 |
| retention ratio (A after B / A learned) | 31.6 | 15.1 | 63.1 |

| gate | result | evidence |
|---|---|---|
| A learning | PASS | every world <= 0.5 and <= half its untaught error; median 0.031 <= 0.25 |
| B state causality | PASS | reset `W` reproduces untaught responses bitwise; learned `W` values transplanted into a fresh state reproduce learned responses bitwise; `W = 0` answers exactly 0 (nmse 1.01-1.51) |
| C persistence | PASS | a clean process given only snapshot bytes and the spec reproduced the learned responses bitwise in all 8 worlds |
| D continual update | PASS | learning B changed the state identity, advanced the epoch by 4000, changed the B responses and acquired B in every world |
| D retention (measured) | INTERFERENCE in 8 of 8, catastrophic in 6 of 8 | A error after B returned to or above the untaught level |
| E negative controls | PASS | untaught responses unchanged; permuted targets and fresh states fail the learning criterion in every world; median shuffled 1.857 >= 4 x 0.031; leak guard holds |
| F determinism | PASS | world qual-0000 learned twice in-process and once in a clean process: identical snapshot and receipt digests |
| G microstate authority | PASS | PTE pairs: `S3` relative difference 0 to 1.2e-16 and response difference <= 3e-16 before learning; after one identical learning step `S3` differs by 7.4e-4 to 1.1e-2 and responses by 5.5e-4 to 1.0e-2 |

## What this supports, and what it does not

Supported, under this frozen synthetic regime only: an ECS_G state learns a
bounded input -> response relationship into its own `W` through the G1
recurrence; the learned behaviour is computed from `W` by the native forward
map, follows `W` (reset, transplant, zero), survives snapshot and a clean
process exactly, changes with further experience, and is deterministic. `S3`
is not sufficient for future dynamics (G), so `W` stays authoritative.

Not supported or not tested:

* **No retention under sequential learning.** Plain sequential G1 learning
  overwrote experience A in every world. R0 establishes no continual-learning
  property; consolidation, replay or any other mechanism is an open question.
* No language, meaning or general cognition; the task has no semantics.
* No promotion of the cubic kernel, `d=6`/`N=36`, `S3`, the G1 recurrence or
  the Branch36 rate beyond this regime.
* Bitwise reproduction is established for one build and numerical profile,
  and only with an FMA OpenBLAS kernel. The v1 profile records neither the
  kernel nor the binary (see Reproduction contract).

## Reproduction contract (CE0, after QUAL; the evidence is unchanged)

Main went red twice after Runtime R1 merged (`61f81e6`, CI run 37161622841,
Python 3.11): `test_qual_measurements_reproduce_exactly` failed in all 8 QUAL
worlds. Metrics differed only in their last bits (e.g. `nmse_A_init`
1.2098485936686503 against the recorded 1.20984859366865), but every learned
snapshot, receipt and response digest changed. The same tree passed on another
runner with the same image, CPython 3.11.16 and NumPy 1.26.4 (run 37161299328).
Python 3.12 passed in both runs. So the runner, not the Python version, decided
the outcome.

Cause: NumPy's bundled OpenBLAS (wheel 1.26.4) chooses its DGEMM kernel per CPU
at load time. It falls back to the generic Prescott kernel on CPUs it does not
recognize, such as Emerald Rapids (family 6, model 207). The non-FMA kernels
(Prescott, Nehalem, Sandybridge, Core2) round the teacher targets `X @ W`
differently in the last bit. Learning carries that perturbation into different
`W` bytes. Forcing those kernels with `OPENBLAS_CORETYPE` reproduces the CI
values exactly, digests included: qual-0003 learned snapshot `59e2f304`,
receipt `89468702`. The FMA kernels (Haswell, Zen, SkylakeX, Cooperlake)
reproduce the record exactly. The native executor is not involved. Its parity
with the scalar reference is bitwise on every host tested, AVX2 and baseline
clones alike.

The v1 record binds Python, NumPy, machine and float format. It does not bind
the native binary, the compiler, the CPU or the BLAS kernel. Its profile is
therefore necessary but not sufficient for bitwise replay. The test suite now
keeps four questions apart:

| question | test | contract |
|---|---|---|
| evidence integrity | `test_lab.py` (byte pins, digests, freeze binding) | always; byte-for-byte |
| current implementation correctness | `tests/ECS_G`, `tests/research/ecs_runtime_r1`, native ctest | always; bitwise executor = reference on this host |
| historical replay | `test_lab.py::test_qual_measurements_reproduce_exactly` | bitwise; demanded only when Python, NumPy and machine equal the recorded ones exactly and the OpenBLAS kernel is one of Haswell, Zen, SkylakeX, Cooperlake (that kernel condition was established after QUAL and is not in the record); skipped with the reason otherwise |
| current-runtime scientific regression | `test_runtime_regression.py` | never skipped for a host or profile; all 8 QUAL worlds |

The regression recomputes every gate (A–G), the aggregate rules, mechanics and
the disposition by the frozen procedure, and requires them to equal the
record. It also requires each world's interference class and catastrophic
flag to equal the record. Each numeric metric must satisfy
`|current - recorded| <= 1e-12 + 1e-9 |recorded|`. That tolerance was fixed
before use:
- The largest deviation measured with each kernel class forced is 5e-15
  relative on nmse and retention metrics, and 4e-13 relative on the PTE
  post-step differences. On ULP-scale quantities it is 4e-16 absolute. No
  class, flag or gate changed. That leaves at least 2,600× headroom.
- The tolerance is still seven orders of magnitude tighter than the smallest
  relative margin of any recorded gated metric from its threshold (3.4%).

The regression compares no snapshot, receipt or response digest with the
record. Those are bitwise identities of one host's arithmetic. The bitwise
properties the pass rule demands (reset, transplant, zero state, clean-process
persistence, determinism) are re-established on the host, current against
current.
