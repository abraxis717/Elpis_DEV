# ECS_C HCA R1 — DEV mechanics record

R1A, R1B and R1C are complete locally. R1D's science/publication boundary is
preserved. **This is a mechanical comparison, not scientific qualification.** B2 is
an unlearned exploratory comparator. No QUAL, training, B3, fractional operator,
neural NCA, trained HCA or port-Hamiltonian/stability claim was executed or made.

Workspace: `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0`; branch
`research/ecsc_hca_r0`; unchanged HEAD
`cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`. All writes are in this research directory.
No commit, checkout, push, merge, install, native build, cache, download, background
worker, external-system write or persisted activation state was introduced.

The user's R1 authorization supersedes the supplied R0 AGENT_TASK prohibition on
B2 and its requirement for prior scientific preregistration: the current sequence
is R1A mechanics, R1B **mechanical** freeze plus **DRAFT_NOT_FROZEN** protocol, then
B2 DEV only. AGENT_TASK.md, RESEARCH_CONTRACT.md and CHECKSUMS.sha256 are unchanged.
The old README's R0 comparative demonstration has been replaced by this record.

`baselines.py` preserves the existing B0 exact-key and B1 introspection APIs. In
particular, B1's inspector still returns unseeded scoped targets, even at zero
salience. **It is not an evaluated answer.** `task.py` supplies the common Task /
Predicate envelope and prediction channel: scope, premise seeds, exact public
relation/type/role/kind/status predicates, output kinds, readout/work budgets,
steps and deterministic schedule parameters. There is no answer key or oracle
field. All methods receive precisely the same bounded graph and checks. Targets
are already graph identities; this is a task-conditioned readout experiment, not
hidden-graph completion. The fixture harness withholds the answer set from solvers.

The task B0 is a strong exact worklist recomputation of weighted conjunction
closure, bounded by work and graph capacity. It is not artificially limited to
B1's number of rounds. B1 uses synchronous weighted min-input/max-output salience.
The old all-keys B0 index remains available as an unscored visibility upper bound.
`evaluation.py` alone receives withheld answer keys and computes deterministic
synthetic key TP/FP/FN, omissions, invalid predictions and label counts. Changing
withheld truth cannot affect predictions. Readout ranks positive eligible cells
by descending salience then key, includes only requested kinds, and reports
truncation. The readout budget counts mandatory constraint records, prediction
records and their direct incoming relation evidence. Insufficient binding capacity
refuses; bindings are never silently dropped or automatically seeded.

Source evidence has distinct VALID, STALE and UNKNOWN states; no-reference bindings
are explicitly AUTHORITY_ONLY, not source-verified. These source labels remain
separate from PROVISIONAL/BINDING and CANDIDATE/ADMITTED. Candidate chains can affect
salience, including an ADMITTED-labelled downstream edge. Direct incoming evidence
is not a transitive native proof; every prediction is explicitly SALIENCE_ONLY.
The immutable graph retains all original references/provenance/epistemic labels.
No production attestation, source locator/version/time contract, or native n-ary
admission is added. The fixtures use synthetic allowlists and synthetic checks;
production native adapters currently admit binary relations.

Successors now require identical Authority (including epoch, complete grants and
schemas) and Limits, plus predecessor/version/counter and binding-retention checks.
There is no migration path in this phase, including to a renamed epoch. A separate
fresh experiment build may choose an explicitly different authority; it is not a
successor. SourceCheck observations may refresh independently and change only the
evidence coordinate; the policy digest stays fixed. The old delayed fixture changes
trusted grants, so its continuity attempt is now refused and each changed fixture
is checked independently. Unchanged-policy successors remain tested. Renaming a
binding is caught by the stricter binding-retention check before policy continuity.

Sidecars use two-pass scalar streaming: measure encoded size under a byte cap,
charge preflight work and reserve hashing work, then incrementally hash. No complete
sidecar JSON string/bytes or recursive asdict copy is allocated. The maximum encoded
scalar label is 386 ASCII bytes (64 characters with worst-case JSON escaping).
Each policy/evidence coordinate has its own sidecar cap. Logical operations count
records, visits, copies and hash bytes; they are **not instructions or FLOPs**.
Python sorting, allocation, recursive object-size checks and interpreter overhead
are reflected in separately measured wall time, not exact instruction accounting.

`b2.py` owns only bounded integer activation snapshots and immutable task/topology
references. A cell is an unsigned big-endian 16-bit value constrained to [0,1000].
Each explicit WARM tick orders cells by SHA-256 of a fixed domain, pinned 64-bit
seed, 64-bit logical step and 32-byte cell identity; ties break by key. It fires
at most updates_per_tick cells. A fired cell is replaced by the maximum of its
seed clamp and valid incoming conjunctive messages, using current asynchronous
neighbor values. All positions and the final output role are explicit; updates
read only incident typed factors and their premises. Global schedule construction,
validation and common readout are separately bounded and metered; no O(1) total
work claim is made. Replay follows deterministic fixed-width encoding, integer
arithmetic, canonical key/edge order and sequential updates. Different Python hash
seeds are tested in separate processes. Partial firing has no finite fairness or
convergence guarantee. The logical counter refuses overflow.

State is bound to the exact evidence and Task digest, preventing scope/evidence
transplants. Failed work requests publish nothing; prior snapshots remain immutable.
Activation damage and reset are explicit synchronous calls; neither changes a
source, graph, binding or status. B2's auxiliary-memory reservation is checked before
cell-state allocation, with a Python-owned-object size backstop. It includes packed
payload, indexing/schedule/copy scratch allowances and bounded metadata. It excludes
the shared graph/task, already bounded for all methods. It is not an OS RSS limit.
No state history is retained by the implementation; callers own any old snapshots.

`BASELINE_R1_FREEZE.json` was written after 28 research tests and two explicitly
selected canonical tests passed, **before b2.py existed**. All frozen hashes still
match. No baseline code, contract, fixture, task, scorer, baseline test, runner or
draft protocol changed after that freeze. `DRAFT_PROTOCOL_R1.md` proposes disjoint
DEV/QUAL seeds, withheld targets, metrics, work matching, timing, controls and
prospective thresholds. It is not frozen scientific authority; QUAL was not run.
A future essential baseline defect requires STOP and a newly authorized freeze.

Declared resource ceilings (independent maxima; work/degree checks may prevent
simultaneously filling every capacity):

| Resource | Default ceiling | Absolute configurable ceiling |
|---|---:|---:|
| Axioms / relations | 32 / 32 | 256 / 512 |
| Arity / degree / references per axiom | 8 / 8 / 4 | 16 / 32 / 16 |
| Rounds or ticks | 8 | 32 |
| Logical work per call / cold comparison | 100,000 | 10,000,000 |
| Encoded bytes per sidecar coordinate | 262,144 | 4,194,304 |
| B2 auxiliary state reservation bytes | 65,536 | 1,048,576 |
| Fired cells per WARM tick | 256 | 256 |

Incidences cannot exceed min(axioms × degree, relations × arity). Index cardinalities
are bounded by those same capacities; B0's pending queue is at most one entry per
cell. Task readout is at most 1,024 records, seeds at most 256, predicates at most
512, and run parameters are checked against the particular graph's limits. The
comparison uses 8 graph cells, 4 factors, 16 incidences, arity 4, degree 2, readout
budget 32, 3 steps, and at most 32 firings per tick (only 4 scoped cells exist).
Its packed activation payload is 8 bytes, auxiliary reservation 15,360 bytes, and
measured start ownership 6,712 bytes in the recorded process. Tests reach 32 cells /
32 relations, 64 incidences, arity 8 on a separate n-ary fixture, 64 activation bytes,
and a 51,456-byte auxiliary reservation; these are mechanics coverage maxima, not
claims to have tested every absolute ceiling.

The final single-run record is `R1_RESULTS.json` (stdout-derived descriptive JSON).
Common graph build: 23,861 logical ops, 2,194,558 ns. At a common 100,000-op cap:

| Method | Logical ops, including setup/readout | Observed wall ns | TP / FP / FN |
|---|---:|---:|---:|
| B0 exact closure | 93 | 84,918 | 1 / 0 / 0 |
| B1, 3 rounds | 89 | 76,662 | 1 / 0 / 0 |
| B2, 3 explicit ticks | 2,766 | 1,112,959 | 1 / 0 / 0 |

At cap 64 all refuse. At 128 and 2,048, B0/B1 complete and B2 refuses atomically.
At 4,096 all complete. B2's coordinate-encoding cost changes slightly with the
number of digits in the task budget; it reports actual logical work (2,762 at
4,096). These are common budget checks, not an instruction-calibrated Pareto study.

After explicit all-zero activation damage, B2's three-tick local repair plus readout
costs 1,109 ops / 497,968 ns; fresh B2 initialization plus recurrence costs 2,766 /
1,111,276 ns. Fresh B0 costs 93 / 83,792 ns and B1 costs 89 / 73,460 ns for the same
readout. Damage itself costs 48 / 113,222 ns and is reported separately. **No useful
repair advantage is demonstrated.** An additional negative-control test shows
unit-weight unsupported cycles retain injected positives; B0/B1 fresh recompute
and B2 reset correctly return empty. This limitation is not hidden or repaired by
changing the baseline. Wall times are noisy single observations, not statistical
performance findings. No scientific PASS/NONPASS is assigned.

Verification commands from workspace root, unless stated otherwise:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m unittest research.ecsc_hca_r0.test_baselines -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m unittest research.ecsc_hca_r0.test_baselines research.ecsc_hca_r0.test_task -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest tests/integration/test_context_lane.py::test_successor_is_pure_and_deterministic tests/integration/test_context_lane.py::test_references_and_relations_keep_their_epistemic_status -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m unittest research.ecsc_hca_r0.test_baselines research.ecsc_hca_r0.test_task research.ecsc_hca_r0.test_b2 -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m research.ecsc_hca_r0.run_dev
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m research.ecsc_hca_r0.run_r1 > research/ecsc_hca_r0/R1_RESULTS.json
```

Successful exit codes: all **0**. Initial R0: 17 tests; R1A freeze: 28 tests; selected
canonical: 2 tests (`.. [100%]`); final combined research: **40 tests, OK**, 3.591 s.
Coverage includes every omitted premise at arities 4/5/8, contradictory scopes,
source invalidation, predicate restrictions, binding retention, two-step candidate
chains, readout/work limits, deterministic replay, independent hash-randomization
processes, state perturbation/reset, atomic refusals, and no core file/worker calls.
Two development-only test corrections preceded their respective successful runs:
R1A expected AUTHORITY_CONTINUITY but correctly got BINDING_RETENTION (exit 1, 28
tests); B2's read-only mapping attribute assignment raised AttributeError rather
than the test's expected TypeError (exit 1, 12 tests). The latter assertion now
attempts item mutation and confirms TypeError. Neither was a scientific result.
No execution blocker remains for R1A/B/C. Future science remains separately
authorized and unspecified; no full canonical suite or native tests were run.

Isolation checks: `git branch --show-current`, `git rev-parse HEAD`,
`git diff --exit-code HEAD -- . ':!research/ecsc_hca_r0'`, and `git diff --check` all
exit 0. The two diff commands produce no output. Research files are untracked, so
`git diff --check` alone does not inspect them; a separate `git diff --no-index
--check /dev/null FILE` check is applied to every research file.
The 18 no-index checks emit no diagnostics; their exit code 1 denotes new-file
differences, not whitespace errors. The first audit wrapper incorrectly required
exit 0 and stopped; the corrected wrapper accepts this documented difference code
only with empty stdout/stderr and exits 0.
Supplied checksum verification (`sha256sum -c CHECKSUMS.sha256` inside this directory) exits 0 with
`RESEARCH_CONTRACT.md: OK` and `AGENT_TASK.md: OK`. No bytecode/cache directories
appear in the research file inventory.

Exact files changed relative to session entry, all relative to this directory:

- Modified: `baselines.py`, `test_baselines.py`, `run_dev.py`, `README.md`.
- Added: `task.py`, `evaluation.py`, `test_task.py`, `BASELINE_R1_FREEZE.json`,
  `DRAFT_PROTOCOL_R1.md`, `b2.py`, `test_b2.py`, `run_r1.py`, `R1_RESULTS.json`.
- Preserved: `AGENT_TASK.md`, `RESEARCH_CONTRACT.md`, `CHECKSUMS.sha256`,
  `fixtures.py`, `__init__.py`; all canonical and external repositories unchanged.

Exact SHA-256 coordinates:

```text
BASELINE_R1_FREEZE.json a5d136abab626f1887a67fbf7bc887ad17e855ad5c2ecf820ca6f1ff97156c10
baselines.py           04719f9c446f83b9de0b0d157f55e29f2aaa2ebfcee80a75272a5fe4fb8aef7a
task.py                6bb07d65cf9bd958b23cbbe2d4fe0118e02cb076424ea14308f7643b87f0892f
evaluation.py          b8901ab3a43d704df4eda08559013d89584d99b905055b5c0770900e1bed7462
canonical context.py   afa96f35c51e8130683eb5502e8e6478d7c113caf1f6116e4a703c6cd0aae2f0
policy digest          fc2876fa9e4daeca14f422ddb79df462caaa3d9a658524944869fc9ec27894db
evidence digest        a968081ebe92b9a39f05963fd40be8aeebb3bdcccc42e535ff12ccebffc7e028
```

The manifest contains the remaining exact source/contract hashes. These coordinates
freeze mechanics and synthetic authority only; they confer no production or science
qualification. No work is committed or merged.
