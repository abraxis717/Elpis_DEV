# ECSC_DYNAMIC_RELATIONAL_DEV_R0: the ECS_C relational view, its verification, and HCA_REVISION_R0

Classification: **DEV, NOT QUAL.** HCA readouts are `SALIENCE_ONLY`. B3 is `NOT_AUTHORIZED`. Nothing here is a
qualification, a production admission or an intelligence claim. The H1A–H1D0 record (`research/hca_system_dev_r0`)
is untouched: all 75 files match `PUBLICATION_PROVENANCE.json`, and its disposition `NO_HCA_ADVANTAGE` stands.

The view itself is specified in `docs/ECS_C_RELATIONAL.md` (source `native/ECS_C/relational`, commit `857e89a`).
This directory holds the verification instruments and evidence beyond its CTest suites, and the one HCA experiment
this milestone justified.

| File | What it is |
|---|---|
| `mutants.py` | Mutation harness: 39 targeted mutants of the view's law, judged by both suites |
| `PROTOCOL_HCA_REVISION_R0.md` | The HCA experiment, predeclared (commit `471ef8a`, pushed before any run) |
| `bench_hca_revision.c`, `build_bench.sh` | Its bench, built from the repository's own sources |
| `results/` | Evidence; hashes in `results/MANIFEST.json` |

## 1. What existed, what was missing, what was built

**Existing** (inspected, not assumed):
- **The memory.** Exact typed evidence slots keyed by relation (type, payload, canonical participants), with polarity
  counts, first/last provenance and the last 32-bit `source_authority`.
- **The authority.** Sources with a version, a scope and a lifecycle, and the pure admission law
  `elpis_ecs_c_check_evidence`.
- **RuntimeCore ABI v7.** Admits a HOT-lane observation when its `source_authority` maps to an ACTIVE source within its
  per-turn bound. It never checks the version or scope, and the observation carries neither.
- **The WARM context lane.** Python, HACF-based; it does not consult ECS_C sources.

**Missing** (shown with the real libraries, `test_ecs_c_relational_authority`):
- *Revision.* After `SOURCE_VERSION`, memory LOOKUP still answers 1→3 at score 2 from v1 evidence, although the
  authority's law now says `SOURCE_STALE`.
- *Dormancy.* After DORMANT it still answers.
- *Reclamation.* After TERMINATED, reclamation and a new source on the same `source_authority`, the old slot's
  provenance resolves to the new source, which never asserted it.
- *Bindings.* Nothing in ECS_C can say which evidence rests on which (source, version, scope), or what a revocation
  would touch.

**Built.** The ECS_C relational view (`docs/ECS_C_RELATIONAL.md`):
- per-binding contributions over the real ECS_C types;
- LIVE = `check_evidence(A, binding) == OK`, and the view equals a memory replay of the LIVE evidence;
- dependency-indexed sync with successor/regression laws;
- compare-and-admit;
- provenance and dependency reads;
- one-allocation bounds, atomic refusals and integrity checks.

## 2. Verification

| Instrument | Result |
|---|---|
| `ECS_C.test_ecs_c_relational` | PASS. The differential: 2,274,048 LOOKUP answers, 189,504 slots and 2,632 states equal a fresh real ECS_C memory replaying only the LIVE observations, with 0 mismatches (8 seeds × 1,200 steps). Non-vacuous: 75% of answers name a filler, 49% carry conflicts, 48% contradictions. |
| `ECS_C.test_ecs_c_relational_authority` | PASS (real authority, production libraries): the gap; the view's law equals `check_evidence` on 750 probes over 5 states; a real `RECONCILE_ROLLBACK` that lowers a version is refused `REGRESSION` until reset |
| `ECS_C.test_ecs_c_relational_alloc` | PASS: 1 heap block per view, 0 per call over 2,000 rounds of every call |
| GCC ASan+UBSan | both suites clean (clang compiles `-Werror` clean; its sanitizer runtimes are not installed on this host) |
| Mutation, round 1 (`results/mutants_round1.*`) | 30/37 killed. Of the 7 survivors, 4 were test gaps: reflip of a damaged contribution, identity coverage of counts, participant canonicalization, non-LOOKUP channels. Tests were added for all 4. |
| Mutation, round 2 | 35/39 killed; 2 mutants were added for the now explicit binding invariant. The new M39 survived (a test gap); a test on a suspended contribution was added. |
| Mutation, round 3 (final, `results/mutants_round3.*`) | **36/39 killed. The 3 survivors (M01, M02, M27) are exactly the mutants predicted equivalent before the run.** In every reachable state the binding invariant makes the terms they remove redundant; `verify` counts and `rebuild` refuses any state that violates it. The control passes both suites. |
| Boundary suite (`ELPIS_NATIVE_BUILD` set) | 189 passed, 10 skipped (donor parity: `ELPIS_DONOR_ROOT` not set), 0 failed; the ABI inventory is checked on the binaries |
| Full native CTest (182 tests) | 181 passed. `runtime.test_autonomous_containment` fails here with `privileged launch refused (errno=1)`, the sandbox's EPERM. No runtime, substrate, ECS_G or `src/` file differs from `d2c1b27`. |
| `tests/research/ecsc_hca_native` | 110 passed (the HCA substrate is unchanged) |

## 3. HCA_REVISION_R0: is there a regime where retained HCA dynamics help after an ECS_C revision?

**The question.** After an ECS_C source revision, can any native HCA comparator re-establish a correct downstream
readout more cheaply than an equally capable exact incremental method, or give a more useful partial readout under a
bounded cost? The interface is one way:

```
ECS_C view → export → research adapter → native HCA → SALIENCE_ONLY readout
```

The protocol (`PROTOCOL_HCA_REVISION_R0.md`, committed and pushed before the bench ran) fixes the workload, the
methods, the costs, the hypotheses H1–H4 and the disposition rule.

**Runs**

* **Predeclared run** (`results/hca_revision_r0_predeclared.json`). The bench of `471ef8a` with one fix
  (`results/bench_predeclared_admit_fix.patch`: a macro evaluated its row argument twice, drawing two random sources
  for one observation). The first attempt aborted before any output, when the view itself refused the misbound
  observation `BINDING_MISMATCH`. The run was reproduced from `471ef8a` plus the patch, with identical logical results.
* **Instrumented run** (`results/hca_revision_r0_instrumented.json`; source `bench_hca_revision.c` as committed). It
  adds three post-hoc instruments, labelled `posthoc_*` or described below, and changes no predeclared quantity:
  * carry-over by graph identity, for H3;
  * a timing-free H4 variant;
  * a breakdown of M-B0's cost.

  Every logical result equals the predeclared run's. Two intermediate runs, each adding one instrument at a time,
  produced the same logical results; they are superseded and not archived.

**Mechanics gates: PASS.** On all 560 events (14 cases × 40):
- M-INC, M-FULL, M-B0, M-B1 and M-B2 are exact (M-B2 by at most 15 ticks).
- B0 equals the independent exact closure.
- The bench's live set equals the view's export.
- Every logical count repeats exactly across the 5 replicates.

### Full-cost wall time per event (median of 5 replicates, µs; predeclared run)

| Regime | Scale | Seed | Cells / factors | Changed relations | M-INC | M-FULL | M-B0 | M-B1 | M-B2 (ticks) |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| R-LOCAL | 32 | 1 | 65 / 40 | 8.0 | 0.9 | 42.2 | 65.3 | 70.8 | 154.9 (2.0) |
| R-LOCAL | 64 | 1 | 99 / 72 | 8.0 | 0.9 | 72.5 | 110.4 | 119.9 | 272.2 (2.5) |
| R-LOCAL | 128 | 1 | 166 / 136 | 8.0 | 1.1 | 152.5 | 235.3 | 250.0 | 701.4 (4.0) |
| R-LOCAL | 256 | 1 | 216 / 264 | 8.0 | 1.9 | 389.2 | 582.7 | 629.3 | 1211.6 (3.0) |
| R-LOCAL | 440 | 1 | 244 / 448 | 8.0 | 1.4 | 527.1 | 751.6 | 786.8 | 1647.0 (5.0) |
| R-RANDOM | 64 | 1–3 | 64 / 128 | 6.2–7.4 | 3.0–4.3 | 151.7–154.7 | 184.2–191.6 | 195.5–203.2 | 345.0–362.9 (3.7–4.8) |
| R-RANDOM | 128 | 1–3 | 128 / 256 | 14.1–15.2 | 8.5–10.1 | 298.7–314.3 | 392.4–415.3 | 410.2–433.9 | 880.0–955.9 (5.5–6.6) |
| R-RANDOM | 256 | 1–3 | 251–253 / 448 | 21.8–23.5 | 12.5–17.3 | 532.9–579.3 | 739.2–803.4 | 768.4–833.7 | 2169.4–2491.5 (7.9–9.3) |

**ECS_C maintenance.** All methods consume the synced view. The view's incremental sync costs 2.6–7.8 µs per event,
against 21.9–272.9 µs for its full `rebuild`. That gap grows with size (logical work 48–142 against 180–2,398).

**The M-B0 cost** (instrumented run, median µs), and the HCA comparators' structural costs:
- `elpis_hca_predict` B0 itself takes 2.0–56.2 µs.
- The rest is the export (17–318), the research adapter (9–239) and `elpis_hca_graph_create` (34–326). The native
  graph is immutable, so every revision rebuilds it.
- Even counting only `predict`, B0 is 2.2–21× slower than M-INC's complete exact update in every case.

### Logical work per event (each library's own units; not comparable across libraries)

| Regime | Scale | M-INC | M-FULL | M-B0 | M-B1 | M-B2 |
|---|---|---:|---:|---:|---:|---:|
| R-LOCAL | 32 → 440 | 30 → 30 | 495 → 3,477 | 6,037 → 52,518 | 12,516 → 113,486 | 17,623 → 162,929 |
| R-RANDOM | 64 → 256 | 95–142 → 712–744 | 1,230–1,275 → 4,235–4,341 | 15,042–15,400 → 53,877–54,349 | 32,435–32,493 → 114,904–115,103 | 36,919–43,849 → 233,501–267,266 |

### Retained bytes between events

Beyond the view, which every method uses (its configured arena is 4,544,192 bytes):
- **M-INC:** closure state of 3,440–15,176 bytes.
- **HCA:** a graph block of 17,136–89,472 bytes and a session block of 2,704 bytes, plus a 560-byte activation for
  B2. B0 and B1 rebuild, so they need not keep them.

Unlike H1D0, B2 does not win on retained bytes here.

### Hypotheses (predeclared) and outcomes

| | Prediction | Outcome |
|---|---|---|
| **H1** exact locality | M-INC's work does not grow with unrelated size | **Confirmed**: 30 units at every F_u from 32 to 440 (0.9–1.9 µs) |
| **H2** HCA locality | B0/B1/B2 work grows with F_u although every event changes 8 relations | **Confirmed**: B0 6,037 → 52,518 (8.7×), B1 9.1×, B2 9.2× |
| **H3** carry-over | the retained B2 activation is refused `INVALID_STATE` on every event | **Falsified as stated, on 10 of 546 attempts.** Carry-over is refused on 536/536 events whose post-event graph differs, and accepted on 10/10 whose graph is identical (no net change of live relations). A B2 activation is owner-bound to one graph and task, so it never survives an actual change of the ECS_C view: it is a cache of one propagation, not memory across revisions. |
| **H4** bounded-cost partial readout | no B2 tick count beats M-STALE's accuracy at less wall time than M-INC (H4 false) | **Confirmed** (no advantage): 0 of 560 events in each run. B2's setup alone exceeds M-INC's whole exact update. |

**Post-hoc, timing-free H4 check.** At any cost, a non-converged B2 readout is more accurate than the stale readout on
485/560 events: mean best partial accuracy 0.941–0.996, against 0.789–0.969 for stale. Every such readout costs more
than M-INC's exact one, so it is dominated.

### Disposition (by the predeclared rule)

```
mechanics:    PASS (exact on 560/560 events, all methods; B0 = independent closure)
scientific:   NO_HCA_ADVANTAGE_ON_ECSC_REVISION_WORKLOAD
H3 note:      prediction falsified only on no-change events; the owner law, not dynamics, decides carry-over
B3:           NOT_AUTHORIZED
```

**What this delimits.** For the source-revision workload over the ECS_C view (binary relations; max–min decayed
closure; up to the native HCA limits of 256 cells and 512 factors), exact dependency-indexed maintenance beats every
native HCA comparator on wall time and logical work. The dominant HCA costs are structural:
- the immutable graph is rebuilt on every revision;
- the activation is owner-bound, so nothing carries across a change;
- even the bare B0 `predict` is slower than the whole exact update.

**What it does not establish.**
- Nothing about HCA designs other than native B0/B1/B2.
- Nothing about workloads whose exact incremental maintenance is provably superlinear. For example, decremental
  closure with deep cyclic support, where B2's documented unsupported-cycle limitation is precisely the failure mode.
  No such workload was built: its justification is not established.
- No general performance guarantee: these are DEV measurements on one 4-CPU host.

## 4. Reproduction

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
(cd build && ctest -R relational --output-on-failure)
python3 -I research/ecsc_dynamic_relational_r0/mutants.py "$PWD" "$PWD/build" WORK_DIR OUT.json
research/ecsc_dynamic_relational_r0/build_bench.sh BENCH_BINARY && BENCH_BINARY > OUT.json
```

Wall times vary by host. Every logical quantity (exactness, work, ticks, carry outcomes, sizes) is deterministic.

## 5. Next technical milestone (one, from this evidence)

**A reviewed proposal for version-bound admission in RuntimeCore.** The proposal comes before any implementation.

The gap is canonical: the HOT lane admits by `source_authority` alone, and the composite memory keeps no binding. The
view shows the exact law and its cost: 2.6–7.8 µs per authority sync at up to 448 relations, all off the HOT lane.

The question the evidence leaves open is persistence. The view is session-local, and the composite memory persisted in
the checkpoint merges bindings away, so after a restart the view cannot be rebuilt from it. A successor must choose
one of two options:
- (a) a bounded per-relation binding table inside the composite envelope. That is a format change that needs new
  authority.
- (b) an explicit, documented rule that pre-restart evidence is unbound, with its query consequences.

That choice, and RuntimeCore's ABI v8 shape for it, should be decided by review before any code. HCA work is not the
next step: this milestone gives it no regime to stand in.
