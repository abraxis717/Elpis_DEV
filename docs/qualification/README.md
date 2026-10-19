# Qualification specifications (scaffolding)

A qualification specification states, **before any run**, the question, the frozen inputs, the procedure, the
gates and the evidence a qualification would produce, and what passing would and would not establish. It grants
nothing by existing. Every specification here is `NOT RUN` and its subject is `UNQUALIFIED` in this repository,
except two:
- **K1 Recovery R0 under ABI v4**, `QUALIFIED` for commit `18f2f4d` only;
- **ECS_C Recovery Q0 under ABI v6**, `QUALIFIED` for commit `117b41a` only.

Each is `RUN PASS` with write-once evidence in this repository, within its stated scope (its "Qualified scope").

| Specification | Subject (implemented) | Status |
|---|---|---|
| [Cognitive Boundary R1](COGNITIVE_BOUNDARY_R1.md) | QUERY/LEARN separation, codec admission, fuel, managed ownership, native admission (RuntimeCore ABI v3) | NOT RUN · UNQUALIFIED |
| [Cognitive Boundary R2](COGNITIVE_BOUNDARY_R2.md) | the same boundary under RuntimeCore ABI v4, plus the durability lane, HOT residency, composite observation authority and context isolation | NOT RUN · UNQUALIFIED |
| [Cognitive Boundary R2 under ABI v5](COGNITIVE_BOUNDARY_R2_V5.md) | the boundary under RuntimeCore ABI v5: R2's laws plus observe-only turn authority, memory mutation without K1 learning, `max_observations`, ABI compatibility and refusal of obsolete callers | NOT RUN · UNQUALIFIED |
| [K1 Recovery R0](K1_RECOVERY_R0.md) | the bounded continuity-authorized K1 checkpoint owner (docs/K1_RECOVERY_R0.md), as pre-registered against ABI v3 | NOT RUN · UNQUALIFIED |
| [K1 Recovery R0 under ABI v4](K1_RECOVERY_R0_V4.md) | the same owner under ABI v4: volatile LEARN, explicit publication, bare K1 and composite lineages | RUN PASS · QUALIFIED (commit `18f2f4d` only; crash consistency under `SIGKILL` and lost-sync faults, testing build; not power loss) |
| [ECS_C Recovery Q0 under ABI v6](ECS_C_RECOVERY_Q0.md) | RuntimeCore ABI v6's durability protocol over the ECS_C authority through the native host: process death, state-directory attacks until explicit operator reconciliation, fixed storage and recovery work; the qualification standalone continuity's retirement waited for | RUN PASS · QUALIFIED (commit `117b41a` only; crash consistency under `SIGKILL` and refusal of the T1 attacks until explicit operator reconciliation, testing build; not power loss) |
| [HACF-ECS Scientific Bridge R0](HACF_ECS_BRIDGE_R0.md) | research/hacf_ecs_bridge_r0 (RESEARCH_ONLY) | NOT RUN · UNQUALIFIED |
| [Evolution Fitness R0](EVOLUTION_FITNESS_R0.md) | research/evolution_fitness_r0 and the evolution policy authority (docs/EVOLUTION_POLICY.md) | NOT RUN · UNQUALIFIED |
| [ECS <-> DSV4 Codec R0](ECS_CODEC_R0.md) | the codec interface and its ECS-dependence gates (no map exists) | NOT RUN · UNQUALIFIED |

**Successors, not rewrites.** RuntimeCore ABI v4 superseded ABI v3's per-turn durability (docs/RUNTIME_CORE.md). The
ABI v3 specifications (Cognitive Boundary R1, K1 Recovery R0) stay exactly as pre-registered, historical authority for
that ABI; their ABI v4 successors are separate documents with their own evidence directories.

**The R1 closure.** [closures/COGNITIVE_BOUNDARY_R1_CLOSURE_20261011.md](closures/COGNITIVE_BOUNDARY_R1_CLOSURE_20261011.md)
records a retrospective, evidence-bound closure of Cognitive Boundary R1 (`QUALIFIED_UNDER_FROZEN_R1_SCOPE`) for commit
`bec6da7` only: RuntimeCore ABI v3, 39 commits before the ABI v4 head at which this closure was recorded (the current
head is ABI v6). Its evidence is held outside this repository, so in this repository R1's status stays `NOT RUN` (the
rule below), and nothing of the closure extends to ABI v4 or later code: Cognitive Boundary R2 is the open gate for
it.

Rules shared by every specification:

* **Write-once evidence.** A qualification run writes its evidence once, under the subject's own evidence directory,
  bound to the exact source and binary digests it measured; it is never edited afterwards. A re-run is a new
  evidence record.
* **DEV before QUAL.** Any choice a run makes (a threshold, a step budget, a map) is made on DEV inputs only, and
  frozen before QUAL inputs are opened. QUAL inputs are disjoint from DEV inputs.
* **No forged status.** A status changes from `NOT RUN` only together with the evidence it names
  (`tests/boundary/test_qualification_specs.py` refuses a `QUALIFIED` status without an evidence file it can
  digest-check). Mechanics tests are not qualification evidence.
* **Bounded runs.** Every run is bounded in worlds, steps and wall-clock by the specification itself; none of
  these specifications asks for a large seed or world cohort, a long simulation, or any H-ECS run.

## Interface readiness (current head)

Categories: **implemented** (code exists), **mechanically verified** (deterministic tests over real native libraries,
with mutation or adversarial negative controls), **mechanically incomplete** (a mechanical piece is missing),
**blocked by scientific authority** (needs a new, separately qualified law or map), **formally qualified** (a frozen
specification run with in-repository evidence: K1 Recovery R0 v4, for commit `18f2f4d` only; ECS_C Recovery Q0, for
commit `117b41a` only), **explicitly
unqualified** (implemented and refused or unreachable by design).

| Interface | Implemented | Mechanically verified | Mechanically incomplete | Blocked by scientific authority | Formally qualified | Explicitly unqualified |
|---|---|---|---|---|---|---|
| Native K1 law `(W, epoch, H, a)` | yes | yes (native suites, legacy differential) | — | — | K1N-v2 attestation (research, frozen regime) | — |
| RuntimeCore v6 QUERY / volatile LEARN | yes | yes | no preemptive deadline (no K1 cancellation point) | — | no (R2 and R2 v5 not run; R2 v5 names ABI v5 and cannot cover v6) | — |
| Durability lane (recoverable / identity-only, tickets, crash law) | yes | yes (simulated and real SIGKILL matrix, ticket law, UNKNOWN window) | — | — | K1 Recovery R0 v4 at commit `18f2f4d` only, over continuity (crash consistency under `SIGKILL`, torn writes and lost-sync faults; testing build); over the ECS_C authority: ECS_C Recovery Q0 at commit `117b41a` only | power-loss durability (never claimed) |
| ECS_C authority through the native host (ABI v6) | yes | yes (the authority's crash, T1 adversary and boundedness suites; the host's end-to-end, crash, containment and fixed-file-set suites; the Q0 instrument's DEV rehearsal) | — | — | ECS_C Recovery Q0 at commit `117b41a` only (crash consistency under `SIGKILL`, the T1 attacks refused until explicit operator reconciliation, fixed storage and recovery work; testing build) | power-loss durability (never claimed); a deep rollback, once accepted, leaves `CHECKPOINT_MISSING` (re-authorization not implemented) |
| HOT residency (FMS holds, NOT_RESIDENT, explicit materialization) | yes | yes (CI-HOTRES-1 mutants) | — | — | no | — |
| ECS composite through the Python offline reference | yes | yes (incl. real SIGKILL; the observe-only turn's laws and mutants) | — | — | no | memory LEARN / QUERY FAST (no capability is ever set), SLOW (not provided) |
| Typed HACF -> memory observation transport | yes | yes (adversarial records, mutants) | — | a HACF -> ECS meaning map | no | semantic interpretation |
| Contextual (WARM) lane | yes | yes (bounds, immutability, isolation, no ECS path) | — | a contextual dynamics law (HCA) | no | — |
| ECS <-> DSV4 codec | protocol and fail-closed refusal only | the refusal and the D1-D3 gate mechanics | token-distribution and memory-LOOKUP protocol methods (proposals) | the map itself | no (ECS Codec R0 NOT RUN) | text generation (`ECS_CODEC_UNQUALIFIED`) |
| HACF-ECS Scientific Bridge R0 | research infrastructure | its harness mechanics | — | real evidence, a map, a meaningful state | no | runtime authority |
| Evolution policy + Fitness R0 | yes | yes (policy gates, evaluator recomputation) | — | meaningful selection advantage, adaptive capability | no (NOT RUN) | autonomous improvement |
| H-ECS hierarchy | research only | — | — | closed: NO_ADVANTAGE_OVER_WIDTH | C2/C2M constituent law (research) | hierarchy integration (NOT AUTHORIZED) |

**Implemented: the observe-only managed turn (RuntimeCore ABI v5).** The proposal recorded here is now
`elpis_runtime_turn_begin_observe` (composite substrates only; no K1 schedule; the K1 component byte-identical; a
commit with nothing observed refused; the turn's observations bounded natively by `budget.max_observations`) and
`Runtime.run_observe` (docs/RUNTIME_CORE.md, "Observe-only turns (ABI v5)"). It is a versioned ABI change (v5): the
budget and counters changed layout, every v4 entry point keeps its meaning, and the envelope and continuity formats are
unchanged.

## Open formal gates

Cognitive Boundary R2 under ABI v5 (its instrument is not built, and it is runnable only at an ABI v5 commit; R2 itself
was superseded before any run); ECS Codec
R0 (blocked by the absence of any map); HACF-ECS Scientific Bridge R0 (blocked: no frozen evidence, map or meaningful
state); Evolution Fitness R0 (runnable as specified; its scope is the instrument only).

**Evolution Fitness R0, what its gates can and cannot reach.** Deterministic policy mechanics (G1: identical reports
and trajectory identities across processes) and evaluator independence (fitness recomputed from replayed trajectories,
self-reports never read; the policy's pinned evaluator) and adversarial candidate rejection (G2) are in its scope, as is
the selection's agreement with its frozen contract (G3) and the absence of side effects (G4). A meaningful selection
advantage and any general adaptive capability are not: the corridor objective is a toy, and no gate measures transfer
(open question S7). The leakage gate matches evaluation-world identifiers and partition digests literally in workspace
files; an encoded copy is not detected, which is harmless here only because a candidate is a declarative table that
never observes a world identifier.

## Open scientific questions (each needs new evidence; none is resolvable by software integration)

| # | Question | Hypothesis form | Controls | Falsifiers | Data | Freeze |
|---|---|---|---|---|---|---|
| S1 | An ECS <-> DSV4 codec map | ECS Codec R0, "Question" | readout-blind, untrained, shuffled-state, byte-matched lookup table, n-gram | any D-gate failure; no margin over lookup / n-gram | a frozen prompt/response corpus, DEV/QUAL/paraphrase | map, parameters, thresholds after DEV |
| S2 | A HACF -> ECS observation map | VALID evidence changes read-only K1 answers in the direction the map predicts, MISLEADING_WELL_FORMED does not | WITHHELD, PERMUTED, STALE, MISLEADING (the bridge harness) | VALID indistinguishable from MISLEADING_WELL_FORMED on QUAL | a pinned corpus, context-graph snapshot and recorded bundles | the map and its pin before DEV |
| S3 | A meaningful trained ECS state for a structural domain | K1 trained on a frozen structural task answers held-out structure queries above controls | untrained W, shuffled training, lookup table | no margin over controls | a frozen structural task (not Cognition R0's synthetic regime) | the training regime before QUAL |
| S4 | A contextual dynamics law (e.g. HCA) over context views | a law over admitted relations improves a pre-registered downstream measure without changing truth status | no-dynamics view, shuffled relations | no improvement; any promotion of a CANDIDATE to authority | frozen context sessions | the law before DEV |
| S5 | Recurrent memory learning (memory FAST) | beats byte-matched exact storage on retention or generalization | exact storage at equal bytes (MF-R0 H4) | H4 NONPASS repeated | a new preregistered regime | a new experiment version |
| S6 | Slow-bank consolidation of the memory | a qualified consolidation law (Retention R1-R3 guards) | plain storage, K1 alone | as the Retention laboratories | a new preregistered regime | a new experiment version |
| S7 | Meaningful selection advantage of evolution | selected candidates improve a held-out objective beyond the noise envelope repeatedly, out of distribution | negative baselines, random edits | no advantage; OOD regression | environments beyond the toy corridor | the objective and evaluator before DEV |
