# Qualification specifications (scaffolding)

A qualification specification states, **before any run**, the question, the frozen inputs, the procedure, the
gates and the evidence a qualification would produce, and what passing would and would not establish. It grants
nothing by existing. Every specification here is `NOT RUN` and its subject is `UNQUALIFIED`.

| Specification | Subject (implemented) | Status |
|---|---|---|
| [Cognitive Boundary R1](COGNITIVE_BOUNDARY_R1.md) | QUERY/LEARN separation, codec admission, fuel, managed ownership, native admission (RuntimeCore ABI v3) | NOT RUN · UNQUALIFIED |
| [K1 Recovery R0](K1_RECOVERY_R0.md) | the bounded continuity-authorized K1 checkpoint owner (docs/K1_RECOVERY_R0.md) | NOT RUN · UNQUALIFIED |
| [HACF-ECS Scientific Bridge R0](HACF_ECS_BRIDGE_R0.md) | research/hacf_ecs_bridge_r0 (RESEARCH_ONLY) | NOT RUN · UNQUALIFIED |
| [Evolution Fitness R0](EVOLUTION_FITNESS_R0.md) | research/evolution_fitness_r0 and the evolution policy authority (docs/EVOLUTION_POLICY.md) | NOT RUN · UNQUALIFIED |

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
