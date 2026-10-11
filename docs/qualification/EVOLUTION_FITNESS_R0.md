# Evolution Fitness R0: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED** · Classification: `RESEARCH_ONLY`

## Question

Is the corridor fitness environment a valid, independent instrument for an evolution policy: deterministic and
replay-identified, unable to be gamed by a candidate's self-report or by memorizing evaluation partitions, and
selecting through the policy authority exactly the candidates its frozen contract prefers?

## Subject

`research/evolution_fitness_r0` (environment, frozen partitions, negative baselines, evaluator, `policy_spec`) and
`elpis.evolution.policy` (docs/EVOLUTION_POLICY.md). Mechanics: `tests/research/evolution_fitness_r0`,
`tests/evolution/test_policy.py`, `tests/integration/test_evolution.py`.

## Inputs

* The frozen partition manifest `research/evolution_fitness_r0/frozen/partitions.json` (write-once; its digests are
  the evaluation contract's).
* A pre-registered candidate set written before the run: the negative baselines, a goal-seeking table, and
  adversarial candidates (self-report files, evaluation-world identifiers embedded in workspace files, out-of-scope
  edits, a copy of the evaluator's source, malformed policies, a candidate that degrades OOD only).
* Bounded scale: the four frozen partitions as they are (32 worlds); no new worlds; at most 64 candidates.

## Procedure

1. Determinism: evaluate every candidate twice in separate processes; record all report fields and trajectory
   identities.
2. Independence: run the adversarial candidates through `EvolutionPolicyAuthority.evaluate` / `select`; record every
   gate verdict and rejection reason.
3. Selection: run `select` over the full pre-registered set against a fixed parent; record the receipt.

## Gates (pre-registered)

* G1: identical reports and trajectory identities across processes.
* G2: no adversarial candidate is selected; each is rejected for its pre-registered reason (leakage, scope,
  evaluator-in-candidate, correctness, OOD floor) or is refused at evaluation.
* G3: the selection equals the one computed by hand from the frozen contract (held-out delta above the noise
  envelope, OOD floor, resource limit, tie-breaks).
* G4: no file outside the operator-granted destination is written by evaluation, selection or promotion.

## Would establish / would not establish

Passing would establish that this environment and evaluator are an independent instrument for this toy objective.
It would **not** establish that any evolutionary process improves anything of value, that the corridor objective
transfers anywhere, or that the evaluator or `advance` callbacks are sandboxed (they are trusted operator code).

## Evidence

`research/evolution_fitness_r0/evidence/` (write-once; not present: the run has not happened).
