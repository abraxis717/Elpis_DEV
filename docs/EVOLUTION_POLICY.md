# Evolution policy authority: evolution may propose; it may not authorize or evaluate itself

Status: implemented systems mechanism (`src/elpis/evolution/policy.py`; `Runtime.evolve`). UNQUALIFIED: no evaluator,
fitness environment or promotion is qualified; the fixtures under `tests/evolution/` are TEST_ONLY. Qualification
specification: `docs/qualification/EVOLUTION_FITNESS_R0.md` (NOT RUN).

## The finding

Before this authority, every authority-bearing value of an evolution attempt came from its caller:

| Value | Who set it |
|---|---|
| allowed component scopes, resource budget digest, evaluation contract digest | `EvolutionPathGate(...)` constructor arguments, passed by the caller of `Runtime.evolve` |
| edit budget | the assertion itself (`edit_count <= edit_budget`, both chosen by the proposer) |
| correctness / leakage / resource / source-scope verdicts, held-out and OOD deltas | `EvaluationEvidence` fields, filled in by whoever built the record |
| partitions | the evidence (checked only against the dict the caller of `select` passed) |
| materialization | `atomic_materialize`, callable with any manifest |

A candidate could therefore configure its own gate, grant itself any budget, report its own evaluation and select
itself. `tests/evolution/test_policy.py::test_self_reported_or_altered_evidence_is_never_weighed` keeps the
reproduction: a forged record still passes the bare promotion law.

## The authority

An `EvolutionPolicy` (`elpis.evolution-policy.v1`) is a JSON document whose SHA-256 must equal an independent pin
from trusted configuration (`RuntimeConfig.evolution_policy_sha256`), never from a candidate. It fixes:

| Field | Law |
|---|---|
| `objective` | the promotion law's `held_out_delta`, `MAXIMIZE` (the only objective the law knows) |
| `evaluator` | `evaluator_id`, `implementation` (`module:qualname`) and `implementation_sha256`; the authority **measures** the evaluator callable's defining source file and refuses anything else (`EVALUATOR_UNPINNED`), on construction and on every use |
| `candidate_scopes` / `protected_scopes` | normalized relative paths; disjoint (no scope equal to, inside or containing a protected one). The evaluator, the policy and the fitness environment belong in protected scopes |
| `budget` | `max_edit_budget` (an assertion may not claim more: `EDIT_BUDGET_NOT_AUTHORIZED`), `max_resource_cost` (at most the promotion law's 120), `max_candidates` (1..64 per selection) |
| `evaluation_contract` | the four partition manifests (EVOLVE, CALIBRATION, HELD_OUT, OOD), pairwise distinct; the law's noise envelope (3); an OOD floor at least the law's (-2). A policy can only narrow the law |
| `side_effects` | `CANDIDATE_WORKSPACE_ONLY`: an evaluation must leave both workspaces byte-identical (`EVALUATION_SIDE_EFFECT`) |
| `confinement` | `TRUSTED_OPERATOR_CALLBACKS`: the evaluator and an attempt's `advance` callback are trusted operator code. **Nothing is sandboxed** |
| `promotion` | `requires_operator_approval: true` (no autonomous promotion) |

The resource-budget and evaluation-contract digests an assertion must carry are derived from the policy (and bind
the policy's own digest and the evaluator identity), so an assertion is bound to exactly one policy.

`EvolutionPolicyAuthority(policy, evaluator)` is the only issuer of:

1. **Gates** (`gate()`): scopes, budgets and contract from the policy alone. `Runtime.evolve` refuses a gate the
   configured policy did not issue, or an issued gate whose values were altered (`EVOLUTION_POLICY_UNAUTHORIZED`),
   and refuses everything without a pin (`EVOLUTION_POLICY_UNPINNED`), before RuntimeCore reserves anything.
2. **Evaluations** (`evaluate(parent, candidate, parent_root, candidate_root)`): both workspaces must be their
   manifests (`WORKSPACE_MANIFEST_MISMATCH`); the authority computes the changed components, the edit count and the
   source-scope verdict itself; a candidate that carries the evaluator's bytes, or an evaluator loaded from a
   workspace under evaluation, is refused (`EVALUATOR_IN_CANDIDATE`); the pinned evaluator returns a typed
   `EvaluatorReport` (`EVALUATION_INVALID` otherwise); the evidence carries the policy's partitions, envelope and
   contract digest, and the resource verdict against the policy's limit.
3. **Selections** (`select(parent, candidates)`): only evaluations this authority issued, unaltered (evidence digest,
   manifest and edit count), are weighed; anything else is rejected `EVALUATION_NOT_INDEPENDENT`, never weighed. The
   policy's OOD floor applies before the law (`OOD_REGRESSION_BEYOND_POLICY`).
4. **Promotion grants** (`approve(receipt, operator_approval_digest=...)`): only for a `SELECT_CHALLENGER` this
   authority issued, against an explicit operator approval digest. `materialize(grant, ...)` writes the child only
   for an issued grant and exactly the granted manifest (`PROMOTION_UNAUTHORIZED` otherwise; nothing is written).

`elpis.runtime.persistence` registers `EvolutionPolicyAuthority.materialize` with `atomic_materialize` as the
operator writer `evolution_child_materialization`; `evolve` stays an operator operation.

## Non-claims

* Pins authenticate exact bytes, not an author. The evaluator digest covers its defining source file only, not its
  imports or the interpreter.
* Issuance registries are process-local. They do not survive restart and are no protection against in-process code
  that builds its own policy, pin and authority; the deployment pin in `RuntimeConfig` is what the managed runtime
  trusts.
* `TRUSTED_OPERATOR_CALLBACKS` means exactly that: no sandbox, no syscall filter, no resource isolation for the
  evaluator or `advance`. The side-effect check covers the two workspaces only.
* `atomic_materialize` remains the low-level operator primitive; it does not itself require a grant.
* No objective is shown to measure anything of value. An evaluator needs an independent fitness environment to
  measure anything real; none is qualified.
