# Cognitive R0: ECS-native stateful learning and recall

Authority for the first cognitive milestone under
[`ELPIS_MISSION.md`](ELPIS_MISSION.md): **DSV4 communicates; ECS computes,
learns and persists.** R0 asks one bounded question:

> Can the ECS_G state itself acquire and retain a bounded input -> response
> relationship through its qualified recurrence, and use that state to
> compute later responses?

**Status: QUALIFIED under the frozen synthetic Cognitive R0 regime.**
ECS_G supports stateful learned input-response computation under the
qualified Cognitive R0 regime: `research/ecs_cognition_r0` v1, frozen authority
`71fba13cdea64551e5d08818e20a1b89cc8b76f0b2fc172ac317d711d9311eb1`, QUAL evidence
`9dfe55adab68bf578511a4e0a463aef318281e1f2ac1073ef3c20910cff10b7d`
(`MECHANICS_PASS`, `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME`; all gates A-G on 8
QUAL worlds). Results: [`docs/research/COGNITION_R0_RESULTS.md`](research/COGNITION_R0_RESULTS.md).

The same QUAL found that plain sequential learning **does not retain**: learning
a second experience interfered with the first in 8 of 8 worlds,
catastrophically in 6 of 8. R0 establishes no continual-learning property.

## The limited claim

* ECS_G state `W` is the learned computational state.
* The native forward map computes responses from `W`.
* The qualified G1 recurrence can modify `W` from experience.
* Cognitive R0 tests stateful learning and recall in ECS-native numerical
  space, on a synthetic regression task.

R0 makes **no** claim that:

* anything here understands, encodes or produces natural language;
* the current cubic kernel (`phi(z) = 0.5z + 0.5z^2 + 0.5z^3`,
  `f_W(x) = sum_i phi(x . w_i)`) is the final architecture;
* `d = 6`, `N = 36` or a single ECS_G kernel is the eventual ECS/EDEN
  organism;
* `S3(W)` is globally sufficient state (it determines the instantaneous
  forward map of this family, not its future transitions);
* gradient descent on `(X, y)` is the eventual general cognitive law;
* `(X, y)` is the permanent semantic input format, or `f_W(x)` / `S3` the
  permanent output representation, of Elpis.

## QUERY and LEARN are different operations

| | QUERY | LEARN |
|---|---|---|
| input | query rows `x` (`rows x dim`, finite) | experience `(X, y)` |
| computation | `f_W(x)` by the native executor forward (bitwise the reference `elpis_ecsg_forward_f64`) on the current authoritative `W`, one native call | `K` qualified G1 steps in one native executor call (bitwise `K` reference steps) |
| effect | none: read-only | one native commit of the candidate: `W -> W'`, epoch `+K` |
| failure | refused; nothing changes | refused; authoritative `W` and epoch unchanged |

A response is never "a state mutation" and never "S3". Learning is never
"run a step and inspect S3".

## What is learned state

The only learned, mutable state is `W` (with its epoch), owned by the native
ECS_G state. A snapshot persists exactly `W` and the epoch. The learning rate
and the step budget are fixed program parameters, not learned state.
Transition receipts are returned to the caller and not retained. `S3` is a
diagnostic projection only; nothing is run from `S3` alone.

No training example, label, lookup table, cache, fixture-side state or
model output may participate in answering a query.

## The runtime scaffold is not R0 semantics

`src/elpis/runtime/cognition.py` (`Stimulus` as a native-ready ordered
experience schedule executed by native K1, `Readout` as `S3` of the candidate
`W`) is the fail-closed mechanical boundary for a future semantic codec. It does not define cognition. R0 uses `(X, y)` because it is
the qualified recurrence's actual input and `f_W(x)` because it is the
qualified forward map; neither choice is promoted beyond R0. Production text
generation stays fail-closed.

## Prohibited shortcuts (gate-enforced)

R0 cannot be satisfied by importing or calling `research/dsv41_tower`,
`DSV41Target`, principal model execution, transformer attention, MoE/expert
inference, or any external learned predictor. `tests/boundary/test_cognition_r0.py`
and `tests/ECS_G/test_cognition_r0_contract.py` enforce that the cognitive
core imports nothing but ECS_G and the standard library, that a clean process
learns and answers with no model machinery loaded, and that responses depend
on ECS state: they follow `W`, change when `W` changes, return when `W` is
restored and vanish when `W` is replaced.

## Qualification plan

`research/ecs_cognition_r0` follows the repository's research protocol: DEV
on DEV worlds only, a write-once freeze of the specification, pass rules and
laboratory source, then one QUAL run on disjoint QUAL worlds with immutable
evidence. A failure is recorded as it is. It measures:

* **A. learning**: held-out error after learning versus the same
  initialization without learning;
* **B. state causality**: learned versus reset `W`, and `W` transplanted into a
  fresh state;
* **C. persistence**: snapshot, a clean process, restore, identical
  predictions;
* **D. continual update**: a second experience changes `W` and future
  responses; acquisition and interference/retention are both reported;
* **E. negative controls**: no learning, shuffled targets, an untrained fresh
  state, and an external-state-leak guard;
* **F. determinism**: same initial `W`, same ordered experience, same
  responses and state;
* **G. microstate authority**: two microstates with identical `S3` answer
  identically but diverge after the same learning step, so `W`, not `S3`, is
  authoritative.

## Canonical surface

* `elpis.ECS_G.native.Executor`: the runtime form of the state
  ([`ECS_RUNTIME_R1.md`](ECS_RUNTIME_R1.md)): `forward` / `forward_into`,
  `learn(X, y, rate, K)` (one native call), `transaction()` (native
  candidate, commit refused `STALE` if the state moved since it began).
  `WorldState` is the scalar reference state the executor is bitwise equal
  to.
* `elpis.ECS_G.cognition.CognitiveCore`: `query` / `query_into`,
  `learn(..., steps=K)` returning a `Transition` receipt (or the native
  `Commit` with `receipt=False`), `snapshot` / `restore`, `identity`,
  `epoch`, `s3` (diagnostic).

Since Runtime R1 the core runs on the native executor. On every host, the
current runtime reaches the recorded gates, disposition and interference
classes, with every metric within the declared tolerance (1e-9 relative)
(`tests/research/ecs_cognition_r0/test_runtime_regression.py`, never
skipped). The recorded QUAL bytes are reproduced exactly only under the
recorded numerical profile and an FMA OpenBLAS kernel
(`test_lab.py::test_qual_measurements_reproduce_exactly`; see
`docs/research/COGNITION_R0_RESULTS.md`, Reproduction contract).

Only these mechanics are promoted. The laboratory harness, the synthetic
task, its teachers and its thresholds stay in `research/` and grant nothing.

## Missing edges

* **ECS <-> DSV semantic codec**: not defined; text generation unavailable.
* **FMS**: current FMS materializes immutable, digest-verified file assets.
  It cannot yet hold mutable ECS state; `W` lives in native ECS_G memory and
  persists through its snapshot bytes.
* **HACF -> ECS**: no structural-memory edge into ECS is qualified.
* **ECS_C**: transitions return receipts but are not recorded in the history;
  cognition does not depend on history or replay.
* **Retention**: no mechanism protects earlier learning from later
  experience; the qualified recurrence alone forgets. Retention R0
  (`docs/research/ECS_RETENTION_R0_RESULTS.md`, RESEARCH_ONLY) found only a
  partial reduction of forgetting at a cost in acquisition; nothing is
  qualified or canonical. Retention R1 v1 (`research/ecs_retention_r1`,
  RESEARCH_ONLY; `docs/research/ECS_RETENTION_R1_RESULTS.md`) stopped at DEV
  as `TASK_INVALID_ON_DEV`: its primary task was not a valid retention test, so
  no candidate was evaluated and nothing is qualified or canonical. Retention R2
  v1 (`research/ecs_retention_r2`, RESEARCH_ONLY;
  `docs/research/ECS_RETENTION_R2_RESULTS.md`) ran on a valid aliasing-conflict
  task and ended `PARTIAL_REDUCTION`: the selected candidate retained every
  experience in 24/24 QUAL worlds, but the causality, native-feasibility and
  numerical-robustness gates failed, so nothing is qualified or canonical.

* **Retention R3** (`research/ecs_retention_r3`, RESEARCH_ONLY;
  `docs/research/ECS_RETENTION_R3_RESULTS.md`) ended `OUTCOME_A`: fixed-size K1
  consolidation retained every experience in 32/32 QUAL worlds and was causal
  (removal and reset lose it, transplant restores it). K1 is eligible for the native
  milestone; until that milestone merges, no retention mechanism is canonical.

* **Native K1 milestone** (`research/ecs_k1_native`, `docs/research/ECS_K1_NATIVE_RESULTS.md`):
  K1N-v1 remains historical `NOT_QUALIFIED`. K1N-v2 is **QUALIFIED**: E1-E5,
  L1, L2, D1_Q and D1_F passed on 32 Q and 32 fresh F worlds. Native K1 is
  canonical through the exact admitted v2 attestation in
  `research/ecs_k1_native/evidence/ecsg-k1-native.v2.attestation.json`.
