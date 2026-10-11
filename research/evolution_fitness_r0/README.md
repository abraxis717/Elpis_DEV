# Evolution Fitness R0: an independent fitness environment (infrastructure)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `UNQUALIFIED` · `NO_CLAIM`

An evolution policy (`elpis.evolution.policy`, docs/EVOLUTION_POLICY.md) can only be as independent as its
evaluator. This laboratory provides an evaluator that measures something outside the candidate: a tiny,
deterministic environment whose fitness the evaluator recomputes itself.

| Component | Where | Law |
|---|---|---|
| State / action / transition | `environment.py` | corridor of `length` cells; `LEFT` / `STAY` / `RIGHT`, clamped; ends at the goal or after `2 * length` steps |
| Feedback | `environment.py` | +10 at the goal, -1 for every other step |
| Terminal evaluation | `environment.py` | an episode's score is the sum of the feedback the environment assigns, recomputed from the actions (`replay`) |
| Replay identity | `environment.py` | hash chain over spec, world and every `(state, action, feedback, next state)`; `verify` replays a recorded trajectory step by step |
| Frozen partitions | `partitions.py`, `frozen/partitions.json` | CALIBRATION (4), DEVELOPMENT (8; the promotion law's EVOLVE), HELD_OUT (12) on a 9-cell corridor; OOD (8) on a 17-cell corridor; pairwise disjoint; digests write-once |
| Negative baselines | `partitions.py` | `ALWAYS_STAY`, `ALWAYS_LEFT`, `AWAY`: a candidate must beat all of them on HELD_OUT to pass correctness |
| Agent | `agent/policy.json` in a workspace | data only: every observation (direction to goal, distance bucket) -> action. Nothing from a workspace is executed |
| Evaluator | `evaluator.py` | refuses partitions other than the frozen ones; self-checks on CALIBRATION; recomputes HELD_OUT and OOD deltas from replayed trajectories; leakage fails when a workspace contains an evaluation world id or partition digest; never reads what a candidate writes about itself |
| Policy binding | `evaluator.policy_spec()` | an `elpis.evolution-policy.v1` spec pinning this evaluator (measured) and these partitions |

The agent never observes a world identifier, so a candidate can only overfit the evaluation partitions by carrying
their identifiers, which the leakage gate detects. Nothing is written and nothing is learned by ECS here.

Tests: `tests/research/evolution_fitness_r0` (scientific lane). The fixtures `SEEK` / `HESITANT` are TEST_ONLY
tables; no claim is made about any evolutionary process. No qualification has been run; the specification is
`docs/qualification/EVOLUTION_FITNESS_R0.md` (NOT RUN).
