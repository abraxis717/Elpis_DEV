# ECS_G Cognitive R0 qualification laboratory

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `SEMANTICS=NONE`

Question (docs/COGNITION_R0.md): can the ECS_G state itself acquire and retain
a bounded input -> response relationship through its qualified G1 recurrence,
and answer later queries from that state?

Every response is the native forward map of the current `W`
(`elpis.ECS_G.cognition.CognitiveCore.query`), and every learning step is the
qualified G1 recurrence committed atomically (`CognitiveCore.learn`). No other
model, table or cache participates. The laboratory imports the canonical core
in one direction only and is never imported by `src/`.

## Regime (fixed before DEV)

`d=6`, `N=36`, `W0 ~ N(0, 0.18^2)`, learning rate `0.002` (the frozen Branch36
reference). Two independent cubic teachers (4 columns, scale 0.5) generate
targets; experience A uses inputs `N(+0.5 e0, 0.5^2 I)`, experience B
`N(-0.5 e0, 0.5^2 I)`; 64 experience rows and 256 held-out rows each. Metric:
`nmse = mean((f_W(x) - y)^2) / var(y)` on held-out rows. 4 DEV worlds, 8
disjoint QUAL worlds.

The pass rule (`experiment.PASS_RULE`) was written before DEV. DEV chooses only
the step budget, by `experiment.DEV_CHOICE`, on DEV worlds only.

| | question | gate |
|---|---|---|
| A | learning | held-out nmse <= 0.5 and <= half the untaught nmse, per world; median <= 0.25 |
| B | state causality | reset `W` reproduces the untaught responses bitwise; the learned `W` values transplanted into a fresh state reproduce the learned responses bitwise; `W = 0` answers 0 |
| C | persistence | a clean process given only snapshot bytes and the spec reproduces the learned responses bitwise |
| D | continual update | experience B changes the state, the epoch and the B responses, and B is acquired; retention of A is **measured and classified, not gated** |
| E | negative controls | untaught responses unchanged; permuted targets and an untrained fresh state fail the learning criterion; median shuffled nmse >= 4x learned; leak guard |
| F | determinism | the same world learned twice in-process and once in a clean process gives identical snapshot and receipt digests |
| G | microstate authority | an exact Prouhet-Tarry-Escott pair with identical `S3` answers identically, then diverges after one identical learning step |

## Protocol

```
PYTHONPATH=src python -m research.ecs_cognition_r0.run dev    --library <libelpis_ecsg_math.so>
PYTHONPATH=src python -m research.ecs_cognition_r0.run freeze --commit <commit holding the lab source>
PYTHONPATH=src python -m research.ecs_cognition_r0.run qual   --library <libelpis_ecsg_math.so>
PYTHONPATH=src python -m research.ecs_cognition_r0.run status
```

`freeze` binds the specification, DEV choice, pass rule, DEV evidence digest,
laboratory source digest and commit, once. `qual` refuses if the laboratory
source changed, runs once and writes its evidence exclusively; a failure is
recorded as it is and a repair needs a new version.
