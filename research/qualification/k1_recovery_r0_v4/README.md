# K1 Recovery R0 under ABI v4: the qualification instrument

Specification: [docs/qualification/K1_RECOVERY_R0_V4.md](../../../docs/qualification/K1_RECOVERY_R0_V4.md). Subject
status: see the specification (`UNQUALIFIED` until its own evidence says otherwise). Nothing here is imported by
`src/`; it is research infrastructure that drives the canonical runtime through its public API.

| File | Role |
|---|---|
| `plan.json` | the pre-registered crash-point list: families, points, lineages, expectations, operator actions, torn offsets, run counts, bounds, DEV and QUAL input seeds |
| `runner.py` | `run`: deploys the testing RuntimeCore build of one operator build (`elpis.runtime.deployment`: pinned, admitted from sealed bytes) and, per point and run, starts a child that drives a real lineage to the point, parks there (the durability worker, or a control operation's caller) and is killed with `SIGKILL`; a fresh process then records what restart decides, performs the pre-registered operator action, resumes, LEARNs once and publishes once. It decides nothing. |
| `check.py` | the gates G1-G6, evaluated from the plan and the records alone (no Elpis import, nothing native) |
| `evidence/` | written once by the QUAL run; absent until then |

```sh
python -I runner.py run --native-build BUILD --work WORK --source-commit SHA --inputs qual \
    --freeze FREEZE.json --out records.jsonl
python -I check.py records.jsonl --out verdict.json
```

A QUAL run refuses to start unless the plan, runner, checker and specification digests, the source commit and every
admitted library equal the freeze record's. DEV inputs (other seeds) exist only to develop the instrument; their
records are not evidence. `SIGKILL` keeps the page cache: nothing here bears on power loss, which only the in-process
lost-sync faults (`uncertain`) model. The stimuli are deterministic numbers and the observations typed fixture
evidence (TRAINING=NONE SEMANTICS=NONE): this measures crash consistency, nothing about meaning or learning.
