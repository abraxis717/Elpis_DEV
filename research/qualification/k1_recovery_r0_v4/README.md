# K1 Recovery R0 under ABI v4: the qualification instrument

Specification: [docs/qualification/K1_RECOVERY_R0_V4.md](../../../docs/qualification/K1_RECOVERY_R0_V4.md). Subject
status: see the specification (`UNQUALIFIED` until its own evidence says otherwise). Nothing here is imported by
`src/`; it is research infrastructure that drives the canonical runtime through its public API.

| File | Role |
|---|---|
| `plan.json` | the pre-registered trials: families, points, lineages, expectations, operator actions, torn offsets, envelope sizes, coordinate and seed rules, bounds, DEV and QUAL inputs |
| `freeze.py` | measures the executing object (checkout, instrument, subject, interpreter and imported package, libraries, deployed copies and pins, reported ABI, toolchain, platform, filesystem) and compares it with a freeze record, fail closed |
| `runner.py` | `freeze` (writes the freeze record once), `run` (verifies, deploys the testing RuntimeCore build, drives every trial through the public API with a real `SIGKILL` at its point, observes restart in a fresh process; it decides nothing), `seal` (writes the evidence directory and `SHA256SUMS` once) |
| `check.py` | `verdict` (gates G1-G6 from the plan and the records alone; no Elpis import) and `evidence` (a sealed directory: exactly its files, every digest, their binding, the verdict recomputed) |
| `evidence/` | written once by the QUAL run; absent until then |

```sh
python -I -B runner.py freeze --native-build BUILD --work-root ROOT --out FREEZE.json
python -I -B runner.py run --native-build BUILD --work ROOT/qual --inputs qual --freeze FREEZE.json --out records.jsonl
python -I -B check.py verdict records.jsonl --inputs qual --out verdict.json
python -I -B runner.py seal --freeze FREEZE.json --records records.jsonl --verdict verdict.json --out evidence
python -I -B check.py evidence evidence
```

A QUAL run refuses to start, before it creates or opens anything, unless everything `freeze.py` measures equals the
freeze record. DEV inputs (other seeds) exist only to develop the instrument; their
records are not evidence. `SIGKILL` keeps the page cache: nothing here bears on power loss, which only the in-process
lost-sync faults (`uncertain`) model. The stimuli are deterministic numbers and the observations typed fixture
evidence (TRAINING=NONE SEMANTICS=NONE): this measures crash consistency, nothing about meaning or learning.
