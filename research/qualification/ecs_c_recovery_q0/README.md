# ECS_C Recovery Q0: the qualification instrument

Specification: [docs/qualification/ECS_C_RECOVERY_Q0.md](../../../docs/qualification/ECS_C_RECOVERY_Q0.md). Subject
status: see the specification (`UNQUALIFIED` until its own evidence says otherwise). Nothing here is imported by
`src/`, and nothing here imports Elpis. It is research infrastructure that drives the installed native host
(`elpis_testing`) as separate processes, exactly as an operator runs it.

| File | Role |
|---|---|
| `plan.json` | the pre-registered trials: families, points, expectations, attack layout, operator actions, stimulus, coordinate and seed rules, bounds, DEV and QUAL inputs |
| `freeze.py` | measures the executing object and compares it with a freeze record, fail closed: the checkout, the instrument, the subject sources, the installed prefix, the interpreter, the toolchain, the platform, the filesystem and the premise |
| `runner.py` | `freeze` writes the freeze record once. `run` verifies, deploys a fresh home per trial from the prefix, drives the trial with a real `SIGKILL` at its point and records what each fresh process observes; it decides nothing. `seal` writes the evidence directory and `SHA256SUMS` once |
| `check.py` | `verdict` judges gates G1-G6 from the plan and the records alone, with no Elpis or runner import. `evidence` checks a sealed directory: exactly its files, every digest, and the verdict recomputed |
| `evidence/` | written once by the QUAL run; absent until then |

```sh
python -I -B runner.py freeze --prefix PREFIX --work-root ROOT --out FREEZE.json
python -I -B runner.py run --prefix PREFIX --work ROOT/qual --inputs qual --freeze FREEZE.json --out records.jsonl
python -I -B check.py verdict records.jsonl --inputs qual --out verdict.json
python -I -B runner.py seal --freeze FREEZE.json --records records.jsonl --verdict verdict.json --out evidence
python -I -B check.py evidence evidence
```

`PREFIX` is an installation of both components (`cmake --install BUILD --prefix PREFIX --component runtime`, then
`--component testing`).

**Before the run.** A QUAL run refuses to start, before it creates anything, unless everything `freeze.py` measures
equals the freeze record. A QUAL run always runs the whole plan.

**Inputs.** DEV inputs (other seeds) exist only to develop the instrument; their records are not evidence, and
`check.py evidence` refuses them.

**The premise.** The crash points exist only in the testing builds, so the production build's protocol is bound by
the premise in `freeze.py`:
- the production views' digests;
- no testing symbol in the production RuntimeCore;
- no testing operation in the production host.

It is a binding, not a proof that the two builds are equivalent.

**Scope.** `SIGKILL` keeps the page cache: nothing here bears on power loss. The stimuli are deterministic numbers
and typed fixture evidence. This measures crash consistency and attack refusal, nothing about meaning or learning.

## DEV rehearsal (not evidence)

**The whole plan** (221 trials, DEV inputs) ran against an installation of the Stage F2 head. `check.py` judged it
PASS on G1-G6 with no mechanical failure. The first full rehearsal stopped the bounded family's history at 900
publications: K1 refused a LEARN of the then turn-varying stimulus as non-finite. The specification records the
stimulus change (revision 4).

**The F1 build.** The same instrument ran against an installation of Stage F1, which lacks two fixes (the anchor and
attack families, one run each). It judged exactly the points those defects touch as failures:
- every anchor point (no interrupted `create` completes);
- both shortened-record attacks (the authority refused open as `CORRUPT`, with no disposition).

Every other attack point passed there too.

`tests/research/qualification_ecs_c_recovery_q0` (scientific lane) repeats a two-run DEV rehearsal. It then shows
that every mutant of the records or the plan fails the gate that exists to catch it, and that every enumeration
defect is mechanical.
