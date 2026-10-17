# R18 retrospective shortcut audit — read-only DEV authority

This diagnostic is **post-hoc**, not R19, not a new blind qualification. It makes no changes to R18's frozen source, fixtures, model, predictions or report. Exact source and evidence SHA-256 pins are enforced. Canonical `src/`, `native/`, `docs/`, `tests/` are read-only. Git is not mutated. No QUAL, models, network or services are accessed.

## Three falsification checks

1. **Root-only heuristic:** reuse R18's supplied parser to find the outermost `AND` or `OR`; ignore all leaves and inner expressions. Predict AND=false, OR=true; invert for enclosing NOT. Compare its binary output and grammar refusals against all 448 frozen predictions. If it reproduces the model, high flip-pair scores cannot establish learned recursive reduction.
2. **Operator and leaf truth table:** compare learned probabilities on independent primitive leaves and all binary truth-table combinations with the declared Boolean oracle. These are controlled-grammar functional tests, not tests of natural-language semantics.
3. **Constant-root counterfactuals:** make 16 new synthetic pairs with identical word order, fixed outermost connective, opposite gold outcome due solely to grouping of an *inner* expression, including two NOT parity settings. This prevents an outermost-operator-only shortcut. The model is frozen; no fitting occurs. These are retrospective probes and cannot be claimed as independently preregistered QUAL evidence.

## Output

`R18_RETROSPECTIVE_SHORTCUT_AUDIT.json` is written once under the existing LoCoMo DEV research directory with byte-identical rerun permitted. It binds all R18 evidence source digests. Report exact matches, truth table mismatches, counterfactual pair-level results and nonclaims. Fail closed on source, model or prediction drift; no automatic repairs.

## Research decision

R18 remains an acceptable mechanical/dev result on a biased fixture, but its supplied-parser structural accuracy cannot be promoted to learned Boolean semantics if the root-only control matches. The next prospective study must change inner truth under identical outer operators and use balanced leaf/operation truth tables. It should include a simple root-operator control, and report pair-level results against a truly raw-token learner separately from the supplied-parser reference.
