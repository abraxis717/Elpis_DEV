# ECS_C R18 — prospective paired bracket-binding DEV evaluation

## Authority and nonclaims

- The R17 scope-ablation report (SHA-256 `49a6a44715dca40a78b36c0395e29d0862deed23a6ba605324b8835e5a29153c`) is **retrospective** and closed. R18 does not edit its model, evidence, prediction records, or prior research files.
- Source checkout remains pinned to `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`. Canonical `src/`, `native/`, `docs/`, and `tests/` stay unchanged. No Git mutations, new worktrees, new venvs, external models, LoCoMo QUAL, or service endpoints.
- R18 tests **formal Boolean expression binding**, not semantic understanding, truth verification of natural-language claims, or automated relation admission. Leaf truth, connective semantics and pairing are authored in the fixture oracle. The candidate model is **provided a deterministic parenthesis parser**; it learns 12 leaf/operator probability parameters by cross-entropy. It does NOT learn parsing. A passed scope result only establishes correct use of this explicitly supplied structural prior.
- R18 does not promote `CANDIDATE` to `ADMITTED` or add any canonical cognition/runtime behavior.

## Explicit falsifier

Compare bracket-counterfactual *pairs* containing the same lexical tokens **in the same order** but with different parentheses and opposite formal labels. Examples: `((A or B) and C)` versus `(A or (B and C))`, with A=true and C=false. Both words and their order are unchanged. A grouping-blind function over non-bracket tokens cannot predict both members correctly; the ceiling is **one correct member per discriminative pair** (50%) by construction. A neutral-control family changes grouping with unchanged labels. This prevents the assessor from assigning success merely because predictions react to the presence of parentheses.

## Frozen DEV splits and controls

Training: **640** authored examples (256 label-flip pairs, 64 grouping-neutral pairs). Held-out DEV: **448** cases:

| Group | Cases | Rule |
| --- | ---: | --- |
| `paired_grouping_flip` | 256 = 128 pairs | Correct labels differ; non-bracket word sequence identical |
| `deeper_grouping_flip` | 64 = 32 pairs | Same, with additional unary nesting beyond usual training depth |
| `same_operator_control` | 64 = 32 pairs | Labels remain the same despite bracket movement |
| `unknown_connective_refusal` | 32 | Unknown `xor` syntax is refused |
| `malformed_parenthesis_refusal` | 32 | Parenthesis mismatch is refused |

Training names and items are disjoint from evaluation names and items; no duplicate complete source text crosses the split. Training rows include gold labels. Prediction-facing evaluation rows contain no `label` or `refuse` field. Gold labels are separate and opened only by `score` after exact model and per-case prediction replay. This is a **code-path separation, not an operating-system security boundary**: the same workstation user can read the files, so it does not prove adversarial label secrecy.

Fixture byte identities are frozen in `r18_bracket_binding.py` as `FIXTURE_PINS`. The three stages are strictly sequential: `prepare`, `infer`, `score`. Never rewrite a conflicting file. Stop on source/hash drift. Re-run is exactly idempotent if all output bytes match. A local end-to-end lifecycle dry-run was performed to test mechanics; its evaluation metric values were deliberately not inspected, but this does not constitute externally blinded qualification.

## Learner and controls

- Candidate: supplied hand-written grammar parser for `not (x)`, `(x and y)`, `(x or y)` and terminal `Person owns / does not own the Item`. Parameters for both terminal classes, a 2-entry NOT probability table and two 4-entry binary operator probability tables are trained; parameter values and all predictions are reproducible on CPU. **The parser is not learned.** There is no external pretrained model or corpus data.
- Comparator: a deterministic **grouping-blind ceiling** for each matched, opposing-label pair: no grouping-blind method using only ordered non-bracket words can exceed 50% pair-member accuracy, regardless of training.
- Negative control: same-operator regrouping, with equal labels, prevents a naive "always flip on moved brackets" heuristic from counting as success.
- Required OOD behavior: unknown operators and malformed grouping cause refusal, not a forced answer.

## Scoring and interpretation

Report per-group `correct`, `wrong`, `refused`; also **pair-level** `both_correct`, `one_correct`, `neither_correct` for each bracket group. Include all refusals in denominators. No after-the-fact threshold adjustment, fixture edit, changing training epochs, or selection on held-out results. Any result is admissible as **descriptive DEV evidence only**. **No scientific qualification** or semantic admission can be claimed from this controlled grammar.

A high pair-level score supports learned operator composition *given a supplied bracket parser*. Failure to exceed grouping-blind performance means the structural prior + learned parameters did not confer the intended capability on this fixture. Either outcome should be preserved. The subsequent, genuinely harder research gate would be to learn bracket/token-to-structure transformations from raw text without providing the parser.
