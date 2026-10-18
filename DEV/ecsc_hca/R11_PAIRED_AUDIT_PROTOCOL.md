# R11 post-hoc paired DEV accounting — NOT R12

This audit does not run, tune, or revise any predictor. It reads *already sealed* R9/R10/R11 DEV releases, scorer annotations and R10/R11 reports. It pins exact source SHA-256 hashes and re-evaluates all 304 DEV questions only, with answer-present and missing-answer cohorts reported separately. The eight reserved QUAL conversations are not evaluated.

## Why

R11 reports 146/310 gold evidence references and 125/235 complete-evidence DEV queries, below the R10 `LEXICAL4_ADJACENT4` control (154/310 and 132/235). Aggregate deltas conceal overlap, paired regressions, per-category failure, and gains on particular questions. This independent retrospective audit counts per-question wins, losses and ties; gained/lost gold references; exact selected-set overlap; and category-level metrics for all three methods. Unresolved original evidence references count as misses, never silently corrected. It checks its reconstructed totals against both frozen R10 and R11 reports.

## Authority and prohibitions

- Descriptive post-hoc analysis only. Neither the method nor its comparisons were preregistered as a confirmatory scientific test. No inferential p-values or performance promotion.
- No QUAL scores, model reruns, science thresholds, parameter search, ground-truth corrections, hidden-model access or canonical Elpis mutations.
- `R11_DEV_PAIRED_AUDIT.json` is immutable once written; a conflicting file is refused.
- Work units are not comparable end-to-end between lexical R10 and HACF/B0 R11. No latency or CPU-speed conclusion follows.
- Graph edges in R11 are candidate source-navigation links, not semantic proofs; evidence-pointer recovery is not answer accuracy.

## Next scientific gate

If R11 is worse on held-out independent annotations, keep the simpler lexical+adjacency DEV baseline; do not tune R11's radius/salience on QUAL. Define a new *prospective* mechanism and independent human-adjudicated factual tasks, pre-register controls, uncertainty, refusal behavior and full WARM costs before opening QUAL. Exact B0 may be maintained as a mathematical reference, not deployed on this retrospective signal.
