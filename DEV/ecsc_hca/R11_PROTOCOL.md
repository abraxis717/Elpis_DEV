# R11 — LoCoMo DEV label-blind candidate-graph B0 projection

**Authority**: descriptive DEV mechanics. R9 and R10 remain frozen. Not scientific qualification or proof of semantic memory.

## Input isolation

- The only prediction input is R9 `DEV_INPUT.json`, R10 attestation, and pinned R10 predictions. The predictor's branch never opens the original LoCoMo raw dataset or `SCORER_GOLD.json`.
- The predictor reconstructs byte-exact quoted-turn-text ranges inside the original, SHA-pinned R9 `DEV_INPUT.json` artifact, without altering source files. One derived corpus manifest references that complete file. The canonical HACF source resolver reopens the file through `RootCapability`, verifies its document digest, the profile-bound claim, and normalized range bytes before B0 graph admission.
- **Profile warning:** The turn JSON-literal ranges are a frozen, explicitly `TEST_ONLY` R11 profile. This is *not* a reproduction of the native HACF chunker's segmentation. Corpus manifest and grants are derived from R9's pinned release and are not independent semantic admission proofs.
- Exact byte verification prevents unnoticed source substitution during the check; it does not prove that source text is true, nor guarantee post-check immutability.

## Frozen model mechanics

- For each of 304 DEV questions, BM25 ranks source turns using R10's unchanged scoring rule and stopwords. First four turns are seeds. Same-session predecessor/successor candidate edges extend outward up to two hops per seed; if overlapping neighborhoods yield fewer than eight distinct turns, additional BM25 seeds fill to eight. Graph bounds are 24 axiom turns, 32 directed binary candidate edges, fixed salience weight 750, no admitted relations and no binding claims.
- All turns are `PROVISIONAL` axioms; all edges `CANDIDATE` navigation only. `TEST_ONLY_R11` authority/grants support mechanical graph admission and do **not** authenticate relationship entailment.
- B0 performs exact closure; the top eight salience readouts are returned. The task has fixed candidate predicate, work budget 500,000 logical units and readout record budget 32. No B0I or B2S evaluation is claimed here.
- The comparison budget is eight returned turns, as in R10. Ranking inside the B0 readout is salience then axiom digest (not BM25 tie order). Unrecognized graph/map/refusal errors stop the experiment rather than silently substituting a fallback.

## Scoring after seal

- Scorer is a separate invocation. It requires R10's exact prediction and report SHA-256 and R9's scorer gold integrity. It checks all 304 question IDs, exactly eight distinct source-valid turn IDs per question, and counts matching raw LoCoMo evidence references, including anomalous unresolved references as misses.
- It reports answer-present and answer-missing cohorts separately, source verification and B0 operation totals **as different units** (chunk count / logical work, not bytes or CPU time), and not a unified end-to-end throughput claim.
- R10 scores are a frozen comparator. Since the R11 graph is *built from lexical BM25 seeds and heuristic adjacency* and its labels are published LoCoMo evidence pointers, apparent wins do not demonstrate semantic inference. Inspect both wins and regressions before making any scientific recommendation.
- No QUAL source, question, annotation, or predicted label is opened. QUAL evaluation is still unauthorized. No threshold, parameter selection, trained model, or semantic result is preregistered from DEV performance.

## Filesystem/operation

- All R11 source and evidence is under the already-established research workspace. No new clone, environment, Git mutation, canonical source changes, or modification of prior reports. The `.tar.gz` in `/home/joe/` is disposable mailbox content only.
