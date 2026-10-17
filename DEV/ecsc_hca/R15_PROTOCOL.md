# R15 — source-bound learned CANDIDATE ingress (DEV mechanics)

## Authority and question

R14 proved that a deterministic sparse linear projector can distinguish several **controlled, templated** affirmative/negative examples; its frozen evaluation also proved **0/32** on changed negation scope and **32/64** on unseen negation vocabulary. These failures remain frozen. R15 does not retrain, refit, tune, calibrate, open QUAL, or invent semantic labels. It asks a narrower architectural question: **can a hypothesis produced by the frozen Elpis-native R14 projector enter ECS_C's typed ContextView as untrusted CANDIDATE and be consumed by B0 without becoming ADMITTED evidence?**

## Frozen inputs

The evaluator pins the existing R14 projector source; R14 model, predictions, and report; R13 null diagnostic; R12N structural source; and released DEV input. All original bytes are read-only. The source sentences are the exact two R13 perturbations, replayed for each of two DEV conversations (4 source variants). A different, fresh temporary HACF document and chunk identity is recomputed for each variant; original DEV_INPUT.json is unchanged.

## Mechanism

1. Take the already trained R14 model, with exact frozen model SHA-256. Compute an uncalibrated candidate polarity for each verified source sentence. This value is **not a probability of semantic truth**.
2. Construct a new provisional (`PROVISIONAL`) hypothesis axiom anchored to the same verified ContextReference as the source turn. Add exactly one `CANDIDATE` relation with an explicit model-derived provenance digest. Separate relation types `1501`/`1502` denote classifier-proposed affirmation/negation, not externally proven entailment.
3. Retain the independently mechanically admitted `NEXT`/`PREVIOUS` structural relations from R12N and the frozen source grant. No extra SourceGrant, fake source bytes, or ADMITTED semantic relation is manufactured.
4. Run canonical HACF verification before Baselines.build, then native B0 task readouts: candidate-permitted matching hypothesis **must** be observable; admitted-only and opposite-relation tasks **must** return no hypothesis. Any readout remains `SALIENCE_ONLY`.
5. Change the source file **after** verification, and confirm old evidence cannot be reused. All temporary documents are removed. Verify the frozen DEV source remains unchanged.

The 4 cases test **safe plumbing**, not whether R14's predictions are semantically correct. The fixture sentences are reused from R13/R14 and are not a new independent holdout. Zero correctness gains, semantic proofs, generalization, calibrated confidence, or ECS_G integration are claimed.

## Qualification boundary

A passing R15 is only `PASS_DEV_ONLY_CANDIDATE_NOT_SEMANTIC_QUAL`. Production integration remains prohibited; model-generated relation provenance cannot be converted to `ADMITTED` merely because its source bytes or model digest verify. A separate independent semantic proof mechanism and held-out compositional evaluation are needed before that claim.

## Hygiene

Single archive, no agents/LLMs, no environment changes, no Git operations that mutate state, no new worktree, no canonical modifications. Existing `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0/research/ecsc_hca_r0/` gets exactly two new research Python files and this protocol. Report is `external/locomo_r0/R15_DEV_CANDIDATE_INGRESS_REPORT.json`, only if research/canonical regressions pass; writes are conflict-refusing/idempotent. `/home/joe/` is download mailbox only.
