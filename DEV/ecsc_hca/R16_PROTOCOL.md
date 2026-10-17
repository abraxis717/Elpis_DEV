# ECS_C R16 — Native Learned Compositional Scope: DEV protocol (frozen before Ouroboros run)

## Inherited authority

R15 mechanical closure (`4a6211f8...17f8c0`) establishes that R14's frozen learned polarity classifier can supply a source-bound `CANDIDATE` relation, which is withheld from `ADMITTED`-only B0 tasks. R14's frozen 0/32 result on shifted negation scope remains a documented failure. R16 does **not** alter either authority, training or result, and does not integrate new code into the canonical runtime. This experiment addresses precisely the weak compositional learning mechanism exposed by R14.

## Research question

Can a small **Elpis-owned, trainable** representation with explicit *syntactic* scope compose operator effects across held-out nesting depths, target/distractor clause swaps and unseen names, where a same-training-data sparse lexical classifier fails? The test cannot establish unsupervised parsing or natural-language semantics, because a strict, authored text grammar and query-to-clause selector are **supplied in advance**.

## Input and target construction

- All data are **synthetic, grammar-generated**, not human-annotated LoCoMo. The R9 DEV directory is used solely as the existing research authority location; R9 reserved QUAL is not opened.
- 288 labeled TRAIN cases: six training names and six training objects, explicit `owns`/`does not own` leaves, operator chains depth 0–2, with and without an independently generated distractor clause. Fixed deterministic ordering/selection and balance; byte-identity-only deterministic resampling removes duplicate question-text/task pairs without consulting labels.
- 384 EVAL cases: 96 unseen names/items at depth ≤2, 96 unseen depths 3–5, 96 distractor/scope swaps, 64 focused double-negation at depth 3–5, and 32 OOD examples with an unsupported operator. EVAL names/objects are disjoint from TRAIN. Groups are **not statistically independent**: entities and operators recur between groups.
- A separately implemented, fully specified grammar oracle assigns binary labels to the first 352 cases; the remaining 32 have no defined oracle answer and must be refused. The oracle is used only in fixture preparation and after prediction. It is NOT a claim of independently discovered human semantic ground truth.
- Three frozen fixture byte identities (SHA-256, `r9_external_intake.wire` encoding):

  - `R16_TRAIN.json`: `9bb482d62fafdc06778b1ce0d51aa0340c191e8b15f84421c8a9bb982d65663c`
  - `R16_EVAL_PUBLIC.json`: `c888833d279287fd0b2fe72ec7fabbbb1daa2911daf664c9a88c6d8a938289a6`
  - `R16_EVAL_GOLD.json`: `f5fc7f767241100ffe4415a4fa9ce8ed8f0ec964705d33c10c35e12758841bc7`

## Learners and controls

**R16 learned composition:** A strict parser extracts matched source clause (selected by exact query name and item), a leaf symbol, and a bracketed operator chain. The parser knows only the supported grammar, and does not infer truth from the phrases. A trainable pair of leaf logits and, for each operator phrase, a trainable two-state stochastic transition matrix maps inner clause polarity outward. Binary cross-entropy, deterministic cyclic SGD, 110 epochs, fixed rate 0.36. No pretrained weights, teacher model, semantic lexicon, LLM, torch/CUDA, or labels from EVAL. Parsing ambiguous, malformed or unknown operator syntax is a refusal, not a fallback answer.

**LEXICAL CONTROL:** Train the unmodified R14 unigram/bigram logistic learner from scratch on the SAME R16 TRAIN labels and text. This is a lexical control, not a parameter-matched architecture. Compare group-wise correct counts, refusals and errors, without confidence-based promotion into semantic authority.

**Identity-only refusal control:** The unsupported operator cases provide an OOD check; a refusal is expected. No answer key is silently invented for an operator with unknown meaning.

## Workflow and scientific discipline

`prepare` checks R14/R15 source/model/report pins; installs precisely the declared TRAIN, predictor-only public EVAL, and private-by-convention EVAL gold files. `infer` reads TRAIN and public EVAL **only**, fits both learners and seals models/predictions. `score` replays the trained models and *every prediction* from the pinned train/public inputs **before opening EVAL gold**, and then records absolute group counts. This is process separation, not a secure OS access barrier: the generating grammar is visible in the research code.

No criterion is tuned after inspection; R16 reports unfavorable outcomes. No parameter search, no opening R9 QUAL, no Git mutation, no source changes in canonical Elpis, no external inference provider, no post-hoc data repair. Existing research files are write-once, collision checked and never overwritten. A refusal or a mechanics fault cannot be relabeled a scientific failure or success. Failed runs remain observable without resetting the workstation.

## Possible observations and nonclaims

High R16 accuracy can indicate that a strongly structured inductive bias learns and composes the **authored grammar's** operator meanings. It does not demonstrate native free-form grammar induction, independent semantic validity, EDEN-ECS attractor behavior, `ADMITTED` relation justification, ECS_G benefit, source-grounded factual reasoning, or natural-language comprehension. Negative results constrain this candidate. R16 is **DEV synthetic science/methods only**, not scientific qualification and not integration authority.

Follow-up, only if warranted, is independently labeled real text with a label-blind learned syntax projector, explicit abstention/calibration, and the existing HACF → CANDIDATE → B0 authority gate. Do not promote a synthetic success.
