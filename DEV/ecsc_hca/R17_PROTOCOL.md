# ECS_C R17 — Raw-Token Learned Composition: prospective DEV protocol

## Authority and scientific question

R16 is closed mechanically and descriptively: 352/352 supervised-grammar labels and 32/32 unknown-operator refusals; those outcomes rely on a supplied parser that extracts an ordered operator scope tree. R16 model SHA-256 `fbe65bd4a4c018e042d4bfbfc4c2cd8b9e6577ae8613c0d91979d28873c7d003`, predictions `ca644c7c3bd4fe341f67fefc1a6780463acfe0655be21cd4c991207f5e9cd6a3`, report `95803624d72930af9d4fa4d4227ed2a7ff5243dbaf134042ca50605308624e3d`. R16 and earlier authorities remain unchanged.

R17 asks whether a **trainable unparsed source-token recurrence** can learn transferable polarity transitions without receiving R16's operator IDs or a precomputed nested scope tree. It is **NOT** asked to discover unrestricted syntax or answer free-form language. The query's exact (name, item) pair selects one sentence, and a fixed tokenizer emits raw word/punctuation tokens after masking that pair; these are major authored priors. Parenthesis balance is checked as an integrity guard but no scope hierarchy is extracted. The two-state transition parameters for individual tokens are fitted to source-text labels only.

## Frozen pre-inference design

- TRAIN is R16's original 288 labels, byte pinned as `R16_TRAIN.json` (`9bb482d62fafdc06778b1ce0d51aa0340c191e8b15f84421c8a9bb982d65663c`). Reusing prior labeled TRAIN is allowed. No R16 EVAL labels enter R17 training.
- R17 EVAL is a new, deterministic 384-row *synthetic* authored-grammar set, not R9/LoCoMo QUAL and not a corpus of human language. Groups are: 64 new identities at shallow depth, 96 nesting depths 3–7, 96 distractor/scope cases, 64 paired polarity counterfactuals, 32 unsupported-operator OOD cases, and 32 malformed grouping cases. Balanced unknown/malformed are explicit refusal targets, not assigned fictional binary semantics.
- This experiment's model, optimizer, EPOCHS=90, LR=0.15, tokenization, fixture generator, controls and scoring are frozen together before the user's R17 run. No tuning after inspecting held-out results; negative results stand.
- Independently pinned fixture bytes: `R17_EVAL_PUBLIC.json` SHA-256 `5ae65b7047c2382b912d4d9cb9898c967f621174ab0f4112f1353a12ca7dfda3`; `R17_EVAL_GOLD.json` SHA-256 `9269f6cd409870f2cb5c10ce8fc7ca57c9d3567c01145be4a20f7f60bd94de5f`.
- Comparison controls: original R16 fitted structured-parse model (upper reference with extra information) and R14's lexical learner fitted to precisely the same R16 TRAIN. These controls do **not** receive R17 EVAL gold. Uncalibrated probabilities must not confer factual/semantic authority.

## Isolation, reproducibility, evidence

`prepare` checks frozen prior authority and writes public and scorer-only-by-convention evaluation files atomically. `infer` loads TRAIN and EVAL PUBLIC only; generates a raw-token model and deterministic ordered prediction record, plus fixed R16-structured and R14-lexical predictions, without opening R17 EVAL gold. `score` checks exact model replay and re-evaluates each prediction from source and model **before** reading EVAL gold, then reports by-group counts and refusals. This is source/process discipline, not secure sandbox isolation: the authored generator is visible in code.

Outputs are write-once/collision-checked within `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0/research/ecsc_hca_r0/external/locomo_r0/`. No writes to `/home/joe` except the handoff tarball, no new venvs, clones or canonically tracked files. The runner holds a single-writer lock, pins source and evidence, runs the research regression and selected canonical context tests, and checks the canonical source diff before and after. Tests assert mechanical invariants, not held-out accuracy thresholds.

## Strict nonclaims

R17 *does not* demonstrate learned sentence selection: exact query matching and boundary detection remain authored. It also cannot establish genuine English semantics because all labels derive from a small generator with declared lexical patterns and the model is a two-state token sequence. Even a perfect result cannot justify `ADMITTED` relations, canonical runtime integration, a claim of grammar induction, a natural-language model, or EDEN-ECS emergence. The next science gate, if warranted, is independently labeled real text with no authored query-to-clause selector and preregistered abstention/error costs.
