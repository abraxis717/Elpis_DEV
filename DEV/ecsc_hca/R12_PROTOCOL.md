# ECS_C R12 — Prospective DEV answer-level comparison (frozen protocol)

**Status:** DEV protocol/mechanics only, **NOT SCIENTIFIC QUALIFICATION**. This phase does **not** alter R0–R11 or canonical Elpis/HACF/ECS_G code. No new ECS_C graph or trained memory algorithm is introduced.

## Fixed question

When a single, pinned local answer model receives a fixed question and exactly eight retrieved dialogue turns, does the previously frozen ECS_C/B0 candidate projection improve *strict answer-string agreement* and benchmark-defined abstention compared with frozen lexical retrieval plus local adjacency?

This phase isolates the **evidence selection** while holding the **answer model, prompt, token cap, and question** constant. It cannot identify true entailment, contradiction, or revocation.

## Prior authority

- Research worktree: `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0`, HEAD `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`.
- R9 LoCoMo DEV: `conv-26`, `conv-30`, exactly **304** questions; remaining eight source conversations do not enter prediction packets.
- R10 predictions SHA-256 `019aa16568d50c4f3eb668d59bd0caeee84ad0796cea1fbcf2b9cf7ea14579b8`.
- R11 predictions SHA-256 `3dee8a4d7042fa0b15f76d48df940508908d35b284c1e8bf8234f7f065b5fe5d`.
- Previously published R10/R11 scored reports and paired audit are checked only by the scorer, **not opened by the predictor**.
- R9 `SCORER_GOLD.json` contains annotations for **all ten conversations**, including eight reserved QUAL conversations; its bytes are opened by the **scorer only**, which filters to the two DEV samples. QUAL annotations are thus *present in the scorer's source file* but never selected for a QUAL score. This is code-path isolation, not an operating-system ACL or a truly unknown hidden test.

## Four prespecified conditions

1. `LEXICAL_TOP8` (R10 untouched).
2. `LEXICAL4_ADJACENT4` (R10 untouched; primary reference).
3. `ECSC_B0_CANDIDATE_RADIUS2_TOP8` (R11 untouched; candidate graph, no admitted semantic relation).
4. `NO_MEMORY` (same answer model with no source passages; serves as a strict abstention / unsupported-answer control).

All source selections are converted into transcript-ordered turns with speaker, date, dialog ID and full text. A fixed system prompt instructs source-backed JSON answers only, including citations for non-null answers and `null` for insufficient evidence. In every comparison, the **same selected model** receives the same 304 questions with identical decoding settings (`temperature=0`, `max_tokens=128`, no streaming). Method request order is **counterbalanced deterministically per question** by the first byte of SHA-256(question ID), independent of labels. Model-call order is not performance-blind; observed latency is purely descriptive.

## Sequential authority

- **Prepare**: re-check the pinned public DEV inputs/predictions and issue `R12_DEV_PREREGISTRATION.json` and `R12_DEV_ANSWER_PACKETS.json` once. This process cannot open gold, QUAL source, or earlier scored reports.
- **Infer**: only after preparation, through an **explicit `http://127.0.0.1:<port>/v1/chat/completions` endpoint** and model ID selected by the user. No internet model, proxy, or redirect is authorized. The server is assumed already running. Inference stores exact raw replies, including malformed JSON, in a single-writer, append-only SHA-256-linked journal. Each invocation processes at most `--steps` calls and resumes only at the next missing index. A malformed response is an invalid result, not repaired or retried. Output does not become scientific evidence merely by being immutable. Request/response timing excludes graph construction and cannot be used for end-to-end latency comparisons.
- **Score**: only after **all 1,216** calls have valid, intact journal records. In a separate process, verify the scorer-only upstream reports and the pinned R9 gold file, then report per-method outcomes and paired deltas against lexical+adjacency. A completed journal with invalid *model output formats* still receives a descriptive score; integrity failure blocks scoring.

**Primary descriptive outcomes:** strict Unicode-NFKC/case-folded alphanumeric answer-string agreement on the **235 answer-present DEV questions**; abstention on the **69 answer-missing DEV questions**. Count malformed responses, nonmatching answers, abstentions on answer-present queries, unsupported answers on answer-missing queries, and malformed/uncited/foreign turn references independently. The exact-string result is a **lower-bound match**, not semantic accuracy; paraphrases are not proven errors. LoCoMo answer absence is *not* independently adjudicated unanswerability.

**Secondary:** citations overlapping LoCoMo's author-supplied evidence references; method-level model latency is separately descriptive and **not** a measure of full HACF/ECS_C WARM work. R8 already established that B0I/B2S recurrence had no measured advantage; this phase does not rerun them.

## Exclusions, failure conditions, and disposition

- No correction/revocation scoring: LoCoMo's annotations do not constitute an independently verified, typed temporal contradiction ledger.
- No independent semantic entailment adjudication or human-verified paraphrase scoring; strict matching must never be presented as semantic QA accuracy.
- No model selection, prompt revision, context-length revision, parameter tuning, or reprocessing of malformed outputs based on DEV scores. These would require a new preregistered lineage and prevent confirmatory attribution.
- No access to QUAL conversations through the predictor. Do not run this R12 scorer on QUAL. No `SCIENCE_PASS` or release claim is possible from this DEV phase alone.
- Stop at `HOLD` for changed pinned bytes, changed protocols, duplicate/conflicting reports, torn or altered journals, unqualified model endpoint, source leakage, incomplete predictions, or canonical drift. No automatic repair; retain evidence.
- A result in which B0 is not better than lexical+adjacency gives no reason to promote the candidate B0 retrieval mechanism. Even if B0 is better by strict string metrics on DEV, this alone does **not** authorize deployment; independent confirmatory material and factual adjudication remain necessary.

## Typical workflow

The bundled launcher installs the research-only module/tests/protocol, runs tests and canonical context checks, and performs **prepare only**. Local inference is a separate, explicitly authorized action:

```bash
cd /mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m research.ecsc_hca_r0.r12_answer_level infer "$PWD" --endpoint http://127.0.0.1:8081/v1/chat/completions --model 'THE_ID_EXPOSED_BY_YOUR_LOCAL_SERVER' --steps 16
```

To continue, rerun the exact same command and model ID. To run a larger batch, change only `--steps`. After all 1,216 calls, the fixed scorer command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m research.ecsc_hca_r0.r12_answer_level score "$PWD"
```

The port and model name above are **examples**, not detected current runtime facts. The user must choose a running local server explicitly. `/home/joe/` is transient mailbox-only; all persistent state remains under the established research workspace.
