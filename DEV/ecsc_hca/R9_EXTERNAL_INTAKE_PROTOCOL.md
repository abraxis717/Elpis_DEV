# ECS_C external-memory evaluation — LoCoMo source intake, r0

**Classification: DEV input acquisition and methodological preparation ONLY. Not a scientific experiment and not a scientific qualification.**

## Frozen upstream

- Dataset: *LoCoMo*, official `snap-research/locomo`, ACL 2024. Contains generated long conversations and independently authored question/answer/evidence annotations. Independent of Elpis does **not** mean annotation quality is perfect, or the conversations are real-world human interactions.
- Commit: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
- Dataset path: `data/locomo10.json`
- Git blob SHA-1: `d95b872480b413d935821fdc3c84f8a8f5f29e73`; local intake also records raw SHA-256.
- Exact 10 samples, 1,986 questions. As observed during protocol development, 444 questions have no `answer` field (primarily abstention), and nine raw evidence reference strings are not exact dialogue IDs. **Report without silently resolving, excluding or rescuing.** These are dataset metadata issues, not HCA failures.
- Source: https://github.com/snap-research/locomo ; committed raw file URL at https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json

## Prevent leakage: fixed first split

- Initial **DEV** samples: `conv-26`, `conv-30` only. `DEV_INPUT.json` contains only transcripts (dialog IDs, speakers, times, texts) and query text/category/ID. **No** QA answer fields, evidence locations or adversarial answers.
- All other eight samples are reserved as **QUAL candidates**, not released to a prediction-facing JSON. The upstream raw JSON and scorer gold are accessible to the same operator account; this is a *procedural* separation, not hard filesystem security. If any agent/adapter sees gold labels or QUAL data before the authority freeze, qualification is contaminated and must not be claimed.
- `SCORER_GOLD.json` contains the original annotations, question IDs and unresolved evidence references. Treat this file as *scorer-side only*; never load it in the prediction process. The source text naturally contains answers as it should; question labels must not leak.
- Frozen QIDs: `<sample_id>/q<index zero-padded to 4>`, original annotation order, no search-derived sorting or filtering. The aggregate 1,986-question denominator is immutable for reporting; separate ineligible/annotation-quality counts must be disclosed.

## What this does *not* authorize

This intake **does not** run B0, B0I, B2S, perform HACF admission, encode conversational text as `Task`/`ContextAxiom`, evaluate language answers, score retrieval, or create ECS_G inputs. It cannot establish cognition, human semantic correctness, temporal updating, external validity, or end-to-end improvements. Before future experiments, freeze the source extraction/projection rules *without reference to answer labels*. Independent annotation adjudication is necessary for claims stronger than the benchmark's published labels permit.

## Preregister for future experiment (NOT YET AUTHORIZED)

1. A blind, bounded conversion from source turns to source-verified HACF references and candidate/admitted relations, with explicit operator/structure authority. An answer/evidence field must NEVER be available to this conversion.
2. Test exact B0 against B0I, B2S and appropriate lexical/source lookup controls on identical source budgets, causal chronology, authority, and output budget. Negative controls must include answers absent from history, updated facts, contradictory turns and cross-conversation contamination.
3. Score against published answers independently of the predictor: supported answer, unsupported answer, abstention, evidence provenance, completeness/refusal, and resource cost including full ingestion, admission, and failed work. No post-hoc exclusions. Case-level records and bounded coordinates.
4. Freeze evaluator/source/scorer/protocol/engine hashes and the full split before exposure to QUAL. Do not tune policies, thresholds, answer formatting or prompts on QUAL. DEV may inform a new immutable prospective protocol; all claims must specify the split.

## Files and paths

Persistent: `research/ecsc_hca_r0/external/locomo_r0/` within existing `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0`. Contains `locomo10.json` (raw pinned upstream), `DEV_INPUT.json` (predictor-facing), `SCORER_GOLD.json` (scorer-facing by convention), `INTAKE_AUDIT.json` (metadata + anomalies). Read-only after freeze by discipline. No code or persistent storage in `/home/joe/`, no commits, no canonical source changes, no new venv/worktree.

An intake PASS is strictly a source and bookkeeping PASS. If network access fails, data pins mismatch, or an output already exists with different bytes, report HOLD. Never substitute sample content or overwrite the evidence.
