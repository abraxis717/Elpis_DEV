# R10: LoCoMo DEV label-blind evidence-turn retrieval controls

## Classification / authority

- DEV-only diagnostic, not science qualification, not a statement of ECS_C utility.
- Frozen R9 LoCoMo source and DEV split; eight QUAL conversations are reserved.
- The R9 original file is pin-verified (Git SHA-1 blob and SHA-256); attestor reconstructs the original R9 DEV, gold and audit outputs exactly, without mutating them.
- The attestor may see the source/labels. A **separate predictor process** can read only the DEV_INPUT and attestation. A **separate scorer process** may see SCORER_GOLD after the prediction file is written immutable. This is protocol separation, **not OS security isolation**; public data cannot be truly hidden from an intentionally malicious process.
- The category is checked for valid R9 shape but **never used to produce predictions**. No answer, adversarial answer, evidence reference, sample summary, or category-specific hint is consumed by the predictor.

## Fixed methods and evaluation BEFORE observing DEV scores

- Full DEV conversations available; not a causal live-session benchmark. No future-turn or contradiction-resolution claim.
- Normalize English word tokens with the source's fixed regex and stopword set; no stemming, embeddings, tuning, external data or model calls.
- Turn-level BM25 with k1=1.2, b=0.75, length normalized per conversation, fixed query-token set, chronological tie-break. Retrieval scores are internally used only to order source turns.
- LEXICAL_TOP8: first 8 ranked source turns.
- LEXICAL4_ADJACENT4: first 4 ranked turns, then immediate same-session neighbors (previous, next by turn order) up to 8, then lexical fallback up to 8. Adjacency is a **CANDIDATE navigation edge**, never a semantic relationship proof or knowledge promotion.
- Scorer uses **unchanged** published LoCoMo evidence strings. Unresolved strings are counted as misses in raw evidence recall and separately accounted; no post-hoc normalization or segmentation.
- Fixed descriptive counts: 304 DEV questions, answer-present vs missing-answer, questions with/without evidence, raw/valid/unresolved evidence items, matching evidence items, any-hit and all-hit queries. No score cutoffs or pass/fail thresholds. No answer generation or correctness scoring.
- All frozen outputs are immutable byte-for-byte on rerun; changing inputs/labels refuses rather than rebuilding an existing report.

## Decision gates

1. Mechanically require a pinned R9 source and exact released input bytes, with no QUAL questions entering the predictor.
2. Mechanically require two 304-question independent retrieval outputs and accurate scorer accounting.
3. Record results descriptively only; compare externally annotated source-turn retrieval to the upcoming ECS_C source/graph projection; **do not** infer HCA utility, semantic correctness, or ECS_G benefit from this baseline.
4. Qualified evaluation will need external proof of authorized relationship extraction, frozen conversion independent of QUAL, full resource costs, preregistered metrics, and a bounded consumer. R10 provides none of those yet.

## File placement

Everything persistent remains under `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0/research/ecsc_hca_r0/external/locomo_r0/` (results) or the existing `research/ecsc_hca_r0/` (source and tests). `/home/joe/` holds only the handoff tarball.
