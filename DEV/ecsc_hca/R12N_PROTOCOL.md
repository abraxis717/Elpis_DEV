# ECS_C R12N — Elpis-native directional source structure

**Status:** frozen DEV structural-mechanics protocol. NO independent semantic qualification; NO language model; NO QUAL evaluation.

## Why this experiment exists

The R12 answer-generation path is **PREPARED but NOT AUTHORIZED**. It would have sent 1,216 packets to an unrelated LLM and measured downstream text answers. That is not a direct test of Elpis as its own model. The R11 B0 mechanism emits *SALIENCE_ONLY* source-axiom activations; it cannot currently truthfully output natural-language factual propositions. R12N therefore tests a narrower capability that Elpis can represent and compute natively: typed, directional, source-proven record-order relations.

## Frozen source and authority

- Existing worktree: `/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0`; frozen HEAD `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`.
- Exactly R9 LoCoMo DEV samples `conv-26`, `conv-30`, public DEV file SHA-256 `89b8ed64d9a3cf83b7efb87611cef017dc65a2775a634851ecd6e6f711f50672`.
- R10 lexical code and frozen prediction authority; R11 exact source-span verification and test-only chunk profile. R12 answer-packet preregistration remains untouched and unused.
- This experiment does **not** read `SCORER_GOLD.json`, R10/R11 gold-scored reports, original upstream raw QUAL corpus, or any LoCoMo answers.

## Registered DEV probe

Choose **12 interior dialogue turns per DEV sample** using frozen SHA-256 ranking of `(sample_id, dialog_id)` (24 centers). Each interior center has a same-session predecessor and successor directly attested by the published ordered turn list.

For each center, construct a three-axiom ContextView (previous, center, next), all `PROVISIONAL`, bound to exact verified R9 JSON text literal spans. Two directional, binary relations are separately admitted under a **TEST_ONLY structural order authority** after exact same-session immediate-order validation:

- `NEXT(origin, destination)` — relation type 1201.
- `PREVIOUS(origin, destination)` — relation type 1202.

`ADMITTED` here proves only **record order in the pinned source document**. It does NOT prove temporal chronology in the world, semantic entailment, textual truth, or admission of any natural-language claim. An R11-style test-only chunk profile is explicitly used, not native HACF chunker segmentation.

Run two Elpis B0 tasks with the **same center seed** and distinct typed predicates; require exact predecessor/successor destinations and nonidentical outputs. Then rebuild with *candidate-only* relations and repeat the admitted-only tasks: both must return no nonseed destination. If any source check, exact order witness, authority, bound, task evidence, or canonical integration fails, **HOLD**, without generating a passing report.

**Sample count:** 24 source windows, 48 admitted-direction readouts, 48 negative-control readouts, 2 × 3 HACF-verified chunk checks per source window. Logical work and graph admission work are logged separately, not converted to wall time.

## Exclusions

- No model calls, no token/embedding inference, no answer scoring, no learned relation extraction, no semantic-accuracy claim, no ECS_G wiring, no source publication mutation.
- Labels (previous/next turn) are deterministically implied by the **same structure used for graph construction**, so correctness is **mechanical internal consistency**, not independent scientific validation. This limitation is structural and must not be disguised by reporting 48/48 as evidence of cognition.
- The `ADMITTED` semantic label is used only as a test of B0's typed status filter, under an exact record-order proof. It cannot be generalized to semantic `ADMITTED` elsewhere.
- The existing R12 external LLM answer packet remains frozen, unscored, and unused. No attempt to reinterpret it as the native Elpis experiment.

## Next genuine research gate

Native task comprehension requires a **separately qualified, source-grounded semantic projection** that maps external representations to typed predicates and propositions, independently of gold labels. Until such a projector exists, Elpis can demonstrate source integrity, relational algebra, preservation, and refusal — but cannot establish open-domain answer correctness. This experiment intentionally makes that boundary explicit rather than borrowing an LLM's language understanding.
