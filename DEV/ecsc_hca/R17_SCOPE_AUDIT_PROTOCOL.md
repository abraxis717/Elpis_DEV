# R17 scope-sensitivity retrospective audit (DEV, nonqualification)

## Authority

Read-only with respect to frozen R16 and R17 evidence. Requires exact pinned R17 source, public fixture, trained model, predictions and scored report. Does not open R17 held-out gold, LoCoMo QUAL, model APIs or external services. A separately named immutable audit report may be written to the existing research evidence location. No model refitting, threshold selection, Git mutation or canonical changes.

## Question

Can the frozen R17 predictor change its binary answers when grouping punctuation is deleted, or when authored operator phrases are reordered into invalid (but vocabulary-known) combinations? The latter are *input-contract violations*, not semantic gold cases. There is no expectation that changed/unchanged answers are morally or epistemically "correct" after violating the grammar. The point is simply to determine whether R17's perfect synthetic accuracy demonstrates parsing/grouping sensitivity or only lexical transition.

## Interventions

On **only the 320 previously answered** R17 cases: (1) remove all parentheses, (2) replace `false that (` with `that false (`, (3) replace `not false that (` with `false not that (`. Changes to the query, label, model, prior predictions and tokenization are prohibited. An unmodified text does not contribute to an intervention denominator. Counts: modified, accepted with identical answer, accepted with changed answer, refused. Previously refused 64 OOD/malformed controls remain in the report and are excluded from new interventions, explicitly.

## Interpretation and limitations

This is a **post hoc mechanistic ablation**, not a prospectively registered scientific experiment. Its success criterion is byte-preserved inputs, exact replay of original predictions, consistent accounting, and canonical source integrity, not a preferred ablation outcome. It does not imply that changes to invalid grammar have independently defined truth values. It cannot show unrestricted language understanding or negate the valid R17 result on its declared grammar. Any observed grouping insensitivity motivates stronger, grammar-independent task definitions before investing in additional model complexity.
