# ECS_G Retention R0 laboratory

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Question: can one ECS_G state acquire a new experience B while retaining a
previously acquired experience A, and what state does that require?
Cognitive R0 found that plain sequential G1 learning does not retain
(interference in 8 of 8 QUAL worlds, catastrophic in 6 of 8).

* `CANDIDATES.md`: the mechanisms (M0 plain G1, M1 rehearsal control, C1
  coarse S3-space functional consolidation, C2 local per-weight
  consolidation, O representability analysis), their state and
  falsification conditions.
* `specs/ecsg-retention-r0.v1.spec.json`: regime, tasks, splits, metrics,
  DEV rules, gates A-J, dispositions, mechanics checks and binding
  requirements, fixed before DEV.

Chronology: RET0A specification and laboratory (no results) -> RET0B DEV
evidence -> RET0C freeze -> RET0D QUAL evidence (once) -> RET0E
interpretation. Every response is the canonical native forward map of the
evaluated `W`; every G1 step is the canonical native step. No DSV, tokenizer,
transformer or other model participates, and nothing here is imported by
`src/`.
