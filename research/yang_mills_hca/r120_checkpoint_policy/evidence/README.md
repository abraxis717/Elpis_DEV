# R120 fresh-seed observation

Registered protocol and scoring source were committed before this evaluation on
`b3f4ca8a00866b506473be5640ea82ac9f29ad5b`. The unchanged R120 `qualify.run()`
was executed exactly once. An external, source-disclosed wrapper captured the
underlying unchanged R118-E raw result prior to the registered R120 score.

- `R120_RAW_FRESH.json` — full graph/checkpoint/source-harness evaluation result.
- `R120_REGISTERED_SCORE.json` — preregistered candidate-vs-controls gate.
- `R120_SECONDARY.json` — per-graph and per-checkpoint descriptive outcomes;
  no additional admission gates were introduced.
- `INDEX.json` — raw/source/protocol and evidence digests.

This is a fresh seed/structure replication on **existing** topology classes,
not a new-topology holdout or evidence to clear R93. The policy changes no
trained HCA parameters and does not read teacher data during inference.
