# H-ECS R3 successor — unresolved DEV 3/8 (2026-10-09)

`RESEARCH_ONLY` · `NOT_QUALIFIED` · `NO_RUNTIME_AUTHORITY` · `DEV_INCOMPLETE` · `NO_QUAL`

This is an **exact read-only Git snapshot**, copied from original sealed native evidence on Ouroboros after verification of the three chained commits and 42 payload/log hashes. It is not a new frozen scientific release and does not alter the original primary evidence. The original `/mnt/primesauce/Elpis_DEV/build/hecs_r3_successor_dev_science_r0/` remains authoritative.

## Scientific disposition

The frozen successor comparison is C0 (historical R3 rollout) versus C1 (joint 8-window HD/L2 + detached horizon-4 consistency with lambda4=0.1). Each used 96,000 K1 steps and 256 updates; the compute work is **not matched**. Only **three of eight** preregistered DEV paired seeds ran. C0 and C1 both passed task, S1 and multistep M at pairs 0 and 1. Both *failed* frozen M at pair 2, HD/L2 rollout horizon 4, below the 0.90 capture threshold. Thus the **universal all-world validity gate cannot pass for either candidate**. No full eight-seed cohort gate was executed. The five remaining DEV pairs are unstarted at the publication census, not repurposed, and must not be represented as tested.

| Pair | Seed | C0 capture | C1 capture | M gate |
|---|---:|---:|---:|---|
| 0 | 15007341021985616615 | 0.977730252 | 0.977241128 | both pass |
| 1 | 12334837942893278080 | 0.918004826 | 0.933688171 | both pass |
| 2 | 5526746150047008937 | **0.852976853** | **0.868555190** | **both fail** |

## What this directory contains

- `evidence/seed_{0,1,2}/COMMITTED/` — exact copies of ten native five-file per-candidate TSV sets, four native child/observer logs, `START.txt` and `COMMIT.txt`.
- `frozen_boundary/` — digest-pinned release/approval/protocol, source and mechanics reports copied from host; **not** the complete historic R3 model source tree and **not** compiled binaries.
- `R3_BW_README.md`, `R3_BW_REPORT.json` — prepublication scientific handoff; its historical `git_mutation_authorized=false` refers to the earlier phase, before explicit operator permission for this research-only Git publication.
- `RECORD.json`, `SHA256SUMS` — public-snapshot manifest and content checksums.

**No scientific continuation, C1 modification, QUAL release, R4 transition, production integration or model superiority claim is authorized by this archival push.** GitHub files are copies; do not replace or rewrite the original immutable host science.

The existing H-ECS R2 raw-output policy remains specific to R2. This R3 snapshot of already committed evidence is intentionally published here by explicit operator instruction; no change to R2 evidence policy is implied.
