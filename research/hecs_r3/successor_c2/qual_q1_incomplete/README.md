# H-ECS R3 successor C2 — QUAL execution Q1: mechanically incomplete (infrastructure interruption)

`RESEARCH_ONLY` · `QUAL_EXECUTION_COMPLETE=NO` · `SCIENTIFIC_FAILURE=NO` · `C2_SCIENTIFIC_QUAL_VERDICT=NOT_ADJUDICATED` · `REGISTRY_SPENT`

**What happened.**
- The one-time Q1 QUAL release (`../dev/QUAL_RELEASE.txt`, `d29b0622…`) was issued after the C2 fresh DEV 8/8 pass.
- Q1 seeds 0–2 were committed.
- Seed 3 (`15447012595138726224`) was started. Its child was then killed by an external container restart. This was an infrastructure event, external to the scientific object.
- The slot remains `STARTED_UNCOMMITTED`. The supervisor never wrote a FAULT record, and none was fabricated afterwards.

**Disposition (operator decision).**
- `C2_DEV = VALID`
- `CURRENT_QUAL_EXECUTION = INCOMPLETE_INFRASTRUCTURE_INTERRUPTION`
- `CURRENT_QUAL_REGISTRY = SPENT` (all 12 seeds, including the unexecuted 4–11)
- `C2_SCIENTIFIC_QUAL_VERDICT = NOT_ADJUDICATED`

This is not `WORLD_MODEL_INVALID_ON_QUAL`. Nothing was replayed (`REPLAY_PERMITTED=NO`), and seeds 4–11 were never run.

**Preserved.**
- The full Q1 tree is preserved byte-for-byte outside Git: committed seeds 0–2 and the interrupted seed 3 with its partial raw files and child stream.
- `PRESERVED_TREE_MANIFEST.txt` lists every file's sha256. Its digest is bound in `INCIDENT.txt` (an operator/infrastructure incident record, separate from any supervisor record).
- `seeds/` holds the compact START, COMMIT and observer records of seeds 0–2 and the START record of seed 3. Seed 3 was not inspected beyond preservation.

**Role of these results.** The Q1 results are historical evidence only. They play no role in any decision, and the spent seeds are never DESIGN or QUAL data again. The unchanged C2 candidate is qualified instead on a fresh registry under Q2 (`../q2_freeze/`).
