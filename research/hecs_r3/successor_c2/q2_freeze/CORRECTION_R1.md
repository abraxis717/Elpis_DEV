# C2-Q2 correction R1 — supervisor phase-key bug, found before any Q2 seed started

**Refusal.** The first `--run-seed QUAL2 … 0` refused with `MECHANICS_OR_SCIENCE_REFUSAL_R3_C2=UNKNOWN_PHASE` before creating anything. No Q2 science root, slot or START record was created, no world was generated and no Q2 seed was consumed.

**Cause.** A bug in the Q2 supervisor derivation. Two call sites passed the internal observer label `QUAL` to `prefix()` where the CLI phase key `QUAL2` is required. These were `run_seed` (line 247) and `close` (line 290). In the C2 supervisor the two strings were identical, which is why C2 DEV ran correctly.

**Fix.** Those two lines only (`prefix(project,ph,…)` → `prefix(project,phase,…)`), recorded in the generator as correction R1.

**Unchanged:** the candidate, the Q2 engine and observer binaries, the protocol, the registry and every other identity. The superseded, unconsumed artifacts are preserved byte-for-byte outside Git:
- supervisor binary `7ea7cbfa…`, source `8f79f427…`;
- release R0 `777747bc…`, which remains in Git as `QUAL2_RELEASE.txt` in commit `39e8c33`.

**Re-issued release.** The corrected supervisor (binary `097dbee5…`, source `2fe02d7d…`) re-issued the write-once Q2 release at the engine's seal path as R1 (`49860123…`, `QUAL2_RELEASE_R1.txt`). R1 binds the same candidate, registry, DEV pass, Q1 incident and flags.

This correction is committed before any Q2 seed runs.
