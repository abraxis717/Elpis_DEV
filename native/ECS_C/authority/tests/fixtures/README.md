# Frozen continuity register v2 vectors

`continuity_v2_vectors.txt` and `continuity_v2_sequence.txt` were generated once, from the qualified Python
continuity authority merged by PR #36 (main `bae6952`), before any Rust existed. The generator
(`generate_v2_vectors.py`) is preserved in commit `RUST-A1` of `migration/rust-continuity-r0` and was removed with the
Python authority it imported.

They moved here, byte for byte, from `native/continuity/tests/fixtures` when standalone continuity retired into ECS_C
(docs/NATIVE_SPLIT_CENSUS.md, section 9). They are reference evidence: tests read them, nothing regenerates them.
ECS_C's register reader and evolution law (`src/register.rs`, `src/legacy.rs`, `src/tests.rs`) must reproduce every
byte and every digest.
