# Frozen continuity register v2 vectors

`continuity_v2_vectors.txt` and `continuity_v2_sequence.txt` were generated once, from the
qualified Python continuity authority merged by PR #36 (main `bae6952`), before any Rust
existed. The generator (`generate_v2_vectors.py`) is preserved in commit `RUST-A1` of
`migration/rust-continuity-r0` and was removed with the Python authority it imported.

These files are reference evidence. Tests read them; nothing regenerates them. The Rust
authority (`src/tests.rs`) and the C ABI (`tests/test_continuity_abi.c`) must reproduce
every byte.
