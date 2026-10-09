# H-ECS R3 original Rust implementation — source-only supplement

`RESEARCH_ONLY` · `NOT_QUALIFIED` · `DEV_3_OF_8` · `NO_RUNTIME_AUTHORITY` · `NO_QUAL`

This is the **original host source**, copied without reconstruction, for the frozen C0/C1 successor DEV engines and R3-BP observer. The three `engines/support/*.rs` modules supply `paired_step`, the native scientific emitter, and C1 `successor_step`; `observer/r3_bp_observer.rs` supplies the independent scientific gate implementation. The engine entry points are identical to the sources already archived in the 3/8 evidence capsule.

The entry points originally lived at `native_source/src/bin/r3_bn_{c0,c1}_dev_engine.rs`. For publication under `engines/`, their `#[path="support/..."]` imports resolve against `engines/support/` unchanged. Dependencies on `hecs_r2` and native C K1 are external to this small supplement and already have separate repository authority. The original Cargo manifest is included under `build/` only if present on Ouroboros; no manifests were reconstructed.

Source-to-binary bitwise rebuild parity is **NOT VERIFIED** here. Frozen binary SHA-256 was checked on the host but no binaries, raw evidence, optimizer checkpoints, or runtimes were uploaded. The 3/8 results and failed universal gate remain unchanged. Remaining five DEV pairs are unrun. Do not promote this archive as valid QUAL or production integration.

See `ORIGINAL_SOURCE_SHA256.json` for the exact captured source hashes.
