# Frozen Yang–Mills/AWR HCA research source snapshot

**Research only. Not production-admitted or a standalone reproduction pass.**

These are the **byte-preserved** source, historical protocols, checkpoint-result archives and selected evaluation helpers collected for R77/R78 research reproduction. The files under `source/` are preserved verbatim for source identity; they have not been modernized, packaged as an installed library or proven to run in a clean checkout.

- `source/frozen_snapshot/r64.py`: HCA construction and historical training mechanics.
- `source/frozen_snapshot/r77.py`: six-model training protocol and result creation.
- `source/frozen_snapshot/r78.py`: frozen variant reconstruction.
- `source/frozen_snapshot/YM_AWR_R77_RESULT.zip` and `YM_AWR_R78_RESULT.zip`: sealed historical checkpoint-result archive inputs. Treat as read-only, audit contents before executing any historical code.
- `source/frozen_snapshot/portable_mechanics.py`: historical geometric/transport mechanics.
- `source/research_helpers/`: selected geometry, quotient and evaluation helpers.
- `INTAKE_MANIFEST.json`: SHA-256 and size for every preserved file, with independent source pins.

This directory **does not contain a high-capacity sparse AWR runtime**. For the proposed bounded participant/relation substrate and the frozen controller's qualification boundary, see the parent `ARCHITECTURE.md`, `RESULTS.md`, `PROVENANCE.md` and `EVIDENCE_INDEX.json`.

Do not conflate the frozen 88-parameter local HCA with the older `native/ECS_C/hca` comparator. R93's `ENERGY_NORMALIZED_DENOMINATOR_HOLD` remains active; local gauge covariance and frozen-shadow gains are not production qualification. No large-graph performance claim, ABI integration or global scheduler has been established.

Original upstream inspiration: [creelie/4Y-M](https://github.com/creelie/4Y-M), v4.21.0 (MIT); upstream authors and bibliography are credited in the parent `PROVENANCE.md`. These collected files are Elpis research snapshots, **not copies of 4Y-M**.
