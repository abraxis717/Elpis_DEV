<!-- Publication adaptation: local host paths redacted; source SHA-256 in manifest. -->
# HCA H1 local operation and handoff

1. Read `H1_PROTOCOL.md` **before** any B3 training. H1A is a compositional anti-shortcut assay; its own simple symbolic control may saturate the task.
2. Extract the one ZIP under `[LOCAL_HOST_PATH_REDACTED]` with `unzip -n`; it places this directory under `hca_system_dev_r0/H1/`. This does not alter the canonical Git checkout.
3. Run `bash [LOCAL_HOST_PATH_REDACTED]` in Konsole. It requires exactly the already-passed H0 `RESULT.json`, source commit and clean tracked checkout. It creates a unique named H1 result directory and writes no source elsewhere.
4. Read `RESULT.json`, `AUDIT.json`, `CONTROLS.json`, and their file digest manifest. `DEV_GOLD.jsonl` and `PAIR_AUDIT_ONLY.jsonl` are developmental scoring files; do not place them in a learner's public input channel.
5. If learned symbolic induction achieves 160/160 on DEV, treat H1A as a **necessary negative control and unit assay**, not sufficient science. Next phase must implement the H1B dynamic stream, matched incremental B0I, factor-node learner and genuine held-out evaluation before attempting HCA B3 claims.
6. If any mechanics HOLD occurs, do not re-run to get a green sample; inspect the failure in the named run directory. Do not overwrite files or retrofit old evidence.
7. If the canonical Git HEAD has advanced beyond the pinned source, STOP; do not silently rebase an authority-bound study. Set up a separately named migration/DEV comparison.

The intended final Konsole output includes `HCA_H1_EXIT=0`, `PASS_HCA_H1A_TASK_CONTRACT_DEV_NOT_SCIENTIFIC_QUAL`, and `H1_SCIENCE_PRIMARY_HOLD=...` when the train-only symbolic model solves every sample.
