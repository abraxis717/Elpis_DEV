<!-- Publication adaptation: local host paths redacted; source SHA-256 in manifest. -->
# H1C1 independent observer / resource ledger (DEV only)

**One Konsole command** (place ZIP in `[LOCAL_HOST_PATH_REDACTED]`):

```bash
unzip -n -q [LOCAL_HOST_PATH_REDACTED] -d [LOCAL_HOST_PATH_REDACTED] && python3 -u [LOCAL_HOST_PATH_REDACTED]
```

No shell failure traps, no Git mutations, no installed dependencies, no network calls, no source modification or historical evidence regeneration. Everything persistent goes under `[LOCAL_HOST_PATH_REDACTED]`. `[LOCAL_HOST_PATH_REDACTED]` is a download mailbox, not an execution workspace. Local optional no-host smoke test: `H1C1_SELFTEST_H1C0_DIR=<unpacked H1C0 directory> python3 -u H1C1_REVIEW_R0_v1/run.py --self-test`.

## Inputs

Pinned tracked source commit `2c7588733180d1ab2e8e978cf277c30f9e4ab371`.

Pinned H1C0 RESULT `84182a32f0269bdd5c60d9fc95e8ac05da835679bf46c887fb09fe2b585b9b63` under `runs/20261023T161341Z_112698_1792772021262764824_H1C0_EXACT_GATE`, evidence manifest `bac528f167908d782bc39342aeb054f9ca2ca750abdbb4c2e95253b7da7fea19`.

The H1C0 source is pinned by package manifest SHA `84e05fb1a68828eddeb2ea12ce46cd20dd6311523426c04220d7911297671932`. Manifest and listed files must match exactly, as must the evidence and result bytes. The script neither reruns nor overwrites H1C0.

## What the independent observer does

- Enumerates legal live `R0->R1->R2` *triples* for every query at every event prefix without calling H1C0's oracle, engine or checker.
- Rechecks complete TRAIN and TUNE gold and prediction maps, entity/factor/event uniqueness, per-case work records, pair identities, opposite-label interventions, nuisance-preservation contracts and reported denominators.
- Runs 18 deliberately corrupted observer inputs through an independent validation path. Only a specific observer rejection counts as success.
- Re-runs **only exact comparator timing/counter measurements** with pinned H1C0 engines in separate child processes; reports 3 samples per engine/split and Python tracing/peak-RSS limitations. No H1C0 regeneration, gold mutation or retraining.
- Creates a new independent immutable evidence directory with a SHA-256 manifest and decisive printed results/hashes.

Scientific interpretation is deliberately negative/limited: this is not a learned HCA trial. Native HCA B0/B0I, trained factor-node learning on H1C0, damaged-state recovery, novel uncertain inference, and B3 are **not** assessed. Sealed DEV remains absent. Failure to verify provenance, parity or accounting is a **mechanics HOLD**, never proof against HCA.
