# H-ECS R3 successor — independent early-gate disposition handoff

**Classification:** Cold-side audit summary. **Not** an eight-seed cohort verdict, scientific release, Git authorization, or modification of any frozen file.

## Result

Both frozen successors **C0 and C1 fail the preregistered universal DEV M gate** at pair 2 (seed `5526746150047008937`), HD/L2 rollout horizon 4. This is an authentic scientific nonpass, not a mechanical refusal.

| DEV index | C0 capture | C1 capture | C1 − C0 | Both pass? |
|---|---:|---:|---:|---|
| 0 | 0.977730252 | 0.977241128 | -0.000489124 | yes |
| 1 | 0.918004826 | 0.933688171 | +0.015683345 | yes |
| 2 | 0.852976853 | 0.868555190 | +0.015578337 | NO |

The frozen M gate requires capture ≥ 0.90 and escape ≤ 0.01, with failure of any registered world/binding/horizon invalidating the candidate. Both arms at pair 2 have escape = 0. The frozen protocol SHA-256 is `9ad9d1d35ce40756aaac91c47a1f26e635265e6115576f820b16d0d000ca262a`.

## Verification boundaries

- Independently opened the **R3-BU raw evidence**, checked the three START/COMMIT identities, predecessor links and 42 committed data/log hashes, and recomputed HD/L2 horizon-4 capture from underlying error terms.
- Opened the **R3-BV host readmission**: the captured log reports fresh native R3-BP reads for all six candidate/seed combinations, unchanged release pins, and no scientific execution.
- R3-BV's bundled `OFFLINE_INDEPENDENT_ADMISSION.json` is byte-identical to the earlier R3-BU offline report. It is reused provenance, not independent new offline data.
- The uploaded R3-BV artifact contains only log + JSON, not fresh raw TSVs; raw arithmetic here was verified against the earlier R3-BU archive.
- Exactly **3/8 pairs are evidenced**, **5/8 remain unstarted as of the host census**; the full frozen eight-seed phase gate was not executed.
- C1 changes **both** horizon-4 regularization and HD/L2 training windows; the candidates are not equal-compute.

## Decision boundary

A failure of a universal all-world criterion is logically irreversible after a single authenticated counterexample. Thus **neither C0 nor C1 can qualify** under the already frozen rule. This does not automatically create a formal eight-seed cohort verdict or an unregistered early-stopping rule.

Two scientifically honest options remain:

1. **Stop and explicitly record the early conclusive universal-gate nonpass** with an incomplete cohort annotation. Preserve all DEV bytes and five unused seeds. Do not issue QUAL.
2. **Finish registered pairs 3–7 sequentially** using the unchanged sealed BQ supervisor, solely to materialize the entire preregistered eight-seed cohort and distributional descriptive observations. The known gate failure cannot be repaired by their results.

**Do not launch pair 3 automatically.** The next scientific-run decision belongs to the operator. No C1 repair on consumed DEV seeds, no QUAL/R4/production promotion, and no Git mutation.

## Authenticity anchors

- DEV release: `cffe1fffab6e26e16e30e4d03fd72727ab55ffe115487703b10715989335a516`
- R3-BR admission: `3425bc453a2d9b53c54f8e054dffd8e4d4cfa09e7f23a3307cb6fb507780ca5c`
- Pair 0 commit: `a731f84c5d77aa556d261e28075c777439839bfb630007e841ebcf18bee85c26`
- Pair 1 commit: `5c48e9ec9446578d4030731fc06a1b437646fcd0317fd4fa0fcd5ed082aeabfd`
- Pair 2 commit: `f5992e836342e1ac11daf7063101a34cc119d23a366e067fa7214adb4e55cf0b`
- R3-BU raw archive: `76bbd1cd64f957c37e58bfac866da15be50335ab01f649afa6a43ce7733b3ea7`
- R3-BV host debrief: `8be5554720156993374f3368c5931b3809a84f354b73942bad913f136135a214`

This handoff should be treated as evidence summary, never as a replacement for the immutable Ouroboros authorities.
