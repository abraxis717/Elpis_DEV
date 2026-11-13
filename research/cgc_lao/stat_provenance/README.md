# R24 statistical qualification caveat + R25A-R8 enumeration registration

This package is a research-only, append-only **annotation**, not a rewrite of R24's historical result or a repaired LAO. It contains only minimal source and receipt material; no original archived logs or runtime models are copied.

- `R24_STATISTICAL_PROVENANCE_ADDENDUM.md`: dated factual caveat, no change to historical R24 disposition.
- `R25A_R7_STATISTICAL_RECEIPTS.json`: 48 retrospectively typed sampler receipts, one per original hypothesis.
- `R25A_R8_ENUMERATION_REGISTRATION.json`: **future-method** freeze, with statistical null space/multiplicity and fail-closed gates; R7 remains post hoc.
- `verify.py`: standard-library integrity/recomputation checks, read-only and bounded.
- `RUN.sh`: stage and verify the bundle under `/dev/shm` (no publication).
- `PUBLISH.sh`: explicit operator-invoked publishing step to a **new Git research branch**; modifies no local canonical project or archived evidence; does not change `main`. Review/merge is separate. Commit precedes all future R8 evaluation.

Pinned Elpis_DEV main: `5b24b39d793311244ef0ea1af9508d8ebde45bf0`. Destination branch: `research/cgc-lao-r24-statistical-provenance-r0`.

For local R24 authority, cite or attach this addendum next to the immutable R24 ZIP; original R24 ZIP SHA-256 is `cd605851e954ce0e07a1ef12854749918dc227ec98d7e7830967d8e473786169`. The original local capsule and ledger are inaccessible in this session; the packaged note does not by itself update that local record. The reviewed R7 input archive SHA-256 is `9b6c287637405ed9de78900995120be5942117b139595b0423624af6f1588402`.

**Hard prohibition:** no R8 qualification can depend on the retrospective R7 corrected p-values as preregistered evidence. A registered method is not a fresh validated dataset. No model, native source, frozen checkpoint, or inference-state mutation is authorized.

## R1 publication-only repair (2026-11-13)

R0's `PUBLISH.sh` required the pinned main commit to exist in the local
`/mnt/primesauce/Elpis_DEV` object store and stopped with
`fatal: Not a valid object name ...^{commit}` on a checkout missing it.
R1 instead verifies the remote `main` SHA, fetches that single branch with
`--depth=1` into a temporary `/dev/shm` repository, verifies the fetched SHA,
and publishes the exact same annotation/receipt/method files. It never fetches
objects into or mutates the local Elpis_DEV checkout.

Only this publishing helper, the package manifest, and this README changed.
The 48 statistical receipts, R8 registration, statistical verifier and R24
addendum are byte-identical to R0. The recorded R7 reanalysis remains
retrospective; no R8 experiment is performed by this package.
