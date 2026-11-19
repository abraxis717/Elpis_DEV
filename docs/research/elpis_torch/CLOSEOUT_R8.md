# Elpis Torch R8 — post-merge closeout record

This is a **post-merge addendum**, written at the start of R9 (19 Nov 2026) against the merged `main`.

- `EVIDENCE_R8.md` and `DESIGN_R8.md` are historical records of the pre-merge work and are not edited.
- Where this record and those documents differ, the difference is stated here; the originals are not changed.

## Authority verified on GitHub

| Fact | Value |
|---|---|
| Repository | `abraxis717/Elpis_DEV` |
| `main` | `4534fbd856ce1b90b7b49e478059f0e67ef9cfaa`, the merge of PR #78 (parents `bfe4a12` and `efb910e`) |
| R8 | PR #78, merged |
| Post-merge CI | run 44553578427 (run 484): 16 of 16 jobs succeeded on `4534fbd`, and it was the latest run on `main` when R9 began |
| Runner policy | 15 jobs that build or run native code on `ubuntu-24.04`; Donor parity on `ubuntu-latest` |
| `docs/CI_POLICY.md` rule 5 | satisfied: post-merge CI on `main` is green |
| Incident CI-RUNNER-IMAGE-1 (`docs/CI_INCIDENTS.md`) | closed: its closure condition was met on the PR head (runs 44538083878 and 44538093798) and again on `main` (run 44553578427) |
| Scheduled check-ins | none remain |

## Content checks

- **R6, R7 and R8 are on `main`.**
  - `efb910e` (the R8 head) is an ancestor of `4534fbd`.
  - R7's commits `a48a4a2` and `b41842c` are ancestors of `4534fbd`; they reached `main` through the R8 merge.
- **Nothing is duplicated or stranded.**
  - No commit subject appears twice between `bfe4a12^` and `4534fbd`.
  - `claude/r6-expert-backbone` ends at `ea81dae`, the merge of PR #77. Its tree is identical to `b41842c`, which is
    on `main`.
  - `claude/r7-ple-admission` (`b41842c`) and `claude/r8-evidence-read-path` (`efb910e`) are ancestors of `main`.
- **The one earlier commit outside `main` is reconciled.** `fe82cc8` on `claude/funny-turing-obbdy5` adds five R5
  records, which are byte-identical on `main`, and a loader change to `run_r5.py`. R6 replaced that change with the
  explicit slot-extension rule (recorded in `EVIDENCE_R6.md` and `EVIDENCE_R8.md` section 0).
- **R8 changed no canonical code.** `git diff bfe4a12 4534fbd -- src native` is empty, and so is the diff from the R6
  head `c05adff`.
- **Stored fine keys stay opt-in.**
  - The stored-key path is still present on `main`, with `EvidenceConfig.fine_keys` in `("reread", "stored")`.
  - The default is `"reread"`; `test_r8.py` asserts it.
  - Stored fine keys remain an **opt-in research optimization**. Making them the default was never decided.
- **The final research job passed on `main`.**
  - The PyTorch research job passed in run 44553578427: the suite on the runner's kernels, again with the AVX2 paths
    forced, and every experiment smoke.
  - Every native-dependent job passed in the same run, on `ubuntu-24.04` (GCC 13.3.0, clang 18.1.3, CMake 3.28.3).

## Dispositions carried forward

- **CUDA qualification: NOT_RUN.** No device was available. Nothing about GPU behaviour of the R5 read or the stored
  keys is established.
- **Stored fine keys:** opt-in. Their resident cost (128 MiB for `enc64`, 256 MiB for `enc256` per stream at 1M, in
  float32) is paid whether or not a session reads.
- **Two native compatibility questions remain open** (CI-RUNNER-IMAGE-1). They are outside R9's scope; R9 does not
  touch native source, compiler warnings, containment semantics or runner selection.
  - GCC 15 `-Wmaybe-uninitialized` in `native/pipeline/ingress/src/regex_hacf_query_ingress.cpp`, lines 269 and 284.
  - The clang 21 builds failing `runtime.native_host_smoke` on Ubuntu 26.04 (Landlock ABI 8). The GCC 15 build
    passes the same test there.

## Correction found while preparing R9

`EVIDENCE_R7.md` section 4 says R7 trained for "1,500 steps of 16 x 128 bytes, about one epoch". The R7 records
(`results/r7/study.json`) show **1,432** completed steps for every run.

- The seeded epoch holds 1,432 full batches for that corpus, and the iterator ran out.
- The cosine schedule, configured for 1,500 steps, therefore stopped at 95.5% of its length.
- The R7 comparisons are unaffected: every arm ran the same 1,432 steps.

R9 states the difference and runs exactly 1,500 steps (`PREREGISTRATION_R9.md` section 3).
