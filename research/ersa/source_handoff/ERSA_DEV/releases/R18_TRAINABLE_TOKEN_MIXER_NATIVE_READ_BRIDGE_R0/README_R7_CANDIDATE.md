# ERSA R7 — Native host-authorized read path

**Scope**: Integrates native eight-feature extraction, score, canonical top-K ranking and signed four-vector reduction into one bounded CPU read path. The *R5 Python bank* retains admission, exact logical IDs, revisions, source and ticket validation, policy, discovery and every mutating operation. This is a **native read operator**, not a native state bank, autonomous selector, trained recurrent model, or SSC-conformant backend.

## Distinctions

- This phase replaces R6's precomputed-score benchmark with a CPU wall-clock measurement covering authoritative discovery + host validation + native features + native scores + native ranking + native reduction. Score-complete R5 versus dense-discovery+native R7, identical insertion state, n=256/1024/8192; insertion excluded from read latencies.
- 8-feature R5 ordering is retained: `qcompat,utility,freshness,credibility,contradiction,goalrel,protected,geomcompat` (the fixed relevance weights remain frozen). Underlying source trust and query permission **are not** inferred from similarity.
- Native feature buffers are all-or-nothing on invalid input; C ABI has no persistent state and never performs admission or memory updates. Host rescans selected identities, revisions and values before releasing outputs. Numeric half-step scores are recalibrated to R5's 12-digit canonical rounding rule.
- Host authorizes every candidate via the frozen bank. An indexed read may miss no-overlap candidates, while dense discovery is O(N). Complete candidate discovery does not mean task-grounded semantic correctness.
- Feature conversion hardens float/int narrowing of relation identifiers and uint8 eligibility, avoiding modular wrapping of attacker-supplied values.
- All R5 fixtures, R2 failure guards, R6 native differential tests are preserved. Mutating the frozen source or expanding permissions is forbidden.
- R2 predecessor file pins are required on the local Ouroboros filesystem. SSC real predecessor files were located by R0 source census at `/mnt/primesauce/RALPH/DEV/Hermes/UnifiedStateConformance_R0/src/elpis_ssc/` but were NOT present in the uploaded source. Conformance remains HOLD.

## Qualification ceiling

`PASS_NATIVE_FEATURE_TO_REDUCE_AUTHORIZED_READ_PATH_DIFFERENTIAL_ONLY`

This stage is *not* a native neural runtime or GDN drop-in. Main remaining gates: exact SSC predecessor conformance using a versioned identity binding; learned recurrent relevance/selection with task-grounded recall; model-side integration. No production trust/rights claim.

## Run

Extract this ZIP under `/dev/shm`, set `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/dev/shm`, `HOME` and `XDG_CACHE_HOME` to your volatile extraction directory, `PYTHONPATH` to `root:root/src:root/native`, and invoke `python3 -B tools/qualify_ersa_r7.py`. Compiler outputs are transient `/dev/shm/lib...so` and removed after each test. No new files are installed under `/mnt/primesauce` except optional user handoff ZIP.
