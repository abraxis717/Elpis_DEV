# ERSA historical source handoff (R18, R35, R36)

**RESEARCH_ONLY · SOURCE_ARCHIVE_ONLY · NO_RUNTIME_AUTHORITY · NO_MODEL_INTEGRATION**

These are byte-for-byte historical ERSA development sources carried from an
Ouroboros source-capture handoff. Files were validated individually against
`source_handoff/SOURCE_MANIFEST.json` before import, with the original archive
SHA-256 recorded below. They have not been rewritten, installed, executed, or
merged into `research/elpis_torch`.

This archive contains:

- **R18:** trainable PyTorch `ERSATokenMixer` and read-only `ERSAState` interface,
  native/SSC research prerequisites, tests, and earlier R4–R18 notes.
- **R35:** donor-backed gated-slot write/read reference, with the MIT FLA
  vendor file and upstream license retained.
- **R36:** CPU C dual-matrix GSA state bridge and differential tests. R11 C
  entity-bank authority is **not** unified with this state.

The sources are archived at `source_handoff/ERSA_DEV/` with their original
internal paths. Install scripts are **archival only**. Do not run them as part
of the Elpis native runtime or default CI. Imported tests are **not certified**
to pass in the Elpis Torch environment; a separately reviewed integration and
upstream license/dependency audit are prerequisites to any model use.

Boundary: no model inference, no persistent inference writes, no production
admission, no CUDA/GPU claim, no canonical runtime imports, no modification of
Claude's existing PyTorch implementation, and no automated changes to `main`.

The R35 donor attribution is to FLA and Raven for the indicated mechanisms;
this archive does not claim original authorship of their math or source.

SOURCE_ARCHIVE_SHA256: 030ad293bcf5ad593fc2d2c1807008b2a02d7b2b092c3fe40fc82f322e70a4db
SOURCE_MANIFEST_SHA256: b67dc98c473d3f307d2efd4890242e6027077b784e774be23dd9b66c6b625ede
SOURCE_FILES: 131
ELPIS_TORCH_BASE: `claude/modest-euler-ip3hj4` (resolved and pinned on push)
