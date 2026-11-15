# ERSA R13 — trained variable-horizon select/continue/halt probe

**Conditional mechanics-only target:** `PASS_FROZEN_NATIVE_VARIABLE_HOP_LEARNED_HALTING_MECHANICS_ONLY`.

This R13 handoff preserves all 87 files from R12 byte-for-byte, the seven frozen SSC predecessor pins and the four frozen ERSA R2 pins. It adds a C-native stateless sigmoid halt kernel, a frozen offline-trained six-weight stop/continue head, and a bounded variable-horizon host runner operating on the actual R11 C-owned SSC RAM state, plus C-native R12 scoring and ranking. **It does not train or run a fully C-native recurrent model.**

## Variable-horizon task

24 synthetic admitted exact-address records, four-vector keys and four-vector values. A source is selected using a query; its stored value is a pointer to the next record, for 1–4 hops. The final record has a distinct goal-like answer value; no explicit hop-count metadata is delivered to the controller. Negative cases include cycles with no terminal, near-goal pointers, and removed goal discrimination. Training oracle path indices are used only to construct offline labels and evaluate held-out answers.

R13 learns six stop-head coefficients offline using torch on synthetic native-compatible observations, and separately trains the original eight selector weights. Weights are frozen in RAM during all inference. The C halt kernel emits CONTINUE / STOP / ABSTAIN, rejects malformed and nonfinite inputs without committing output, and enforces the four-hop ceiling. An **observable exact-pointer veto** prevents early stops on values that perfectly resolve to an admitted key. The deterministic comparator uses the same veto.

### Controls and interpretation

Compare: learned selector + learned halt, strong key-plus-goal selector + learned halt, strong key-plus-goal selector + deterministic stop, and fixed-two-hop stop. All use the same C RAM, admission and step budget. Report success, false terminal stops, abstentions and reads, per family, plus three independent training-seed replications. No learning advantage is implied when the deterministic comparator ties or wins; R12 had the same limitation. No safety claim outside the synthetic law.

**Qualifications excluded:** native full-runtime autonomy, unscoped semantic selection, a GDN baseline at matched throughput/parameter budget, GPU/VRAM, independent SSC transport and OS-level disk-write attestation. Runtime is bounded and intended to write only tmpfs, with no project state mutation. Metadata and full 24-address discovery are host-supplied and not secure against untrusted host code.

The packet contains only a single ZIP deliverable and its internal manifests. Execute with source pins verified before and after. Check the `=== SEND BACK ===` section and preserve HOLD if a predecessor is unavailable or modified.
