# R123 preregistration verification erratum R1

Original R0 preregistration: `d52ccc20c56bd67ef4770ca020145900f834a9cf`
(base: `501be02d33201ce9c95e211d6ba9a1b531656f43`).

R0 was published remotely before an attempted prospective execution.
The attempt aborted in `prereg_guard` with `HOP_DRIFT`, before the first
`R123_GRAPH_START` and before any new evaluation state was generated.
The published graph definitions were correct; `make_protocol.py` inserted
graphs as H1_A, H2_A, H1_B, H2_B, but JSON `sort_keys=True` ordered them
H1_A, H1_B, H2_A, H2_B. The R0 verifier incorrectly depended on the
insertion order and expected [1, 2, 1, 2] from dictionary iteration.

R1 changes only: (1) verify hops by explicit graph labels, independent of
serialization order; (2) require branch R1 and ancestry via the immutable R0
preregistration and base; (3) regenerate the protocol with the new verifier
source hash. All graph values, seeds, arms, teacher limit, metrics, success
thresholds and runtime budgets compare byte-semantically equal to R0.

R0 remains published and immutable. R1 must be published before its single
prospective evaluation. R93 remains ENERGY_NORMALIZED_DENOMINATOR_HOLD.
