# R125 registration erratum R1 — late output-path shadowing

Immutable R0 preregistration: `1bac95bebf96843f98adb1200aa70fa0584df700`, parent R124 evidence `7e8536ab2deafae6aabb423f3639ba4d218ce357`.

The first R0 execution replayed all 14 selected fixture groups, completed the model diagnostics, but failed before writing the final result. In `evaluate(out)` the name `out` initially denoted the volatile result destination `Path`; inside the per-sequence counterfactual loop the same name was rebound to a NumPy state vector. The final `out.exists()` therefore raised `AttributeError: 'numpy.ndarray' object has no attribute 'exists'`. No R125 evidence result was written or published, and R0's failed workspace and log are retained.

R1 repairs the *output serialization path only* by renaming the `evaluate` argument and the three output-write references to `output_path`. The local candidate vector `out` is not changed. The only other edits update the research branch, verify R1 ancestry via R0, and regenerate the corresponding source hashes. The scientific protocol, 40 consumed events, six checkpoints, original numerical kernel, counterfactual sequence definitions, metrics, thresholds, seeds, budgets, negative controls and R93 hold are unchanged. Semantic protocol equality is checked by the runner with only the branch and two source-hash fields excluded.

This R1 reruns an **already-consumed retrospective dataset**, not a new prospective holdout or newly registered discovery. The R1 preregistration is independently published and remotely verified before execution. No trained weights or original R119 source are modified. R93 remains `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.
