# R125 — Early-prefix causal forensics + real preclamp proposal census

**RESEARCH_ONLY; CONSUMED DATA.** R125 is not a prospective holdout or a new learned correction. R93 stays `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.

Base: `7e8536ab2deafae6aabb423f3639ba4d218ce357` (R124 evidence). R123 prospectively qualified a 25% offline teacher-target correction to k4, but it also created 40 high-amplitude readout reversals. R124 reproduced all 40 and showed their one-step inherited-state sufficiency; 38 occur at H3.

R125 enumerates **all 8 operator histories (N/K at steps H1,H2,H3)** for each H3 reversal. The two later events (H9/H11) use the same H1/H2 assignments under uniform N and K middle-step contexts and N/K final operators (16 histories each). N means native frozen HCA; K means fixed 25% post-sigmoid k4 limit intervention, on private float64 bias arrays only. All actual trajectory interventions use the original R119 EXACT_GUARDED kernel. A counterfactual from the same initial state and teacher is assessed with R122's existing direction classifier; teacher is never used to select an inference action.

To distinguish clamp activation from postreconciliation state norms, a separate **diagnostic C instrument** takes an in-memory copy of the original pinned R119 C source, inserts two read-only `preclamp` counters immediately before its existing clamp guard, compiles it under `/dev/shm`, loads the `.so`, then deletes the compile files. It measures actual **unclamped A and B proposal norms** at H1/H2/H3 for both original N and K incoming paths. Every instrumented step must be bitwise identical to its original R119 step; any mismatch is NONQUAL. These counters do **not** establish that clamping caused a reversal, because no clamp intervention is tested.

## Scientific integrity

- Frozen R118/R119 sources and all six HCA checkpoints unchanged.
- R124 event inventory (40) and R123 original H8/H20 scores replay.
- 38 H3, one H9, one H11; all selected by **previously published** evidence.
- No new graphs, seeds, hyperparameter search, trainable weights, online teacher, or unbounded memory.
- All execution scratch and compile state volatile. Only intentional research commits are published.
- Bounded 15,000 native passes, 1,800 seconds, 2,450 MiB RSS and 750 KB evidence.
- First commit preregisters code/protocol, **pushed and verified before replay**. Second commit records evidence (including negative findings).
- Result is `FORENSIC_PREFIX_PASS_BOUNDED` only for integrity. It is not `MECHANISM_PROVED`, an ablation demonstrating clamp causation, or production qualification.

## Run

The archive's `RUN.sh` is executed with a single verified ZIP in the transient handoff mailbox. It reuses the RAM Git cache at `/dev/shm/elpis-r123-reuse/repo` and incrementally fetches only missing R124 history, leaving `/mnt/primesauce/Elpis_DEV` and `main` untouched. The result workspace and compact log remain in `/dev/shm/elpis-r125-prefix-clamp-r0` for inspection. The runner refuses a duplicate published branch or an existing workspace rather than rerunning a failed phase autonomously.
