# H-ECS R3 successor C2 — frozen DEV/QUAL protocol (prospective; science unstarted at freeze)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_R4` · `NO_INTEGRATION`

## Authority and question

The R3 successor DEV cohort closed 8/8 with C0 and C1 both `WORLD_MODEL_INVALID_ON_DEV`. Each failed only one cell: seed `5526746150047008937`, HD/L2 ROLLOUT h=4 (`research/hecs_r3/closure/successor_dev_8_of_8/`, commit `d26c78b78b4cbedbc412f6a410076f15bdadf350`). Those eight worlds are burned for qualification and were used, together with 80 DESIGN-only worlds, solely as DESIGN evidence (`DESIGN_RECORD.json`).

This protocol asks one question. Does the **fixed** candidate C2 pass the unchanged frozen validity hierarchy (TASK → S1 → M) at **every** seed, binding and bound horizon of a fresh, never-executed 8-world DEV registry? If and only if it does, the question is asked once more on the 12-world held-out QUAL registry. No comparison arm is run. No superiority, causal or compute-matched claim is made.

## Candidate C2 (frozen)

Engine `r3_bn_c2_dev_engine.rs`: the frozen successor C1 engine source, byte-for-byte outside the documented C2 diff, with support modules `r3_training_core.rs` and `r3_bd_emitter.rs` unchanged and the new `r3_c2_terminal.rs`.

Unchanged from the R3 successor law: native K1 G1 96,000 steps; every binding (L1/N72 h16, L1/N36 h4, TS/L2 h4, HD/L2 h4), its windows and its measurement; the G1 arm; the TEACHER arm; all task rows and the V5 census; RATE = 0.0005·36/width; CLIP = 10 (global gradient norm); LAMBDA = 0.37. At HD/L2 the TEACHER arm runs exactly the teacher half of the historical paired step (256 updates on the seven picks `windows[(7s+j) mod n]`, j = 0..6, objective_gradient(Teacher), update), so its trajectory is the C0 TEACHER trajectory; no rollout step is evaluated alongside it.

Changed: only the HD/L2 ROLLOUT arm. Starting from the native G1 W, it runs `C2_UPDATES = 4096` updates. Update s uses the six C0 rollout picks `windows[(7s+j) mod n]`, j = 0..5. Its loss is the unchanged R3 rollout objective plus a true-target terminal anchor:

`L = mean_windows[ e_1 + (λ/3)(e_2 + e_3 + e_4) + w_T · e_4 ]`, with `e_t = 0.5·‖ẑ_t − z_t‖²/(LATENT·var)`, `λ = 0.37`, `w_T = C2_TERMINAL_WEIGHT = 1.0`.

- ẑ is free-running from the true z_0 through the cubic transition G_W with the true actions. z_t are true training latents (never a teacher prediction).
- var is the train-only latent variance.
- The gradient is exact reverse-mode through all four recursive transitions; the base term and the terminal term are computed separately and summed.
- Each update uses the unchanged R3 update and clip rule.
- Numerical escape (|ẑ| > 1e3 or non-finite) in a C2 update is a committed scientific failure of the ROLLOUT arm (event stage `C2_OPTIMIZER`, failure rows for ROLLOUT only; G1 and TEACHER stay measured), never a retry.

Selection basis (DESIGN only; `DESIGN_RECORD.json`): a bounded grid over terminal weight, update budget, window count and per-step weighting, across the 8 burned worlds plus 32 DESIGN worlds. The selection rule, declared before the held-out check, was: within the C2 family (unchanged base objective plus true-target terminal anchor, with update budget as the only other axis), choose the highest minimum HD/L2 h4 capture. Terminal weight 1.0 with 4096 updates (minimum 0.9718 over 40 worlds) was then confirmed on 48 held-out DESIGN worlds (48/48, minimum 0.9764). The DESIGN margin 0.92 is a selection margin only; it is not a qualification threshold.

## Registries

**DEV (fresh, never executed):** `seed_i = u64 big-endian of SHA-256(ASCII("HECS|R3-SUCCESSOR|C2|V1|DEV|" + decimal(i)))[0..8]`, i = 0..7, with no rejections.

0: `684004886576317082`
1: `1703137016795210571`
2: `10088827697117853828`
3: `8960933697477120594`
4: `134760612940272399`
5: `7871577751992445249`
6: `17179523806848842400`
7: `13174565413729237818`

**QUAL (held out, never executed):** the reserved successor QUAL registry, `R3_BN_ADMISSION.json` key `qual_seeds_reserved`, designated by the operator as this lineage's held-out QUAL. It is kept untouched until a DEV pass:

`10405860717610874745`, `17760219202278299362`, `14564880931009575015`, `15447012595138726224`, `8070250311440129494`, `7015268779680225457`, `12995885642664273032`, `4683001325509031006`, `5700582020531935958`, `15439875992717053813`, `2654106804044592861`, `1474235648324239766`.

**Disjointness.** DEV ∪ QUAL is disjoint from:
- every H-ECS registry in Git (93 values: R0, R1, R2, historical R3 DESIGN/DEV/QUAL, the R3 synthetic census, successor DEV/EXPLORATORY/QUAL);
- the first 4096 indices of every known derivation domain;
- the burned successor DEV cohort;
- all 80 C2 DESIGN seeds (`ELPIS/HECS/R4/C2/DESIGN/V1/<i>`, i = 0..79).

The engine's DESIGN mode refuses every DEV/QUAL world.

## Evidentiary law (unchanged arithmetic)

Per seed, the engine writes the five-file set: 4 observations, 60 metrics, 2 task rows, 14 V5 rows, events.

The C2 observer is the frozen R3-BP observer (source `c63f71f5…`) with an explicit cohort registry and candidate label C2. Its per-seed recomputation and cohort decision law are byte-for-byte R3-BP. On the legacy 8/8 successor cohort its outputs are byte-identical to R3-BP.

Gates, in order:
1. **TASK:** T1, T3, V3, V5 per seed, and no SCIENTIFIC_NONPASS event for the seed; V2 and V4 as cohort means over all registry seeds. Under the frozen R3-BP law any event fails TASK_LOCAL for that seed. This covers a G1 native refusal, a measurement failure, and a C2 `C2_OPTIMIZER` escape. Such a seed therefore yields `TASK_INVALID_ON_*`, which also closes C2 as failed.
2. **S1:** ROLLOUT one-step capture ≥ 0.95 at every binding.
3. **M:** ROLLOUT capture ≥ 0.90 and escape ≤ 0.01 at every binding and every horizon h ≤ consumed.

**Every seed means every seed.** One failure at any world, binding or horizon fails the phase.

**Pre-registered, candidate-independent TASK risk.** Task validity depends only on the world, through the generator and the frozen task law, never on the candidate. Across the 40 worlds examined during DESIGN (32 full-hierarchy C2 DESIGN worlds and the 8 burned worlds), the MATCHED world's T3 maximum observation L/G had median 1.084, 90th percentile 1.181 and maximum 1.2355. The other 31 full-hierarchy C2 DESIGN worlds passed every gate. One world exceeded the frozen 1.20 limit.

A TASK-invalid DEV or QUAL world fails the phase under the frozen law. If that happens, the phase is recorded as TASK-invalid, which is not a world-model verdict, and it is reported for an operator decision. No DEV or QUAL world is screened, replaced or re-drawn for task validity.

DEV pass yields disposition `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE`. QUAL pass on every seed yields `WORLD_MODEL_VALID`. Otherwise the disposition is `TASK_INVALID_ON_*`, `ONE_STEP_MODEL_INVALID_ON_*` or `WORLD_MODEL_INVALID_ON_*`, in the R3-BP priority order.

## Execution law

The C2 supervisor (derived from the frozen R3-BQ supervisor) does the following.

**Release.**
- It writes a write-once release per phase. The release binds:
  - the engine binary, observer binary and native K1 digests;
  - every engine, support and observer source digest, the hecs_r2 library tree digest (built copy and repository copy), and the registries file;
  - this protocol's digest;
  - the supervisor source and binary digests;
  - the DESIGN record digest;
  - the reconstruction qualification digest;
  - the registry;
  - the C2 constants.
- The QUAL release is refused unless the committed DEV disposition reads `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE`.

**Per-seed runs.**
- Seeds run strictly in index order.
- Before a seed starts, the supervisor re-admits the full predecessor START/COMMIT chain and refuses any existing slot.
- Each slot holds:
  - the START record;
  - the five-file set;
  - the child and observer streams;
  - the COMMIT record, binding every digest, followed by an atomic rename.
- A failure writes `FAULT.txt` with `REPLAY_PERMITTED=NO`.

**Closing a phase.** After the last seed, the R3-BP cohort calculation runs once over plain copies of the committed sets. The supervisor then writes a write-once disposition record.

**Prohibitions.**
- No retry, replay, seed replacement, rescue, retuning, threshold change or gate reordering.
- No parameter change after a DEV seed starts.
- No QUAL rerun after a QUAL seed starts.
- A failed DEV closes C2; any successor candidate needs a new fresh DEV registry.
- A failed QUAL closes C2 as failed QUAL; any future lineage needs a new held-out QUAL registry.

## Build and identity

Toolchain: rustc 1.97.0 (2d8144b78 2026-07-07); cargo 1.97.0; gcc 13.3.0; cmake 3.28.3.

- **Engine.**
  - Layout: hecs_r2 crate (`research/hecs_r2/rust/src`, library tree digest `f030ed4978567f6520443748463ea7fec4deb39f20321499c996822aea021d90`, plus the frozen `Cargo.toml`/`Cargo.lock`).
  - Sources: engine `r3_bn_c2_dev_engine.rs` `5345d0acc4c37cdff832f08a418bde475a0f30d637404e65ed56fdcea36871f0`; support `r3_c2_terminal.rs` `344879473c17a10193b1b0422b9eea68e33bc66fb98dea4cb1beb9902e3eb86f`, `r3_training_core.rs` `51488b07…`, `r3_bd_emitter.rs` `35bdfb02…`.
  - Build command: `cd /mnt/primesauce/Elpis_DEV/build/hecs_r3_successor_c2_engine_r0/native_source && CARGO_TARGET_DIR=/mnt/primesauce/Elpis_DEV/build/hecs_r3_successor_c2_engine_r0/cargo_target cargo build --offline --locked --release --bin r3_bn_c2_dev_engine`. The binary is copied to `build/hecs_r3_successor_c2_engine_r0/r3_bn_c2_dev_engine`.
  - Release-profile numerics are bit-identical to the frozen debug engines: the full C0 engine source built in release reproduces all five committed raw files of burned seed 0. The C2 engine's HD/L2 ROLLOUT rows equal the DESIGN harness rows bit-for-bit on all 32 full-hierarchy DESIGN worlds.
- **Observer.**
  - Source: `r3_c2_observer.rs` `62282dad588c82e6475c8e8d08070cd51b123f102c64cb44d1866aa721308e5b`.
  - Build command: `cd /mnt/primesauce/Elpis_DEV/build/hecs_r3_successor_c2_engine_r0 && rustc --edition 2021 -O -o r3_c2_observer r3_c2_observer.rs`. The relative source path matters, because the binary embeds it.
  - Cohort registries are bound to arms: the legacy successor registry admits only C0/C1, and C2_DEV/C2_QUAL admit only C2.
- **Supervisor.** `cd /mnt/primesauce/Elpis_DEV/build/hecs_r3_successor_c2_supervisor_r0 && rustc --edition 2021 -O -o r3_c2_supervisor r3_c2_supervisor.rs`. The supervisor pins this protocol's digest, so its own digests are bound by the release and not listed here.
- **K1:** the qualified reconstruction K1 (`8d8b7ffc…`), qualified by exact raw-output identity of the successor pairs 0–2.

The project root is the historical `/mnt/primesauce/Elpis_DEV`, resolved by a private mount namespace onto the scratch reconstruction root (path mechanics only). Raw telemetry stays out of Git; compact records go to `research/hecs_r3/successor_c2/`.
