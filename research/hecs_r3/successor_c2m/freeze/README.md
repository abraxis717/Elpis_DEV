# H-ECS R3 successor C2M — matched constituent family, pre-registered before DEV (2026-10-10)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_RUNTIME_READINESS_CLAIM` · `DEV_RELEASED_SCIENCE_UNSTARTED` · `NO_QUAL_RELEASE` · `NO_INTEGRATION`

This directory records the frozen matched family C2M and its DEV release. It was committed before any C2M DEV seed was executed.

**Why.** C2 (`../successor_c2/`) is `WORLD_MODEL_VALID`, but only its HD/L2 constituent was trained by the C2 law. A hierarchy comparison over its models would therefore confound hierarchy with unequal training. Under operator option 1, C2M applies the same C2-derived learning law to every constituent that H2 consumes:
- L1/N72 at h16, which serves FLAT;
- L1/N36 at h4, which serves TS and HD;
- TS/L2 at h4;
- HD/L2 at h4.

The family then runs its own DESIGN → fresh DEV → fresh QUAL. The R2 H2 comparison is adjudicated only on a `WORLD_MODEL_VALID` QUAL.

**Law.** C2M uses the C2 law (4096 updates, six C0 picks, R3 rollout objective with λ = 0.37, true-target terminal anchor, exact gradient, unchanged rate and clip) with terminal weight **w_T(h) = (4/h)^1.5**.
- At h4 this is exactly the qualified C2 weight, 1.0.
- At h16 (L1/N72) it is 0.125.

DESIGN showed that the unchanged weight of 1.0 fails at h16: 38/40, min M16 0.884, from clip-saturating terminal gradients. The exponent was chosen by a rule declared before the grid ran. It was confirmed at every binding on 48 held-out DESIGN worlds (every world passes; min bound M 0.960 at TS/L2, 0.992 at L1/N36 and L1/N72).

**Planning.** H2 planning is QUERY-only native forward computation over the materialized trained weights. It is ported verbatim from R2 and verified bitwise against `hecs_r2::experiment::run_seed`.

**Learning versus query.** The training inside a seed is an offline LEARN qualification replicate, never per-turn QUERY work (`../successor_c2/ARCHITECTURE_CONSTRAINTS.md`). A hierarchy adjudication is not a runtime-readiness claim.

**Contents.**
- `PROTOCOL_R3_C2M.md` — the frozen law: family, planning, registries, validity and H2 laws, execution, build.
- `C2M_REGISTRIES.json` — the fresh 8-seed DEV and 12-seed QUAL registries, with derivation and exclusions.
- `DESIGN_RECORD.json` — DESIGN evidence, the selection rule, held-out confirmation, full-engine checks and the supervisor rehearsal.
- `DEV_RELEASE.txt` — the write-once DEV release.
- `source/` — the C2M engine and the new planning module, the C2M observer, the C2M supervisor, and the counted-replacement generators that derive them from the frozen C2 sources. The unchanged support modules and the hecs_r2 library are at the C2 freeze and source-closure commits.

Raw telemetry is not in Git.
