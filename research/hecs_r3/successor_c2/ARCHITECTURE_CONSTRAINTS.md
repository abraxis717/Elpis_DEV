# H-ECS: binding architectural constraints and limitations (operator direction, 2026-10-10)

`RESEARCH_ONLY` · `NO_RUNTIME_READINESS_CLAIM`

## 1. LEARN is offline mechanics; QUERY is read-only

The C2 training law and every matched training law derived from it are **TRAIN/LEARN mechanics, never QUERY mechanics**. A scientific seed rebuilds its models from G1:
- 96,000 native G1 steps;
- 4096 C2 optimizer updates;
- full model and hierarchy reconstruction.

Such a seed is an independent qualification replicate. It is never the work required for one Elpis turn.

Required runtime model:

```
TRAIN / LEARN
    ↓
persistent qualified state for each hierarchy level
    ↓
snapshot/materialize through the existing retained-state/FMS authority
    ↓
QUERY
    ↓
forward / recurrent / planning computation only
    ↓
readout
```

Ordinary QUERY must not perform G1 training, C2 optimizer updates, full model reconstruction, or hierarchy retraining. An architecture that needs any of these per query or context turn is a runtime-design failure; its latency is not to be accepted. QUERY and LEARN stay semantically distinct.

## 2. Order of milestones

1. **Hierarchy science.** Run a matched constituent family: the same C2-derived law for every compared constituent, under DESIGN → fresh DEV → fresh QUAL. Then a frozen hierarchy comparison (H2A/H2B/H2C) that is attributable to hierarchy rather than to unequal training.
2. **Only if the hierarchy adjudication passes: H-ECS persistent-runtime qualification.** It must prove at least the following.
   1. Each hierarchy constituent is trained or materialized once.
   2. Its complete authoritative retained state is preserved.
   3. That state is snapshotted and restored through the existing K1/FMS-compatible authority.
   4. Repeated QUERY operations run with zero training updates.
   5. QUERY results are numerically identical before and after restore.
   6. QUERY latency is measured independently of training time.
   7. Bounded incremental LEARN is exercised explicitly and separately from QUERY.
   8. Learning one constituent does not silently reconstruct or retrain unrelated hierarchy levels.
   9. A restart does not require retraining from zero.
   10. The maximum bounded work of a live LEARN operation is defined and measured.

A passing hierarchy experiment is **not** a runtime-readiness claim.

## 3. Limitation of the C2 result (`../successor_c2/qual_q2/`, `WORLD_MODEL_VALID`)

C2's true-target terminal anchor uses true future latent states during training. C2 therefore establishes a **learning law from observed trajectories**. It is not an instant, query-time method for creating a new world model from arbitrary novel context.

## Desired endpoint

```
qualified learning law
        +
qualified hierarchy
        +
persistent materialized state
        +
cheap read-only QUERY
        +
bounded explicit LEARN
```

The endpoint is not a training run on every turn.
