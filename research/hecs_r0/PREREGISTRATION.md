# H-ECS R0: hierarchical ECS world models — preregistration

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

The machine-readable authority is `specs/hecs-r0.v1.spec.json`, rendered by `hecs spec`
and frozen in the same commit as this document. Every evidence file embeds it. Where this
text and the specification differ, the specification decides. Both are write-once: no
parameter, condition, measurement or gate below may change after any DEV or QUAL
observation exists.

## 1. Question

> Width scaling inside one ECS keeps one representation and one update regime. Does a
> hierarchy of canonical ECS states at distinct temporal scales, optionally with distinct
> learned representation spaces between levels, show behaviour that width scaling of one
> ECS does not show?

The conceptual reference is H-JEPA (arXiv:2610.06805): per-level latent spaces at coarser
strides, top-down subgoals, and the observation that higher levels discard fast detail
only when the data's timescales are separated. R0 borrows those principles, not the
architecture. **ECS is not a JEPA**: there is no learned neural encoder trained end to
end, no SIGReg, no transformer predictor. A positive result would not show that ECS
learns representations the way H-JEPA does.

## 2. What is fixed (and not touched)

- **The ECS primitive.** Each level owns exactly one native K1 state `(W, epoch, H, a)`
  created and driven through the existing C ABI (`native/ECS/include/elpis/ecsg_k1.h`):
  `create`, `forward` (QUERY), `learn` (LEARN) and `state_digest`. The qualified cubic
  equations are unchanged. Learning uses LEARN only, so H stays zero and every step is
  exactly G1. `S3` is never used as state.
- **One state per level.** No shared giant `W`. A level sees only its lower level's
  latents and actions, never another level's W, H or a, and never the ground truth.
- **Research only.** No canonical inference path, no ECS<->DSV codec, no HACF edge.

## 3. Design

### World (known ground truth)

- Slow factor `s` in [-1, 1], driven by the action: `s' = clip(s + 0.04 a + 0.005 n)`.
- Second factor `f`, never influenced by the action: AR(1),
  `f' = rho f + sqrt(1 - rho^2) 0.5 n`.
  - **SEPARATED** world: `rho = -0.7` (fast, high frequency; predictable one step
    ahead, not over a stride of 4).
  - **MATCHED** control: `rho = 0.995` (a timescale deliberately close to `s`).
    Fixed before freezing from generator statistics alone, on design seed 9999 with no
    ECS involved. The f/s spectral-centroid ratio was 2.12 at 0.97, 1.50 at 0.99 and 1.36
    at 0.995; SEPARATED's was 32.7.
- Observation: a world-specific rotation of `(s, f)` plus noise 0.01. Exploration data:
  16 training and 8 held-out episodes of 256 steps, actions uniform in [-1, 1] held for
  1 to 8 steps.

### Levels

- **Level 1 (every condition):** observation whitening; clock = environment step.
- **Level l > 1:** ticks every `stride = 4` lower steps. Its latent at tick `t` encodes
  the window of the last `window` lower latents ending at lower step `4 t` (past only).
  Its action is the mean of the lower actions until the next tick.
- **ECS input** (`d = 6`, the qualified dimension): `[z0, z1, b, sel0, sel1, 1]`, target
  `z'_k`. The one-hot selector lets one scalar ECS predict both coordinates.
- **Training:** a fixed budget of K1 steps on shuffled mini-batches of 128 transitions,
  25 steps per call. Rate `0.002 x 36 / width` (the reference rate at width 36).
  Initial `W ~ N(0, 0.18^2)`.
- **Bounded controller:** per level, a ring of `window` lower latents and a stride
  counter. Its resident size is a constant of the specification, and it stores no
  trajectory.

### Conditions

Total ECS width is matched: FLAT_N72 matches the two-level hierarchies (2 x 36), and
FLAT_N108 matches the three-level hierarchies (3 x 36).

| Condition | Levels (width) | Upper representation |
|---|---|---|
| FLAT_N36 / FLAT_N72 / FLAT_N108 | 1 (36 / 72 / 108) | none: the **width baseline** |
| TEMPORAL_SHARED_L2 / _L3 | 2 / 3 (36 each), stride 4, window 1 | none: the upper latent is the lower latent at the tick (one shared space; the temporal-hierarchy control) |
| HIERARCHICAL_DISTINCT_L2 / _L3 | 2 / 3 (36 each), stride 4, window 2 | a separately learned affine map, window (4) -> 2 |

**Distinct encoder.**
- Fit a linear one-tick model of the next window from (window, pooled action).
- Let R be its residual covariance and S the window covariance.
- Keep the two directions minimising `p^T R p` subject to `p^T S p = 1` (closed form,
  deterministic). These are the most predictable unit-variance components.
- This is a linear predictability proxy trained before the ECS, not end-to-end training
  through it.

### Measurements (every level of every condition)

- Prediction NMSE at horizons 1, 2, 4 and 8 level steps (held out, open loop, true level
  actions), and the fraction of diverged rollouts.
- Slow-factor and fast-factor recoverability: NMSE of a ridge linear probe from the level
  latent to `s` and `f` at the window-end time (fit on training, scored on held out).
- State trajectory: K1 epoch, training MSE and `||W||_F` at 0, 25, 50, 75 and 100% of the
  budget, plus the final K1 state digest.
- Attractors: free run at `b = 0` from a 5 x 5 grid on [-2, 2]^2 for 200 steps; reported
  as fixed points, occupancy (fixed, diverged, other) and steps to settle.
- Compute: training multiply-adds, and planning native forward rows and multiply-adds.
- Memory: K1 image and workspace bytes, encoder parameters, controller resident bytes.

### Planning (24 paired episodes per world, start and goal shared by every condition)

- **Task.** Reach the goal slow value within 64 steps (success: `|s - s_goal| <= 0.1`).
  The start is in [-0.9, 0.9] and the goal at least 0.8 away. The goal is given as an
  observation rendered from `(s_goal, f_goal)`, with `f_goal` drawn from the second
  factor's law, so a planner that matches every latent coordinate also matches an
  uncontrollable one.
- **Deterministic random shooting** over the levels' own ECS models.
  - FLAT plans 16 steps (actions held in blocks of 4).
  - Hierarchies plan top-down. The top level plans 4 of its steps toward the goal latent
    in its own space. Each lower level plans one upper tick to its subgoal, scoring the
    encoded predicted window.
  - Every condition executes 4 environment steps per replan.
  - Before each replan, every level starts from the encoding of its most recent window,
    as H-JEPA encodes the current observation at every level. At its own tick that is
    exactly the level's tick latent.
- **Compute-matched budgets.** Native forward rows per replan are about
  {256, 1024, 4096} for every condition:
  - FLAT: K {8, 32, 128};
  - two levels: K {16, 64, 256};
  - three levels: K {11, 43, 171} (within 4%).
- **Subgoal tracking:** the normalized squared distance between the level-2 latent
  reached after a replan's actions and the subgoal level 1 was steered to.
- **Oracle:** the flat planner over the true slow dynamics (validity only).

## 4. Phases

1. **Mechanics.** The fast tests (`cargo test`) check structure only: spec validity,
   width and compute matching, streaming controller = offline lift, level-state
   independence, exact row counting and determinism. Before freezing, one wall-time
   probe (seed 9999, outside DEV and QUAL) measured training time only; no outcome was
   read.
2. **DEV calibration (`hecs calibrate`).**
   - Grid of per-level training budgets: {6000, 24000, 96000} K1 steps.
   - Choose the smallest budget for which FLAT_N36's level-1 one-step NMSE (no divergence)
     is at most 0.25 in every DEV world (seeds 0-3, both worlds).
   - That criterion is the only one consulted. If none qualifies: `TASK_INVALID_ON_DEV`,
     stop.
3. **DEV run (`hecs run --phase dev`)** at the chosen budget.
   - If the validity gates fail: `TASK_INVALID_ON_DEV`, stop.
   - DEV hypothesis values are reported as non-binding and change nothing.
4. **QUAL run (`hecs run --phase qual`), once,** on fresh seeds 100-107 at the DEV
   budget. Its disposition is final.

## 5. Gates (QUAL decides; means over seeds unless stated)

**Validity.** All of the following must hold, or the result is `TASK_INVALID`.

| Gate | Requirement |
|---|---|
| V1 | FLAT_N36 level-1 one-step NMSE <= 0.25 with no divergence, in every world instance |
| V2 | Oracle success >= 0.8 in each world |
| V3 | Level-1 probe NMSE <= 0.1 for both `s` and `f`, every instance |
| V4 | Spectral-centroid ratio `f/s` >= 4 (SEPARATED) and <= 1.5 (MATCHED) |
| V5 | No K1 refusal in any training |

**H1, selective abstraction (HIERARCHICAL_DISTINCT_L2).** The fast loss is
`dF = NMSE_f(L2) - NMSE_f(L1)`.
- **H1:** in SEPARATED, mean `dF >= 0.3`, `NMSE_s(L2) <= 0.2`, and `dF >= 0.3` in at least
  75% of seeds.
- **H1C (data dependence):** mean `dF` in MATCHED <= mean `dF` in SEPARATED - 0.2.
- **H1T (temporal control):** TEMPORAL_SHARED_L2 mean `dF` in SEPARATED < 0.1.

**H2, planning (SEPARATED; success means over seeds at each compute point).**
- **H2A, beyond width:** HIERARCHICAL_DISTINCT_L2 - FLAT_N72 >= 0.15 at >= 2 of 3 compute
  points.
- **H2B, beyond temporal decomposition:** HIERARCHICAL_DISTINCT_L2 - TEMPORAL_SHARED_L2
  >= 0.10 at >= 2 of 3 compute points.

**Reported, not gated:**
- width: success of FLAT_N36 / N72 / N108;
- depth: L1 -> L2 -> L3 per representation at the middle compute point, classed
  IMPROVES (>= +0.05), NO_CHANGE or DEGRADES (<= -0.05);
- every measurement in the MATCHED world.

**Disposition.**

| Disposition | Condition |
|---|---|
| `OUTCOME_A` | H1, H1C, H1T and H2A, H2B all hold |
| `OUTCOME_B` | the H1 family holds, H2 does not |
| `OUTCOME_C` | H2 holds, the H1 family does not |
| `OUTCOME_D` | neither holds |
| `TASK_INVALID` | a validity gate fails |

No disposition promotes H-ECS into the canonical runtime. R0 makes no cognitive,
language or product claim.

## 6. What would not rescue a failure

No gate, threshold, world parameter, budget, seed or condition changes after DEV or QUAL
data exist. The calibration grid is the only DEV-time choice, and its rule is fixed above.
A failed hypothesis is reported as failed.
