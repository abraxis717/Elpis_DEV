# Bounded causal learning problem: specification and hypotheses

## Task and observables

The synthetic LAO benchmark uses three experts (`K=3`), public contexts `x_t ∈ R^4`, frozen public router weights `W ∈ R^{3×4}`, and hidden per-expert reward biases `b ∈ R^3`. At each step a selector chooses *one* expert `a_t` and receives *only* that expert's binary reward `y_t ∈ {0,1}`. Unselected potential rewards, true biases, corruption labels/rates and future terminal-query outcomes are not available to the deployed update function. The current qualified acquisition experiment has **T=8** steps. Terminal expert/STOP utility is supplied by the benchmark harness and must not be silently redefined.

A representative clean reward probability is `p_t(a) = sigmoid((W x_t)_a + b_a)`. If the actual corruption mechanism is independent symmetric Bernoulli flipping with probability `q`, then `P(y_t=1 | p_t,q) = q+(1-2q)p_t`. **Treat this as a candidate analytic model, not a verified universal description of each simulator regime.** Under `q=0.4`, the difference in probabilities contracts by factor 0.2; at `p≈0.5` the Fisher information per independent observation about p is roughly 4% of the clean rate. With only eight observations shared across three experts, a mathematically plausible limitation is weak early identifiability. The experiments below test this; they do not prove an impossibility theorem.

## Existing CGC–LAO mechanical contract

CGC's **three-output** adaptation operator has 120 parameters: weights `θ ∈ R^{3×5×8}`. Its microscopic function is `φ(z)=(z+z²+z³)/2`. At deployment, `θ` stays **frozen**. Episode-local state `s_t ∈ [-1.5,1.5]^3` is initialized at zero, contains no persistent memory, and is reset at episode completion. The selected action is chosen using predicted logits `W x_t+s_t` and a frozen `PROBE_EARLY` schedule. A selected-observation residual constructs five inputs per episode; the model proposes an update with cap 0.25, applied to s with clipping. The exact mechanically tested interface lives under `reference/src/cgc/lao_q4e5.py`.

`PROBE_EARLY` uses greedy selection at the first step, forces the predicted best **unseen** expert at steps 1 and 2 if available, and ensures coverage by deadline whenever unseen experts equal remaining steps. Otherwise it is greedy. This is a **fixed acquisition policy**, not learned active querying and not a HALT policy.

The public reference's output is an array of terminal expert probabilities; downstream STOP choice, query distribution, costs, confidence intervals and other evaluation mechanics belong to the original experimental harness, not the reference module.

## Required invariants for new algorithms

- No inference-time CGC/FFN weight gradients or mutation. Allow only explicitly bounded **volatile episode state**. If a candidate adapts fast weights at deployment, it is a different, presently inadmissible contract.
- Bounded state footprint (target: tens of scalars, not the 918-float diagnostic posterior); bounded updates, bounded episode length and explicit reset. An exact memory budget and runtime must be declared before testing.
- No autonomous persistence, NVMe writes, user-visible artifacts, unbounded logs, network retrieval or checkpoint accumulation from the inference engine.
- Strictly causal observations and selection. If a controller changes its future selected expert, use the simulator's correctly indexed potential outcomes to produce new selected-only feedback; do not replay an incompatible logged reward.
- No hidden `b`, `q`, regime identity, unselected potential reward, future query label or oracle decision in the deployed policy. Privileged variants must be labeled **inadmissible references**.
- Do not conflate corruption classification with decision quality. Calibrated posterior mass with poor discrimination is not sufficient; an oracle with superior AUC need not imply that an admissible regret-minimizer is deployable.

## Three concrete research hypotheses

**H1: observational sufficiency.** Under T=8, unknown q is insufficiently identified to choose update attenuation reliably. Challenge this with an admissible analytical filter and matched-log regret; estimate a minimal-observation curve, not merely AUC. Test q in/out of the hypothesized support and misspecified mode–q coupling.

**H2: utility-aware small state.** A bounded filter tracking only per-expert counts, prediction residuals and a calibrated global reliability belief may achieve a better clean/FLIP40 frontier than a 120-weight meta-optimizer. This is an open hypothesis. Compare against NOOP, pinned CGC, static attenuation, a matched Beta-Bernoulli estimator, and a privileged known-q oracle. Do not use oracle q to tune the production algorithm.

**H3: acquisition matters more than update complexity.** If the posterior receives too little independent feedback, a bounded information-directed selector or controlled paired probe may improve expected terminal decision quality relative to the frozen scheduler. Test in a **separate, explicitly unqualified diagnostic branch** before any policy substitution; keep Q4E5's qualified acquisition and the reserved Q4E14 holdout unchanged.

## Falsifiable objective

Seek an algorithm `A` that minimizes empirical **terminal decision regret** under corrupted feedback while satisfying noninferiority on CLEAN and ALL and a conservative online-regret budget. Its quality must be reported as paired differences against pinned CGC **and NOOP**, with independent router-seed confidence intervals and inference-state/time costs. Surrogate BCE, MSE, posterior classification and differentiable soft regret may be useful training metrics but **cannot replace actual discrete terminal regret**.

A useful initial method is an **amortized analytical reliability estimate** (unknown-q) coupled to a finite-state action-value calculation, with the operator's proposal optionally given a bounded weight `λ∈[0,1]`. Include a frozen no-update baseline and sample-count-matched controls. Do not claim a valid confidence sequence, Bayesian optimality or certified robustness unless its assumptions and proof hold.
