# LAO/Q4E5 — Opt-in synthetic acquisition reference

**Status:** `QUALIFIED_SYNTHETIC_ACQUISITION_ONLY` — not production or RALPH-qualified.

The independently evaluated Q4E5 `PROBE_EARLY` schedule passed its declared
sixteen-router-seed holdout gate with a fixed horizon of 8, exactly three
experts, endogenous observed-selected-expert-only feedback, and an externally
trained frozen 120-scalar CGC update controller.

- Terminal regret versus GREEDY: paired Δ = **-0.000152450435**;
  approximate 95% CI **[-0.000226330143, -0.000078570727]**; 14/16 wins;
  both cohorts favorable (−0.0001625903 / −0.0001423105).
- Online expert regret mean **0.056615294**, approximate 95% upper bound
  **0.057917808**, below the predefined bound **0.075**.
- Versus PREFIX2_DEADLINE8: no established terminal difference (mean
  Δ=+0.000004829) but online regret approximately **0.01784675 lower**.
- FLIP40-specific improvement **not established**; terminal Δ versus GREEDY
  **+0.0000565514**, CI upper +0.000245201.
- CGC-versus-FFN advantage in the Q4E5 cohort was small. Do not infer
  architecture-wide CGC superiority or RALPH-level performance.

The `cgc.lao_q4e5` package contains a strict, separately importable torch
reference that preserves the Q4E5 selector and episode-update equations.
The repository's historical `CGC120` is a **four-output float64** reference,
whereas Q4E5 uses a **three-output float32 [3,5,8]** operator. They must
not be silently interchanged. No frozen Q4E5 trained checkpoint is shipped.
`FrozenCGC3` requires externally obtained, validated weights.

Example:

```python
from cgc.lao_q4e5 import FrozenCGC3, ProbeEarlyEpisode

# frozen_w: verified float32 tensor [3,5,8] from offline training
# router_w: public float32 tensor [3,4] for this episode
controller = FrozenCGC3(frozen_w).eval()
episode = ProbeEarlyEpisode(router_w, controller, batch_size=32)
for t in range(8):
    expert_id = episode.select(context[t])     # context[t]: [32,4]
    reward = environment.observe_only(expert_id)  # selected binary reward [32]
    episode.observe(reward)
probs = episode.terminal_expert_probabilities(terminal_queries)
# STOP/terminal decision and economic calibration belong to caller.
```

**Strict boundaries:** fixed 8 observations; 3 experts; one selected binary
reward per acquisition; no privileged latent bias or unchosen outcomes;
no adaptive evidence HALT; no terminal STOP policy; no model-weight mutation;
reset episode state by constructing a new `ProbeEarlyEpisode`.

**Primary source and provenance:** Q4E5 qualification archive
`CGC_LAO_Q4E5_PROBE_EARLY_QUALIFIED_R0.zip` SHA-256
`8652b9d1dab3aef8a7b1fe0e5d4d94dfc1ec69f7d009395bba2bcb960c5c4692`.
Q4E4 source member SHA-256:
`56c13de6a4873722e1fb98f59068041156cd34136d958b1288be9cfabd3358aa`.
The original archival evidence stays immutable and external to the package.

**Provenance caveat:** the archived `FROZEN_SPEC.md` clock time conflicts
with the reported Hermes completion time. Archive entry ordering places
specification before holdout results, but the preregistration timestamp has
not been independently reconciled. Do not claim independent timestamp proof.

**Required next gates:** independently test FLIP40 reliability gating on
fresh seeds; calibrate terminal STOP; measure on-policy RALPH adaptation and
integration memory/latency; audit model-weight provenance. Do not promote
this opt-in reference to default inference based on Q4E5 alone.
