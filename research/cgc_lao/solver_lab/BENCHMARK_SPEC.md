# SLB1: selected-feedback learning benchmark (frozen before experiments)

Machine-readable form: [`specs/slb1.spec.json`](specs/slb1.spec.json). The test suite checks that the
simulator's constants equal that file. This benchmark is an **independent illustrative benchmark**. It is
not the Q4E5/Q4E14/Q4F simulator, whose source, utility and normalisation were not published. Its numbers
must not be compared with historical qualification measurements.

## World

| Quantity | Definition | Visible to the learner |
|---|---|---|
| Experts | `K = 3` | yes |
| Router `W ∈ R^{3×4}` | `0.65·N(0,1)` per entry, one draw per router seed, rounded to float32 | yes (public, frozen) |
| Context `x_t ∈ R^4` | `N(0, I_4)`, float32-representable | yes |
| Hidden bias `b ∈ R^3` | `N(0, 1)` per expert, one draw per episode | **no** |
| Clean reward probability | `p_t(a) = σ(W_a·x_t + b_a)` | **no** |
| Clean event | `C[t,a] = 1[U^r[t,a] < p_t(a)]` | **no** |
| Reading `r` of an event | `Y[t,a,r] = C[t,a] XOR F[t,a,r]` | only `Y[t,a_t,r]` for the selected `a_t` |

Every potential outcome `C[t,a]`, `F[t,a,r]` and `Y[t,a,r]` is generated for all `t < 16`, all experts and
two readings **before** any policy runs, from counter-based streams. A policy that selects `a_t` reads
`Y[t, a_t, 0..R_t-1]`, so any two controllers are compared on exactly indexed potential outcomes and never
on a replayed incompatible log.

## Corruption mechanisms

| Mechanism | `F[t,a,r]` |
|---|---|
| `symmetric(q)` | `1[U^f[t,a,r] < q]`, independent per reading |
| `qunif(qmax)` | per episode `q ~ U(0, qmax)`, then `symmetric(q)` |
| `asym10(q)` | `C[t,a] · 1[U^f[t,a,r] < q]` (successes reported as failures only) |
| `burst(enter, exit, f)` | hidden per-episode Markov state `c_t` (stationary start, `P(0→1)=enter`, `P(1→0)=exit`); `c_t · 1[U^f < f]` |
| `context(q)` | `symmetric(q)` when `x_t[0] > 0`, clean otherwise |
| `event(q)` | `1[U^f[t,a,0] < q]` for **every** reading (all readings of an event share the corruption) |

Design regimes (the deployment distribution a candidate may be designed for): `CLEAN` (q=0), `FLIP20`
(q=0.2), `FLIP40` (q=0.4, the **primary corrupted regime**). `ALL` is the equal-weight mixture of the three
(per router seed, the mean of the three regime means). Shift regimes (`QUNIF`, `FLIP30`, `FLIP45`, `BURST`,
`ASYM40`, `CTXQ40`, `BIASWIDE` (bias scale 1.6), `BIASWIDE40`, `ROUTERWIDE40` (router scale 1.0),
`EVENT40`) are never used to design or select a candidate; they are run on VALIDATION and HOLDOUT only.
`QUNIF` and `BURST` are the designated primary shift tests.

## Decisions and utility (fixed for the whole study)

After acquisition the learner receives 16 fresh query contexts `x'_j` and chooses, for each, one expert or
`STOP`. Choosing expert `a` is worth its clean success probability `p(a|x'_j)`; `STOP` is worth the fixed
payoff **`s = 0.6`** (a known safe fallback that is better than a coin flip). The corruption acts on the
feedback channel only: the terminal utility is the clean one.

- **Terminal regret** of an episode: `(1/16) Σ_j [max(max_a p(a|x'_j), s) − U(d_j)]`, in probability units.
- **Online regret** of an episode: `(1/T) Σ_t [max_a p_t(a) − p_t(a_t)]` (acquisition steps only; STOP is
  not available during acquisition).
- Both use expected (clean) utilities, not sampled rewards, to remove irrelevant outcome noise.
- **Acquisition cost:** no explicit per-observation price in any contract. Contracts are compared by
  reporting terminal regret, online regret, expert queries and feedback reads side by side.

## Observation contracts

| Contract | Horizon | Readings per acquisition | Status |
|---|---|---|---|
| `ORIGINAL` | 8 | 1 | the original eight-step task |
| `DUP2` | 8 | 2 (independent readings of the selected event) | changed contract: +8 reads, +0 expert queries |
| `DUP2_4` | 8 | 2 at steps 0–3, then 1 | changed contract: +4 reads |
| `T12`, `T16` | 12, 16 | 1 | changed contract: +4 / +8 expert queries |

All contracts read the same pre-generated potential outcomes, so a contract comparison is paired episode by
episode. Results for changed contracts are reported separately and never as a solution of `ORIGINAL`.

## Seeds, pairing and statistics

Every random array is a stream keyed by
`SHA-256("elpis/cgc_lao/solver_lab/v1/" + partition/router-seed/regime/stream labels)`; element `i` is the
splitmix64 finaliser of `key + (i+1)·0x9E3779B97F4A7C15 (mod 2^64)`, top 53 bits. Gaussians use
Box–Muller (cosine branch). An episode's data does not depend on how episodes are batched.

| Partition | Router seeds | Use |
|---|---:|---|
| `TRAIN` | 64 | fit baseline coefficients and gate scales |
| `DEV` | 256 (rounds of 16) | the research loop; every round uses fresh seeds |
| `VALIDATION` | 32 | confirm the selected candidates |
| `HOLDOUT` | 64 | opened **once**, for frozen candidates only |

The pairing unit is the router seed: for each seed, the mean regret over its episodes; a comparison is the
per-seed difference, summarised by its mean and a two-sided 95% Student-t interval with `n_seeds − 1`
degrees of freedom. Episode counts never enter the degrees of freedom.

## Deployment contract every learner obeys

- Inputs: the public router, the contexts, its own selected readings and the terminal queries. Never `b`,
  `q`, the regime, corruption labels, unselected outcomes or future queries (except `KNOWN_Q_ORACLE`, which
  is labelled privileged and is never a candidate).
- Frozen coefficients: no gradient, no weight mutation, no checkpoint, no file or network access during an
  episode.
- Bounded volatile state: a fixed number of scalars per episode (budget **64**), reset at episode start;
  fixed work per update and per decision; no growing history.
