# Elpis Torch R13 P5 — preflight and DEVELOPMENT-ONLY feasibility pilot

**DEVELOPMENT ONLY. This report decides nothing.**
- No number here is evidence for or against any reading of `PREREGISTRATION_R13_P5.md`.
- None of it is pooled with confirmation data.
- No confirmation run has started; one needs the owner's further authorization.

**What ran.**
- `run_r13_p5 preflight`, then `run_r13_p5 pilot`.
- Both ran on the clean freeze commit `f57d252`, under the frozen plan `094adb45…cd432`.
- Records: `research/elpis_torch/results/r13_p5/preflight.json` and `dev_pilot.json`. `MANIFEST.json` pins both,
  the plan, and every source by name.
- `tests/research/elpis_torch/test_r13_p5_records.py` checks them, reading JSON only.

**The pilot's scope** (owner decision, compute):
- one development seed, 9001;
- at most 200 updates per trainable arm, batch 1, 2 CPU threads;
- the full backbone budget;
- 60 episodes per primary cell and 40 per in-distribution hop count.

**Source pins.**

| What | Pin |
|---|---|
| Geodome | HexaFlake `4e5fe56670c61a0d8906770b03033658c3087c3a`, Reference API v1.0.0, vendored byte for byte |
| R13 corrected sources and supplemental receipts | `fffe0cc`; receipts in `c565097` (`results/r13_hardening/MANIFEST.json`, SHA-256 `282657f8…`) |
| P5 harness | `e817c77` |
| frozen plan | `results/r13_p5/preregistration.json`, SHA-256 `094adb4500857103bbac6f7d215e1873c4167d2dbfb86daa1844a05999dcd432`; frozen in `f57d252` |
| records | produced on `f57d252` (clean), committed in `bf9018f` |
| environment | CPU, PyTorch 2.15.0+cpu, Python 3.13, 2 threads |

## 1. Preflight (`preflight.json`; before the pilot)

| Check | Value |
|---|---|
| plan frozen (the file equals the runner's constants) | yes |
| RAM | 15.7 GiB total; 14.7 GiB available |
| swap | none |
| scratch free | 30.7 GB |
| CPUs / threads | 4 / 2 |
| measured probes | backbone step 0.57 s; observing one episode 0.061 s; arm training step 16 ms; arm evaluation 5 ms |
| projected pilot | 0.21 h; peak RSS 0.69 GB |
| projected confirmation | 5 seeds × 19.4 min = 1.6 h; peak RSS 0.82 GB (view cache 156 MB for 2,900 episodes per seed) |
| limit | 5 GiB (no operation above it starts) |
| verdict | **go** |

## 2. The frozen backbone (seed 9001)

**The run.** R5's recipe, 1200 steps of 8 episodes in 420 s. It has 441,549 parameters, and its digest
`167742b3…` was unchanged after every arm had trained.

| Step | Answer CE | Coarse source loss | Fine evidence loss | Total |
|---|---|---|---|---|
| 0 | 4.17 | 2.81 | 2.83 | 10.02 |
| 100 | 3.91 | 2.13 | 2.07 | 8.37 |
| 600 | 3.87 | 2.11 | 2.07 | 8.09 |
| 1199 | 3.82 | 2.12 | 2.11 | 8.09 |

**What this means.**
- **The answer stays at chance.** Its cross-entropy stays near ln 48 = 3.87 from step 100 on: 3.82 to 4.08,
  per batch of 8.
- **Both evidence losses settle near ln 8 = 2.08** (2.07 to 2.14). The backbone learned which blocks are FACT
  blocks. It did not
  learn which of the eight facts has the query's start entity as its subject: uniform weight over the 8 facts
  gives exactly ln 8.
- **NATIVE is therefore at chance.** 6 of 240 primary episodes (2.5%) and 1 of 120 in-distribution episodes
  (0.8%), against 1/48 = 2.1%.

## 3. Retrieval coverage (the deployable evidence)

The deployable evidence is the backbone's own top 8 fine-ranked live blocks at the query, captured from the same
forward.

| Set | Complete-set recall | By hops | Fact blocks among the 8 | Hop-1 fact ranked first |
|---|---|---|---|---|
| primary (240) | **1.000** | 2 hops 1.000; 3 hops 1.000 | 0.9995 | 0.117 |
| in-distribution (120) | 0.992 | 1: 1.000; 2: 1.000; 3: 0.975 | 0.9990 | 0.125 |
| training episodes (200) | 0.980 | 1: 0.985; 2: 1.000; 3: 0.955 | 0.9969 | 0.115 |

- **Coverage is complete on the primary set.** The pooled recall is 1.000 on this seed; the frozen threshold is
  0.90.
- **The ranking within the facts is at chance.** The hop-1 fact is ranked first in 11.7% of primary episodes, the
  same as 1/8. Binding k receives an effectively random fact.
- **The oracle adds little here.** With complete recall, DIAGNOSTIC_ORACLE receives the same 8 blocks as the
  deployable arms, only in a seeded order instead of rank order. The oracle diagnostic separates little unless
  recall drops.

## 4. Training behavior (200 updates per arm)

**The zero gate.** Every arm starts with `g = 0` exactly.
- **At step 0 only the gate receives a gradient.** The ingress, message, update, readout and injection gradients
  are exactly 0.0, as expected with a closed gate.
- **From step 1 every group receives a nonzero gradient.** By then Adam's first step has moved the gate to ±0.001.
  EQUAL_BUDGET's groups are ingress, `DirectReader`, readout and injection; it has no message or update kernels.

The initial gate-only gradient is therefore the closed gate, not a learning failure.

| Arm | Trainable parameters | Gate after step 0 | Max \|gate\| over the log | Final gate | Loss, first fifth → last fifth |
|---|---|---|---|---|---|
| GEODOME | 16,449 | −0.0010 | 0.0086 | −0.0081 | 3.886 → 3.868 |
| EQUAL_BUDGET | 16,457 | +0.0010 | 0.0055 | +0.0040 | 3.886 → 3.868 |
| COMPUTE_MATCHED | 16,449 | +0.0010 | 0.0127 | −0.0035 | 3.886 → 3.868 |
| FLAT_RECURRENT | 16,449 | +0.0010 | 0.0111 | +0.0015 | 3.886 → 3.868 |
| RELABEL_ISOMORPHIC | 16,449 | −0.0010 | 0.0086 | −0.0082 | 3.886 → 3.868 |
| REWIRED | 16,449 | −0.0010 | 0.0072 | −0.0058 | 3.886 → 3.868 |
| DIAGNOSTIC_ORACLE | 16,449 | −0.0010 | 0.0057 | −0.0045 | 3.886 → 3.868 |

A logged gate is the value at that step, before its own update. Step 1 therefore shows the gate after step 0's
update.

**Gradient norms at the log steps** (GEODOME; the other arms are of the same order):

| Step | Loss | Gate | Gate grad | Ingress | Message | Update | Readout | Injection |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.850 | 0 | 3.6e-3 | 0 | 0 | 0 | 0 | 0 |
| 1 | 3.938 | −0.0010 | 1.5e-3 | 6.2e-6 | 2.6e-5 | 2.1e-5 | 2.5e-5 | 3.0e-5 |
| 10 | 3.963 | −0.0012 | 3.3e-3 | 1.6e-5 | 1.7e-4 | 1.2e-4 | 3.3e-5 | 3.6e-5 |
| 100 | 3.697 | +0.0036 | 5.9e-3 | 1.4e-5 | 1.2e-4 | 9.1e-5 | 1.6e-4 | 1.5e-4 |
| 199 | 3.780 | −0.0086 | 2.0e-2 | 3.0e-4 | 2.1e-4 | 2.4e-4 | 5.1e-4 | 4.7e-4 |

**No arm learned anything in 200 updates.**
- **The gate stays small and wanders.** It stays within ±0.013 and changes sign (GEODOME: −0.001, +0.006,
  −0.009), so no consistent direction has formed.
- **The losses are the backbone's.** The first-fifth and last-fifth losses are identical for every arm, because
  they are NATIVE's losses on the same episodes: the injected change is too small to move them.
- **The predictions are NATIVE's.** Every arm's per-episode predictions equal NATIVE's in every primary,
  in-distribution and held-out-region episode. The primary NLL differs from NATIVE's by at most 5e-6.

## 5. Development accuracies and interventions (not a test)

| | Primary (240) | In-distribution (120) | Held-out region (240) |
|---|---|---|---|
| NATIVE | 0.025 | 0.008 | — |
| every trained arm | 0.025 | 0.008 | 0.025 |

**The primary cells for NATIVE:**

| Cell | Correct |
|---|---|
| held-out combination, 2 hops | 0/60 |
| held-out combination, 3 hops | 2/60 |
| held-out structure, 2 hops | 2/60 |
| held-out structure, 3 hops | 2/60 |

**Every GEODOME intervention leaves the predictions exactly at NATIVE's:** evidence removal, swapped evidence,
depth 4 and depth 2, interrupted communication, intermediate-state erasure and port permutation. There is nothing
yet for them to remove. The disabled injection is bit-exact NATIVE in all 240 primary episodes, as the protocol
requires.

## 6. Regions and controls (seed 9001; owner decision B: plan)

**The regions.** Four regions were drawn: training regions 0–2 and held-out region 3, with centers 106490,
127476, 79929 and 37703.
- Each has 48 vertices, 120 edges (240 arcs), is connected, and has no degree-5 vertex.
- Each has the same induced degree histogram (6 of degree 3, 15 of degree 4, 27 of degree 6), diameter 7 and
  readout eccentricity 4.
- Every evidence vertex is at distance 2 from the readout.
- Pairwise evidence distances: min 1, max 4, mean 2.79–2.86.
- The port histograms are uniform: 40 per port, except region 1 at 34–44.

**The held-out region has the same shape statistics as the training regions.** Away from its twelve degree-5
vertices the geodesic lattice is locally uniform. The held-out-region test therefore checks transfer to another
position and another vertex identity, not to another shape.

**The controls, matched as closely as possible, with every mismatch recorded.**

| Control | Matched | Mismatches |
|---|---|---|
| COMPUTE_MATCHED (Havel-Hakimi + 10 \|E\| swaps) | vertices, edges, degree sequence (equal), connectivity, evidence-to-readout (all 2) | 15–24 of 120 edges shared with the region; evidence pairwise mean 2.54–2.75 |
| REWIRED (\|E\| / 2 swaps) | vertices, edges, degree histogram, connectivity, evidence-to-readout (all 2) | 40–48 of 120 edges kept; evidence pairwise mean 2.39–2.68 (max 3 in one region) |
| RELABEL_ISOMORPHIC | the region under a seeded permutation; arc order and ports identical | none in its inputs. Its training log equals GEODOME's to 3 significant digits through step 100; floating-point differences reach about 2% in the step-199 gradient norms |
| FLAT_RECURRENT | 8 evidence leaves, 8 rounds | by design: 9 vertices, 8 arcs, distance 1, port 0 everywhere |
| interrupted communication (GEODOME variant) | — | 36 arcs removed per region |

## 7. Runtime and resources (MEASURED; one CPU container, not a benchmark)

| Quantity | Value |
|---|---|
| pilot total | 486 s: backbone 420 s, plans 0.6 s, observing 560 episodes at 0.053 s each, arms 36 s |
| per arm (200 updates plus evaluation) | GEODOME 11.4 s (with 8 intervention passes); others 2.4–4.8 s |
| peak RSS (the whole process) | 664,080,384 bytes (0.62 GiB) |

**The projected confirmation envelope** scales the pilot's own times linearly by steps and episodes, with the same
backbone budget.

| Arm | Train, 2,000 steps | Evaluation | Within 60 min |
|---|---|---|---|
| GEODOME | 23.8 s | 22.7 s | yes |
| EQUAL_BUDGET | 11.1 s | 3.2 s | yes |
| COMPUTE_MATCHED | 22.7 s | 5.2 s | yes |
| FLAT_RECURRENT | 17.5 s | 4.5 s | yes |
| RELABEL_ISOMORPHIC | 24.0 s | 5.7 s | yes |
| REWIRED | 23.5 s | 6.0 s | yes |
| DIAGNOSTIC_ORACLE | 24.3 s | 5.8 s | yes |

- **Per seed:** 773 s, including the 420 s backbone (limit 60 min).
- **Five seeds:** 1.07 h.
- **Peak RSS:** the preflight projects 0.82 GB, against the 5 GiB limit. The pilot measured 0.66 GB with a view
  cache five times smaller (30 MB, against 156 MB).

The compute limits are far from binding. The binding constraint is the frozen step budget.

## 8. Findings that bear on the protocol (reported; nothing was changed)

These are the pilot findings the preregistration (§9) asks to be reported before any confirmation. Any change is
the owner's decision, frozen as a separate amendment before any confirmation data.

1. **There is no learning signal at the pilot's budget.**
   - After 200 updates every arm's predictions equal NATIVE's, and the gate is still within ±0.013.
   - The pilot cannot show whether 2,000 batch-1 updates are enough to learn the task.
   - If they are not, the frozen confirmation would end `NOT_ESTABLISHED` because of the training budget, not
     the substrate.
   - Compute would allow far more: 2,000 updates take about 25 s per arm.
2. **The native backbone is at chance**, and it cannot rank the facts.
   - The R5 recipe learned FACT versus non-FACT (both evidence losses at ln 8), but not the start-entity match.
     The answer stayed at ln 48.
   - Every trained arm therefore receives a complete but unordered fact set, and must learn the whole composition
     itself, including the hop-1 match.
   - The +5-point margin over NATIVE is measured from about 2%.
3. **Retrieval is not the limit on this seed.** Complete-set recall on the primary set is 1.000; the frozen
   threshold is 0.90. The oracle arm adds little separation while recall stays complete.
4. **The held-out region has the training regions' shape statistics** (§6). Its test is transfer across position
   and identity, not across shape.
5. **A provenance key collision in the frozen runner.**
   - `run_r13_p5.provenance()` writes the plan file's digest under `plan_sha256`. That shadows the digest of
     the source named `plan` (`geodome_cognitive/plan.py`).
   - The runner was not changed after the freeze. `plan.py` is pinned instead by each record's clean revision,
     and by name in `MANIFEST.json`.
   - `test_r13_p5_records.py` checks both. It compares every other source by name and treats `plan_sha256` as
     the plan file's digest.
   - Fixing the key would change the runner's digest, so it belongs in any amendment the owner freezes.

## 9. Reproduce

```bash
python -B -m research.elpis_torch.experiments.run_r13_p5 preflight --out research/elpis_torch/results/r13_p5/preflight.json
```

```bash
python -B -m research.elpis_torch.experiments.run_r13_p5 pilot --out research/elpis_torch/results/r13_p5/dev_pilot.json
```

```bash
python -B -m research.elpis_torch.experiments.run_r13_p5 manifest --out research/elpis_torch/results/r13_p5/MANIFEST.json
```

The pilot takes about 8 minutes on 2 CPU threads. Its results are deterministic given the seed; only the timings
and RSS vary. `run_r13_p5 confirm` refuses to run without `--authorization`.
