# Elpis Torch R13 P5 — preregistration: representation and distributed competence of the Geodome substrate

**Status: FROZEN.**
- **The plan.** `research/elpis_torch/results/r13_p5/preregistration.json`, SHA-256
  `094adb4500857103bbac6f7d215e1873c4167d2dbfb86daa1844a05999dcd432`, written by `run_r13_p5 plan` from the runner's
  own constants.
- **Drift is refused.** Every other command refuses a plan file that differs from them.
- **Every record carries** the plan digest, the digests of the sources that produced it, and the manifest digest of
  the R13 supplemental receipts.
- **Authority.** The owner's review of PR #83 (head `343dc92`) approved the task, the plan family, the size, the
  evidence sources, the oracle, the baselines, the statistical decisions, the compute envelope and the mandatory
  controls. This document resolves every remaining decision and threshold.
- **What may run now.** Only a DEVELOPMENT-ONLY feasibility pilot. It decides nothing. No confirmation run starts
  without the owner's further authorization: `run_r13_p5 confirm` refuses without `--authorization`.

**Base.**
- The R13 mechanics, re-qualified on the corrected sources: `results/r13_hardening`, `fffe0cc`.
- The P5 harness: `e817c77`.
- Elpis base `d73cdde`.

## 1. Question

Does a learned, bounded, distributed computation over admitted evidence, on Geodome's physical topology and
injected at the EvidenceLM final block, improve the final answer of multi-hop record composition? It must improve
it over:
- the native backbone;
- controls matched in parameters, compute, recurrence and graph statistics.

And does the gain survive the causal interventions that should remove it?

## 2. Task (owner decision B; `composition_tasks.py`)

**The episode.** 18 blocks of 16 tokens (288):
- a header block;
- 16 content blocks: 8 FACT blocks (`FACT x REL y`) and 8 NOTE blocks (three entity mentions), in a seeded order;
- the query block, which ends `QUERY s HOPk ASK`.

The query gives the start symbol and the hop count, never the answer. The answer is the next token.

**The vocabulary.** 10 special tokens and 48 entities (58 in all). The vocabulary is fixed, and every entity occurs
in training facts, so every embedding is trained.

**Invariants**, checked from the tokens by `solve` for every episode:
- every required edge sits in its own earlier FACT block;
- every entity has at most one outgoing edge, so ambiguous paths are rejected and there is no shortcut;
- the hard negatives are an overshoot edge out of the answer, converging edges into path entities, unrelated edges,
  and the NOTE blocks.

**The strata.** 1, 2 and 3 hops.

| Split | Use | Pairs on the path | Fact order |
|---|---|---|---|
| `train` | the arms' training episodes (indices 0..steps − 1, hop counts cycling 1, 2, 3) and the backbone's (indices from 1,000,000) | none held out | training structures |
| `iid` | in-distribution evaluation | none held out | training structures |
| `heldout_combination` | **primary** (2 and 3 hops) | at least one of the 451 held-out pairs (20% of ordered pairs, seed 1300; never in any training fact) | training structures |
| `heldout_structure` | **primary** (2 and 3 hops) | none held out | held out: 2 hops `(1,0)`; 3 hops `(1,2,0)`, `(2,0,1)` |

The training structures are `(0)`; `(0,1)`; and `(0,1,2)`, `(2,1,0)`, `(1,0,2)`, `(0,2,1)`.

## 3. The frozen backbone (one per seed, identical for every paired arm)

**Architecture.**
- EvidenceLM: d 64, 2 blocks, 4 ERSA heads (key and value 16, convolution 4, chunked kernel, chunk 72);
- CGC channel with routed experts (shared + 4, top 2) in both blocks; float32;
- evidence block 16, ring 512 tokens, coarse top 17 (every eligible block), fine top 8, read width 3, encoder 32,
  supersession off.

**Training.** It is trained once per seed with **R5's recipe** (`run_r5.dense_loss`):
- the answer's cross-entropy;
- plus 0.05 × the mean next-token cross-entropy of every position;
- plus R5's training-only `coarse_source_loss` and `fine_evidence_loss`.

The schedule: 1200 steps of 8 episodes, AdamW lr 3e-3 (0.9, 0.95), weight decay 0.01, 50 warmup steps, cosine to
10%, clip 1.0. The time limit is 60 minutes.

**The evidence label.** It is the hop-1 fact block, the only block whose subject is the query's start entity, at
the ASK position only. **No hop-2 or hop-3 fact is ever labelled**, so there is no intermediate-route label. The
labels exist only inside the backbone's training. No label, gold address or answer-bearing information reaches any
deployable arm's inputs, in training or in evaluation.

**Why this recipe, disclosed.** R6 states that label-free learning of addressing is not claimed. Two short
engineering probes before the freeze (300 to 400 steps on development seeds 1 and 2, not recorded as evidence, no
arm involved) agreed:
- **Label-free training.** It left the backbone's fine ranking at chance: fact blocks were about half of its
  top 8.
- **R5's recipe.** It put exactly the 8 FACT blocks in its top 8 in every split and stratum.

After the freeze the backbone is never trained again. Its parameter digest is recorded, and is checked unchanged
after every arm has trained.

## 4. The substrate, regions and evidence (owner decisions B: plan, size, evidence source, oracle)

**Size.**
- K = 48 vertices, R = 8 synchronous rounds, latent width 32, hidden width 64, 8 evidence bindings.
- The R13 caps are unchanged: capacity 64, 16 rounds, 1024 arcs, 16,384 visits, 8 evidence items.

**Regions.**
- **Per seed:** three training regions and one held-out region. Each is a breadth-first ball of 48 physical
  vertices around an independent uniform center, disjoint from the others, with every physical arc admitted.
- **The bindings are fixed by the region alone** (no episode, route or answer):
  - the query and the readout at the center;
  - the 8 evidence vertices taken evenly from the ring at distance 2.
- **Distance 2, because R = 8.** At distance 3 a 2-hop chain cannot complete in the worst case: center to fact 1,
  fact 1 to fact 2, fact 2 back to the center take 3 + up to 6 + 3 rounds.
- **Assignment.** Training and primary episode i uses training region i mod 3. The held-out-region test evaluates
  every primary episode on the held-out region.
- **Every topology statistic is recorded:**
  - induced degrees, degree-5 vertices, arcs and connectivity;
  - evidence-to-readout and pairwise evidence distances, eccentricity and diameter;
  - the port histogram.

**Evidence.**
- **Deployable.** The frozen backbone's own top-8 fine-ranked live blocks at the query.
  - They are captured from the same forward by `CognitiveSession(capture_candidates=8)`, which the R13 hardening
    receipt shows is bit-exact with no extra token.
  - They are admitted through the exact reread; binding k receives rank k.
  - Complete-set recall (every required fact among them) is measured, per stratum, before any end-to-end result is
    interpreted.
- **Oracle** (DIAGNOSTIC_ORACLE only, never a deployable result). The episode's 8 FACT blocks in a seeded order.

**Isolation** (mandatory controls C).
- Every episode is observed from a reset session (backbone and cognitive state).
- Every arm starts each episode from rows of zero.
- The cached observation of each episode carries no graph and is shared by every arm of the seed.

## 5. Arms and controls (owner decision B: baselines)

| Arm | What it is |
|---|---|
| NATIVE | the frozen backbone's own prediction (not trained) |
| GEODOME | the R13 kernels on the physical region, Geodome-validated plans, deployable evidence |
| EQUAL_BUDGET | the same ingress, readout and injection; `DirectReader` (one attention read of the evidence rows by the query row, an MLP of hidden 72) in place of the graph: 9,128 parameters for the kernels' 9,120 message, port and update parameters (trainable totals 16,457 against 16,449) |
| COMPUTE_MATCHED | a fully mixed random graph with the region's degree sequence (Havel-Hakimi, then 10 \|E\| degree-preserving swaps), simple and connected; bindings placed at distance 2 |
| FLAT_RECURRENT | a hub (query and readout) reading 8 evidence leaves for 8 rounds; port 0 everywhere |
| RELABEL_ISOMORPHIC | the region under a declared seeded vertex permutation; arc order and port ordinals kept, so the kernels' inputs are identical, permuted |
| REWIRED | the region after \|E\| / 2 successful degree-preserving double-edge swaps (simple, connected); bindings placed at distance 2 |
| DIAGNOSTIC_ORACLE | GEODOME with the oracle evidence |

**The controls.**
- **What is matched.** COMPUTE_MATCHED and REWIRED match the region's vertex count, arc count, degree distribution
  and connectivity. Their evidence-to-readout distance is matched as closely as possible, so their compute equals
  GEODOME's.
- **Labelling.** Control graphs are labelled as not Geodome topology and never pass Geodome authorization.
- **Mismatches.** Every mismatch is recorded: shared edges, pairwise distances, and anything else.
- **Port changes are tested separately.** `port_permutation` is an evaluation-only diagnostic on trained GEODOME.

**Training.** Every trainable arm:
- **Initialization:** the same seeded kernel initialization;
- **Data:** 2000 steps (the pilot: at most 200) of batch 1 on the same episodes and the same frozen backbone;
- **Optimizer:** AdamW lr 1e-3 (0.9, 0.999), no weight decay, clip 1.0;
- **Supervision: final answer only.** The cross-entropy of the refined logits at the query; no route or
  intermediate labels.

**The gate.** It starts at exactly 0. The training log records the gate and the gradient norm of every group
(ingress, message, update, readout, injection, gate) at steps 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 and 1999.
This shows the other kernels receiving gradients once the gate opens, so the initial gate-only gradient is not
mistaken for a learning failure.

## 6. Evaluation and interventions (mandatory controls C)

**The sets.**
- **Primary:** per seed, 150 episodes in each of held-out combination × {2, 3} and held-out structure × {2, 3}
  (600).
- **In-distribution:** 100 per hop count.
- **Held-out region:** the primary episodes on the held-out region.

**The metric.** Accuracy: the argmax of the refined logits (NATIVE: of the backbone's) is the answer.

**Interventions.** These are on the trained GEODOME arm, on the primary set:

| Intervention | Definition | Role |
|---|---|---|
| evidence removal | no evidence ingress | decisive |
| swapped evidence | the deployable evidence of the next primary episode | decisive |
| reduced depth | 4 rounds instead of 8 (2 rounds reported) | decisive |
| interrupted physical communication | every arc between the readout's 1-hop disc and the rest of the region removed | decisive |
| intermediate-state erasure | after 4 of 8 rounds every row but the query and evidence rows set to zero | reported |
| disabled cognitive injection | the backbone's own tail | decisive: must equal NATIVE bit for bit |
| port permutation | ports remapped by (1,2,3,4,5,0) | reported |

## 7. Statistics (owner decision B: statistical decisions; fully specified)

**The comparison.** The paired difference per seed and episode is correct(GEODOME) − correct(X), on the same
episodes with the same frozen backbone.
- The estimate is the mean over seeds of the per-seed mean.
- **Bootstrap** (hierarchical, seed-aware): 10,000 draws, generator seed 13013. Each draw takes 5 seeds with
  replacement, then each drawn seed's episodes with replacement. The statistic is the mean over the drawn seeds of
  their mean paired difference.
- **Bound.** One-sided lower confidence bound: the α/m quantile of that distribution, with α = 0.05 and m the size
  of the comparison's family (Bonferroni).

**Families.**
- **Distributed** (m = 3): NATIVE, EQUAL_BUDGET, FLAT_RECURRENT.
- **Geometry** (m = 2): COMPUTE_MATCHED, REWIRED.
- **Interventions** (m = 4): removal, swap, reduced depth, interrupted communication.
- **Substrate-agnostic:** each of COMPUTE_MATCHED and REWIRED against the distributed family (m = 3).

**Criteria.**
- **A comparison passes** when its estimate is at least +0.05, it is positive in at least 4 of 5 seeds, and its
  lower bound is above 0.
- **An intervention passes** when its drop has an estimate above 0, is positive in at least 4 of 5 seeds, and has a
  lower bound above 0.
- **Disabled injection** must be bit-exact NATIVE in every primary episode of every seed.
- **The relabel check** requires |estimate(GEODOME − RELABEL_ISOMORPHIC)| ≤ 0.02.
- **Recall threshold:** 0.90 pooled complete-set recall on the primary set.

**Readings** (evaluated in this order):
1. **`GEOMETRY_SPECIFIC_GAIN`**: GEODOME passes the distributed family, the decisive interventions and the relabel
   check, **and** the geometry family.
2. **`DISTRIBUTED_COMPETENCE_ESTABLISHED`**: GEODOME passes the distributed family, the decisive interventions and
   the relabel check; the geometry family does not pass, and no nongeometric graph control passes the distributed
   family.
3. **`SUBSTRATE_AGNOSTIC_GAIN`**: a nongeometric graph control passes the distributed family, and GEODOME does not
   pass the geometry family.
4. **`RETRIEVAL_LIMITED`**: none of the above, and the pooled complete-set recall is below 0.90. The oracle arm is
   then reported as computation conditional on complete evidence.
5. **`NOT_ESTABLISHED`**: every other case.

**Claims.** None of these readings is a cognitive-advantage claim. A geometric claim needs reading 1.

## 8. Compute (owner decision B: compute)

**The pilot (DEVELOPMENT ONLY).**
- Development seed 9001, at most 200 updates per trainable arm, the full backbone budget.
- Evaluation sets: 60 per primary cell and 40 per in-distribution hop count.
- It records retrieval recall, the gate and gradient log, loss movement, latency, RSS and the confirmation
  envelope projected from its own measurements.
- Its records are labelled `DEVELOPMENT ONLY` and are never pooled with confirmation.

**The confirmation (not authorized yet).**
- Seeds 1301 to 1305, paired; at most 2,000 steps per trainable arm, batch 1, 2 CPU threads.
- Hard limits: 60 minutes per arm and seed, and 60 minutes for each backbone.
- An exhausted run stops and is reported, never retried silently.

**RAM.**
- `preflight` checks RAM, swap, scratch and threads, and projects the pilot and the confirmation. It refuses any run
  projected above 5 GiB.
- Every run also stops if its peak RSS passes 5 GiB.
- No operation above 5 GiB starts without explicit authorization.

## 9. What would change this plan

Nothing, without the owner. A pilot finding that bears on the protocol, such as a retrieval or budget limitation, is
reported with the pilot. Any change would be a separately frozen amendment, before any confirmation data.
