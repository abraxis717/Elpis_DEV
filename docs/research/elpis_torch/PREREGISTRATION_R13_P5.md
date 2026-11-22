# Elpis Torch R13 P5 — protocol draft: representation and distributed competence of the Geodome substrate

**Status: DRAFT FOR OWNER REVIEW. NOT FROZEN. No data has been collected.**
- **The instruction.** The owner asked for the P0–P4 results before admitting any large competence study.
- **What it would take to freeze this.** The owner's approval. The protocol would then be frozen as a JSON plan
  whose SHA-256 every confirmation record carries, as in R9–R12, before any confirmation run.
- **Until then** `DISTRIBUTED_COMPETENCE` stays `HOLD`, and no competence code exists.
- **The base.** The R13 mechanics (`DESIGN_R13.md`, `EVIDENCE_R13.md`), all of whose gates pass.

## 1. Question

Does a learned, bounded, distributed computation over admitted evidence on Geodome's physical topology, injected at
the final block, improve the final answer on a task that requires composing several pieces of evidence? It must
improve it:
- **(a)** over the native backbone;
- **(b)** over controls that match its parameters, compute, recurrence and graph statistics without Geodome's
  geometry.

And does any gain survive the causal interventions that should remove it?

## 2. What would be held fixed

| Element | Proposal |
|---|---|
| Backbone | An EvidenceLM trained once with R6's `lm_training` on the task's streams, then FROZEN for every arm. One backbone per seed, shared by all arms of that seed |
| Seam | R13's, unchanged: the final block's `pre`, `P_query`, the latent, `g · P_out`, and the actual final channel over the chunk's rows. The gate is initialized at 0 |
| Substrate | Kernels as in R13 (width 32, hidden 64). Synchronous schedule, fixed rounds R. No halting |
| Plan family | ONE generic plan for every episode, with no task-specific routes: a BFS ball of K = 48 vertices around a fixed centre; all its physical arcs; the query at the centre; evidence item k at the k-th of 4 fixed vertices at hop distance 3; the readout at the centre; R = 8. The ball is chosen by a seed fixed before training. Its geometry is recorded |
| Evidence (deployable) | Addresses chosen by the backbone's OWN retrieval (R5 coarse index and fine reranker, inference mode): the top 4 candidates before the query's block. No labels |
| Evidence (diagnostic only) | The gold addresses, in an arm labelled `DIAGNOSTIC_ORACLE`. It is never part of a deployable claim |
| Supervision | **Primary: final-answer-only** cross-entropy at the query position. No intermediate, route or node supervision in the primary arms |
| Optimizer | AdamW, lr 1e-3, 2000 steps, batch one episode at a time, no backbone gradient. The same for every arm |

## 3. Task (proposal)

**Multi-hop record composition**, generated synthetically in R5's record format, with hard negatives.
- A stream holds many single-fact blocks (`a → b`).
- The query asks for a 1-, 2- or 3-hop composition (`a → ? → ? → d`). Each hop's fact is in a different, earlier
  block, separated by at least 64 tokens.
- The answer is one token.
- Held-out evaluation uses unseen symbols and unseen compositions.
- The hop count is stratified and reported separately. 1-hop is the control stratum, where native R5 evidence
  should already suffice.

## 4. Arms

| Arm | What it is | What it controls for |
|---|---|---|
| `NATIVE` | The frozen backbone with its own R5 read (the bypass logits) | Everything the substrate adds |
| `GEODOME` | The substrate on the fixed physical plan | — (the arm under test) |
| `EQUAL_BUDGET` | Same trainable-parameter count, no graph: the same ingress, attention pooling over the query and evidence rows, an MLP readout, the same injection | Extra parameters |
| `COMPUTE_MATCHED` | The same kernels, rounds and arc count, on a uniformly random directed graph over the same 48 vertices (no locality) | Extra computation and message passing as such |
| `FLAT_RECURRENT` | One vertex with a global mean message, iterated R rounds with the same update kernel | Recurrence without topology |
| `RELABEL_ISOMORPHIC` | `GEODOME` on an isomorphic copy of the plan: vertex identities and port ordinals permuted consistently | Label and port-ordinal artifacts. Expected to equal `GEODOME` within noise |
| `REWIRED` | A degree-preserving rewiring of the plan's arcs (the same degree sequence and arc count, a different structure) | Geodome's specific geometry, as against any sparse graph of its degrees |
| `DIAGNOSTIC_ORACLE` | `GEODOME` with gold evidence addresses | The composition capacity given perfect evidence (never deployable) |

**Graphs outside Geodome.** `COMPUTE_MATCHED`, `FLAT_RECURRENT` and `REWIRED` run on control graphs that are not
Geodome topology. They must go through an explicitly separate control path in the tensor executor, labelled as
such. Their edges are never presented as Geodome physical arcs, and Geodome's authorization rules apply only to
`GEODOME`, `RELABEL_ISOMORPHIC` and `DIAGNOSTIC_ORACLE`. Building that control path is P5 implementation work, after
approval.

## 5. Causal interventions (evaluation only, on trained `GEODOME` models)

| Intervention | Prediction if the substrate computes the answer |
|---|---|
| Gate set to 0 | Returns exactly `NATIVE` (already a gate) |
| Evidence ingress zeroed | Accuracy falls to about `NATIVE` on 2- and 3-hop items |
| Evidence swapped with another episode's | The answer follows the swapped evidence |
| Rounds below the evidence-to-readout distance | Accuracy falls to about `NATIVE` (the hop bound) |
| The evidence-to-vertex binding permuted | Accuracy drops if the learned computation is position-specific |
| Shortest-path arcs removed (the ball stays connected) at fixed R | Accuracy drops if information must travel those paths in R rounds |

## 6. Statistics and decision rules (proposed; frozen only on approval)

- **Seeds.** 5 per arm, paired: the same backbone, episodes and evidence per seed.
- **The primary metric.** Held-out answer accuracy on 2- and 3-hop items. Secondary: answer NLL, and the 1-hop
  stratum.
- **The test.** `GEODOME` against each of `NATIVE`, `EQUAL_BUDGET`, `COMPUTE_MATCHED`, `FLAT_RECURRENT` and
  `REWIRED`: a paired episode-level bootstrap within each seed, and a sign count across seeds, Holm-adjusted over
  the five comparisons.
- **Readings.**

| Reading | Condition |
|---|---|
| `ESTABLISHED` (distributed competence) | `GEODOME` beats every one of the five by at least +5 accuracy points in at least 4 of 5 seeds, with every Holm-adjusted p < 0.05; `RELABEL_ISOMORPHIC` is within 2 points of `GEODOME`; the interventions behave as predicted |
| `SUBSTRATE_AGNOSTIC` | `GEODOME` beats `NATIVE` and `EQUAL_BUDGET`, but not `COMPUTE_MATCHED` or `REWIRED`: message passing helps, and Geodome's geometry is not shown to matter |
| `NOT_ESTABLISHED` | Otherwise |

- **No `COGNITIVE_ADVANTAGE` claim** follows from any reading. A geometric-advantage claim would need
  `ESTABLISHED` and a margin over `REWIRED` specifically.

## 7. Integrity gates (every evaluation episode)

- No oracle coordinate, label or gold address reaches a deployable arm. They are checked by construction and by a
  run-time assertion on the address source.
- The backbone parameter digest is unchanged, the external clock is unchanged by reasoning, and the gate at 0 is
  bit-exact to `NATIVE`. These are the R13 gates, re-run per seed.
- Every plan was validated by `compile_plan`, and every refusal is recorded. No plan depends on the episode's
  answer.

## 8. Budget and stopping

- CPU only, with a per-run wall-clock cap and a memory preflight before every run (R13's `projected_bytes`).
- Development runs are labelled `DEVELOPMENT ONLY` and never pooled with confirmation.
- A failed integrity gate stops that arm's seed. It is reported, never re-run silently.

## 9. Open decisions for the owner

1. The task: the proposed multi-hop record composition, or an existing R5 or R10 task family.
2. The plan family: one fixed ball (proposed), or a small preregistered set of generic plans.
3. The size: K = 48, R = 8, width 32 (proposed), and the number of evidence items (4).
4. The margins and thresholds of §6.
5. Whether `DIAGNOSTIC_ORACLE` runs at all, or is deferred.
6. The compute envelope (wall-clock and seeds).
