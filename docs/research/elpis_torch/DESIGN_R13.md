# Elpis Torch R13 — design: the Geodome cognitive substrate admission

**What R13 adds.** The first opt-in, differentiable cognitive computation backed by Geodome, wired into the existing
EvidenceLM final block. It does five things:
- consumes the backbone's own final-block representation and admitted evidence;
- addresses and activates latent working-state nodes on Geodome's frequency-120 identity index;
- performs bounded, multi-round, distributed computation over admitted physical arcs;
- reads out a latent;
- injects that latent into the existing language-model readout through a gate that starts at exactly zero.

It is research-only and opt-in. It is not a parallel language model and it replaces nothing.

**The phase plan.** The owner's directive has seven phases. R13 implements and qualifies P0 to P4: the seam audit,
the dependency admission, the tensor backend, the session bridge and the twelve gates. P5 (a competence study) is
drafted as a protocol for review, `PREREGISTRATION_R13_P5.md`. P6 (learned addressing) and P7 (CGC and HCA88
admission) are not started. The evidence is in `EVIDENCE_R13.md`.

**What R13 does not change.**
- Nothing under `src/` or `native/`.
- No CI policy and no R0 to R12 record.
- No default configuration.
- No existing source file of `research/elpis_torch`: the R13 runner checks against the base `d73cdde` that no
  `.py` file there is modified, and that no earlier record is modified (`identity.json`).
- The one changed test is `test_isolation.py`, which registers the R13 runner's single `--out` write site.
- The upstream Geodome sources are vendored byte for byte and never edited.

## 1. P0: the seam, audited

The audit is measured, not only read: `audit.json`.

| # | Question | Answer (source; measurement) |
|---|---|---|
| 1 | Where is the final-block pre-channel representation? | `ErsaLM.forward`: per block, `pre = x + mixed`, then `x = pre + channel(channel_norm(pre))` (`ersa_model.py`). With `return_pre=True` the call returns the last block's `pre` restricted to `logit_positions` |
| 2 | Does the R5 evidence read reach it? | Yes. `EvidenceLM._extra_mix` adds the read to `mixed` in the last block (`read_layer = n_layers - 1`), only at the query positions (`logit_positions`). With `mode="off"` against `"learned"`, every earlier row and the whole recurrent state are bit-identical; only the final row differs, by at most 0.0063 |
| 3 | Is the final channel batch-variant? | Yes. The routed channel's grouped dispatch (R6) groups the rows it is given. Recomputing the channel over the chunk's own T rows equals the forward bit for bit; over the single final row it differs by 5.6e-17 (float64). The bridge therefore records all T rows the channel consumed (`lastblock.capture_pre`, a forward pre-hook on the last `channel_norm`) and recomputes over exactly those rows |
| 4 | Is `logit_positions` a different computation? | Only after the channel: the head runs on the selected rows. A full-logits forward agrees at the last position to 1.1e-16 (final norm and head over T rows against one). The bridge's reference is the identical `logit_positions=[T-1]` call |
| 5 | When does the state commit? | After the logits. ERSA cells and tails, window K/V and tokens, PLE history, ring tokens, keys and fine keys, and positions are all written after the last block. A refused call leaves the state bit-identical. Tested refusals: an oversize chunk (`ChunkError`), out-of-range tokens, and the oracle mode without labels |
| 6 | Diagnostics and chunking | `diagnostics=True` returns a triple. The chunk limit is the model's `ersa.chunk`, and the session refuses more. Partitioning one 70-token chunk as 35 + 35 moves the final logits by 3.3e-16 (the same batch variance) |
| 7 | Which channel is "the final channel"? | `blocks[-1].channel`, the configured `SparseChannel` (shared expert plus 4 routed experts, top 2) in the fixture. The bridge calls that module; it never re-implements or approximates it |

**The seam.**

```
tokens [1, T] ─► EvidenceLM.forward(tokens, state, return_pre=True, logit_positions=[T-1])   once per chunk
   last block:  pre = x + mixed (+ the R5 read at T-1)     rows [1, T, d] ◄── capture_pre (the channel's input)
                x   = pre + channel(channel_norm(pre))
   logits = head(final_norm(x[:, T-1]))                    the cached BYPASS; the state is committed by the model

CognitiveSession (no further backbone call, no state write):
   query     q  = tanh(P_query pre[T-1])                    ─► the plan's query vertices
   evidence  e_k = tanh(W_e pool(encoder(exact reread of admitted block k)))   ─► the plan's evidence vertex k
   rounds    1..R over admitted arcs (synchronous): m(u→v) = MLP(h_u, port(u→v), port(v→u));  h_v' = GRU-like(h_v, Σ m)
   latent    z  = tanh(W_r mean(h over the readout vertices))
   refined   pre'[T-1] = pre[T-1] + g · P_out z             g is a scalar that starts at exactly 0
   logits'   = head(final_norm(pre' + channel(channel_norm(pre'))))[T-1]   the ACTUAL channel, over the same T rows
```

With `g = 0` the refined logits equal the bypass bit for bit, in every tested variant (`bridge.json`).

## 2. P1: Geodome as a frozen dependency

**Source authority.**
- HexaFlake, https://github.com/abraxis717/HexaFlake, commit `4e5fe56670c61a0d8906770b03033658c3087c3a`, directory
  `Geodome/`, Reference API v1.0.0, Apache-2.0.
- The API names its own frozen upstream commit, `993429cc94e9e3dcfd42df40502fa5b57175d883` (`SOURCE_PIN`).
- The public manifest `PUBLIC.sha256` is pinned at `e44d288fec19a09c17be0ce503f011d400b499e4d393f1275d305e2b23d31cb8`.
- The identity fingerprints: mapping `abe96202…d1724f51b`, edges `712a8fc3…d014231`.

**What is vendored** (`research/elpis_torch/upstream/geodome_v1/`, byte for byte):
- the seven modules the core imports: `geodome.py`, `geodome_index.py`, `geodome_state.py`,
  `geodome_scheduler.py`, `Hexaflag.py`, `GEO144K/g1c_compact.py`, `GEO144K/g2b_bridge.py`;
- with them, the upstream `LICENSE`, `NOTICE` and `PUBLIC.sha256`.

Not vendored, because nothing uses them: the optional C4 adapter (`adapters/geodome_compute.py`), CGC120, HCA88,
the qualification replays and the receipts.

**The import mechanism** (`geodome_cognitive/source.py`, `load_geodome`):
1. Every vendored file's SHA-256 is checked against its pin and against its line in the vendored `PUBLIC.sha256`,
   which is itself pinned. An unpinned file in the tree is refused.
2. A Geodome module name already loaded from anywhere else is refused, so nothing can shadow it.
3. The vendored directory is on `sys.path` only during the import, with bytecode writing disabled. Both are
   restored in `finally`.
4. Every Geodome module must resolve inside the vendored tree. The `GEO144K` namespace package's path must too.
5. The API constants (`API_VERSION`, `SOURCE_PIN`, `MAPPING_SHA256`, `EDGE_SHA256`) must equal the pins.

**The identity.** `substrate()` calls Geodome's own `build_index()`, which rejects any identity drift. It builds the
compact index once per process (about 1.1 s) and caches it. It is never rebuilt per token or per round. The
invariants are checked again on construction: 144002 vertices, 432000 undirected edges, 864000 directed arcs,
address capacity 823543 (depth 7), 12 degree-5 vertices, and an index payload of 11,502,238 bytes.

**What Geodome provides, used as Geodome defines it.**
- Hexaflag identity and coordinate resolution: `address`, `resolve`, `coordinate`, `resolve_coordinate`.
- Physical adjacency: `neighbors`, `arc_id`, `arc_endpoints`. A non-edge raises `KeyError('NONEDGE')`.
- Arc identity: `arc = 2·edge + (u > v)`. An arc identity is not authorization.
- Logical prefix groups: `prefix_count`, `prefix_page` (at most 1024 per page). A prefix is logical, not
  necessarily a connected region. Measured: 16 of 27 depth-3 prefix groups are physically disconnected; Geodome's
  C1 receipt found 236 of 343.
- `physical_arcs(vertices)` returns real edges only. No edge is ever derived from address proximity.
- Node ownership, allocation generation, epochs and bounded capacity come from Geodome's own `NodeState`
  (`acquire_vertex`, `handle`, `read`, `release`, `reset`). `NodeState.read` is the handle check: it refuses
  another instance's handle, an earlier epoch and a reused slot's earlier generation.

## 3. P2: the differentiable tensor backend (`research/elpis_torch/geodome_cognitive/`)

| Module | Concern |
|---|---|
| `source.py` | the pins, source verification, the isolated import |
| `topology.py` | `GeodomeSubstrate`: identity, physical adjacency, ports, physical arc sets, components, bounded shortest paths |
| `plan.py` | `ExecutionPlan` (host-supplied authorization) and `compile_plan` (validation into integer tensors) |
| `state.py` | `TensorWorkingState`: Geodome `NodeState` identity plus a tensor of latent rows |
| `kernels.py` | `CognitiveKernels`: the shared trainable ingress, message, update, readout and injection |
| `activation.py` | the two schedules: `run_synchronous` (trainable) and `run_changed_destination` (Geodome's semantics) |
| `reference.py` | Geodome's own `SparseScheduler` over its own `NodeState`, float64: the numerical reference |
| `binding.py` | `CognitiveSession`: the session-owned bridge to the actual `EvidenceLM` |

**The runtime state.**
- `node_ids [K]` are the plan's vertices. Each resident vertex has a Geodome `StateHandle`.
- `latent [K, D]` is float32 for training (float64 for reference checks). Rows are indexed by Geodome slot: sparse
  residency has K slots (at most 4096, Geodome's enrolment cap); dense residency has 144002, for the parity gate.
- `admitted_arcs [E]` are directed arc identities, sorted, so each destination aggregates in Geodome's arc order.
- `local_endpoints [2, E]` are `src` and `dst` in plan order, with each arc's two port ordinals.
- The Geodome `NodeState` is created with width 1. Its rows are never written: Geodome owns the identity and
  lifecycle, and the tensor owns the values. Nothing of Geodome's logic is re-implemented.

**The kernels.** They are shared by every vertex and arc; there are no per-vertex parameters.
- message: `m(u→v) = W₂ gelu(W₁ h_u + P_out[port(u→v)] + P_in[port(v→u)] + b₁) + b₂`
- update: `z = σ(W_z[h_v; a_v])`, `c = tanh(W_c[h_v; a_v])`, `h_v' = h_v + z(c − h_v)`, where `a_v` is the sum of
  incoming messages in arc-identity order
- readout: `tanh(W_r mean(h_readout))`
- injection: `g · P_out z`, with `g` starting at 0

A port is the position of a neighbour in the index's neighbour list (0 to 5). It is an identity ordinal, not a
learned or geometric direction. The message depends only on the source's old row and the two ports, exactly
Geodome's transport signature `(arc, source, destination, source_row)`. The update depends only on the old row and
the inbox, exactly Geodome's update signature. That is what makes the two executors comparable on matched
semantics.

**The two schedules, never presented as numerically equivalent.**
- `synchronous` (the trainable path). Fixed rounds. Each round, every admitted arc carries a message from its
  source's old row, and every enrolled vertex with at least one admitted incoming arc updates. There is no
  data-dependent control flow and no host extraction (`.item()`, `.tolist()`, NumPy, callbacks, ctypes), so
  gradients reach every kernel across the bounded horizon. Information moves at most one physical hop per round.
- `changed_destination` (Geodome's `SparseScheduler` semantics; the reference mode).
  - A frontier starts at the plan's `start`. Only arcs out of the frontier carry messages, and only their
    destinations update. A destination whose row changed joins the next frontier.
  - The budgets and statuses are Geodome's: `QUIESCENT`, `ACTIVE_BUDGET`, `ARC_BUDGET` and `ROUND_BUDGET`.
  - It reads scalars to decide the frontier and the status, as Geodome must. That is why it is not the training
    path.
- **Where they coincide.** If every enrolled vertex has an admitted incoming arc, the frontier starts with every
  enrolled vertex, and every destination's row changes in every round, the two perform the same computation round
  by round. The parity gate checks this class against Geodome's scheduler, reading the class membership from
  Geodome's own round records. Outside it the policies differ, and a witness records by how much (0.87 on a
  one-directional chain).

**Bounds.**
- Geodome's caps (4096 enrolled, 24576 arcs, 256 rounds, 6,291,456 visits, 1024 touched per round) are intersected
  with the session's configuration (by default 64 resident vertices, 1024 arcs, 16 rounds, 16384 visits and 8
  evidence items).
- A synchronous plan must fit `rounds × arcs` in its visit budget at compile time.

**Gradient horizon.** The working state keeps values only: published rows are stored detached. A training graph
therefore spans one `reason()` call (its ingress, rounds and readout) plus the injection of its `readout()`, never
two calls and never an external token step. Graph size was measured: 185 autograd nodes per call, constant over
repeated calls. `observe()` drops every result of the previous observation.

## 4. P3: the session bridge (`CognitiveSession`)

**Scope.** Batch one, the final position as the query, a frozen backbone (eval mode, no parameter requires
gradient), and kernels of the backbone's own width and dtype.

| Operation | Contract |
|---|---|
| `observe(tokens)` | Feeds `[1, T]` once through `EvidenceLM.forward`. The model commits its own state. Keeps the final-position logits (the bypass), `pre[T-1]` and the T channel rows. Runs under `no_grad`, even inside `enable_grad` |
| `enrol(vertices)` | Makes Geodome vertices resident with zero rows, all or none. Refuses an already-resident vertex, a duplicate, an out-of-range vertex or an overflow before anything changes |
| `admit_evidence(addresses)` | Takes 1 to 8 exact `BlockAddress` values. Checks the type, duplicates, causality, the canonical form and liveness. Then rereads the original tokens and encodes them through the existing `_reread_encoded` path. All or none |
| `reason(plan)` | Compiles and validates the plan, then runs the bounded computation. Publishes every enrolled row or nothing. Does not advance the external clock |
| `readout()` | Returns the latent and the refined final-position logits, through the actual final channel over the same T rows |
| `prediction()` | Returns the refined logits of the current observation's last committed readout, else the backbone's cached logits. Says which (`"cognitive"` or `"backbone"`) |
| `reset()`, `close()` | `reset` gives a clean session in the same identity: a new ring epoch and a new Geodome epoch, so every earlier handle and address is refused. `close` drops everything |

**The admission rules.** All checks run before any reread.
- The address must be the ring's own `BlockAddress` type.
- It must be strictly before the query's own block: `block < position // block_size`. This refuses incomplete,
  future and same-query blocks, including the block that holds the query token (`EVIDENCE_CAUSALITY`).
- It must be canonical: `ring.address(block) == address`. This refuses another session's or epoch's address, a
  tampered slot, generation or span, and an evicted block (`StaleAddressError`).
- It must be live at the query: `ring.check(address, at=position)`.
- The encoder sees the ORIGINAL tokens (the exact reread). The R8 stored fine keys are never read for admission:
  across 18 variants, `blocks_reencoded` rises by exactly the admitted count and `fine_tier_reads` by zero.

**Vertex and address are different identities.** A Geodome vertex ID is never treated as an EvidenceRing address.
The binding is explicit and positional, per `reason()` call: `plan.evidence[k]` names the vertex that receives
admitted item `k`. Nothing maps block numbers to vertices implicitly.

**The clock invariants.** `admit_evidence`, `reason` and `readout` change none of:
- the external position;
- ERSA cells and tails;
- the window cache and its tokens;
- the PLE history;
- ring tokens, index keys, ids and stored fine keys.

They may move the ring's accounted reread counters (`blocks_reencoded`, `ring_rereads`). The state digest covers
every one of those tensors and clocks.

**Anchoring and transactions.**
- Every operation checks the session identity, the backbone revision (`RevisionWatch`) and the current
  observation. A backbone revision refuses the session until `reset()`, and the refusal is sticky.
- `reason()` is atomic per operation, which is stricter than Geodome's per-round atomicity. A refused plan, a
  non-finite computation, a stale plan, or a working state reset during the computation publishes nothing.
- A refusal never rewinds or repeats the observation: the cached logits remain the safe bypass.

**Inference and training.** `observe` and `admit_evidence` always run the backbone without gradients. Only
`reason` and `readout` build a graph, and only under `enable_grad`. The R8 stored fine keys serve the backbone's own
evidence read in inference only. They are never a differentiable training path: the backbone runs without
gradients, and the cognitive path's evidence is the exact reread.

## 5. P4: the gates and where each is tested

| Gate | Record (`results/r13/`) | Test |
|---|---|---|
| 1 import and dependency identity | `identity.json` | `test_the_vendored_geodome_is_the_pinned_upstream_bytes`, `test_a_damaged_vendored_tree_is_refused`, `test_a_shadowing_module_is_refused_before_anything_is_imported`, identity tests |
| 2 exact disabled-path logits and state parity with unmodified R12 | `bridge.json` (18 variants) | `test_the_bridge_on_actual_evidence_lm` |
| 3 tensor against the Geodome reference on matched semantics | `parity.json` | `test_the_changed_destination_executor_is_geodomes_scheduler`, `test_the_synchronous_schedule_equals_the_scheduler_only_where_they_coincide` |
| 4 dense against sparse residency | `parity.json`, `bridge.json` | the parity test (both residencies), `test_sparse_and_dense_residency_give_the_same_bridge_result` |
| 5 finite nonzero gradients through a multi-hop path | `gradients.json` | `test_gradients_reach_every_kernel_over_the_hops_and_stop_at_the_hop_bound` |
| 6 gradients against a float64 numerical reference | `gradients.json` | `test_float64_gradcheck_of_the_executor_and_readout`, `test_end_to_end_float64_gradcheck_through_the_actual_final_channel` |
| 7 no propagation beyond the synchronous hop distance | `parity.json`, `gradients.json` | `test_information_moves_at_most_one_physical_hop_per_round` and the hop-bound test |
| 8 refusal of unauthorized arcs, stale and cross-session handles, invalid tensors, overflows | `bridge.json` (`refusals`, `admission`) | `test_every_plan_refusal_names_its_rule`, `test_handles_are_owned_epoch_scoped_and_generation_checked`, `test_every_refusal_is_coded_and_changes_nothing` |
| 9 state rejection and transaction atomicity | `bridge.json` | `test_publish_validates_everything_first_and_keeps_values_only`, `test_working_state_acquisition_is_all_or_none_and_bounded`, the refusal battery |
| 10 backbone parameter immutability | `bridge.json` (`immutability`) | `test_training_the_cognitive_kernels_never_writes_the_backbone` |
| 11 internal reasoning does not advance the external clock | `bridge.json` (`context_clock`) | `test_the_bridge_on_actual_evidence_lm` |
| 12 reset restores clean-session behavior | `bridge.json` (`reset`) | `test_the_bridge_on_actual_evidence_lm` |

**The EvidenceLM variants** (`bridge.json`) form the full matrix of:
- evidence `off` and `learned`;
- fine keys `reread` and `stored`;
- PLE off and on;
- origin 0 and 64.

Two window-attention variants are added: (learned, reread, PLE off, origin 0) and (learned, stored, PLE on,
origin 64). Every stream of 560 to 640 tokens wraps the 256-token ring and evicts its earliest blocks. The evicted
addresses are refused.

**Host extraction** is scoped to the cognitive path: ingress, synchronous rounds, readout and injection. It is
detected with a `TorchFunctionMode` that records `item`, `tolist`, `numpy` and the scalar conversions. A positive
control proves the detector sees them. The backbone's routed channel keeps its own integer routing bookkeeping
(`SparseChannel._grouped` uses `tolist()`, R6). That bookkeeping is part of the existing final channel, which the
directive requires to be used as it is. The reference path converts rows by design.

## 6. Resource policy

- Everything is volatile. There are no writes of runtime state, plans or checkpoints, and the session refuses
  pickling and copying.
- The frozen index (11,502,238 bytes) is shared by the process.
- Per session, `projected_bytes()` is exact: the backbone state, the working state (latent rows plus Geodome's
  `NodeState` identity metadata, which includes an O(V) vertex-to-slot map), and the kernel parameters. An
  optional `budget_bytes` refuses the session before use.
- The plan, the state, the visits and the autograd graph are all bounded by construction.
- `resources.json` records, separately, parameters, state bytes, topology bytes, gradient scratch (saved tensors
  per training step), scheduler metadata (compiled plan bytes), message visits, active vertices, latency and the
  process's peak RSS.

## 7. Deliberately absent

- Competence training of any kind. P5 is a draft protocol for the owner's review.
- Learned addressing (P6). Plans are host-supplied and validated: "the controller proposes, the runtime validates"
  is a P6 concern, and there is no controller here.
- Autonomous halting. Rounds are fixed.
- CGC volatile weight adaptation, CGC120 and HCA88 (P7). HCA88 is qualified upstream only for C4 two-arc relation
  windows.
- Geodome's C4 adapter, GPU execution and native kernels.
- A substrate KV cache. Geodome allocates no token-indexed KV, and Elpis's window and evidence memory keep their
  duties.
- Any claim of cognitive, geometric or semantic advantage. Mechanical tests cannot carry one.
