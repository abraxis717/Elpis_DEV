# Provenance of the CGC mathematics in `research/elpis_torch`

## Upstream source

| | |
|---|---|
| Repository | https://github.com/abraxis717/CGC |
| Commit | `0895c2bc732a377b5a5f0d063ae13cc72684ec9a` (2026-11-09, `main`) |
| License | Apache-2.0 (the upstream `LICENSE` is byte-identical to this repository's `LICENSE`, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`) |
| File | `src/cgc/ecsg_e0.py` |
| File SHA-256 | `92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084` (equals the digest the upstream `SOURCE_SHA256SUMS.txt` and its R97W experiment pin as `SHA_E0`) |

The same mathematics is stated in this repository by `native/ECS_G/README.md` (the qualified native kernel)
and implemented by `native/ECS_G/src/ecsg_math.c`. Neither is modified by this work.

## What was copied

`upstream/cgc_ecsg_e0.py` is the upstream file, byte for byte. It is never edited. Its digest is checked by
`tests/research/elpis_torch/test_e0_reference.py`, which loads it from the file and uses it only as an oracle.
Nothing else was copied: not the `CGC120` wrapper, not the R97W experiment, not the native C sources, no
weights, no evidence.

## What was written here, and how it relates to the upstream

| File | Relation to the upstream mathematics | Changes |
|---|---|---|
| `e0.py` | A separate float64 implementation of the same E0 cell (`phi`, `phi'`, `f_W`, the mean-squared-error gradient, one explicit gradient step). Written after reading the upstream file and the ECS_G README; it follows their formulas and the upstream operation order. | Equal to the upstream bitwise on every tested case. Adds the unpacked moment form `(mu, M, T3)` (the upstream packs the upper triangle). Raises `E0Error` with descriptive messages instead of the upstream's code strings. No K1, no FMS, no state. |
| `cgc.py` `CGCCore` | `C` E0 cells sharing one input, evaluated as one matrix product. With `preact_bound=None` each output channel equals `e0.forward` of that channel's `[d, N]` weight. | New tensor layout `[d, C, N]` (the upstream `CGC120` holds four separate `[5, 6]` cells). `phi` in Horner form (agrees with the reference to round-off, not bitwise). Runs in float32 or float64 and on inputs of any leading shape. |
| `cgc.py` `preact_bound` | **Extension, not E0.** `u -> b tanh(u/b)` before `phi`. Deviates from E0 by at most `\|u\|^3 / (3 b^2)` in the pre-activation and converges to E0 as `b -> inf`. | New. Exists to cap the cubic's growth. |
| `cgc.py` `CGCTransform` | **Architecture around E0.** Optional linear bottleneck `P_in` and linear readout `P_out`. Without the bound, cell `c` is exactly E0 on the full hidden input with weight `P_in^T W_c` (a rank constraint on `W`). | New. E0 has neither a readout nor a factorized weight. |
| `model.py`, `session.py` | Use `CGCTransform` inside a transformer block and adapt the core weights `W` per session. | New. The session's update is plain gradient descent on a language-model loss; it is not the G1 recurrence, not K1 and not the R97W Reptile procedure. |

## What is not claimed

* None of the tensorized or bounded operators is a qualified ECS_G kernel. The qualified kernel is
  `native/ECS_G`, in its frozen regime `d = 6`, `N in {36, 48, 72}`, float64.
* The R97W result (a 120-parameter CGC against a parameter-matched GELU network on synthetic meta-learning
  tasks) says nothing about language models, and nothing here reproduces or extends it.
* Agreement with the upstream file is established on the tested shapes and inputs
  (`tests/research/elpis_torch/test_e0_reference.py`), not for all inputs.

---

# R2 (ERSA): provenance, licences and what was not copied

## Sources studied

| Source | Identity | Licence | Use in R2 |
|---|---|---|---|
| Historical ERSA archive, `research/ersa/` | commit `7b423a174fa58e6f0d2f3156ed496dcb67f53361` ("research(ersa): archive pinned R18 R35 R36 sources"), 133 tracked files (131 sources, the manifest, a README), manifest SHA-256 `b67dc98c473d3f307d2efd4890242e6027077b784e774be23dd9b66c6b625ede` | first-party research archive; contains one third-party file (next row) | **Read only.** It is the starting research authority for what ERSA is meant to be: R18 a read-only multi-hop slot reader over host-supplied state, R35 gated slot writes, R36 a C state owner. Nothing in it is edited, imported, installed or executed by R2. |
| `research/ersa/.../R35_.../third_party/fla_gsa/naive.py` | adapted from `fla-org/flash-linear-attention`, `fla/ops/gsa/naive.py`; Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li | MIT; the licence text is retained beside it in the archive (`LICENSE.txt`) | **Not used.** R2 does not import it, call it or reproduce it. R2's update is a gated delta rule on one matrix, not the two-matrix gated-slot recurrence of that file. |
| `divagr18/qwenputer` | commit `0cab57038a8b218dabe481f994174c3437405e3d` | MIT, Copyright (c) 2026 Divyansh Agrawal | README and design notes read. **Idea adopted:** the state of a gated delta-rule layer is exact RAM when keys are one-hot, the gate is open and the write strength is 1; gate 0 wipes a head, gate 1/2 halves it. No code, compiler, weights or trace format copied. |
| `vixhal-baraiya/qwenGPU` | commit `e2695f6a0a818ebc68e2e14b6b990eddf04c51b3` | MIT, Copyright (c) 2026 vixhal-baraiya | README read: registers, program ROM and memory as rows of the recurrent state; the token stream as execution log. No code copied. |
| `Bobliuuu/qwentpu` | commit `856720f3e5d27c0fc1c19f370a39487853e722ae` | MIT, Copyright (c) 2026 Bobliuuu | README read: exact integer computation replayed against a reference, position by position. The replay-against-a-reference test method was adopted. No code copied. |
| The gated delta rule | the published recurrence used by delta-rule linear attention and Gated DeltaNet | mathematics, not code | R2's kernel is that recurrence written from its definition. No reference implementation was consulted for the code, and no chunk-parallel kernel was ported. |

The three external repositories were cloned read-only outside this repository for study and are not vendored.

## What R2 wrote

| File | Content | Relation to the sources |
|---|---|---|
| `ersa.py` | `delta_step` (one state transition), `ERSAMixer` (trainable multi-head recurrent mixer), `ERSACells` (exact addressable bank), `compile_cell_mixer` (weights computed so the mixer's forward pass is exact RAM), `recall_capacity` | New. Shares the name and the purpose of the historical ERSA (a recurrent associative token mixer over bounded volatile state) and none of its code. Unlike R18 it writes its own state; unlike R35 it needs no donor routine and no host-supplied admission mask. |
| `ersa_model.py` | `ErsaLM`: ERSA as the token mixer of the CGC transformer, optionally beside R1's ring-buffer window attention | New; reuses R0/R1 modules of this package only. |
| `experiments/run_r2.py`, `tests/research/elpis_torch/test_ersa.py` | experiments and qualification tests | New. |

No new dependency: PyTorch only, as for R0 and R1.

## Not claimed, and left to the owner

* R2 is not the historical ERSA lineage's R11/R14 native authority, selector, halting controller or admission
  model. None of those is implemented, replaced or qualified here.
* The archive's third-party MIT file is carried with its licence inside the archive. This change does not edit
  the repository's top-level `NOTICE` or `LICENSES/`; whether the archive's presence calls for an entry there
  is the owner's decision.
* The archive's own tests and install scripts were not run.

---

# R3 (long-context addressing): references and what was not copied

No third-party code was copied, vendored, imported or executed. No repository listed in the directive was cloned
for R3. The following were used as research references for design principles only:

| Reference | Licence (code) | Principle taken | Not taken |
|---|---|---|---|
| Native Sparse Attention, arXiv 2502.11089 | paper | select blocks by a learned score and read their raw tokens; keep a local branch beside it | its three-branch gated attention, compression MLP, kernels |
| StreamingLLM / attention sinks, arXiv 2309.17453 (`mit-han-lab/streaming-llm`, MIT) | paper; code not consulted | a "read nothing" sink (R3's learned null logit) and position semantics relative to the cache, not the text | sink tokens in a dense model |
| Kimi Linear, arXiv 2510.26692 (`MoonshotAI/Kimi-Linear`, MIT) | paper; code not consulted | a hybrid of recurrent layers with few attention layers (configurable `attn_layers`; 3:1 is the scaling default) | Kimi Delta Attention, its kernels |
| Chunkwise delta rule (the WY/UT form of DeltaNet; Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", 2024) | mathematics | that the delta-rule recurrence has an exact block-parallel form with a unit-triangular solve | any code; R3's `chunked_scan` was derived from R2's own recurrence (with untied erase and exact zero gates via masked cumulative products) and is tested against the serial loop |
| RULER (`NVIDIA/RULER`, Apache-2.0) | benchmark reference | the idea of a graded synthetic ladder with multi-hop, distractor and position variants | its tasks, code and data |
| FlashAttention (`Dao-AILab/flash-attention`, BSD-3) | not used | | no fused kernel was written or imported |

What R3 wrote: `history.py`, `history_model.py`, `longrange_tasks.py`, `program_tasks.py`, `scaling.py`,
`experiments/run_r3.py`, `tests/research/elpis_torch/test_r3.py`, the `chunked_scan` kernel and the device and
precision changes in `ersa.py`, `ersa_model.py`, `scroll.py`, and the tokenizer interface. All new and original.

---

# R4 (context runtime): references and what was not copied

No third-party code, weights or data were copied, imported, vendored or executed. No new dependency.

| Reference | Licence | Used for | Not taken |
|---|---|---|---|
| XHToken Spark-X2.5-4B and Spark-X2.6-4B model cards and `config.json` (huggingface.co/XHToken), fetched 2026-11-16 | Apache-2.0 (weights, code) | the published geometry only: 36 layers (27 sliding, 9 full), window 512, 16 query / 4 KV heads, head dim 256, head-wise sigmoid attention-output gate, partial rotary 0.25 (theta 5e6) on full layers, full rotary (theta 1e4) on sliding layers, 1,048,576 positions; the cards' statements that X2.6 keeps X2.5's architecture and improves long context through post-training | its modeling code (never downloaded), weights, tokenizer, training data, benchmark numbers |
| StreamingLLM (arXiv 2309.17453; mit-han-lab/streaming-llm, MIT) | paper | the distinction between a bounded streaming window and retained information | code not consulted |
| Native Sparse Attention (arXiv 2502.11089), Kimi Linear (arXiv 2510.26692) | papers | sparse placement of expensive attention; selection over compressed blocks (already in R1/R3) | code |

What R4 wrote (all original): `context_runtime.py` (deterministic exact top-K, tiled exact scoring, the
encoded-block cache, runtime options, the streaming session), `spark_reference.py` (an independent, untrained,
scaled-down comparator written from the published configuration), the R4 parts of `history_model.py`,
`history.py`, `scroll.py`, `ersa_model.py`, `scaling.py`, `experiments/run_r4.py` and
`tests/research/elpis_torch/test_r4.py`.


# R5 (learned evidence retrieval): references and what was not copied

No third-party code, weights or data were consulted, copied, imported, vendored or executed, and no dependency was
added. By directive, nothing of DeepSeek-V4.1-Flash or Spark-X2.6 was reproduced and no VGGT-Omega code, weight or
dependency was used. The design follows general retrieval practice (a cheap coarse candidate scan followed by an
exact reread and a learned reranker; token-level matching of a query against candidate tokens), written from
scratch for this repository; no specific paper's implementation was followed.

What R5 wrote (all original): `evidence_ring.py`, `evidence_model.py`, `evidence_tasks.py`,
`experiments/run_r5.py`, `tests/research/elpis_torch/test_r5.py`, the `--evidence` scenario of
`tests/research/elpis_torch/zero_write_scenario.py`, and `docs/research/elpis_torch/BASELINE_R5.md`,
`DESIGN_R5.md`, `EVIDENCE_R5.md`. It reuses this repository's ERSA/CGC core unchanged (`ersa_model.ErsaLM`
through its extension hooks), R3's channel module and R4's deterministic `exact_topk`.


# R6 (expertized backbone): references and what was not copied

No third-party code, weights or data were copied, imported, vendored or executed, and no dependency was added.
The directive named three papers as motivation for sparse expert capacity, granularity, routing stability and
the reuse of dense checkpoints: OLMoE (arXiv 2409.02060), Sparse Upcycling (arXiv 2212.05055) and DeepSeek-V3
(arXiv 2412.19437). They were not consulted for code during R6; the routing (softmax top-K with renormalized
weights), the load-balance loss (`E sum f_e P_e`), the router z-loss and the upcycling-by-cloning idea are
standard techniques implemented here from their mathematical definitions, and the expert anatomy (each expert a
SwiGLU plus its own CGC and gate) is Elpis's own. No architecture of those papers is reproduced.

What R6 wrote (all original): `sparse_channel.py`, `lm_training.py`, `experiments/run_r6.py`, the R6 parts of
`ersa_model.py`, `context_runtime.py`, `history_model.py`, `evidence_model.py` and `experiments/run_r5.py`,
`tests/research/elpis_torch/test_r6.py`, and `docs/research/elpis_torch/EVIDENCE_R5.md`, `DESIGN_R6.md`,
`EVIDENCE_R6.md`. The training fixture is this repository's own source code.


# R7 (parametric lookup embeddings): the Engram reference and what was not copied

The R7 directive named DeepSeek-AI's Engram as the reference for conditional n-gram memory:
github.com/deepseek-ai/Engram at commit `fb7f84a21f91223715394a33a1dc24bbfb7f788e` (Apache License 2.0; its
`README.md`, `engram_demo_v1.py` and `Engram_paper.pdf` were read), arXiv:2601.07372 (v1, 12 Jan 2026; v2, 12 Jul
2026) and the vLLM Engram documentation. The repository was cloned read-only outside this repository, and nothing
in it was executed. No Engram code, weights, data or tokenizer was copied, vendored or imported, and no dependency
was added.

`ple.py` implements the published mechanism from its mathematical description: hashed suffix n-grams with
multiplicative-XOR heads over prime-sized ranges of one table, concatenated rows, and a sigmoid gate on
RMS-normalized query and key in front of a value projection added to the residual stream. Every divergence is
listed in `docs/research/elpis_torch/DESIGN_R7.md` section 3, including:
- exact byte ids instead of tokenizer compression;
- sentinel and boundary handling;
- per-head multipliers derived from SHA-256 instead of NumPy;
- no short convolution and no multi-branch keys;
- lazy Adam updates.

What R7 wrote (all original): `ple.py`, `ple_store.py`, `experiments/run_r7.py`, the R7 parts of
`ersa_model.py`, `lm_training.py`, `scaling.py` and `sparse_channel.py`, `tests/research/elpis_torch/test_r7.py`,
the PLE scenario of `zero_write_scenario.py`, `docs/research/elpis_torch/DESIGN_R7.md` and `EVIDENCE_R7.md`. The
training corpora are this repository's own source and documentation, plus synthetic streams generated by the
harness.


# R8 (the evidence read path): no external source

R8 measures and accelerates Elpis's own R5 evidence read. No external code, paper mechanism, weights or data was
used. The stored fine keys reuse a projection the R5 reranker already defines, computed earlier (at block
completion) and kept in the ring's own index; nothing else changes.

What R8 wrote (all original): `experiments/run_r8.py`, the R8 parts of `evidence_ring.py` and `evidence_model.py`,
the stored-keys scenario of `zero_write_scenario.py`, `tests/research/elpis_torch/test_r8.py`, and
`docs/research/elpis_torch/DESIGN_R8.md` and `EVIDENCE_R8.md`. The timing streams are seeded random byte ids; the
test streams are the R5 synthetic record generators.


# R11 (the bounded expert-local CGC specialist): the R97W protocol, re-implemented

**Sources read, in this repository and in the public CGC repository** (`abraxis717/CGC`, read-only clone at commit
`0895c2b`):
- the pinned E0 cell already vendored here (`upstream/cgc_ecsg_e0.py`, unchanged);
- R97O's archived result (`evidence/phase97o_ecsg_e0_differentiable_unroll/run_YwHfIlCG/result.json`);
- R97W's archived result and its portable port (`research/benchmarks/r97w/`).

**What was taken.** No code was copied, vendored or imported from the CGC repository in R11. `specialist_tasks.py`
re-implements R97W's portable protocol from its specification, under R11's own seeds and namespaces:
- the teacher families;
- the 120-parameter E0 and GELU learners;
- the 8-step clipped inner SGD;
- first-order Reptile.

It is a new experiment, never a re-run of R97W's held-out episodes. The geometry (5 inputs, 3 cells, 8 units) is
R97W's.

**What R11 wrote (all original).**
- New modules: `cgc_specialist.py`, `specialist_tasks.py`, `specialist_stack.py`.
- Experiment scripts: `experiments/run_r11.py`, `experiments/run_r11_stage_c.py`,
  `experiments/bench_r11_operators.py`.
- Opt-in additions to existing modules: in `sparse_channel.py`, `model.py` and `ersa_model.py`.
- Tests: `tests/research/elpis_torch/test_r11.py`.
- Documents: the R11 documents under `docs/research/elpis_torch/`.

**Data.** Every task is generated synthetically by the harness. No external data, weights or checkpoints were used.


# R12 (the micro-CGC volatile fast state): no external code

**What R12 is built from.**
- **The update.** The CGC lineage's own explicit local update: R97W's and R11 Stage B's 8 clipped SGD steps, with
  the analytic gradient of the PDF's adaptive specialist contract. It is checked against autograd in `parity.json`.
- **The transaction.** It runs through this repository's `session.bounded_update` (R0), unchanged.
- **The research context.** The owner supplied an analysis, "CGC, LAO, and the Missing Specialization Mechanism in
  Elpis". R12 implements its first minimal discriminating experiment: fast-update restoration inside the frozen
  ERSA stack. The analysis was read as context; nothing in it was copied as code.
- **P5 reuses R11's own Stage B procedure** (`run_r11.train_stage_b`, `specialist_tasks.evaluate_arm`), unchanged.
- **Nothing from LAO or K1** is used. No external code, weights or data were used.

**What R12 wrote (all original).**
- `fast_state.py`, `fast_episodes.py`, `experiments/run_r12.py` and `experiments/run_r12_p5.py`.
- `tests/research/elpis_torch/test_r12.py`, `test_r12_p5.py` and `test_r12_records.py`.
- `DESIGN_R12.md`, `CGC_FAST_STATE_CONTRACT_R12.md`, `PREREGISTRATION_R12.md`, `PREREGISTRATION_R12_P5.md` and
  `EVIDENCE_R12.md`.

**Data.** Every task is the R11 teacher's synthetic data. R12 retrains R11's Stage C models from their recorded
seeds with R11's own code; it neither downloads nor loads any checkpoint.


# R13 (the Geodome cognitive substrate): Geodome v1.0.0, vendored byte for byte

**Upstream source.**

| | |
|---|---|
| Repository | https://github.com/abraxis717/HexaFlake |
| Commit | `4e5fe56670c61a0d8906770b03033658c3087c3a` ("Release Geodome Reference v1.0.0"), directory `Geodome/` |
| API | Geodome Reference API 1.0.0; its frozen upstream commit (`SOURCE_PIN`) is `993429cc94e9e3dcfd42df40502fa5b57175d883` |
| Public manifest | `PUBLIC.sha256`, SHA-256 `e44d288fec19a09c17be0ce503f011d400b499e4d393f1275d305e2b23d31cb8` |
| Identity fingerprints | mapping `abe96202b554608bd8f5e702b70459daaaf4ab33eb7dfd058723fd4d1724f51b`, edges `712a8fc3d94685aeb614456ba8c88f6df7e6c6ea8a4c19e86d4d3865ad014231` |
| License | Apache-2.0. The upstream `LICENSE` is byte-identical to this repository's `LICENSE` (SHA-256 `c71d239d…0ab4`) |

**What was copied.** `upstream/geodome_v1/` holds these files, each equal to its line in the upstream
`PUBLIC.sha256`, and never edited:
- the seven modules the Geodome core imports: `geodome.py`, `geodome_index.py`, `geodome_state.py`,
  `geodome_scheduler.py`, `Hexaflag.py`, `GEO144K/g1c_compact.py`, `GEO144K/g2b_bridge.py`;
- the upstream `LICENSE`, `NOTICE` and `PUBLIC.sha256`.

`geodome_cognitive/source.py` pins every file by SHA-256 and checks it against both the pin and the manifest line.

Nothing else was copied. That leaves out the optional C4 adapter (`adapters/geodome_compute.py`), `CGC120.py`,
HCA88, the qualification replays and receipts, `run.py` and `example.py`.

The upstream manifest also lists `src/ecsg_e0.py` with SHA-256 `92c987e0…7d084`. That is the same digest as the CGC
E0 file vendored here in R0 (`upstream/cgc_ecsg_e0.py`), so the two lineages agree on that file.

**Attribution** (from the vendored upstream `NOTICE`, kept with the files):
- `Original HexaFlake concept and geometric architecture: @Akitti (https://x.com/Akitti).`
- `Computational implementation and experimental development: Elpis / abraxis717 (https://github.com/abraxis717).`
- "Attribution does not assign or extinguish any contributor rights."

The `NOTICE` also retains the upstream Elpis notice and its third-party statements, which concern components not
vendored here (HCA88's CPython-derived summation, the inference subsystem's references). It is preserved unchanged,
as the Apache License requires on redistribution.

**The import mechanism.** `load_geodome` admits the sources in five steps:
1. It verifies every file against its pin and the manifest, and refuses unpinned files.
2. It refuses a Geodome module name already loaded from elsewhere.
3. It puts the vendored directory on `sys.path` only for the import, with bytecode writing disabled, and restores
   both.
4. It requires every Geodome module, the `GEO144K` namespace path included, to resolve inside the vendored tree.
5. It checks the API constants against the pins.

The frequency-120 index is then built once per process by Geodome's own `build_index()`, which rejects identity
drift.

**What R13 wrote (all original).**
- `geodome_cognitive/` (`source`, `topology`, `plan`, `state`, `kernels`, `activation`, `binding`, `reference`).
- `experiments/run_r13.py`.
- `tests/research/elpis_torch/test_r13.py` and `test_r13_records.py`.
- `DESIGN_R13.md`, `EVIDENCE_R13.md` and `PREREGISTRATION_R13_P5.md` (a draft).

The tensor executor, the kernels and the bridge are new. Geodome's scheduler is used unchanged, as the float64
reference only.

**Data.** Synthetic token streams from seeded generators. No external data, weights or checkpoints were used, and
nothing is written but the runner's `--out` records.
