# PROPOSAL (NOT ADOPTED): a CGC-native generative transformer as the ECS/EDEN computation

| | |
|---|---|
| Status | **PROPOSAL. NOT ADOPTED.** This file is not repository authority. `docs/ELPIS_MISSION.md` is unchanged and governs. |
| Would become | Amendment 4 of `docs/ELPIS_MISSION.md`, only by the repository owner's explicit decision |
| Scope of the evidence behind it | `research/elpis_torch` (Elpis Torch R0): a research prototype, toy scale, `docs/research/elpis_torch/EVIDENCE_R0.md` |
| Changes made by this proposal | none: no canonical document, source, test or gate is edited |

## 1. Why a proposal is needed

The mission says where cognition may happen:

    DSV4 COMMUNICATES.
    ECS / EDEN COMPUTES, LEARNS AND PERSISTS.        (Amendment 3: in volatile memory only)

and it prohibits one topology in canonical code: a full DSV transformer/MoE model as the cognitive path,
with ECS as a sidecar that conditions it or is updated from its output statistics. `ELPIS_SYSTEM.json`
adds that computation is meant to happen in the ECS substrate, "not in a transformer", and the mission gate
(`tests/boundary/test_mission.py`) fails if a DSV4.1 tower module exists in the canonical tree.

The work in `research/elpis_torch` is a trainable autoregressive transformer whose nonlinear channel
computation is the ECS_G cubic operator. If it were ever to become the Elpis model, it would be the middle
of the topology: the thing that computes. Under the mission as written that is not permitted, and it
should not become permitted by drift, by a relabelling, or because its mechanics pass their tests
("Mechanics pass is not mission pass"). So the question is put to the owner as an amendment, and until it
is decided the prototype stays in `research/`, which the gate already treats as noncanonical.

## 2. What is proposed

Recognize a second admissible form of the ECS/EDEN computation:

> ECS/EDEN may be realized as a **CGC-native sequence model**: a model over token positions in which the
> ECS_G cubic operator is a constituent, jointly trained neural operator of every computational block, and
> whose only adaptive state at inference is a bounded, session-local, volatile deformation of the CGC
> weights. Such a model is the ECS/EDEN computation. It is not a DSV model and DSV4 remains the codec.

The topology would read:

```
text / bytes
    |
DSV4 ENCODE (token boundary: tokenizer only)
    |
admitted stimulus = token embeddings                     <- part of the trained substrate
    |
==================================================
ECS / EDEN = CGC-native blocks
  attention (routing between positions)
  CGC channel transformation (the geometric operator; W_base immutable)
  session fast weights dW (volatile, bounded, discarded at close)
==================================================
    |
ECS readout = final norm and vocabulary projection       <- part of the trained substrate
    |
DSV4 DECODE (token boundary: detokenizer only)
```

## 3. Why this is not the prohibited topology

| Prohibited | In the proposed form |
|---|---|
| ECS state (`S3`) projected into a DSV model as conditioning | There is no ECS state outside the model to project. CGC weights are the model's parameters. |
| A DSV turn's output statistics (entropy, top-k shape, token ids, Engram rows) mapped into an ECS update | The fast-weight update is the gradient of a next-token loss on **observed** tokens. No statistic of the model's own output is a learning signal, and its own samples are never targets (enforced and tested in R0). |
| The loop `ECS -> DSV conditioning -> DSV cognition -> ECS feedback` | One model, one forward pass. There is no second system to loop with. |
| An autonomous DSV transformer replacing the ECS | Nothing of DSV4.1 is used: no MoE, Engram, mHC, shared compressed attention, donor weights or donor numerics. The donor tower stays in `research/dsv41_tower` as a donor. |

The honest counter-argument is that it is still a transformer: attention is not geometric mathematics, and
a reader of "not in a transformer" would say the sentence was written precisely to exclude this. That is
why the change is an amendment and not an interpretation. Section 6 lists it as open.

## 4. What the amendment would have to change

1. **Mission statement.** Add the CGC-native sequence model as an admissible realization of
   "ECS / EDEN COMPUTES, LEARNS". Keep "DSV4 COMMUNICATES" unchanged.
2. **`ELPIS_SYSTEM.json` purpose.** Replace "not in a transformer" with a statement that excludes a DSV
   model and any model in which ECS is auxiliary, without excluding a model built from the ECS operator.
3. **Prohibited topology.** Keep it, word for word. Add a second prohibited form so the new one cannot be
   hollowed out: a model whose blocks compute without the CGC operator (CGC present but gated to zero,
   bypassed, or confined to a side path that the output does not depend on) is not CGC-native.
4. **Roles.** ECS_G: owns the cubic operator's mathematics and its qualified native kernel, as now; the
   tensorized operator would be a new, separately qualified form of it, never a relabelling of the `d = 6`
   kernel. Codec: tokenizer and detokenizer only, as now. Runtime: orchestrates codec -> model -> codec.
5. **The fail-closed text path.** `ECS_CODEC_UNQUALIFIED` exists because no mapping between token space
   and ECS state is qualified. A trained embedding and readout would be that mapping. The refusal would be
   lifted only for a specific checkpoint whose qualification record the codec authority pins.
6. **Enforcement.** New gate items; none of the existing ten is weakened. The forensic sidecar commits
   must still be rejected by the rewritten gate.

## 5. What stays exactly as it is

* **Amendment 3.** The checkpoint is an immutable static resource, read and never written. Every mutable
  state of a session lives in volatile memory, is bounded, and is lost at close. No fast weight, optimizer
  state or cache is exported, at shutdown or on failure. R0 implements and tests these mechanics in
  PyTorch; a canonical version would have to repeat the zero-write qualification of
  `research/volatile_runtime_v1` on the native path.
* **Training is offline.** Producing a checkpoint is an `OFFLINE_ARTIFACT_PRODUCER` with its own
  authorization and its own entry in the persistence census. It never runs in a runtime.
* **No PyTorch in the canon.** `torch` stays a forbidden import of `src/elpis`. A canonical model would
  execute natively; the PyTorch code would remain the research reference and the trainer.
* **The qualified kernels.** `native/ECS_G` sources, digests, formats and evidence are untouched. K1's
  state `(W, epoch, H, a)` and its consolidation law are a different mechanism and are not claimed to be
  the fast weights of this model.
* **Authority.** A session's permission to learn, its bounds and its fuel would be transitions of the
  volatile ECS_C session, as `LEARN` is today. ECS_C would still not compute.

## 6. Open questions the owner should decide, not the implementer

1. Is conventional attention acceptable inside ECS/EDEN, or must routing between positions also be
   geometric before the model may be called CGC-native?
2. Which configuration is the substrate: CGC as the only channel nonlinearity (B), or an FFN with a CGC
   path (C)? In C the model computes without CGC when its gate is zero, which item 3 of section 4 would
   forbid in canonical code. R0's evidence does not favour C (`EVIDENCE_R0.md`).
3. Is the pre-activation bound, which is not E0 mathematics, admissible in a canonical operator, or must a
   canonical model be faithful E0 with stability obtained another way?
4. What replaces the retired persistent learning? With volatile fast weights a session's learning is lost
   by design. Whether anything a session learned may ever reach a future checkpoint is an offline training
   question this proposal does not answer.

## 7. Conditions before adoption could even be considered

None of these is met by R0. Each is a gate with evidence, not a milestone to be declared.

| | Condition |
|---|---|
| P1 | The tensorized operator is qualified against the native ECS_G kernel on a shared regime, with a declared float32 profile. |
| P2 | CGC against FFN at matched compute (not only matched parameters) on real text at a scale where language modelling is meaningful, preregistered, with negative results kept. |
| P3 | A native executor of the forward pass and of the bounded update, bitwise or tolerance-bound to the PyTorch reference, under the sanitizer jobs. |
| P4 | The session law bound to the volatile ECS_C session, and a zero-write qualification of the composed native path. |
| P5 | A fast-weight design whose per-session memory is bounded at deployment scale (`DESIGN.md`, section 6). |
| P6 | The interference of adaptation with unchanged knowledge characterized, with a stated acceptance limit. |
| P7 | The rewritten mission gate, reviewed, that still rejects the forensic sidecar topology and rejects a CGC-bypassed model. |
| P8 | The owner's explicit instruction. |

## 8. If the proposal is declined

Nothing has to be undone. `research/elpis_torch` remains a research laboratory like `research/dsv41_tower`:
never packaged, never imported by `src/`, never part of the Elpis cognitive path.
