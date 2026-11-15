# ERSA R12 — Offline-trained relevance selector & bounded recurrent two-hop probe

**Narrow qualification target:** `PASS_FROZEN_NATIVE_STATE_AND_TRAINED_TWO_HOP_PROBE_ONLY`.

This phase **does not** qualify a Gated DeltaNet replacement, learned halting,
a fully C-native neural runtime, unconstrained semantic retrieval, or production trust.

## Genuine predecessor coupling

- Frozen R11: 80 source/package files, all previous source pins.
- The real 14-operation typed C SSC state machine is instantiated on every
  evaluation episode. Each record is populated into C-owned exact-address RAM;
  both search steps use its actual `read_exact` receipts. There is no side
  database acting as the answer authority.
- Frozen R7 C feature extraction/ranking, plus a new 8-weight C score kernel.
  C cannot create admission rights, modify relations, or train at runtime.
- Host supplies static query/key/goal metadata and a pre-authorized mask. It
  gates generation integrity between materialization and selection.

## Task and methodology

Synthetic 24-record two-hop pointer-retrieval episodes. Each episode has a
source and target with near-key aliases carrying contradictory goal context.
Step 1 selects the source from its query, step 2 uses the *stored source
value* to select the target. Exact target value is returned from C-owned state.
The oracle source and target indices occur **only in offline training** and
in the independent evaluation scorer, not in the runtime selector.

Train 8 float64 score weights by cross-entropy on 480 disjoint synthetic
training episodes (960 supervised steps), using torch CPU only. After training
freeze these eight weights; inference uses only C scoring and native-state
read-through. No weights or checkpoints are written persistently.

Four held-out test regimes: IID alias resolution, misleading metadata, noisy
goals and removed goal signal. 256 episodes per regime. Report complete
success and abstentions. Every comparator uses an L2-normalized confidence
margin to avoid unfair absolute-score-scale differences.

**Controls:** original R7 fixed scoring weights, key-only, goal-only, AND a
competent hand-coded key-plus-goal scorer. The learned head is expected to
outperform the original R7 scoring on this fixture law but **does not need to
beat key-plus-goal** to qualify mechanics. If it fails to beat that deterministic
baseline, `LEARNED_SUPERIORITY` remains HOLD, as it should.

The missing-goal control and shuffled-training-label control prove that the
successful runs depend on genuine task signals and supervision rather than an
implicit target-label metadata field. Training labels are not present in C
scorer inputs. No GDN backend has been tested.

## Boundaries

- The two-hop recurrent loop is *host-orchestrated*, not a trainable recurrent
  cell. Max 2 selection steps and 24 C-owned exact addresses per episode.
- Evaluator materializes all 24 values from exact native RAM, so discovery
  remains O(N). The separate scorer rejects over 8192 candidates.
- R12 only proves this synthetic law. No open-world retrieval, model-scale
  throughput, causal reasoning, or learned stopping is claimed.
- The read-only-query intent does not mean the C state is literally immutable:
  SSC read-accounting/control records may mutate volatile counters.
- Explicit test/attestation exceptions: external official SSC transport still
  unpinned; OS-level NAND write ban not attested; trust and rights remain HOLD.
- Regression runner executes 732 tests (672 inherited + 60 new), plus frozen
  8192-capacity stress. All development is isolated from canonical authority.
