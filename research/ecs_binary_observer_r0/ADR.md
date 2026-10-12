# ADR EBO-R0-1: where a binary observer may look at ECS, and what it may see

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `UNQUALIFIED` · `NO_CLAIM` · Phase A of ECS Binary Observation R0

Written before any observer code exists.

## Pinned source

* Mainline `bec6da7327bc164f8f18f38274223fb89969b89b` (merge of PR #44, `claude/busy-gates-4bg595`). It is not
  `fd9c128`; the 90-file delta between them includes the K1 header and source, RuntimeCore, and `src/elpis/ECS/k1.py`.
* K1 ABI v1 as declared in `native/ECS/include/elpis/ecsg_k1.h`, implemented in `native/ECS/src/ecsg_k1.c`.

## Question

Which numerical results of an actual native ECS computation can a deterministic encoder
`B = Encode(observation, frozen_law)` read? The encoder must work without:

* re-executing the computation;
* an extra native crossing into ECS;
* exposing privileged mutable state;
* a device synchronization.

## What the canonical computation is

The canonical read-only operation is QUERY:

* `docs/COGNITION_R0.md` defines it.
* Managed callers reach it through `Runtime.run_query` → RuntimeCore `Core::query` (`native/runtime/src/core.rs`).
* Unmanaged callers reach it through `elpis.runtime.cognition.run_query` → `K1State.query_identity`.

Both end in exactly one guarded native call:

```
elpis_ecsg_k1_query_identity(state, dim, x, rows, out, digest)
  out[r]  = f_W(x_r) = sum_i phi(x_r . w_i),  phi(z) = 0.5z + 0.5z^2 + 0.5z^3     (binary64, rows values)
  digest  = SHA-256 retained-state identity over header || W || H || a              (32 bytes)
```

The call takes the single-writer guard, reads W and the image, writes only the caller's `out` and `digest`, and
allocates nothing. The managed path also compares `digest` with the durable expected identity. On a mismatch,
RuntimeCore withholds the answer (zeroes `out`) and fail-stops.

The computation runs on the CPU in binary64. There is no device, so there is nothing to synchronize.

## Candidates

| Candidate | Shape | Kind | Observable without re-execution / extra crossing / privileged state? | Notes |
|---|---|---|---|---|
| Query input `X` | `rows x dim` binary64, `1 <= rows <= 4096`, `dim <= 64` | caller-owned input; exact bytes | **yes**: the caller already holds it (`QueryStimulus.x`, the `x` slice in RuntimeCore) | Not computed by ECS. Its identity is exact; it carries nothing about W. |
| Forward answer `f_W(X)` | `rows` binary64 | **learned representation**: the only quantity ECS computes from its learned W on a QUERY | **yes**: written by the one `query_identity` call into caller memory | A scalar per row. Determined by `S3(W)` (83 moments at dim 6), so `rows >= 83` generic rows determine the instantaneous forward map. Invariant under neuron (column) permutation of W, up to summation-order rounding. |
| Retained-state digest | 32 bytes | **exact identity bytes**, not a representation | **yes**: same call | Distinguishes every state exactly but carries no geometry: any change gives an unrelated digest, and neuron-permuted W gets a different digest. |
| Pre-activations `x . w_i`, neuron activations `phi(.)` | `rows x width` | internal intermediates | **no** | They exist only transiently in K1's private `row` scratch (`ROW_BLOCK = 8` rows at a time, overwritten). Exposing them would need a K1 ABI change or a second forward pass; both are prohibited. |
| W, H, a | `dim x width`, `F(F+1)/2`, `F` | privileged retained state | **no** | Only via `copy_w` / `copy_h_packed` / `copy_a`: extra crossings on the cold path that expose privileged state. They may be used only as laboratory ground truth, never by the observer. |
| S3 readout of a candidate | `F` binary64 | diagnostic projection on the legacy LEARN turn path | **excluded** | Produced only inside a LEARN transaction (`txn_run_schedule`). The observer must have no access to LEARN. |
| Transformer hidden activations | n/a | n/a | **do not exist** | No qualified ECS<->DSV codec exists (`ECSCodecMap` is UNQUALIFIED; fixtures are `TRAINING=NONE SEMANTICS=NONE`). The DSV4.1 tower is research donor code, not Elpis cognition. Nothing is simulated in their place. |

## Measured native cost of the observed call

`build_rel`, GCC Release, this host (Xeon 2.1 GHz, SHA-NI present but unused by the portable SHA-256), dim 6,
width 36, one exploratory run.

| rows | `query_identity` | bare `forward` |
|---|---|---|
| 1 | ~139 us | ~0.07 us |
| 64 | ~136 us | ~4.8 us |
| 4096 | ~370 us | ~225 us |

The canonical QUERY's floor is the retained-state SHA-256, about 30 KB per call.

**Disclosed now:** any "about 1% p99" budget is generous only because of that digest. Against a bare `forward`, the
same observer would be a large fraction of the cost. Phase D measures and reports both.

## Decision

1. **Observation point.** The observer reads only:
   * the caller-owned buffers of one completed, accepted QUERY: `X`, `f_W(X)` and the 32-byte digest;
   * on the managed path, only after RuntimeCore accepted the identity (`Ok(identity)`).

   It never reads a refused or withheld answer, and it never observes a LEARN.
2. **No new access.** The observer:
   * does not link, import or call K1 or RuntimeCore;
   * adds no K1 crossing;
   * does not touch W/H/a, continuity or LEARN;
   * performs no I/O, allocation, synchronization or device work.
3. **Exact identity and learned representation are kept separate in the record:**
   * a truncated state-identity field (digest bytes, copied);
   * a bounded input-identity hash (exact, non-cryptographic);
   * a learned-value payload (an encoding of `f_W(X)`).

   Identity bits are never presented as information about W's function.
4. **Bounded, fixed record.**
   * Only the first `min(rows, 64)` answer values are encoded, so the work is constant beyond 64 rows.
   * Rows beyond the window are a declared loss.
   * The record has a fixed width of 64 bytes and an independent format version.
5. **Two encoders, both with a fixed 256-bit payload:**
   * `GRAY4`: a fixed-threshold 16-level quantization, Gray-coded, 4 bits per row.
   * `SPROJ`: a fixed sparse ±1 projection of the normalized values plus a threshold, 256 bits.

   Calibration (thresholds, normalization, projection thresholds) uses DESIGN worlds only and is frozen into a pinned
   law blob before DEV. A raw binary16/binary64 readout at the same 256-bit budget is the matched comparator. It is
   not an encoder.
6. **Prototype location.**
   * `research/ecs_binary_observer_r0`: a research C library, a Python reference law, a laboratory, and a
     direct-native performance driver.
   * Nothing in `src/`, `native/runtime` or the K1 library changes.
   * No production wiring.
   * The handoff is a caller-allocated, fixed-capacity single-producer/single-consumer ring with an explicit
     `DROP_NEWEST` overflow policy. It exists only in the prototype.

## Consequences

* Feasible as an observation point. The record observes one answer of the forward map on caller-chosen inputs. It
  does not observe the state, its history, its retained H/a, or any hidden layer.
* Information that is structurally unavailable: anything not in `f_W` on the observed rows. Because `f_W` is
  invariant under neuron permutation, W itself is never identifiable from any number of answers.
* The scientific usefulness question remains open. It is decided only by the preregistered probes against
  controls, at a matched budget.

## Status

ACCEPTED for Phase B under the constraints above. It grants nothing to the runtime.
