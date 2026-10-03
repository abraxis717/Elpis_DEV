# DSV4.1 Yielded Token Stream, R0 (YTS-R0)

YTS-R0 runs the whole DSV4.1 tower through a post-compiled accelerator provider. It is
built entirely above the existing generic execution port: `native/substrate/**` is
unchanged. Elpis keeps:

- FMS as the only reader and verifier of model asset bytes;
- Engram row authority and decoding;
- bounded host staging;
- the terminal semantics of `ELPIS_EXEC_BACKEND_ONLY`.

The provider owns everything on the accelerator side: attention caches, the token
continuation and expert compute.

This document describes what is implemented. The architecture decision is
`DSV41_ACCELERATOR_STREAM_ADR` (Yielded Token Stream R0, `NO_GENERIC_PORT_CHANGE`).

## Shape

All traffic uses ordinary `BACKEND_ONLY` submissions of one adapter-defined operation
(`ELPIS_DSV41_STREAM_OPERATION = 41`). The message kind lives in a fixed 128-byte
little-endian header. Each R0 runtime has:

- `workers=1`
- `capacity=1`
- one active stream
- strict alternation: submit, take, validate, then the next submit

The message sequence is:

```
MODEL_ADMIT_BEGIN, MODEL_ADMIT_TENSOR* (resident tensors + RoPE tables), MODEL_ADMIT_END -> ADMIT_ACK
STREAM_OPEN -> OPENED
per token:  TOKEN_BEGIN -> NEED | COMPLETE
            EXPERT_SUPPLY (one part) -> NEED | COMPLETE        (repeat)
STREAM_RELEASE -> RELEASED
MODEL_RELEASE  -> MODEL_RELEASED
```

A materialization yield is the completed, successful result of a submission (a `NEED`).
While Elpis reads FMS, no port token is outstanding. FMS work happens between
submissions, on the principal thread. It never runs inside a provider callback or on a
pool worker, and the provider never calls back into Python.

Wire layout and the provider attachment ABI: `research/dsv41_tower/native/include/elpis/dsv41_stream.h`.
Pure codec and validation: `research/dsv41_tower/stream_protocol.py`.
Host adapter: `research/dsv41_tower/provider_stream.py`.
Port binding: `src/elpis/substrate/execution.py`.

## State machine (implemented)

Runtime states:

- `ATTACHED`, then `READY` after a valid `ADMIT_ACK`.
- `MODEL_RELEASE` returns the runtime to `ATTACHED`, and the same runtime may admit again.
- `close` leads to `CLOSED`.
- Any provider-originated failure leads to `QUARANTINED`, which is terminal.

Stream states:

- `BOUNDARY(p)`: the stream sits between tokens.
- `IN_TOKEN`: the host is preparing a token, has a submission in flight, is parked on a
  `NEED`, is staging, or is validating `COMPLETE`. A valid `COMPLETE` returns the stream
  to `BOUNDARY(p+1)`.
- Terminal states:
  - `RELEASED`: normal release;
  - `DISCARDED`: host-originated failure after `TOKEN_BEGIN`;
  - `QUARANTINED`.

### Failure classes

| When | Provider state | Disposition | Surfaces as |
|---|---|---|---|
| Before `TOKEN_BEGIN`: hashing, Engram row lookup or decode, or cancellation | untouched | `STREAM_RELEASE` (normal); model and cache kept | the row/FMS code, or `CLOSED` |
| Host-originated after `TOKEN_BEGIN`: FMS `INTEGRITY`/`IO`/`MISSING`, `LIMIT`/`BUSY` past the host deadline, cancellation | advanced, mid-token | `STREAM_RELEASE` (discard); model and cache kept | that code, or `CLOSED` |
| Provider-originated: rejection, terminal poll, timeout, or a malformed or non-finite successful reply | unknown | Quarantine: `elpis_exec_destroy` (shutdown with cancellation, then `backend.shutdown`), then detach. Nothing survives. | `DEVICE`, `INTEGRITY` or `ENCODING` |

A provider sequence is never continued on CPU. The only recovery is a fresh principal
sequence from committed state.

Transient FMS `LIMIT`/`BUSY` while parked is retried until the host's own deadline
(`staging_deadline_s`, default 5 s). Retrying is safe because nothing is in flight and
waiting does not change provider state.

## Engram

1. Before `TOKEN_BEGIN`, the host computes `scheme.stream_hash`.
2. It then runs every `RowEngine.lookup`. These are FMS leases with page verification
   and E4M3/E8M0 to BF16 decoding.
3. The decoded F32 rows go into the `TOKEN_BEGIN` body, for exactly the admitted Engram
   layers in ascending order.

There is no Engram yield. A row failure therefore happens before the provider advances.

## Experts and FMS

- **What the provider names.** Only `(layer, expert index)`, where index `expert_count`
  means the shared expert. It never receives an asset ID, path, descriptor, offset, page
  map or authority.
- **Validation.** Elpis validates every `NEED` before acting on it:
  - stream, epoch, sequence number and position echoes;
  - strictly increasing layers;
  - expert bounds and uniqueness;
  - `need ⊆ sorted(selected) ∪ {shared}`, excluding resident experts;
  - the cache-hit map;
  - the cursor and part order;
  - per-token reply and supply bounds.
- **Resolution and reading.** `TensorStore.expert_image(roles)` gives the binding digests
  and sizes. `TensorStore.stage_image_range` copies an arbitrary byte range of the
  canonical image `w1 ‖ w3 ‖ w2` through FMS range leases, directly into the port request
  buffer. FMS verifies every page it loads. Staging is accounted against the same budget
  and busy guard as `expert()`. There is no host expert cache.
- **Chunking.** An expert can be supplied in parts of at most `part_bytes`. This matters
  because one production F32 expert is 135 MiB, above the port's 64 MiB buffer limit.
  The provider assembles the parts in one bounded expert slot.
- **Accumulation order.** Ascending routed expert ID, then the shared expert, with the
  route weight applied before `w2`. The order is the same whether an expert was supplied,
  cached or resident. The route trace is kept in router-score order.
- **Optional provider cache.** It is filled only from Elpis-supplied bytes. Entries are
  immutable, carry their binding digests and are bounded by the admitted byte budget. A
  re-supply under a different binding identity is refused. The cache lives no longer than
  one model admission and is cleared by `MODEL_RELEASE` and by quarantine. A layer whose
  experts are all cached or resident is elided: it yields no `NEED`, and its route is
  reported in `COMPLETE`.

## Derived tables and numerical identity

RoPE tables (host NumPy `cos`/`sin`) are admitted as digest-bound tensors. The provider
never recomputes them.

A provider-backed target has its own `numerical_profile`: the base profile plus
`{protocol, provider_library SHA-256, kernel_profile}` from `ADMIT_ACK`. Its
`model_identity` is unchanged. A provider commit therefore never replays on the CPU or
native target: `begin` fails with `UNSUPPORTED`.

## Provider attachment

The provider exports four functions:

- `elpis_dsv41_stream_provider_abi_version`
- `attach(host table, &elpis_exec_backend)`
- `bind_runtime(context, runtime)`
- `detach(context)`

Elpis seals and loads every native object with `RTLD_LOCAL`. `execution.h` forbids a
provider from linking its own copy of the execution runtime. The host table therefore
hands the provider the buffer and notify entry points of the single sealed runtime
instead.

This is the only new attachment surface. It is DSV4.1-stream specific and narrow, so a
future generic post-compiled attachment layer can absorb it without changing YTS
semantics.

The host owns the attached context through detach. If runtime construction fails,
it detaches that context; if runtime binding fails or raises, it first destroys the
runtime (quiescing callbacks and notifications), then detaches. Successful owners
must be closed explicitly. The sealed loader retains both DSOs and their sealed
FDs for process lifetime, independently of Python object collection; close never
unloads code. Generic Python runtime handles are borrowed until destroy, and
post-destroy access is refused before entering C. Other API users must stop before
destruction; these rules do not add concurrent close or multiple YTS streams.

## Reference provider (test-only)

`research/dsv41_tower/native/tests/dsv41_reference_provider.c` implements YTS-R0 behind the real
`elpis_exec_backend`. It compiles the qualified DSV4.1 kernel sources with the same
`-ffp-contract=off` policy and calls them in the order `DSV41NativeBackend.apply_layer`
uses. It is registered as a test target, never as a production library.

Test controls:

- synchronous mode;
- a fake asynchronous device thread, optionally with notify;
- fault injection by request index and kind;
- live-resource counters;
- a completion-to-observation delay probe.

## Qualification

| Q | What | Result |
|---|---|---|
| Q1 | 24 positions, 7 layers | **Bitwise** equal to `DSV41NativeBackend`: logits, every layer stream, route trace, selected positions, attention counts, history. Against NumPy: worst 2.98e-7 on streams and 8.34e-7 on logits (rtol 1e-4, atol 1e-5), identical argmax. |
| Q2 | 20 ms FMS delay per supply, poll budget 3 | Bitwise identical; polls equal submissions; no timeout. |
| Q3 | 3× `LIMIT` before every supply; persistent `LIMIT` | Transient: bitwise identical, no extra polls. Persistent: `LIMIT`, stream `DISCARDED`, model `READY`; a fresh stream on the retained admission is bitwise identical. |
| Q4 | Expert page corrupted after admission | `INTEGRITY` at the first staging after `TOKEN_BEGIN` (exactly 2 messages); stream discarded; model and slot kept; 0 pins; 0 staged bytes. |
| Q5 / Q13 | Faults on every message of a token, sync and async | 37 sync cases (poll failure at all 22 messages; reject, bad echo and timeout at the boundaries; bad need, bad cursor, NaN) and 4 async cases. Every case quarantines with the right code, `backend_fallback == 0`, no host arithmetic, and zero live provider objects. |
| Q6 | Cancellation at `BOUNDARY`, `PREPARING`, in flight, `PARKED`, `STAGING` | As specified (see deviations). |
| Q7 | Parts of 128 B, 384 B and the whole image; port input 4 KiB | Bitwise invariant. Host staging high-water = 128 + 144 + part. Provider slot high-water = 1152 B. |
| Q8 | Cache budgets 0, one image, all experts; then a second model differing in one expert on the same runtime | Bitwise invariant across budgets. Elided layers equal the analytic count. After `MODEL_RELEASE` the cache is empty, the first token of model B has zero hits, and B is bitwise equal to B-native. |
| Q9 | Engram | `TOKEN_BEGIN` carries exactly the host-decoded rows of layers 1 and 5. A row failure sends no `TOKEN_BEGIN` and releases the stream normally. |
| Q10 | Topology | Ratio-0 (layer 0); ratio-2 owner, index consumer and pure consumer (1–3); ratio-1 owner and candidate source, masked consumer and pure consumer (4–6); group-start positions included. |
| Q11 | `PrincipalEngine` | Outputs and trace equal the native run; replay on the provider is identical; the CPU target refuses with `UNSUPPORTED` / `IDENTITY`. |
| Q12 | Traffic | Only YTS messages. Per fixture token: 22 submissions, 31,608 B host→provider, 522,276 B provider→host (including 896 B of observed layer streams). Matches the analytic formula and the provider's own byte counters. No NumPy or synchronous native arithmetic. |
| Q14 | Notify | Completion observed 38–59 µs after the device finishes with notify, ~981–1007 µs without. A 2 s device step against a 20-poll budget times out, and the quarantine follows. Abort interrupts the step, so the token returns in under 1 s instead of waiting it out. |
| Q15 | C protocol test | Passes in the gcc/clang Debug/Release and ASan+UBSan/TSan matrices. |
| Q16 | Port | `EXECUTION.md`, `execution.h` and `execution.c` are byte-identical to `ba1e4f2`. |
| Q17 | `ELPIS_DSV41_QUALIFY=1` | The donor differential, tower and YTS suites all pass with 0 skips. |

### Deviations from the ADR, with reasons

- **Q14 criterion.** The ADR asked for "≤ 2 polls per submission with notify". That only
  holds for device steps shorter than the runtime's own ≥ 1 ms re-poll interval: a longer
  step collects timer polls with or without notify. The `COMPLETE` step here computes the
  129,280-row vocabulary head, which can take a third poll. The qualified criterion is
  what notify actually provides: the completion-to-observation delay, measured in the
  provider. Poll counts are still reported. Every `NEED` step took at most 2 polls.
- **Cancellation in `PREPARING`.** The provider stream is released normally, exactly as
  the ADR specifies. But the token surfaces as a typed `CLOSED` failure rather than a
  stop, because `principal.py` is frozen and `window_step` cannot end a sequence early.
- **Host staging depth.** It is 1 (no pre-staging of the next part while one is in
  flight). The ADR allowed depth 1 or 2.
- **`release_window` never raises.** A failed `STREAM_RELEASE` quarantines the runtime
  and is recorded in `provider.failure`, so `finalize` and the failure paths of the
  principal keep their result.
- **Re-admission.** The same runtime may admit a model again after `MODEL_RELEASE`. Q8
  uses this to prove cross-model cache isolation.

## Nonclaims

- No vendor provider ships. The reference provider is CPU-only and test-only.
- R0 has one active stream per runtime and no pipelining.
- No packed or FP4 format, no FMS device-domain lease, and no mid-token rollback.
- No performance claim. At production scale, per-token expert streaming is viable only
  with a high provider cache hit rate or a packed expert format.
