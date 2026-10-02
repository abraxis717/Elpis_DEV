# DSV4.1 Native Clock R0

The opt-in native clock is a recurrence controller, not another numerical
backend or scheduler. The Python YTS, target and Principal implementations
remain the oracle. Wire operation 41, YTS-R0 and the generic execution runtime
are unchanged.

`native/inference/include/elpis/dsv41_clock.h` is ABI v1. Cold admission happens
through `DSV41StreamProvider`. `NativeClock` then borrows exclusive use of that
idle, admitted runtime. A single `advance` can execute the entire prefill and
generation without an interpreter callback. The optional `run_principal`
control-plane helper validates Principal inputs and uses the existing
`PrincipalEngine.finalize` only after native recurrence returns.

## Ownership and binding

```
Pinned Python control-plane owner
  +-- sealed execution DSO --> runtime --> attached provider context/model/cache
  +-- sealed clock DSO --> bounded clock state/trace storage
  +-- sealed Elpis materializer DSO --> host service context and authority
```

The clock DSO does **not** link `elpis_execution`. Its host table contains the
buffer, submit, take, metrics and shutdown entry points of the already-loaded
sealed execution DSO. It always submits BACKEND_ONLY with a NULL compute
callback. The execution runtime owns worker scheduling and provider polling.
There is no CPU replay path, device discovery, backend registry or new pool.

The config arrays and service context are immutable borrows through clock
destruction. The Python wrapper retains their owners. Handles are monotonically
assigned integer identities in a bounded table of 64 live clocks; identities
are never reused. Stale access is refused without dereferencing freed memory.
Repeated destroy of a previously issued handle succeeds. The handle table
does not select or register providers.

One caller owns advance/read/stop/close. `cancel` may race advance. A concurrent
close or destroy requests cancellation and returns BUSY: join the advancing
caller and retry. Diagnostic trace views are borrowed until the next mutation
or destruction; callers must serialize access. Runtime destruction by another
owner while a clock borrows it violates the ABI.

## State and coarse results

```
CREATED --open--> BOUNDARY --prepare rows--> PREPARING
                        --TOKEN_BEGIN--> IN_TOKEN
                        <--COMPLETE-----+
IN_TOKEN --NEED--> PARKED --expert part--> IN_TOKEN
```

`advance(handle, token_budget, metrics)` counts completed input positions,
including prefill. Its semantic outcomes are:

| Outcome | Meaning |
|---|---|
| PROGRESS | The completed-token budget ended at a boundary. |
| MATERIALIZATION_NEEDED | A native host service deferred; retry advance. No lease or port submission is outstanding. |
| COMPLETE | Prefill and the requested number of generated tokens completed. |
| STOP_TOKEN | A generated stop token completed its tower step and entered output/history. |
| STOPPED | Explicit graceful stop at a completed-token boundary, retaining trace/output. |
| CANCELLED | Cancellation acknowledged; partial token discarded. |
| FAILED | Typed host or provider failure. |

Greedy selection rejects non-finite logits and retains the first index on ties,
as NumPy argmax does. Prefill consumes context then prompt without recognizing
stop tokens. A generated token is published only after its COMPLETE validates;
stop recognition follows that step, including at the max-token boundary.
History advances only after success. `stop` is graceful only at a boundary;
`cancel` is the operation for abandoning a partial token. `run_principal` also
supports a deterministic `stop_after` boundary for YIELD commits/replay.

The clock validates headers, exact body lengths, echoes, reply kinds,
monotonic layers, selected/needed membership and ordering, residency/cache-hit
claims, unchanged within-layer plans, part cursors, per-token reply/supply
bounds, COMPLETE geometry/routes/elisions/positions, and finite outputs.
Header integers and floats are explicitly decoded little-endian.

## Failures and cleanup

| Origin/boundary | Stream | Model/cache | Runtime |
|---|---|---|---|
| Cold create rejection | Not opened | Unchanged | Borrow remains with owner |
| Host preparation failure before TOKEN_BEGIN | Normally released | Retained | Usable |
| Host failure/cancellation after TOKEN_BEGIN | Discarded via STREAM_RELEASE | Retained | Usable |
| Completion/stop/normal close | Normally released | Retained | Usable |
| Submit/take/terminal provider failure, timeout or malformed success | Quarantined | Freed by native shutdown | Shut down, unusable |
| Failed stream release | Quarantined | Freed by native shutdown | Shut down, unusable |

Native shutdown quiesces the provider before FAILED returns from C. The external
owner then destroys the borrowed runtime and detaches the provider. The Python
wrapper performs this final ownership cleanup before its `advance` returns a
failure; C callers have the same documented obligation. Clock destruction frees
all clock-owned buffers. Closing a provider also closes its native clock.

Release failure is independently recorded in `provider_code` and QUARANTINED
state. It preserves the already determined logical outcome/code: a completed
Principal commit remains completed, and an earlier host error remains that
error. This matches the oracle's non-raising `release_window` behavior.

The clock observes cancellation before token preparation, after each acquired
row block, before submission, and after ordered retirement/validation. It does
not asynchronously interrupt a provider token or claim rollback. Provider
failure takes precedence over cancellation. Exchange waits and materialization
pressure have separate monotonic deadlines; the runtime retains its own poll
budget. Shutdown relies on the existing bounded backend/abort contract.
An exchange deadline marks failure; quiescence can extend to the runtime's
bounded provider completion/poll budget. It is not an asynchronous device abort.
Any completed result retained by shutdown is retired and released natively.

## Native materialization service

This is an **Elpis host** service. The accelerator provider receives neither
the table nor filesystem/FMS authority.

* `rows`: pinned bank digest, layer, ordered row IDs, count and dimension;
  returns decoded F32 little-endian values in request order.
* `expert`: validated layer/expert, canonical image offset and part length;
  returns `w1 || w3 || w2` bytes and three binding digests. Resident experts
  cannot be requested. Parts may cross tensor boundaries.
* `release`: releases each successful borrowed span exactly once, before a
  submission or coarse return, including malformed/non-finite host spans.
* `quiesce`: bounded, synchronous, idempotent cancellation of host service work;
  leaves no leases. The caller retains ownership of the service context.

BUSY/DEFER retain nothing. Expert LIMIT is transient staging pressure, matching
the Python staging semantics, and becomes terminal LIMIT at the host deadline.
Rows still fail before TOKEN_BEGIN. Retrying advance resumes the same row or
expert range; completed row blocks and expert parts are not repeated.
There is no separate supply API because the native service supplies the span
directly at this boundary. No neural primitive is exposed as a clock event.

The clock lowers active-token `AddressScheme.stream_hash`: mapped-token
history, dead-tail blocking, pad substitution, signed-int64-bounded products,
XOR, modulus, offsets and layer/column order. Cold normalization, prime-layout
construction and bank identity stay with authority intake. Row codecs and
BF16 rounding have **not** been reimplemented.

The test-only materializer receives cold-verified decoded banks and canonical
expert images. It exercises bounded part supply and leases, not a production
host expert cache. Its complete resident backing is an explicit synthetic-test
condition. The provider can still use one bounded expert slot and optional
cache. Neither test materializer nor reference provider is a production library
in `ELPIS_SYSTEM.json`.

## Bounds and diagnostics

Create preallocates token IDs, row identities, per-position COMPLETE bodies and
latencies, one request scratch and the token plan. The caller supplies a memory
budget capped at 512 MiB. Over-budget geometry is refused before open; output
storage never grows or silently truncates. Port request/reply limits remain
64 MiB each. Clock storage accounting excludes borrowed cold assets, runtime
slots, provider storage and temporary port buffers, each independently bounded
by its owner. The per-position full-logit trace is intentionally conservative
for R0 qualification and may require a smaller sequence budget at large vocab.

Metrics include completed position/output count, current/argmax token, YTS
sequence, submissions/polls, traffic, request allocations/consumption, released
outputs, service acquire/release/yield counts, failure disposition and timings.
Raw COMPLETE trace bodies preserve logits, route, attention selections/counts,
optional layer streams and provider telemetry for post-call inspection.

## Qualification and nonclaims

`test_dsv41_clock` uses the existing native reference-provider admission fixture
and real execution runtime, including chunking/cache/async and fault injection.
`test_native_clock.py` compares the unchanged Python YTS, native backend and
NumPy arithmetic, Principal commits/replay, active-token hashes and ownership.
Its profile instrument has a positive Python-entry control and surrounds a
direct multi-token C advance, alongside traps on Python recurrence services.

The test suite distinguishes the raw-token synthetic fixture from the original
production-shaped tokenizer fixture. The former uses the exact checked-in
recurrence methods on explicitly synthetic cold tensors and address parameters;
it makes no tokenizer/donor-environment claim. Required production/donor skips
are not qualification passes. Compiler, sanitizer and environment outcomes must
be reported with the run's external evidence, not inferred from checked-in code.

R1 still needs a sealed Elpis file-asset service: transfer already-pinned asset
descriptors/page maps, perform native bounded verified page loads and FMS lease
accounting, implement ordered row acquisition with the exact FP8/E8M0/BF16
decoder, and stage canonical expert ranges under the existing budgets. It must
qualify corruption, pressure, lease cancellation/quiescence and numerical parity.
There is no production Python-free file-materialization claim in R0, and no
accelerator speedup claim from the CPU reference provider.
