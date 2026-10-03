# DSV4.1 Native Clock R0 and Native Materializer R1

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

Frozen turn conditioning uses the config layout `CONFIG_V2`
(`abi_version = 2`): a tail of `const float *conditioning; uint32_t
conditioning_count, reserved2`. `conditioning_count` is 0 or `dimension` and
requires granted `FEATURE_CONDITIONING`; values must be finite. The clock copies
them at create and sends them once in `STREAM_OPEN`, never per token. A
`CONFIG_V1` caller passes the struct prefix ending at `max_new_tokens`; the clock
never reads past it. The clock ABI version stays 1. `run_principal(...,
conditioning=...)` preprojects through the target and calls the factory with
`conditioning=<vector>` only when conditioned.

## Ownership and binding

```
Pinned Python control-plane owner
  +-- sealed execution DSO --> runtime --> attached provider context/model/cache
  +-- sealed clock DSO --> bounded clock state/trace storage
  +-- sealed Elpis materializer DSO --> host service context and authority
        (R1 production: elpis_dsv41_materializer, see "Native Materializer R1")
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
construction and bank identity stay with authority intake. The R1 production
materializer (below) natively implements the row codecs and BF16 rounding.

The R0 test-only materializer receives cold-verified decoded banks and canonical
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

Native Clock R0 itself makes no production Python-free file-materialization
claim and no accelerator speedup claim from the CPU reference provider. The
production file-backed path is Native Materializer R1.

## Native Materializer R1

`native/inference/include/elpis/dsv41_materializer.h` (ABI v1) is the sealed
production Elpis host service behind the **unchanged** `elpis_dsv41_materializer_v1`
table; the clock, YTS wire protocol and generic execution port are unchanged.
It is the pinned library `elpis_dsv41_materializer`, embedding the
substrate-generic native file-asset page service (`fms_file_service.h`) and
FMS core; its exported surface is only the materializer ABI and the pure row
codec (linker version script).

### Two phases

1. **Cold admission (Python control plane, `FileMaterializer`).** Inputs are
   the admitted target's `FMSFileAssets`, `RowEngine` tables and `TensorStore`
   bindings. For each row table and file-backed expert asset,
   `FMSFileAssets.transfer_asset` lends the retained descriptor (stamp
   re-checked under the provider lock) with the pinned size, page size, page
   map and identity stamp bound by `register` against the deployment catalog.
   The service duplicates it (`F_DUPFD_CLOEXEC`), refuses writable,
   non-regular, changed or mis-mapped objects, and never sees a path. Banks
   (layer, pinned bank digest, rows, dimension, codec, asset, offset,
   RowEngine bounds) and experts (three `(asset, offset, size)` roles and
   admitted binding digests, exact canonical image geometry) are added, then
   `seal` ends admission. Admission creates no authority; it consumes
   `FMSFileAssets`'.
2. **Runtime (native).** `rows`, `expert`, `release`, `quiesce` only. No
   discovery, no admission, no Python.

### Ownership

```
Python control plane (cold only)
  FMSFileAssets --transfer_asset (borrowed fd)--> FileMaterializer --create/admit/seal--+
  RowEngine / TensorStore records ---------------------------------------------------- |
                                                                                       v
elpis_dsv41_materializer identity (64 live, never reused; clock context = identity)
  +-- file service: duplicated read-only fds, page maps, stamps, private FMS ctx
  |      (WARM budget, max pages, page-staging bound, LRU page table, range leases)
  +-- banks, expert table, one row scratch, one expert staging slot
NativeClock --table--> rows/expert/release/quiesce        provider: decoded bytes only
```

The accelerator provider receives neither the table, the descriptors, page
maps, asset identities nor FMS authority. It cannot select page maps or
ranges: the clock requests only validated `(layer, expert, offset, length)`
from a NEED that it has already checked against residency and the plan.

### Runtime semantics (exactly the Python oracle)

* **Pages** (`FMSFileAssets._load/acquire`): stamp check on every load (also on
  resident hits) → `INTEGRITY`; bounded `pread` loop, `EINTR` retried, zero-byte
  return or error → `IO`; page digest mismatch → `INTEGRITY` (nothing
  registered); stamp re-check → `INTEGRITY`; FMS registration refused
  (NOMEM/LIMIT/UNSUPPORTED) → evict the least-recently-used unleased page and
  retry, none → `LIMIT`; a range wider than the WARM budget → `LIMIT`. Any
  failure releases every lease the call acquired; loaded pages may stay
  resident, unleased.
* **Rows** (`RowEngine.lookup` + `decode_row`): batch and output bounds →
  `LIMIT`; bank identity/dimension → `INTEGRITY` (Python `IDENTITY`); row
  outside the bank → `INVALID`; sorted, deduplicated physical reads; output in
  request order, F32 little-endian. `F32_LE` copies bits and rejects
  non-finite values. `DS4_E4M3_E8M0_BF16` rejects codes with `(c & 127) == 127`
  and scale 255, decodes `ldexp(base, scale-127)` with sign, rejects overflow,
  applies the uint32 BF16 round-to-nearest-even of the oracle and rejects
  non-finite results; all as `ENCODING`.
* **Experts** (`TensorStore.stage_image_range/_copy`): canonical
  `w1 || w3 || w2` ranges across tensor and page boundaries, copied through
  range leases of at most half the WARM budget; `length > staging budget` →
  `LIMIT`; a borrowed span outstanding → `BUSY`; out-of-range/invalid →
  `INVALID`; resident experts → `INTEGRITY` (never staged). The span carries
  the three admitted binding digests. One bounded staging slot; no host
  expert cache.

Clock codes are the clock ABI's; Python `IDENTITY` maps to `INTEGRITY`,
`MISSING`/`UNSUPPORTED` to `INVALID`.

### Spans, leases, cancellation and lifecycle

FMS page leases never outlive a call: a successful acquisition copies verified
bytes into service-owned scratch and returns a span whose lease is a
generation token. At most one span is borrowed at a time.

| Operation | Contract |
|---|---|
| `rows`/`expert` OK | One span borrowed; no FMS lease outstanding. |
| BUSY / DEFER / LIMIT / failure | Nothing retained (no span, no lease). |
| `release` | Exactly once; a stale or forged span is a counted no-op. |
| `quiesce` | Bounded, synchronous, idempotent: interrupts in-flight acquisitions at their next page boundary (they return DEFER retaining nothing), waits for them, invalidates any borrowed span and leaves no FMS lease. The service stays usable. |
| `destroy` | BUSY while a call is in flight or a span is borrowed (and interrupts in-flight work); idempotent for issued identities; STALE otherwise. |
| After destroy | Every table entry returns STALE or is a no-op without dereferencing freed memory; a clock bound to it fails STALE. |

Clock cancellation is unchanged (observed between row blocks/submissions).
The materializer adds interruption only through `quiesce`/`destroy`.
Independent services share nothing mutable; the identity table mutex is held
only for lookup, never across I/O.

### Failure dispositions through the clock

| Fault | Clock outcome / state | Resources |
|---|---|---|
| Changed or truncated row asset | FAILED `INTEGRITY`, RELEASED (before TOKEN_BEGIN) | no span/lease; model retained |
| Changed/truncated/corrupt expert asset | FAILED `INTEGRITY`, DISCARDED | no span/lease; model retained |
| Invalid FP8 code / E8M0 scale / overflow | FAILED `ENCODING`, RELEASED | no span/lease |
| Expert staging pressure past deadline | MATERIALIZATION_NEEDED, then FAILED `LIMIT`, DISCARDED | no span/lease |
| Destroyed service | FAILED `STALE`, RELEASED | no dereference |
| Provider/STREAM_RELEASE failure | QUARANTINED, logical result preserved | service quiesced, no span/lease |
| Cancellation before/after acquisition | CANCELLED (RELEASED/DISCARDED) | no span/lease |

### Bounds and accounting

WARM budget, max pages, page staging (`page_size <= min(staging/4, warm)`),
page-map storage, range-lease table, expert staging budget, row scratch and
the identity table are fixed at create. Statistics report resident
current/high-water and pinned bytes, page hits/misses, `pread` bytes and
calls, semantic bytes, row/expert bytes, page and span leases, evictions,
forced releases and per-acquisition latency/resident samples. Buffered `pread`
does **not** charge or bound the Linux page cache.

### Qualification

* `substrate.test_fms_file_service`: raw-digest identity against Python,
  admission refusals (bad/closed descriptor, writable, directory, pipe,
  stamp, geometry, staging bound, allocation failure, descriptor closure),
  LRU order, all-pages-leased and over-budget `LIMIT`, eviction `BUSY`, stale
  range handles, interrupted/short/EIO/fragmented reads, corrupt page,
  changed and truncated objects, interrupt, concurrency.
* `inference.test_dsv41_materializer`: exhaustive codec (every code × scale)
  against an independently formulated reference, rows/experts/lifecycle,
  faults, quiesce/destroy races, independent instances.
* `inference.test_dsv41_clock_production`: real runtime + reference provider;
  bitwise COMPLETE bodies, tokens and row identities against the R0 test
  materializer for F32 and FP8 banks, parts 128/1152, provider cache, and a
  concurrent quiesce race; fault table above.
* `tests/inference/dsv41/test_native_materializer.py`: exhaustive codec
  differential against `decode_row`; rows and expert ranges bitwise against
  `RowEngine.lookup`/`stage_image_range` with equal hits, misses, `pread`
  bytes, reads, semantic bytes, lease counts and LRU resident page order
  (including eviction pressure); complete Native Clock runs bitwise against
  the Python YTS oracle and the native backend under traps on
  `AddressScheme.stream_hash`, `RowEngine.lookup`, `FMSFileAssets.acquire/_load`,
  `RangeLease.read/readinto`, `TensorStore.stage_image_range/_copy`,
  `DSV41StreamProvider.exchange`, `os.pread` and host arithmetic, with a
  profiler positive control and zero Python frames during `clock_advance`;
  Principal commit/replay/stop/YIELD equality; fault dispositions equal to
  the oracle; provider and release faults; lifecycle/stale/busy; concurrent
  independent instances; characterization.

R1 changes no tolerance: tower arithmetic keeps the existing qualification;
row and expert bytes are exact.

### Nonclaims

No vendor accelerator provider ships; all provider runs use the test-only CPU
reference provider. No learned parameter artifact or trained Engram table is
admitted: R1 qualifies the runtime path on deterministic fixtures, not model
quality. The Python implementation remains the oracle. Characterization
numbers are CPU-reference measurements, not speed claims. Linux page-cache
bytes are not accounted. Cold admission, authority and final Principal
result construction stay in Python by design.
