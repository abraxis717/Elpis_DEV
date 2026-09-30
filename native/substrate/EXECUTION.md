# Bounded native execution R0

`elpis_execution` is one C11/pthreads shared library in the substrate. It has no
accelerator SDK, Python dependency, background device probe or global worker pool.
All adapters share its resource accounting. Runtime contexts reserve 1..4 workers
from a four-worker module budget; creating additional contexts cannot exceed it.
Reservations are returned after join, including partial startup failure. Default
workers are `min(3,max(1,available_cpus-1))`, using Linux process affinity before
online CPU count. Applications with other native pools must budget those separately;
this API deliberately does not wrap NumPy, BLAS or the existing Python RowEngine.

The fixed slot window counts queued, running and finished/unretired tasks. Each
worker has a FIFO list of slot indices, assigned by caller affinity. No stealing,
busy-spin, malloc per queue insertion or growing completion map. Capacity is
1..4096; input/output limits are each <=64 MiB and the maximum sum of retained
input/output payloads is <=512 MiB per context. These limits exclude caller-owned
unsubmitted buffers, operation scratch, pthread stacks and small metadata. The
Regex adapter inherits the existing bounded lexer/evidence scratch allocation;
it is not a total-RSS allocator or an incremental source transport API.

Worker metadata and task slots have 128-byte alignment and stride assertions;
shared metrics are separately aligned and lock-protected. This is conservative
false-sharing isolation, not a hard-coded cache capacity assumption. Input spans
remain contiguous and immutable. Affinity keeps related independent work on the
same worker. Queue synchronization is a short common mutex critical section;
compute and device operations run outside it. No domain/authority lock is acquired.

## Ownership and progress

Allocate/fill `elpis_exec_buffer`, then submit an explicit task and `buffer **`.
Successful submit consumes that reference and NULLs the caller variable. Failure
retains caller ownership. A producer may retain an additional reference before
sharing; first successful submission seals mutable access. Previously acquired
writable pointers must not be used after sharing (C cannot revoke raw pointers).
Workers cannot receive arbitrary borrowed source pointers through this ABI.

Callbacks return one owned result buffer, including on failure; the runtime
releases failed outputs. Successful `take` transfers result ownership independently
of runtime lifetime. Cancellation and destroy release all retained references.
Descriptors are copied, large input buffers are not. Output materialization is an
explicit adapter cost, not a queue-crossing source copy.

Sequence identifiers reflect successful submit linearization. One consumer retires
results in that order; stage/tag fields are echoed. Concurrent producer scheduling
itself is not deterministic. A slow first result can intentionally fill the window
with completed later work; producer gets WOULD_BLOCK and must consume or retry
from its own event loop. No implicit wait/inline execution. `take` can poll or sleep
for at most 60 seconds using a monotonic condition variable. Slot capacity is
released only by ordered retirement, not compute completion.

Cancellation unlinks only queued work and produces one ordered CANCELLED result.
Running work returns WOULD_BLOCK and finishes normally. Shutdown closes admission,
drains or cancels queued work, joins workers and leaves results available. Joining
does not require a consumer even with a full result window. One lifecycle owner
must stop other API callers before destroy. Callbacks must terminate normally;
status failure is isolated, process faults/undefined behavior/pthread_exit are not.
No arbitrary thread interruption is attempted.

## Optional backend

NULL backend means a zero capability mask and no initialization or poll. A supplied
backend initializes once; failure disables routing but does not fail CPU startup.
Only explicit PURE tasks with a supported operation bit are eligible. submit
rejection retains no token. Only accepted tokens are polled, with a configured
finite poll count (<=1000) and 1ms sleeps. Terminal poll consumes the token. A
timeout calls abort, which MUST quiesce device access synchronously, before CPU
fallback. Failed partial outputs are discarded. Side-effecting tasks never retry
on this route. Backend callbacks must be bounded, thread-safe and nonblocking;
an actual device implementation must qualify these contracts and exact CPU parity.
The base build contains no real device backend, PCIe enumeration or vendor runtime.
No performance or disappearance guarantees are claimed for an unqualified backend.

## Initial adapter and boundaries

`elpis_regex_execution` schedules independent complete byte sources. Every task
creates its own existing V2 lexer, feeds <=64 KiB spans serially, finalizes, and
returns the full canonical JSON. It retains existing default grammar, evidence and
source semantics. Input is memory only. The adapter grants no semantic/admission
authority and does not schedule query-overlay publication. The original APIs and
Python runtime are unchanged; opt-in native hosts pass `elpis_regex_execute` as
their compute function. There is no callback into Python or new Python thread pool.

ECS mutation, append, fsync, replay application and inference token recurrence stay
serial. HACF scoring remains unchanged because a new adapter must first prove its
leased-span lifetime and benefit without holding a shard gate around expensive
parallel work. Inference rows/experts remain outside this budget because existing
Python/NumPy pools and numerical contracts need separate qualification.

## Tests and measurements

Root CMake/CI automatically includes the runtime, Regex parity and benchmark smoke
tests in gcc/clang Debug/Release and ASan+UBSan/TSan matrices with warnings as errors.
`test_execution` uses a separate test-only archive to inject pthread_create failure;
the production ABI has no injection hook. Test gates force out-of-order completion
and queue pressure without timing-dependent cancellation assumptions. Six producers
check exactly-once delivery. Fake backends test capability selection, rejection,
initial absence, disappearance, timeout/quiescence and output equivalence.

`benchmark_regex_execution WORKERS FAMILY TASKS` prints JSON. WORKERS=0 calls the
original V2 API directly; 1..4 use the runtime. FAMILY is small or large; data derive
from existing Regex fixtures including no-match, long whitespace, ambiguity and
invalid UTF8. Setup and producer input copying are included. The runtime queue
itself never copies inputs. Wall, process CPU, source/output bytes, throughput,
time to first ordered result, p50/p95 retirement latency, compute/queue delay,
high-water and process peak RSS are reported. Allocator calls inside the existing
lexer are not instrumented. No inference acceleration is inferred from this test.
