# Bounded native execution R1

`elpis_execution` is one C11/pthreads shared library in the substrate. It has no
accelerator SDK, Python dependency or background device probe. It owns exactly
one module-wide pool of at most four worker threads, which every runtime context
shares: Regex, HACF read-only scoring and future independent kernels consume one
bounded CPU budget. The pool grows on demand to the largest live context's
`workers` and is joined when the last context is destroyed (also after a partial
startup failure). Contexts never reserve private threads, so creating another
context cannot fail for lack of budget and total CPU concurrency stays <=4.
Default `workers` are `min(3,max(1,available_cpus-1))`, using Linux process
affinity before online CPU count. Applications with other native pools must
budget those separately; this API deliberately does not wrap NumPy, BLAS or the
existing Python RowEngine, and it never nests pools.

A context's `workers` is its concurrency cap (at most that many of its CPU tasks
run at once; `workers=1` keeps a context serial) and its number of affinity
lanes. The fixed slot window counts queued, running, parked and
finished/unretired tasks. Each lane is a FIFO list of slot indices (no malloc per
insertion, no growing completion map). A pool thread takes, in order: a parked
accelerator token that is due; the oldest task in its own lane of any context
below its cap; otherwise the oldest runnable task of any lane of any context
(stealing, oldest first across contexts, so no context starves). Affinity is a
locality preference only and can never strand work: a thread sleeps only when no
context below its cap has queued work and no parked token is due. No task has a
hard-affinity requirement; one would need an explicit API. Capacity is 1..4096;
input/output limits are each <=64 MiB and the maximum sum of retained
input/output payloads is <=512 MiB per context. These limits exclude caller-owned
unsubmitted buffers, operation scratch, pthread stacks and small metadata.

Worker metadata and task slots have 128-byte alignment and stride assertions.
This is conservative false-sharing isolation, not a hard-coded cache capacity
assumption. Input spans remain contiguous and immutable. Synchronization is one
pool mutex held only for queue/slot transitions, never across compute, backend
calls or buffer copies; contention is counted (`lock_contended`, `lock_wait_ns`).
A submit wakes at most one idle thread (its lane's thread when idle, otherwise
any idle one; a thread already being woken is not woken twice), and only when
the context is below its cap. Completion signals a context's consumer only when
the completed slot is that context's retirement head. No domain/authority lock
is acquired.

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
itself is not deterministic, and neither is internal completion order: a slow
task delays only the publication of later results, never their execution. A slow
first result can intentionally fill the window with completed later work;
producer gets WOULD_BLOCK (explicit backpressure) and must consume or retry from
its own event loop. No implicit wait/inline execution. `take` can poll or sleep
for at most 60 seconds using a monotonic condition variable. Slot capacity is
released only by ordered retirement, not compute completion. Each result reports
the executing pool thread, its completion time and its completion-to-retirement
wait.

Cancellation unlinks only queued work and produces one ordered CANCELLED result.
Running work returns WOULD_BLOCK and finishes normally. Shutdown closes admission,
drains or cancels queued work, waits until the context has no queued, running or
parked work and leaves results available. This does not require a consumer even
with a full result window. One lifecycle owner must stop other API callers before
destroy; the last destroy joins the pool. Callbacks must terminate normally;
status failure is isolated, process faults/undefined behavior/pthread_exit are not.
No arbitrary thread interruption is attempted.

## Optional backend

NULL backend means a zero capability mask and no initialization or poll. A supplied
backend initializes once; failure disables routing but does not fail CPU startup.
Only explicit PURE tasks with a supported operation bit are eligible. submit
rejection retains no token. Only accepted tokens are polled, at most the
configured number of times (<=1000). No worker ever sleeps or spins on a token:
submit is followed by one poll, and an undecided token is parked and polled again
by whichever pool thread is free, at least 1 ms later, while CPU work runs. A
backend with a completion event (fence, eventfd, interrupt) calls
`elpis_exec_backend_notify` to make its parked tokens due immediately. Terminal
poll consumes the token. Exhausting the poll count calls abort, which MUST quiesce
device access synchronously, before CPU fallback. Failed partial outputs are discarded. Side-effecting tasks never retry
on this route. Backend callbacks must be bounded, thread-safe and nonblocking;
an actual device implementation must qualify these contracts and exact CPU parity.
The base build contains no real device backend, PCIe enumeration or vendor runtime.
No performance or disappearance guarantees are claimed for an unqualified backend.

## Initial adapter and boundaries

`elpis_regex_execution` schedules independent complete byte sources. Every task
creates its own V2 lexer, feeds <=64 KiB spans serially, finalizes, and returns
the full canonical JSON. It retains existing default grammar, evidence and
source semantics. Input is memory only. The adapter grants no semantic/admission
authority and does not schedule query-overlay publication. The original APIs and
Python runtime are unchanged; opt-in native hosts pass `elpis_regex_execute` as
their compute function. There is no callback into Python or new Python thread pool.

ECS mutation, append, fsync, replay application and inference token recurrence stay
serial. HACF vector search schedules one bound task per immutable shard on the same
pool; the per-shard gate now covers only WARM lease acquisition (residency work),
so scoring runs outside it. Inference rows/experts remain outside this budget
because existing Python/NumPy pools and numerical contracts need separate
qualification.

## Tests and measurements

Root CMake/CI automatically includes the runtime, Regex parity and benchmark smoke
tests in gcc/clang Debug/Release and ASan+UBSan/TSan matrices with warnings as errors.
`test_execution` uses a separate test-only archive to inject pthread_create failure;
the production ABI has no injection hook. Test gates force out-of-order completion
and queue pressure without timing-dependent cancellation assumptions. Six producers
check exactly-once delivery. Fake backends test capability selection, rejection,
initial absence, disappearance, timeout/quiescence and output equivalence. The
scheduling tests require that 40 tasks queued on a blocked task's own lane all
complete while it blocks; that eight long tasks on one lane run four at once on
all four threads; that `workers=1` stays serial next to a wider context sharing the
pool; that any number of contexts share at most four threads; and that CPU work
completes while an undecided accelerator token stays parked, without abort.

`elpis_exec_get_pool_metrics` reports per pool thread: tasks, steals, polls,
wakeups, busy and idle wall time and thread CPU time; plus lock acquisitions,
contended acquisitions and wait time.

`benchmark_regex_execution WORKERS FAMILY TASKS` prints JSON. WORKERS=0 calls the
original V2 API directly; 1..4 use the runtime. FAMILY is small or large; data derive
from existing Regex fixtures including no-match, long whitespace, ambiguity and
invalid UTF8. Setup and producer input copying are included. The runtime queue
itself never copies inputs. Wall, process CPU, source/output bytes, throughput,
time to first ordered result, p50/p95 retirement latency, compute/queue delay,
high-water and process peak RSS are reported. Allocator calls inside the existing
lexer are not instrumented. No inference acceleration is inferred from this test.
