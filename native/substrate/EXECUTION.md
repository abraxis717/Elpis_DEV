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
insertion, no growing completion map). A pool thread takes, in order: due parked
work (an accelerator token to poll, or a deferred task to re-run while its context
is below its cap); the oldest task in its own lane of any context below its cap; otherwise the oldest runnable task of any lane of any context
(stealing, oldest first across contexts, so no context starves). Affinity is a
locality preference only and can never strand work: a thread sleeps only when no
context below its cap has queued work and no parked work is due. No task has a
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

## Deferral instead of waiting

A CPU operation never sleeps or spins on a resource held elsewhere. When one is
temporarily unavailable and the operation has made no externally visible change,
it returns `ELPIS_EXEC_DEFER` (never a result status). The task is parked without
holding a pool thread or its context's cap and runs again from the start, by
whichever thread is free, at least 1 ms later or at once on `elpis_exec_notify`.
Its sequence position is unchanged, so later results still wait for it to retire.
The runtime does not bound deferrals; the operation does, with its own deadline
(HACF: the existing 5 s residency deadline). `metrics.deferred` counts them.

Cancellation unlinks queued work and deferred tasks waiting to run again and
produces one ordered CANCELLED result. Running work and parked backend tokens
return WOULD_BLOCK and finish normally. Shutdown closes admission, drains or
cancels queued and deferred work (a task that defers after a cancelling shutdown
began is cancelled, not parked), waits until the context has no queued, running
or parked work and leaves results available. This does not require a consumer even
with a full result window. One lifecycle owner must stop other API callers before
destroy; the last destroy joins the pool. Callbacks must terminate normally;
status failure is isolated, process faults/undefined behavior/pthread_exit are not.
No arbitrary thread interruption is attempted.

## Optional backend

The execution ABI is the accelerator port. It is deliberately vendor-neutral:
the base Elpis build contains no CUDA/HIP/Metal/Vulkan SDK dependency, performs
no accelerator enumeration and ships no device kernel. A host may attach an
explicit provider built after Elpis for that user's substrate. The provider owns
its device runtime and kernel implementation behind `elpis_exec_backend`.

NULL backend means a zero capability mask and no initialization or poll. A supplied
backend initializes once; failure disables routing but does not fail CPU startup.
The provider advertises operation bits and receives only explicit tasks whose
operation it supports.

There are two disjoint task modes:

- `ELPIS_EXEC_PURE`: opportunistic acceleration. The task has a CPU compute
  callback. Backend rejection, terminal failure or poll timeout synchronously
  quiesces the token and executes that CPU oracle. This mode is only for work
  whose retry is semantically safe.
- `ELPIS_EXEC_BACKEND_ONLY`: provider-owned/stateful stream work. The task has no
  CPU callback. The operation must be advertised before admission; otherwise
  submission is refused without consuming the input or a sequence number. Once
  admitted, backend rejection/failure/timeout is terminal and **never** becomes a
  CPU retry. `abort` only has to quiesce device access; it does not claim to roll
  provider state back. An adapter using recurrent provider state must therefore
  discard the stream after a terminal backend failure unless that provider has a
  separately qualified recovery contract.

Only accepted tokens are polled, at most the configured number of times (<=1000).
No worker ever sleeps or spins on a token: submit is followed by one poll, and an
undecided token is parked and polled again by whichever pool thread is free, at
least 1 ms later, while CPU work runs. A backend with a completion event (fence,
eventfd, interrupt) calls `elpis_exec_notify` to make its parked tokens due
immediately. Failed partial outputs are discarded. Backend callbacks must be
bounded, thread-safe and nonblocking. Every concrete accelerator provider must
qualify its stream/lifecycle contract and numerical parity against the applicable
CPU oracle before admission.

For recurrent inference, the intended composition is one explicitly admitted
provider stream with ordered submissions; the provider may retain device-resident
weights and sequence state internally. Elpis exposes the stream boundary, not a
vendor runtime. The post-compiled provider is where CUDA, HIP, Metal, an NPU SDK,
or a future substrate belongs.

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
so scoring runs outside it. A shard task try-locks the gate and makes one FMS WARM
READ lease attempt per run: a held gate, FMS BUSY (move in flight) or LIMIT
(headroom pinned by other readers) defers the task instead of blocking or
sleeping on a pool thread, within the same 5 s deadline. The serial search path
still waits on the caller's own thread. Inference rows/experts remain outside this budget
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
pool; that any number of contexts share at most four threads; that CPU work
completes while an undecided accelerator token stays parked, without abort; and
that a deferring task leaves a serial context's other work running, is re-run no
more than once per millisecond, and is cancelled by cancel or a racing shutdown.
`test_vector_concurrency` pins most of a tight WARM ceiling so every shard task is
refused, checks that unrelated CPU work still runs on all four threads and that the
search then returns the serial digest, and races four executors against demotion.

`elpis_exec_get_pool_metrics` reports per pool thread: tasks, steals, polls,
wakeups, busy and idle wall time and thread CPU time; plus lock acquisitions,
contended acquisitions and wait time.

`benchmark_regex_execution WORKERS FAMILY TASKS [CAPACITY]` prints JSON. WORKERS=0
calls the original V2 API directly; 1..4 use the runtime. FAMILY is small, large or
skew (one 1 MiB whitespace source ahead of small ones); data derive from existing
Regex fixtures including no-match, long whitespace, ambiguity and invalid UTF8.
Setup and producer input copying are included. The runtime queue itself never
copies inputs. Wall, process CPU, source/output bytes, throughput, time to first
ordered result, p50/p95/p99 end-to-end, compute, queue and retirement-wait latency,
tasks completed before the head (skew), high-water, steals, pool lock
acquisitions/contention/wait, per-thread tasks/steals/wakeups/busy/idle/CPU time
and process peak RSS are reported. No inference acceleration is inferred from this
test.
