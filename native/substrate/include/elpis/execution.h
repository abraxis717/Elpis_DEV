#ifndef ELPIS_EXECUTION_H
#define ELPIS_EXECUTION_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ELPIS_EXEC_OK = 0, ELPIS_EXEC_INVALID = 1, ELPIS_EXEC_CLOSED = 2,
    ELPIS_EXEC_WOULD_BLOCK = 3, ELPIS_EXEC_CANCELLED = 4,
    ELPIS_EXEC_BACKEND_UNAVAILABLE = 5, ELPIS_EXEC_BACKEND_REJECTED = 6,
    ELPIS_EXEC_INTERNAL = 7,
    /* Returned by a CPU operation only: see elpis_exec_compute. Never a result status. */
    ELPIS_EXEC_DEFER = 8
} elpis_exec_status;
typedef struct elpis_exec_runtime elpis_exec_runtime;
typedef struct elpis_exec_buffer elpis_exec_buffer;

/* Buffers own their storage. Fill exclusively before sharing/submission; data()
 * is const. retain/release protect lifetime, not concurrent producer writes.
 * alloc is capped at 64 MiB. No borrowed source pointer enters the queue.
 * mutable_data returns NULL after the buffer has been sealed by submission.
 * Previously obtained writable spans must cease use before sharing. */
elpis_exec_buffer *elpis_exec_buffer_alloc(size_t bytes);
void *elpis_exec_buffer_mutable_data(elpis_exec_buffer *);
const void *elpis_exec_buffer_data(const elpis_exec_buffer *);
size_t elpis_exec_buffer_size(const elpis_exec_buffer *);
void elpis_exec_buffer_retain(elpis_exec_buffer *);
void elpis_exec_buffer_release(elpis_exec_buffer *);

/* CPU operation: no Python callbacks, no secondary pools, no reentry into this
 * runtime. Return normally on failure (no pthread_exit/longjmp/exceptions).
 * Input is immutable and owned through return. Output is a separately owned
 * buffer <= max_output; transfer exactly one reference via *out, also on error.
 * Work must terminate; cancellation never interrupts arbitrary C code.
 * An operation must not sleep or spin waiting for a resource held elsewhere. If
 * one is temporarily unavailable, and the operation has made no externally
 * visible change, it returns ELPIS_EXEC_DEFER (any *out is released): the task
 * is parked without occupying a pool thread and runs again, from the start, at
 * least 1 ms later or at elpis_exec_notify. A deferred task counts as parked for
 * shutdown and can be cancelled like queued work. The runtime does not bound the
 * number of deferrals; the operation must (e.g. by a deadline of its own). */
typedef elpis_exec_status (*elpis_exec_compute)(const elpis_exec_buffer *input,
                                               size_t max_output,
                                               elpis_exec_buffer **out);
/* Bound CPU operation for stateful native adapters. Context is borrowed: it must
 * remain valid from successful submission through ordered retirement (or
 * shutdown). The runtime never dereferences, frees or retains it. Bound tasks
 * are CPU-only in R0; optional accelerator backends are not consulted. This
 * additive API leaves elpis_exec_task and elpis_exec_submit unchanged. The
 * ELPIS_EXEC_DEFER contract of elpis_exec_compute applies unchanged. */
typedef elpis_exec_status (*elpis_exec_bound_compute)(void *context,
                                                     const elpis_exec_buffer *input,
                                                     size_t max_output,
                                                     elpis_exec_buffer **out);
enum {
    ELPIS_EXEC_PURE = 1u,
    /* Provider owns the operation. No implicit CPU fallback/retry is permitted.
     * This is for stateful accelerator streams whose provider state may advance
     * after an accepted submission. */
    ELPIS_EXEC_BACKEND_ONLY = 2u,
    ELPIS_EXEC_REGEX = 1u
};
typedef struct {
    uint32_t operation;       /* <64, capability bit; adapter defines semantics */
    uint32_t stage;
    uint32_t affinity;        /* locality hint: lane = affinity % workers. Preferred by that
                                 lane's pool thread, stolen by any idle one. Never pinning,
                                 never a correctness requirement. */
    uint32_t flags;           /* PURE = opportunistic backend + CPU fallback.
                                 BACKEND_ONLY = provider-owned stream, no fallback. */
    uint64_t tag;             /* caller identifier, echoed unchanged */
    elpis_exec_compute compute; /* required except BACKEND_ONLY; NULL there */
} elpis_exec_task;
typedef struct {
    uint32_t operation;
    uint32_t stage;
    uint32_t affinity;
    uint32_t flags;
    uint64_t tag;
    elpis_exec_bound_compute compute;
    void *context;            /* borrowed through retirement/shutdown */
} elpis_exec_bound_task;

/* Optional backend. Callbacks and context must live until shutdown returns.
 * init runs once, returns supported operation bitmap (zero disables backend).
 * shutdown runs once after every init attempt, including a failed init.
 * All callbacks must be thread-safe, bounded and nonblocking. No owned CPU
 * pools. submit OK alone transfers one device token; every other status means
 * no retained input/token. poll WOULD_BLOCK retains it; any terminal poll
 * consumes it and guarantees no later accesses. abort MUST synchronously
 * quiesce the token and relinquish ALL buffer access; only then may CPU retry.
 * Backend output must match the corresponding CPU oracle exactly. Partial output
 * on any failed poll is released. PURE tasks are opportunistic: backend
 * rejection/failure/timeout falls back to their CPU compute callback.
 * BACKEND_ONLY tasks are the stateful-stream mode: compute must be NULL, the
 * backend must advertise the operation before admission, and every backend
 * failure is terminal with no implicit CPU retry. After terminal failure/abort,
 * a stateful provider stream must be discarded unless that provider's own
 * separately qualified contract proves recovery. An invalid/hung backend
 * violates this trusted ABI; hardware integrations must prove their abort/fence
 * contract before use.
 * No worker ever sleeps on a token: submit is followed by one poll; an undecided
 * token is parked and polled again by whichever pool thread is free, at least
 * 1 ms later, for at most backend_poll_limit polls in total before abort and
 * CPU fallback. A backend with a completion event (fence, eventfd, interrupt)
 * calls elpis_exec_notify to make parked tokens due at once. */
typedef struct {
    void *context;
    elpis_exec_status (*init)(void *, uint64_t *capabilities);
    elpis_exec_status (*submit)(void *, const elpis_exec_task *,
                               const elpis_exec_buffer *, size_t, void **token);
    elpis_exec_status (*poll)(void *, void *token, elpis_exec_buffer **out);
    void (*abort)(void *, void *token);
    void (*shutdown)(void *);
} elpis_exec_backend;
typedef struct {
    unsigned workers;        /* 0 default; explicit 1..4, otherwise INVALID. The
                                context's concurrency cap and lane count; the
                                threads come from the shared pool. */
    unsigned capacity;       /* 1..4096, includes completed unretired results */
    size_t max_input_bytes;
    size_t max_output_bytes; /* sum of worst-case slot payloads <=512 MiB */
    unsigned backend_poll_limit; /* 1..1000 when backend set; polls >=1 ms apart */
    const elpis_exec_backend *backend; /* NULL first-class CPU-only path */
} elpis_exec_config;
typedef struct {
    uint64_t sequence, tag;
    uint32_t operation, stage;
    elpis_exec_status status;
    uint64_t queue_ns, compute_ns;
    elpis_exec_buffer *output; /* caller owns; release independently of runtime */
    /* Diagnostics (CLOCK_MONOTONIC): pool thread that executed the task (UINT32_MAX
     * when cancelled), completion time, and completion-to-retirement wait. */
    uint32_t worker;
    uint64_t completed_ns, retire_ns;
} elpis_exec_result;
typedef struct {
    unsigned workers, capacity, outstanding, high_water;
    uint64_t submitted, completed, retired, cancelled, queue_full;
    uint64_t queue_ns, compute_ns, backend_accepted, backend_fallback;
    /* Current queued/running/parked (backend tokens and deferred tasks) counts;
     * tasks executed off their lane;
     * summed completion-to-retirement wait; backend polls and submit-to-terminal time. */
    unsigned queued, running, parked;
    uint64_t steals, retire_wait_ns, backend_polls, backend_wait_ns;
    uint64_t deferred;        /* DEFER returns (each parks the task once) */
} elpis_exec_metrics;
/* Shared pool diagnostics. Times are CLOCK_MONOTONIC except cpu_ns
 * (CLOCK_THREAD_CPUTIME_ID of the pool thread while executing tasks). */
typedef struct {
    uint64_t tasks, steals, polls, wakeups, busy_ns, idle_ns, cpu_ns;
} elpis_exec_worker_metrics;
typedef struct {
    unsigned threads, contexts;
    uint64_t lock_acquisitions, lock_contended, lock_wait_ns;
    elpis_exec_worker_metrics worker[4];
} elpis_exec_pool_metrics;

unsigned elpis_exec_default_workers(unsigned available_cpus);
/* Every context runs on one module-wide pool of at most four threads, grown on
 * demand to the largest live context's workers and joined when the last context
 * is destroyed. Contexts never reserve private threads, so any number may be
 * live; total CPU concurrency stays <=4. The production target is shared so all
 * adapters use one pool. Embedding private copies of the implementation in
 * additional DSOs is unsupported. */
elpis_exec_status elpis_exec_create(const elpis_exec_config *, elpis_exec_runtime **);
/* Success consumes *input and sets it NULL. Failure leaves ownership intact.
 * Accepted sequence numbers are contiguous from zero; rejection consumes none.
 * Multiple producers supported. Their mutex acceptance order defines sequence.
 * No submit wait and no caller-assisted execution. BACKEND_ONLY requires a
 * configured backend advertising the operation; synchronous unavailability
 * consumes neither the input nor a sequence number. */
elpis_exec_status elpis_exec_submit(elpis_exec_runtime *, const elpis_exec_task *,
                                  elpis_exec_buffer **input, uint64_t *sequence);
/* Bound-task variant for native adapters that operate on caller-owned state
 * without copying that state through the queue. Success consumes *input exactly
 * like elpis_exec_submit. The borrowed context is never passed to a backend. */
elpis_exec_status elpis_exec_submit_bound(elpis_exec_runtime *,
                                          const elpis_exec_bound_task *,
                                          elpis_exec_buffer **input,
                                          uint64_t *sequence);
/* Single ordered consumer. wait_ms 0 polls, 1..60000 bounded monotonic wait.
 * WOULD_BLOCK also means the earliest task is still running, even if later
 * tasks finished: ordering delays publication only, later tasks keep executing.
 * The consumer is woken only when its retirement head completes. Metadata/output
 * are written only on OK; result.status is the task outcome. Results remain
 * available after shutdown. */
elpis_exec_status elpis_exec_take(elpis_exec_runtime *, unsigned wait_ms,
                                elpis_exec_result *);
/* Cancels queued or deferred (parked CPU) tasks only. Running/completed/parked
 * backend token => WOULD_BLOCK; unknown => INVALID. Successful cancel still
 * produces exactly one ordered completion. */
elpis_exec_status elpis_exec_cancel(elpis_exec_runtime *, uint64_t sequence);
void elpis_exec_get_metrics(elpis_exec_runtime *, elpis_exec_metrics *);
void elpis_exec_get_pool_metrics(elpis_exec_pool_metrics *);
/* Makes every parked token and deferred task of the context due now. Thread-safe;
 * may be called from a backend's completion thread/handler or by whatever
 * releases a resource a deferred task waits for. */
void elpis_exec_notify(elpis_exec_runtime *);
/* Single lifecycle owner; never call from workers or concurrently with another
 * shutdown/destroy. Submit/take/metrics/cancel may run concurrently with shutdown.
 * cancel_queued=0 drains; =1 cancels queued and deferred tasks and lets running
 * tasks and parked backend tokens finish. Waits
 * until this context has no queued, running or parked work, without needing a
 * consumer. Idempotent; rejects future submits. Destroy requires all other API
 * users stopped; releases unconsumed results; the last destroy joins the pool. */
elpis_exec_status elpis_exec_shutdown(elpis_exec_runtime *, int cancel_queued);
void elpis_exec_destroy(elpis_exec_runtime *);
#ifdef __cplusplus
}
#endif
#endif
