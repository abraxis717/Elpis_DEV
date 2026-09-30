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
    ELPIS_EXEC_INTERNAL = 7
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
 * Work must terminate; cancellation never interrupts arbitrary C code. */
typedef elpis_exec_status (*elpis_exec_compute)(const elpis_exec_buffer *input,
                                               size_t max_output,
                                               elpis_exec_buffer **out);
/* Bound CPU operation for stateful native adapters. Context is borrowed: it must
 * remain valid from successful submission through ordered retirement (or
 * shutdown). The runtime never dereferences, frees or retains it. Bound tasks
 * are CPU-only in R0; optional accelerator backends are not consulted. This
 * additive API leaves elpis_exec_task and elpis_exec_submit unchanged. */
typedef elpis_exec_status (*elpis_exec_bound_compute)(void *context,
                                                     const elpis_exec_buffer *input,
                                                     size_t max_output,
                                                     elpis_exec_buffer **out);
enum { ELPIS_EXEC_PURE = 1u, ELPIS_EXEC_REGEX = 1u };
typedef struct {
    uint32_t operation;       /* <64, capability bit; adapter defines semantics */
    uint32_t stage;
    uint32_t affinity;        /* worker = affinity % worker_count; not CPU pinning */
    uint32_t flags;           /* PURE permits retry, never authority/side effects */
    uint64_t tag;             /* caller identifier, echoed unchanged */
    elpis_exec_compute compute;
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
 * Backend output must match the CPU oracle exactly. Partial output on any
 * failed poll is released. An invalid/hung backend violates this trusted ABI;
 * hardware integrations must prove their abort/fence contract before use. */
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
    unsigned workers;        /* 0 default; explicit 1..4, otherwise INVALID */
    unsigned capacity;       /* 1..4096, includes completed unretired results */
    size_t max_input_bytes;
    size_t max_output_bytes; /* sum of worst-case slot payloads <=512 MiB */
    unsigned backend_poll_limit; /* 1..1000 when backend set, 1ms sleeps */
    const elpis_exec_backend *backend; /* NULL first-class CPU-only path */
} elpis_exec_config;
typedef struct {
    uint64_t sequence, tag;
    uint32_t operation, stage;
    elpis_exec_status status;
    uint64_t queue_ns, compute_ns;
    elpis_exec_buffer *output; /* caller owns; release independently of runtime */
} elpis_exec_result;
typedef struct {
    unsigned workers, capacity, outstanding, high_water;
    uint64_t submitted, completed, retired, cancelled, queue_full;
    uint64_t queue_ns, compute_ns, backend_accepted, backend_fallback;
} elpis_exec_metrics;

unsigned elpis_exec_default_workers(unsigned available_cpus);
/* Contexts share a four-worker budget within one linked module instance.
 * create returns WOULD_BLOCK if other live contexts reserve that budget.
 * The production target is shared so adapters use one budget. Embedding private
 * copies of the implementation in additional DSOs is unsupported. */
elpis_exec_status elpis_exec_create(const elpis_exec_config *, elpis_exec_runtime **);
/* Success consumes *input and sets it NULL. Failure leaves ownership intact.
 * Accepted sequence numbers are contiguous from zero; rejection consumes none.
 * Multiple producers supported. Their mutex acceptance order defines sequence.
 * No submit wait and no caller-assisted execution. */
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
 * tasks finished. Metadata/output are written only on OK; result.status is the
 * task outcome. Results remain available after shutdown. */
elpis_exec_status elpis_exec_take(elpis_exec_runtime *, unsigned wait_ms,
                                elpis_exec_result *);
/* Queued task cancellation only. Running/completed => WOULD_BLOCK; unknown =>
 * INVALID. Successful cancel still produces exactly one ordered completion. */
elpis_exec_status elpis_exec_cancel(elpis_exec_runtime *, uint64_t sequence);
void elpis_exec_get_metrics(elpis_exec_runtime *, elpis_exec_metrics *);
/* Single lifecycle owner; never call from workers or concurrently with another
 * shutdown/destroy. Submit/take/metrics/cancel may run concurrently with shutdown.
 * cancel_queued=0 drains; =1 cancels queued and lets running tasks finish. Joins
 * all workers without needing a consumer. Idempotent; rejects future submits.
 * Destroy requires all other API users stopped; releases unconsumed results. */
elpis_exec_status elpis_exec_shutdown(elpis_exec_runtime *, int cancel_queued);
void elpis_exec_destroy(elpis_exec_runtime *);
#ifdef __cplusplus
}
#endif
#endif
