#ifndef ELPIS_DSV41_CLOCK_H
#define ELPIS_DSV41_CLOCK_H
#include "elpis/dsv41_stream.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Native Clock R0. Admission is a cold host operation. The clock borrows
 * EXCLUSIVE use of an admitted, idle runtime. It links no execution runtime.
 * All table functions MUST be native, never interpreter trampolines.
 * Config arrays/context/DSOs live through destroy. The owner destroys/detaches
 * the runtime/provider after quarantine; native shutdown quiesces and frees
 * model/cache/stream resources before FAILED returns.
 * Handles are non-reused identities, not pointers (64 live clocks maximum).
 * One caller may advance/read/close; cancel may race advance. close/destroy
 * during advance request cancel and return BUSY; join the caller and retry.
 * Destroy is idempotent. Other access to a destroyed handle returns STALE. */
enum { ELPIS_DSV41_CLOCK_ABI_V1 = 1 };
/* Config layout versions. V1 callers pass the struct prefix ending at
 * max_new_tokens; the clock never reads the V2 tail for them. */
enum { ELPIS_DSV41_CLOCK_CONFIG_V1 = 1, ELPIS_DSV41_CLOCK_CONFIG_V2 = 2 };
typedef uint64_t elpis_dsv41_clock;
typedef enum {
    ELPIS_CLOCK_OK = 0, ELPIS_CLOCK_INVALID = 1, ELPIS_CLOCK_STALE = 2,
    ELPIS_CLOCK_BUSY = 3, ELPIS_CLOCK_LIMIT = 4, ELPIS_CLOCK_DEVICE = 5,
    ELPIS_CLOCK_INTEGRITY = 6, ELPIS_CLOCK_ENCODING = 7, ELPIS_CLOCK_IO = 8,
    ELPIS_CLOCK_CLOSED = 9, ELPIS_CLOCK_DEFER = 10
} elpis_clock_code;
typedef enum {
    ELPIS_CLOCK_PROGRESS = 0, ELPIS_CLOCK_MATERIALIZATION_NEEDED = 1,
    ELPIS_CLOCK_COMPLETE = 2, ELPIS_CLOCK_STOP_TOKEN = 3,
    ELPIS_CLOCK_CANCELLED = 4, ELPIS_CLOCK_FAILED = 5, ELPIS_CLOCK_STOPPED = 6
} elpis_clock_outcome;
typedef enum {
    ELPIS_CLOCK_CREATED = 0, ELPIS_CLOCK_BOUNDARY = 1, ELPIS_CLOCK_PREPARING = 2,
    ELPIS_CLOCK_IN_TOKEN = 3, ELPIS_CLOCK_PARKED = 4, ELPIS_CLOCK_RELEASED = 5,
    ELPIS_CLOCK_DISCARDED = 6, ELPIS_CLOCK_QUARANTINED = 7
} elpis_clock_state;
typedef struct {
    uint32_t abi_version, reserved;
    elpis_exec_buffer *(*buffer_alloc)(size_t);
    void *(*buffer_mutable_data)(elpis_exec_buffer *);
    const void *(*buffer_data)(const elpis_exec_buffer *);
    size_t (*buffer_size)(const elpis_exec_buffer *);
    void (*buffer_release)(elpis_exec_buffer *);
    elpis_exec_status (*submit)(elpis_exec_runtime *, const elpis_exec_task *, elpis_exec_buffer **, uint64_t *);
    elpis_exec_status (*take)(elpis_exec_runtime *, unsigned, elpis_exec_result *);
    void (*metrics)(elpis_exec_runtime *, elpis_exec_metrics *);
    elpis_exec_status (*shutdown)(elpis_exec_runtime *, int);
} elpis_dsv41_clock_host_v1;

/* Elpis-owned service, never accelerator-provider authority. A successful
 * acquire borrows a byte span/lease; release occurs exactly once before submit
 * or coarse return. BUSY/DEFER retain nothing. rows returns decoded F32 LE in
 * request order, bound to the bank identity. expert supplies validated ranges
 * of w1||w3||w2 with the three binding digests. Calls must be bounded. quiesce
 * synchronously cancels service work, is idempotent and leaves no live leases.
 * Expert LIMIT (temporary staging pressure), BUSY and DEFER return
 * MATERIALIZATION_NEEDED; advance retries the same request
 * with a monotonic deadline. No port submission is outstanding at that return. */
typedef struct { const uint8_t *data; size_t bytes; void *lease; uint8_t digests[96]; } elpis_clock_span;
typedef struct {
    uint32_t abi_version, reserved;
    void *context;
    elpis_clock_code (*rows)(void *, uint32_t, const uint8_t bank[32], const uint64_t *, size_t,
                             uint32_t dimension, elpis_clock_span *);
    elpis_clock_code (*expert)(void *, uint32_t, uint32_t, uint64_t, size_t, elpis_clock_span *);
    void (*release)(void *, elpis_clock_span *);
    void (*quiesce)(void *);
} elpis_dsv41_materializer_v1;
typedef struct {
    uint32_t abi_version, reserved;
    elpis_exec_runtime *runtime;
    elpis_dsv41_clock_host_v1 host;
    elpis_dsv41_materializer_v1 materializer;
    uint64_t sequence, stream_id, epoch;
    uint8_t manifest[32];
    uint32_t vocab, layers, active_experts, expert_count, index_topk, dimension, hc_mult;
    uint32_t max_tokens, features, engram_count, order, heads, row_dimension, pad;
    const uint32_t *ratios, *layer_flags;
    const uint8_t *resident; /* [layers][expert_count+1] */
    const uint32_t *engram_layers; /* [engram_count], ascending */
    const uint8_t *banks; /* [engram_count][32] */
    const uint32_t *token_map; /* [vocab], normalized IDs */
    const uint64_t *multipliers; /* [engram_count][order] */
    const uint64_t *primes, *offsets; /* [engram_count][(order-1)*heads] */
    uint64_t image_bytes, part_bytes, memory_budget;
    uint32_t max_input_bytes, max_output_bytes, materialization_timeout_ms, exchange_timeout_ms;
    const uint32_t *prefill, *stop_tokens;
    uint32_t prefill_count, stop_count, max_new_tokens;
    /* CONFIG_V2 tail. Frozen per-sequence turn conditioning: conditioning_count
     * is 0 (none) or dimension, requiring granted FEATURE_CONDITIONING. Values
     * are finite F32, copied at create and sent once in STREAM_OPEN; never per token. */
    const float *conditioning;
    uint32_t conditioning_count, reserved2;
} elpis_dsv41_clock_config_v1;
typedef struct {
    /* code belongs to the sequence; provider_code records quarantine separately.
     * A failed release must not replace an already determined Principal result. */
    uint32_t outcome, code, state, position, generated, token, argmax, provider_code;
    uint64_t sequence, submissions, bytes_h2p, bytes_p2h, polls, provider_ns, advance_ns;
    uint64_t materialization_yields, acquires, releases, allocations, consumed, outputs_released;
    uint64_t discards, normal_releases, quarantines, storage_bytes;
    uint64_t requests_released; /* unconsumed inputs: allocations == consumed + this */
} elpis_dsv41_clock_metrics_v1;
/* Views borrowed until the next mutation/destroy. Traces include only completed
 * tokens. COMPLETE is the unchanged YTS body, including optional layer streams.
 * Storage is preallocated at create within memory_budget; no growing queues. */
typedef struct {
    uint32_t token, position;
    const uint64_t *rows;
    const uint8_t *complete;
    size_t complete_bytes;
    uint64_t elapsed_ns;
} elpis_dsv41_clock_trace_v1;
uint32_t elpis_dsv41_clock_abi_version(void);
elpis_clock_code elpis_dsv41_clock_create(const elpis_dsv41_clock_config_v1 *, elpis_dsv41_clock *out);
elpis_clock_code elpis_dsv41_clock_open(elpis_dsv41_clock);
elpis_clock_code elpis_dsv41_clock_advance(elpis_dsv41_clock, uint32_t token_budget, elpis_dsv41_clock_metrics_v1 *);
elpis_clock_code elpis_dsv41_clock_cancel(elpis_dsv41_clock);
/* Graceful Principal stop, at a completed token boundary only. Mid-token
 * stop returns BUSY; cancel is the discard operation. Preserves trace/output. */
elpis_clock_code elpis_dsv41_clock_stop(elpis_dsv41_clock);
elpis_clock_code elpis_dsv41_clock_metrics(elpis_dsv41_clock, elpis_dsv41_clock_metrics_v1 *);
elpis_clock_code elpis_dsv41_clock_trace(elpis_dsv41_clock, uint32_t position, elpis_dsv41_clock_trace_v1 *);
elpis_clock_code elpis_dsv41_clock_close(elpis_dsv41_clock);
elpis_clock_code elpis_dsv41_clock_destroy(elpis_dsv41_clock);
/* Pure AddressScheme primitive. tail newest-first, -1 dead. No mutation or
 * decoding. Products must fit signed int64 as required by admitted geometry. */
elpis_clock_code elpis_dsv41_clock_hash(uint32_t current, const int64_t *tail, uint32_t order,
    uint32_t heads, uint32_t pad, const uint64_t *multipliers, const uint64_t *primes,
    const uint64_t *offsets, uint64_t *rows);
#ifdef __cplusplus
}
#endif
#endif
