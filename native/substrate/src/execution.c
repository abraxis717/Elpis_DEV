#define _GNU_SOURCE
#include "elpis/execution.h"
#include <pthread.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#ifdef __linux__
#include <sched.h>
#endif

#define BUFFER_LIMIT ((size_t)64 * 1024 * 1024)
#define RUNTIME_LIMIT ((size_t)512 * 1024 * 1024)
#define NONE UINT32_MAX
/* Conservative isolation granule, not an assertion about L1/L2 capacity. */
#define ISOLATION 128
/* Shared by contexts linked to this module. Reserve before creating any thread;
 * adapters share contexts rather than multiplying the CPU budget. */
static atomic_uint worker_budget;
static int reserve_workers(unsigned n) {
    unsigned current = atomic_load(&worker_budget);
    do {
        if (current > 4 - n) return 0;
    } while (!atomic_compare_exchange_weak(&worker_budget, &current, current + n));
    return 1;
}
struct elpis_exec_buffer {
    atomic_uint references;
    atomic_int sealed;
    size_t size;
    max_align_t alignment;
    unsigned char data[];
};
enum slot_state { FREE, QUEUED, RUNNING, DONE };
typedef struct {
    alignas(ISOLATION) elpis_exec_task task;
    elpis_exec_bound_compute bound_compute;
    void *bound_context;
    elpis_exec_buffer *input;
    elpis_exec_result result;
    uint64_t accepted_ns;
    unsigned next;
    enum slot_state state;
} slot;
_Static_assert(sizeof(slot) % ISOLATION == 0, "slot stride");
typedef struct {
    alignas(ISOLATION) pthread_t thread;
    struct elpis_exec_runtime *runtime;
    unsigned head, tail;
    pthread_cond_t wake;
} worker;
_Static_assert(alignof(worker) >= ISOLATION, "worker isolation");
_Static_assert(sizeof(worker) % ISOLATION == 0, "worker stride");
struct elpis_exec_runtime {
    alignas(ISOLATION) pthread_mutex_t lock;
    pthread_cond_t changed;
    elpis_exec_config config;
    elpis_exec_backend backend;
    uint64_t capabilities;
    slot *slots;
    worker workers[4];
    alignas(ISOLATION) elpis_exec_metrics metrics;
    unsigned started;
    int closed, joined, backend_initialized;
};

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
elpis_exec_buffer *elpis_exec_buffer_alloc(size_t n) {
    if (n > BUFFER_LIMIT) return NULL;
    elpis_exec_buffer *b = malloc(sizeof(*b) + n);
    if (!b) return NULL;
    atomic_init(&b->references, 1);
    atomic_init(&b->sealed, 0);
    b->size = n;
    return b;
}
void *elpis_exec_buffer_mutable_data(elpis_exec_buffer *b) {
    return b && !atomic_load(&b->sealed) ? b->data : NULL;
}
const void *elpis_exec_buffer_data(const elpis_exec_buffer *b) { return b ? b->data : NULL; }
size_t elpis_exec_buffer_size(const elpis_exec_buffer *b) { return b ? b->size : 0; }
void elpis_exec_buffer_retain(elpis_exec_buffer *b) {
    if (b) atomic_fetch_add_explicit(&b->references, 1, memory_order_relaxed);
}
void elpis_exec_buffer_release(elpis_exec_buffer *b) {
    if (b && atomic_fetch_sub_explicit(&b->references, 1, memory_order_acq_rel) == 1) free(b);
}
unsigned elpis_exec_default_workers(unsigned n) { return n <= 2 ? 1 : n <= 4 ? n - 1 : 3; }
static unsigned available_cpus(void) {
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0)
        return (unsigned)CPU_COUNT(&set);
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (unsigned)n : 1;
}

static elpis_exec_status compute(elpis_exec_runtime *r, slot *s,
                                 unsigned *accepted, unsigned *fallback) {
    elpis_exec_buffer **out = &s->result.output;
    if (s->bound_compute)
        return s->bound_compute(s->bound_context, s->input, r->config.max_output_bytes, out);
    if ((s->task.flags & ELPIS_EXEC_PURE) &&
        (r->capabilities & (UINT64_C(1) << s->task.operation))) {
        void *token = NULL;
        elpis_exec_status code = r->backend.submit(r->backend.context, &s->task,
                                                  s->input, r->config.max_output_bytes, &token);
        if (code == ELPIS_EXEC_OK) {
            ++*accepted;
            for (unsigned i = 0; i < r->config.backend_poll_limit; ++i) {
                code = r->backend.poll(r->backend.context, token, out);
                if (code != ELPIS_EXEC_WOULD_BLOCK) break;
                elpis_exec_buffer_release(*out);
                *out = NULL;
                struct timespec delay = {0, 1000000};
                while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
            }
            if (code == ELPIS_EXEC_WOULD_BLOCK) r->backend.abort(r->backend.context, token);
            if (code == ELPIS_EXEC_OK && (!*out || (*out)->size <= r->config.max_output_bytes))
                return ELPIS_EXEC_OK;
        }
        elpis_exec_buffer_release(*out);
        *out = NULL;
        ++*fallback;
    }
    return s->task.compute(s->input, r->config.max_output_bytes, out);
}
static void *worker_main(void *arg) {
    worker *w = arg;
    elpis_exec_runtime *r = w->runtime;
    pthread_mutex_lock(&r->lock);
    for (;;) {
        while (w->head == NONE && !r->closed) pthread_cond_wait(&w->wake, &r->lock);
        if (w->head == NONE) break;
        slot *s = &r->slots[w->head];
        w->head = s->next;
        if (w->head == NONE) w->tail = NONE;
        /* Cancellation unlinks slots under the same lock, so only QUEUED here. */
        s->state = RUNNING;
        uint64_t start = now_ns();
        s->result.queue_ns = start - s->accepted_ns;
        pthread_mutex_unlock(&r->lock);
        unsigned accepted = 0, fallback = 0;
        s->result.status = compute(r, s, &accepted, &fallback);
        if (s->result.status < ELPIS_EXEC_OK || s->result.status > ELPIS_EXEC_INTERNAL ||
            (s->result.output && s->result.output->size > r->config.max_output_bytes))
            s->result.status = ELPIS_EXEC_INTERNAL;
        if (s->result.status != ELPIS_EXEC_OK) {
            elpis_exec_buffer_release(s->result.output);
            s->result.output = NULL;
        } else if (s->result.output) atomic_store(&s->result.output->sealed, 1);
        elpis_exec_buffer_release(s->input);
        s->input = NULL;
        s->result.compute_ns = now_ns() - start;
        pthread_mutex_lock(&r->lock);
        r->metrics.backend_accepted += accepted;
        r->metrics.backend_fallback += fallback;
        r->metrics.queue_ns += s->result.queue_ns;
        r->metrics.compute_ns += s->result.compute_ns;
        ++r->metrics.completed;
        s->state = DONE;
        pthread_cond_broadcast(&r->changed);
    }
    pthread_mutex_unlock(&r->lock);
    return NULL;
}

/* Failure injection is compiled only into a separate test support archive. */
#ifdef ELPIS_EXEC_TESTING
extern int elpis_exec_test_thread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#define CREATE_THREAD elpis_exec_test_thread_create
#else
#define CREATE_THREAD pthread_create
#endif
elpis_exec_status elpis_exec_create(const elpis_exec_config *c, elpis_exec_runtime **out) {
    if (!out) return ELPIS_EXEC_INVALID;
    *out = NULL;
    if (!c || c->workers > 4 || !c->capacity || c->capacity > 4096 ||
        !c->max_input_bytes || !c->max_output_bytes || c->max_input_bytes > BUFFER_LIMIT ||
        c->max_output_bytes > BUFFER_LIMIT ||
        c->max_input_bytes + c->max_output_bytes > RUNTIME_LIMIT / c->capacity)
        return ELPIS_EXEC_INVALID;
    if (c->backend && (!c->backend->init || !c->backend->submit || !c->backend->poll ||
                      !c->backend->abort || !c->backend->shutdown ||
                      !c->backend_poll_limit || c->backend_poll_limit > 1000)) return ELPIS_EXEC_INVALID;
    unsigned count = c->workers ? c->workers : elpis_exec_default_workers(available_cpus());
    if (!reserve_workers(count)) return ELPIS_EXEC_WOULD_BLOCK;
    elpis_exec_runtime *r = aligned_alloc(alignof(elpis_exec_runtime), sizeof(*r));
    if (!r) { atomic_fetch_sub(&worker_budget, count); return ELPIS_EXEC_INTERNAL; }
    memset(r, 0, sizeof(*r));
    r->config = *c;
    r->config.workers = count;
    r->metrics.workers = r->config.workers;
    r->metrics.capacity = c->capacity;
    r->slots = aligned_alloc(alignof(slot), c->capacity * sizeof(slot));
    if (!r->slots) { free(r); atomic_fetch_sub(&worker_budget, count); return ELPIS_EXEC_INTERNAL; }
    memset(r->slots, 0, c->capacity * sizeof(slot));
    if (pthread_mutex_init(&r->lock, NULL)) {
        free(r->slots); free(r); atomic_fetch_sub(&worker_budget, count); return ELPIS_EXEC_INTERNAL;
    }
    pthread_condattr_t attr;
    if (pthread_condattr_init(&attr)) goto mutex_failure;
    if (pthread_condattr_setclock(&attr, CLOCK_MONOTONIC)) { pthread_condattr_destroy(&attr); goto mutex_failure; }
    if (pthread_cond_init(&r->changed, &attr)) { pthread_condattr_destroy(&attr); goto mutex_failure; }
    pthread_condattr_destroy(&attr);
    unsigned initialized = 0;
    for (; initialized < r->config.workers; ++initialized) {
        worker *w = &r->workers[initialized];
        w->runtime = r;
        w->head = w->tail = NONE;
        if (pthread_cond_init(&w->wake, NULL)) goto startup_failure;
    }
    if (c->backend) {
        r->backend = *c->backend;
        r->backend_initialized = 1;
        if (r->backend.init(r->backend.context, &r->capabilities) != ELPIS_EXEC_OK)
            r->capabilities = 0;
    }
    for (; r->started < r->config.workers; ++r->started)
        if (CREATE_THREAD(&r->workers[r->started].thread, NULL, worker_main, &r->workers[r->started]))
            goto startup_failure;
    *out = r;
    return ELPIS_EXEC_OK;
startup_failure:
    pthread_mutex_lock(&r->lock);
    r->closed = 1;
    for (unsigned i = 0; i < r->started; ++i) pthread_cond_signal(&r->workers[i].wake);
    pthread_mutex_unlock(&r->lock);
    for (unsigned i = 0; i < r->started; ++i) pthread_join(r->workers[i].thread, NULL);
    if (r->backend_initialized) r->backend.shutdown(r->backend.context);
    for (unsigned i = 0; i < initialized; ++i) pthread_cond_destroy(&r->workers[i].wake);
    pthread_cond_destroy(&r->changed);
mutex_failure:
    pthread_mutex_destroy(&r->lock);
    free(r->slots);
    free(r);
    atomic_fetch_sub(&worker_budget, count);
    return ELPIS_EXEC_INTERNAL;
}
elpis_exec_status elpis_exec_submit(elpis_exec_runtime *r, const elpis_exec_task *t,
                                  elpis_exec_buffer **input, uint64_t *sequence) {
    if (!r || !t || !t->compute || t->operation >= 64 || (t->flags & ~ELPIS_EXEC_PURE) ||
        !input || !*input || !sequence || (*input)->size > r->config.max_input_bytes)
        return ELPIS_EXEC_INVALID;
    pthread_mutex_lock(&r->lock);
    elpis_exec_status code = ELPIS_EXEC_OK;
    if (r->closed || r->metrics.submitted == UINT64_MAX) code = ELPIS_EXEC_CLOSED;
    else if (r->metrics.outstanding == r->config.capacity) {
        ++r->metrics.queue_full;
        code = ELPIS_EXEC_WOULD_BLOCK;
    } else {
        uint64_t seq = r->metrics.submitted++;
        unsigned index = (unsigned)(seq % r->config.capacity);
        slot *s = &r->slots[index];
        s->task = *t;
        s->bound_compute = NULL;
        s->bound_context = NULL;
        s->input = *input;
        atomic_store(&s->input->sealed, 1);
        *input = NULL;
        s->result = (elpis_exec_result){.sequence=seq, .tag=t->tag,
                                      .operation=t->operation, .stage=t->stage};
        s->state = QUEUED;
        s->accepted_ns = now_ns();
        s->next = NONE;
        worker *w = &r->workers[t->affinity % r->config.workers];
        if (w->tail == NONE) w->head = index;
        else r->slots[w->tail].next = index;
        w->tail = index;
        ++r->metrics.outstanding;
        if (r->metrics.outstanding > r->metrics.high_water) r->metrics.high_water = r->metrics.outstanding;
        *sequence = seq;
        pthread_cond_signal(&w->wake);
    }
    pthread_mutex_unlock(&r->lock);
    return code;
}
elpis_exec_status elpis_exec_submit_bound(elpis_exec_runtime *r,
                                          const elpis_exec_bound_task *t,
                                          elpis_exec_buffer **input,
                                          uint64_t *sequence) {
    if (!r || !t || !t->compute || t->operation >= 64 || (t->flags & ~ELPIS_EXEC_PURE) ||
        !input || !*input || !sequence || (*input)->size > r->config.max_input_bytes)
        return ELPIS_EXEC_INVALID;
    pthread_mutex_lock(&r->lock);
    elpis_exec_status code = ELPIS_EXEC_OK;
    if (r->closed || r->metrics.submitted == UINT64_MAX) code = ELPIS_EXEC_CLOSED;
    else if (r->metrics.outstanding == r->config.capacity) {
        ++r->metrics.queue_full;
        code = ELPIS_EXEC_WOULD_BLOCK;
    } else {
        uint64_t seq = r->metrics.submitted++;
        unsigned index = (unsigned)(seq % r->config.capacity);
        slot *s = &r->slots[index];
        s->task = (elpis_exec_task){
            .operation = t->operation,
            .stage = t->stage,
            .affinity = t->affinity,
            .flags = t->flags,
            .tag = t->tag,
            .compute = NULL,
        };
        s->bound_compute = t->compute;
        s->bound_context = t->context;
        s->input = *input;
        atomic_store(&s->input->sealed, 1);
        *input = NULL;
        s->result = (elpis_exec_result){.sequence=seq, .tag=t->tag,
                                      .operation=t->operation, .stage=t->stage};
        s->state = QUEUED;
        s->accepted_ns = now_ns();
        s->next = NONE;
        worker *w = &r->workers[t->affinity % r->config.workers];
        if (w->tail == NONE) w->head = index;
        else r->slots[w->tail].next = index;
        w->tail = index;
        ++r->metrics.outstanding;
        if (r->metrics.outstanding > r->metrics.high_water) r->metrics.high_water = r->metrics.outstanding;
        *sequence = seq;
        pthread_cond_signal(&w->wake);
    }
    pthread_mutex_unlock(&r->lock);
    return code;
}

elpis_exec_status elpis_exec_take(elpis_exec_runtime *r, unsigned ms, elpis_exec_result *out) {
    if (!r || !out || ms > 60000) return ELPIS_EXEC_INVALID;
    uint64_t deadline = now_ns() + (uint64_t)ms * 1000000;
    struct timespec ts = {(time_t)(deadline / 1000000000), (long)(deadline % 1000000000)};
    pthread_mutex_lock(&r->lock);
    elpis_exec_status code;
    for (;;) {
        slot *s = &r->slots[r->metrics.retired % r->config.capacity];
        if (r->metrics.outstanding && s->state == DONE) {
            *out = s->result;
            s->result.output = NULL;
            s->state = FREE;
            ++r->metrics.retired;
            --r->metrics.outstanding;
            code = ELPIS_EXEC_OK;
            break;
        }
        if (r->closed && !r->metrics.outstanding) { code = ELPIS_EXEC_CLOSED; break; }
        if (!ms || now_ns() >= deadline) { code = ELPIS_EXEC_WOULD_BLOCK; break; }
        int e = pthread_cond_timedwait(&r->changed, &r->lock, &ts);
        if (e && e != ETIMEDOUT) { code = ELPIS_EXEC_INTERNAL; break; }
    }
    pthread_mutex_unlock(&r->lock);
    return code;
}
static void cancel_locked(elpis_exec_runtime *r, slot *s) {
    worker *w = &r->workers[s->task.affinity % r->config.workers];
    unsigned target = (unsigned)(s - r->slots), prev = NONE, at = w->head;
    while (at != target) { prev = at; at = r->slots[at].next; }
    if (prev == NONE) w->head = s->next;
    else r->slots[prev].next = s->next;
    if (w->tail == target) w->tail = prev;
    elpis_exec_buffer_release(s->input);
    s->input = NULL;
    s->result.status = ELPIS_EXEC_CANCELLED;
    s->result.queue_ns = now_ns() - s->accepted_ns;
    r->metrics.queue_ns += s->result.queue_ns;
    s->state = DONE;
    ++r->metrics.cancelled;
    ++r->metrics.completed;
    pthread_cond_broadcast(&r->changed);
}
elpis_exec_status elpis_exec_cancel(elpis_exec_runtime *r, uint64_t seq) {
    if (!r) return ELPIS_EXEC_INVALID;
    pthread_mutex_lock(&r->lock);
    elpis_exec_status code = ELPIS_EXEC_INVALID;
    if (seq >= r->metrics.retired && seq < r->metrics.submitted) {
        slot *s = &r->slots[seq % r->config.capacity];
        code = ELPIS_EXEC_WOULD_BLOCK;
        if (s->state == QUEUED) { cancel_locked(r, s); code = ELPIS_EXEC_OK; }
    }
    pthread_mutex_unlock(&r->lock);
    return code;
}
void elpis_exec_get_metrics(elpis_exec_runtime *r, elpis_exec_metrics *out) {
    if (!r || !out) return;
    pthread_mutex_lock(&r->lock);
    *out = r->metrics;
    pthread_mutex_unlock(&r->lock);
}
elpis_exec_status elpis_exec_shutdown(elpis_exec_runtime *r, int cancel) {
    if (!r || (cancel != 0 && cancel != 1)) return ELPIS_EXEC_INVALID;
    if (r->joined) return ELPIS_EXEC_OK;
    pthread_mutex_lock(&r->lock);
    r->closed = 1;
    if (cancel) for (unsigned i = 0; i < r->config.capacity; ++i)
        if (r->slots[i].state == QUEUED) cancel_locked(r, &r->slots[i]);
    for (unsigned i = 0; i < r->started; ++i) pthread_cond_signal(&r->workers[i].wake);
    pthread_cond_broadcast(&r->changed);
    pthread_mutex_unlock(&r->lock);
    for (unsigned i = 0; i < r->started; ++i) pthread_join(r->workers[i].thread, NULL);
    if (r->backend_initialized) r->backend.shutdown(r->backend.context);
    atomic_fetch_sub(&worker_budget, r->config.workers);
    r->joined = 1;
    return ELPIS_EXEC_OK;
}
void elpis_exec_destroy(elpis_exec_runtime *r) {
    if (!r) return;
    elpis_exec_shutdown(r, 1);
    for (unsigned i = 0; i < r->config.capacity; ++i) {
        elpis_exec_buffer_release(r->slots[i].input);
        elpis_exec_buffer_release(r->slots[i].result.output);
    }
    for (unsigned i = 0; i < r->started; ++i) pthread_cond_destroy(&r->workers[i].wake);
    pthread_cond_destroy(&r->changed);
    pthread_mutex_destroy(&r->lock);
    free(r->slots);
    free(r);
}
