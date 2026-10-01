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

/* R1 scheduling model.
 *
 * One module-wide pool of at most four worker threads serves every live runtime
 * context. Contexts no longer reserve private threads: Regex, HACF and future
 * adapters share one bounded CPU budget and none can be refused for lack of it.
 * Each context keeps its own ordered slot window, consumer and backend; its
 * `workers` value is its concurrency cap (at most that many of its CPU tasks run
 * at once) and the number of affinity lanes it maps to.
 *
 * A pool thread takes, in order: due parked work (an accelerator token to poll,
 * or a deferred task to re-run while its context is below its cap); the oldest
 * task in its own lane (affinity preference) of any context
 * below its cap; otherwise the oldest runnable task of any lane (stealing).
 * Affinity is therefore a locality hint only and never strands work: a thread
 * is idle only when no context below its cap has queued work and no parked
 * work is due. Retirement stays strictly in sequence order per context, but a
 * slow task delays only publication, never the execution of later tasks.
 *
 * Synchronisation is one pool mutex held for short queue/slot transitions,
 * never across compute or backend calls. A submit wakes at most one idle thread
 * (its lane's thread when idle); completion signals a context's consumer only
 * when the completed slot is that context's retirement head. Accelerator tokens
 * are never waited for inside a worker: an undecided token is parked and polled
 * again (at most backend_poll_limit polls in total, at least 1 ms apart) by
 * whichever thread is free, so CPU work runs meanwhile. Likewise a CPU task that
 * finds a resource temporarily unavailable returns DEFER instead of sleeping; it
 * is parked and re-run >=1 ms later by a free thread. elpis_exec_notify makes
 * all parked work of a context due at once. */

#define BUFFER_LIMIT ((size_t)64 * 1024 * 1024)
#define RUNTIME_LIMIT ((size_t)512 * 1024 * 1024)
#define NONE UINT32_MAX
#define MAX_WORKERS 4u
#define POLL_INTERVAL_NS UINT64_C(1000000)
/* Conservative isolation granule, not an assertion about L1/L2 capacity. */
#define ISOLATION 128

struct elpis_exec_buffer {
    atomic_uint references;
    atomic_int sealed;
    size_t size;
    max_align_t alignment;
    unsigned char data[];
};
enum slot_state { FREE, QUEUED, RUNNING, PARKED, POLLING, DONE };
typedef struct {
    alignas(ISOLATION) elpis_exec_task task;
    elpis_exec_bound_compute bound_compute;
    void *bound_context;
    elpis_exec_buffer *input;
    elpis_exec_result result;
    void *token;
    uint64_t accepted_ns, started_ns, backend_ns, next_poll_ns, ticket;
    unsigned next, lane, polls;
    int deferred; /* the CPU operation returned DEFER; re-runs skip the backend */
    enum slot_state state;
} slot;
_Static_assert(sizeof(slot) % ISOLATION == 0, "slot stride");

struct elpis_exec_runtime {
    alignas(ISOLATION) elpis_exec_config config;
    elpis_exec_backend backend;
    uint64_t capabilities;
    slot *slots;
    unsigned head[MAX_WORKERS], tail[MAX_WORKERS];
    unsigned parked_head, parked_tail;
    unsigned queued, running, parked, polling;
    pthread_cond_t changed, drained;
    alignas(ISOLATION) elpis_exec_metrics metrics;
    int closed, cancelling, joined, backend_initialized;
    struct elpis_exec_runtime *prev, *next;
};

typedef struct {
    alignas(ISOLATION) pthread_t thread;
    pthread_cond_t wake;
    unsigned index;
    int idle, woken;
    elpis_exec_worker_metrics m;
} worker;
_Static_assert(sizeof(worker) % ISOLATION == 0, "worker stride");

static struct {
    pthread_mutex_t mu;   /* all scheduling state, every context */
    pthread_mutex_t life; /* serialises pool membership changes (create/destroy) */
    pthread_condattr_t monotonic;
    int attr_ready;
    unsigned started, live;
    int stopping;
    uint64_t ticket;
    elpis_exec_runtime *contexts;
    uint64_t acquisitions, contended, lock_wait_ns;
    worker workers[MAX_WORKERS];
} pool = {.mu = PTHREAD_MUTEX_INITIALIZER, .life = PTHREAD_MUTEX_INITIALIZER};

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static uint64_t thread_cpu_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static struct timespec deadline(uint64_t ns) {
    struct timespec ts = {(time_t)(ns / 1000000000), (long)(ns % 1000000000)};
    return ts;
}
static void lock(void) {
    if (pthread_mutex_trylock(&pool.mu)) {
        uint64_t t = now_ns();
        pthread_mutex_lock(&pool.mu);
        pool.lock_wait_ns += now_ns() - t;
        ++pool.contended;
    }
    ++pool.acquisitions;
}
static void unlock(void) { pthread_mutex_unlock(&pool.mu); }

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

/* ---- queues (pool.mu held) ------------------------------------------------- */
static void cancelled(elpis_exec_runtime *r, slot *s);
static void lane_push(elpis_exec_runtime *r, slot *s) {
    unsigned index = (unsigned)(s - r->slots), lane = s->lane;
    s->next = NONE;
    if (r->tail[lane] == NONE) r->head[lane] = index;
    else r->slots[r->tail[lane]].next = index;
    r->tail[lane] = index;
    ++r->queued;
}
static void lane_pop(elpis_exec_runtime *r, unsigned lane) {
    slot *s = &r->slots[r->head[lane]];
    r->head[lane] = s->next;
    if (r->head[lane] == NONE) r->tail[lane] = NONE;
    --r->queued;
}
/* The parked list stays in due order (appends never move a due time earlier). */
static void park(elpis_exec_runtime *r, slot *s, uint64_t due) {
    unsigned index = (unsigned)(s - r->slots);
    if (r->parked_tail != NONE && r->slots[r->parked_tail].next_poll_ns > due)
        due = r->slots[r->parked_tail].next_poll_ns;
    s->state = PARKED;
    s->next_poll_ns = due;
    s->next = NONE;
    if (r->parked_tail == NONE) r->parked_head = index;
    else r->slots[r->parked_tail].next = index;
    r->parked_tail = index;
    ++r->parked;
}
static void parked_unlink(elpis_exec_runtime *r, unsigned prev, unsigned at) {
    if (prev == NONE) r->parked_head = r->slots[at].next;
    else r->slots[prev].next = r->slots[at].next;
    if (r->parked_tail == at) r->parked_tail = prev;
    --r->parked;
}
/* A deferred CPU task counts against its context's cap when it runs again; a
 * backend token poll does not. */
static int parked_ready(const elpis_exec_runtime *r, const slot *s) {
    return !s->deferred || r->running < r->config.workers;
}
static int context_idle(const elpis_exec_runtime *r) {
    return !r->queued && !r->running && !r->parked && !r->polling;
}
/* Wake at most one idle pool thread, preferring `preferred`. */
static void wake_one(unsigned preferred) {
    worker *w = NULL;
    if (preferred < pool.started && pool.workers[preferred].idle && !pool.workers[preferred].woken)
        w = &pool.workers[preferred];
    for (unsigned i = 0; !w && i < pool.started; ++i)
        if (pool.workers[i].idle && !pool.workers[i].woken) w = &pool.workers[i];
    if (w) { w->woken = 1; pthread_cond_signal(&w->wake); }
}
static void wake_all(void) {
    for (unsigned i = 0; i < pool.started; ++i) pthread_cond_signal(&pool.workers[i].wake);
}
/* Next work for pool thread `i`, or NULL. Marks the slot RUNNING or POLLING.
 * Due parked work (a backend token to poll, a deferred task to re-run) first. */
static slot *pick(unsigned i, elpis_exec_runtime **owner, int *stolen) {
    uint64_t now = now_ns();
    for (elpis_exec_runtime *r = pool.contexts; r; r = r->next) {
        for (unsigned prev = NONE, at = r->parked_head; at != NONE; prev = at, at = r->slots[at].next) {
            slot *s = &r->slots[at];
            if (s->next_poll_ns > now) break;
            if (!parked_ready(r, s)) continue;
            parked_unlink(r, prev, at);
            if (s->deferred) { ++r->running; s->state = RUNNING; }
            else { ++r->polling; s->state = POLLING; }
            *owner = r; *stolen = 0;
            return s;
        }
    }
    elpis_exec_runtime *best_r = NULL;
    unsigned best_lane = 0;
    uint64_t best = UINT64_MAX;
    for (int pass = 0; pass < 2 && !best_r; ++pass) {
        for (elpis_exec_runtime *r = pool.contexts; r; r = r->next) {
            if (!r->queued || r->running >= r->config.workers) continue;
            for (unsigned lane = 0; lane < r->config.workers; ++lane) {
                if (pass == 0 && lane != i) continue;
                if (r->head[lane] == NONE) continue;
                uint64_t t = r->slots[r->head[lane]].ticket;
                if (t < best) { best = t; best_r = r; best_lane = lane; }
            }
        }
    }
    if (!best_r) return NULL;
    slot *s = &best_r->slots[best_r->head[best_lane]];
    lane_pop(best_r, best_lane);
    ++best_r->running;
    s->state = RUNNING;
    s->started_ns = now;
    s->result.queue_ns = now - s->accepted_ns;
    *owner = best_r; *stolen = best_lane != i;
    return s;
}
/* Earliest time parked work becomes runnable. A deferred task held back by its
 * context's cap is not counted: the thread that finishes one of that context's
 * running tasks picks it up. */
static uint64_t earliest_poll(void) {
    uint64_t due = UINT64_MAX;
    for (elpis_exec_runtime *r = pool.contexts; r; r = r->next)
        for (unsigned at = r->parked_head; at != NONE; at = r->slots[at].next)
            if (parked_ready(r, &r->slots[at])) {
                if (r->slots[at].next_poll_ns < due) due = r->slots[at].next_poll_ns;
                break;
            }
    return due;
}

/* ---- execution (no lock held) ---------------------------------------------- */
enum outcome { COMPLETE, PARK };
static enum outcome cpu(elpis_exec_runtime *r, slot *s) {
    elpis_exec_status code = s->bound_compute
        ? s->bound_compute(s->bound_context, s->input, r->config.max_output_bytes, &s->result.output)
        : s->task.compute(s->input, r->config.max_output_bytes, &s->result.output);
    if (code == ELPIS_EXEC_DEFER) {
        elpis_exec_buffer_release(s->result.output);
        s->result.output = NULL;
        s->deferred = 1;
        return PARK;
    }
    s->result.status = code;
    return COMPLETE;
}
static elpis_exec_status backend_terminal_status(elpis_exec_status code) {
    switch (code) {
    case ELPIS_EXEC_INVALID:
    case ELPIS_EXEC_BACKEND_UNAVAILABLE:
    case ELPIS_EXEC_BACKEND_REJECTED:
    case ELPIS_EXEC_INTERNAL:
        return code;
    default:
        return ELPIS_EXEC_INTERNAL;
    }
}
/* Accelerator exchange for a token that was just submitted or is due for a poll.
 * PURE tasks preserve the existing behaviour: terminal backend failure or timeout
 * synchronously aborts/quiesces and falls back to the CPU oracle. BACKEND_ONLY
 * tasks never retry on CPU because provider-owned stream state may already have
 * advanced; failure is terminal and the owning adapter must discard that stream. */
static enum outcome backend_poll(elpis_exec_runtime *r, slot *s, unsigned *fallback, uint64_t *polls) {
    elpis_exec_buffer **out = &s->result.output;
    elpis_exec_status code = r->backend.poll(r->backend.context, s->token, out);
    int backend_only = (s->task.flags & ELPIS_EXEC_BACKEND_ONLY) != 0;
    int timed_out = 0;
    ++*polls;
    ++s->polls;
    if (code == ELPIS_EXEC_WOULD_BLOCK) {
        elpis_exec_buffer_release(*out);
        *out = NULL;
        if (s->polls < r->config.backend_poll_limit) return PARK;
        r->backend.abort(r->backend.context, s->token);
        timed_out = 1;
    }
    s->token = NULL;
    if (code == ELPIS_EXEC_OK && (!*out || (*out)->size <= r->config.max_output_bytes)) {
        s->result.status = ELPIS_EXEC_OK;
        return COMPLETE;
    }
    elpis_exec_buffer_release(*out);
    *out = NULL;
    if (backend_only) {
        s->result.status = timed_out ? ELPIS_EXEC_BACKEND_UNAVAILABLE
                                     : (code == ELPIS_EXEC_OK ? ELPIS_EXEC_INTERNAL
                                                              : backend_terminal_status(code));
        return COMPLETE;
    }
    ++*fallback;
    return cpu(r, s);
}
static enum outcome execute(elpis_exec_runtime *r, slot *s, unsigned *accepted, unsigned *fallback,
                            uint64_t *polls) {
    if (s->state == POLLING) return backend_poll(r, s, fallback, polls);
    int backend_only = (s->task.flags & ELPIS_EXEC_BACKEND_ONLY) != 0;
    int backend_candidate = (s->task.flags & ELPIS_EXEC_PURE) || backend_only;
    if (!s->deferred && !s->bound_compute && backend_candidate &&
        (r->capabilities & (UINT64_C(1) << s->task.operation))) {
        void *token = NULL;
        s->backend_ns = now_ns();
        elpis_exec_status code = r->backend.submit(r->backend.context, &s->task, s->input,
                                                  r->config.max_output_bytes, &token);
        if (code == ELPIS_EXEC_OK) {
            ++*accepted;
            s->token = token;
            s->polls = 0;
            return backend_poll(r, s, fallback, polls);
        }
        elpis_exec_buffer_release(s->result.output);
        s->result.output = NULL;
        if (backend_only) {
            s->result.status = backend_terminal_status(code);
            return COMPLETE;
        }
        ++*fallback;
    }
    if (backend_only) {
        s->result.status = ELPIS_EXEC_BACKEND_UNAVAILABLE;
        return COMPLETE;
    }
    return cpu(r, s);
}
static void finish(elpis_exec_runtime *r, slot *s) {
    if (s->result.status < ELPIS_EXEC_OK || s->result.status > ELPIS_EXEC_INTERNAL ||
        (s->result.output && s->result.output->size > r->config.max_output_bytes))
        s->result.status = ELPIS_EXEC_INTERNAL;
    if (s->result.status != ELPIS_EXEC_OK) {
        elpis_exec_buffer_release(s->result.output);
        s->result.output = NULL;
    } else if (s->result.output) atomic_store(&s->result.output->sealed, 1);
    elpis_exec_buffer_release(s->input);
    s->input = NULL;
}

static void *worker_main(void *arg) {
    worker *w = arg;
    lock();
    for (;;) {
        if (pool.stopping) break;
        elpis_exec_runtime *r = NULL;
        int stolen = 0;
        slot *s = pick(w->index, &r, &stolen);
        if (!s) {
            uint64_t due = earliest_poll(), since = now_ns();
            w->idle = 1; w->woken = 0;
            if (due == UINT64_MAX) pthread_cond_wait(&w->wake, &pool.mu);
            else {
                struct timespec ts = deadline(due);
                pthread_cond_timedwait(&w->wake, &pool.mu, &ts);
            }
            w->idle = 0;
            w->m.idle_ns += now_ns() - since;
            ++w->m.wakeups;
            continue;
        }
        int polling = s->state == POLLING;
        unlock();
        uint64_t start = now_ns(), cpu0 = thread_cpu_ns();
        unsigned accepted = 0, fallback = 0;
        uint64_t polls = 0;
        enum outcome outcome = execute(r, s, &accepted, &fallback, &polls);
        if (outcome == COMPLETE) finish(r, s);
        uint64_t end = now_ns(), cpu = thread_cpu_ns() - cpu0;
        lock();
        w->m.busy_ns += end - start;
        w->m.cpu_ns += cpu;
        if (outcome == COMPLETE) ++w->m.tasks;
        w->m.steals += (uint64_t)stolen;
        r->metrics.steals += (uint64_t)stolen;
        w->m.polls += polls;
        r->metrics.backend_accepted += accepted;
        r->metrics.backend_fallback += fallback;
        r->metrics.backend_polls += polls;
        /* Idle threads that slept while this context was at its cap ignored its
         * deferred tasks; if this completion frees the cap, one re-arms. */
        int uncapped = 0;
        if (polling) --r->polling;
        else uncapped = r->running-- == r->config.workers && r->parked;
        if (outcome == PARK && s->deferred && r->cancelling) {
            ++r->metrics.deferred;
            cancelled(r, s); /* shutdown already swept the parked list */
        } else if (outcome == PARK) {
            if (s->deferred) ++r->metrics.deferred;
            park(r, s, end + POLL_INTERVAL_NS);
            wake_one(NONE); /* an idle thread re-arms its timed wait for the token */
        } else {
            if (s->backend_ns) r->metrics.backend_wait_ns += end - s->backend_ns;
            s->result.compute_ns = end - s->started_ns;
            s->result.worker = w->index;
            s->result.completed_ns = end;
            r->metrics.queue_ns += s->result.queue_ns;
            r->metrics.compute_ns += s->result.compute_ns;
            ++r->metrics.completed;
            s->state = DONE;
            if (s->result.sequence == r->metrics.retired) pthread_cond_signal(&r->changed);
        }
        if (uncapped) wake_one(NONE);
        if (r->closed && context_idle(r)) pthread_cond_broadcast(&r->drained);
    }
    unlock();
    return NULL;
}

/* Failure injection is compiled only into a separate test support archive. */
#ifdef ELPIS_EXEC_TESTING
extern int elpis_exec_test_thread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#define CREATE_THREAD elpis_exec_test_thread_create
#else
#define CREATE_THREAD pthread_create
#endif

/* Stop and join every pool thread; pool.life held, pool.mu not held, no contexts. */
static void stop_pool(void) {
    lock();
    pool.stopping = 1;
    wake_all();
    unsigned started = pool.started;
    unlock();
    for (unsigned i = 0; i < started; ++i) {
        pthread_join(pool.workers[i].thread, NULL);
        pthread_cond_destroy(&pool.workers[i].wake);
    }
    lock();
    pool.started = 0;
    pool.stopping = 0;
    memset(pool.workers, 0, sizeof(pool.workers));
    unlock();
}
static void unlink_context(elpis_exec_runtime *r) {
    if (r->prev) r->prev->next = r->next;
    else pool.contexts = r->next;
    if (r->next) r->next->prev = r->prev;
    r->prev = r->next = NULL;
}
static void free_context(elpis_exec_runtime *r) {
    pthread_cond_destroy(&r->changed);
    pthread_cond_destroy(&r->drained);
    free(r->slots);
    free(r);
}

elpis_exec_status elpis_exec_create(const elpis_exec_config *c, elpis_exec_runtime **out) {
    if (!out) return ELPIS_EXEC_INVALID;
    *out = NULL;
    if (!c || c->workers > MAX_WORKERS || !c->capacity || c->capacity > 4096 ||
        !c->max_input_bytes || !c->max_output_bytes || c->max_input_bytes > BUFFER_LIMIT ||
        c->max_output_bytes > BUFFER_LIMIT ||
        c->max_input_bytes + c->max_output_bytes > RUNTIME_LIMIT / c->capacity)
        return ELPIS_EXEC_INVALID;
    if (c->backend && (!c->backend->init || !c->backend->submit || !c->backend->poll ||
                      !c->backend->abort || !c->backend->shutdown ||
                      !c->backend_poll_limit || c->backend_poll_limit > 1000)) return ELPIS_EXEC_INVALID;
    unsigned count = c->workers ? c->workers : elpis_exec_default_workers(available_cpus());
    elpis_exec_runtime *r = aligned_alloc(alignof(elpis_exec_runtime), sizeof(*r));
    if (!r) return ELPIS_EXEC_INTERNAL;
    memset(r, 0, sizeof(*r));
    r->config = *c;
    r->config.workers = count;
    r->metrics.workers = count;
    r->metrics.capacity = c->capacity;
    for (unsigned i = 0; i < MAX_WORKERS; ++i) r->head[i] = r->tail[i] = NONE;
    r->parked_head = r->parked_tail = NONE;
    r->slots = aligned_alloc(alignof(slot), c->capacity * sizeof(slot));
    if (!r->slots) { free(r); return ELPIS_EXEC_INTERNAL; }
    memset(r->slots, 0, c->capacity * sizeof(slot));
    pthread_mutex_lock(&pool.life);
    if (!pool.attr_ready) {
        if (pthread_condattr_init(&pool.monotonic) ||
            pthread_condattr_setclock(&pool.monotonic, CLOCK_MONOTONIC)) {
            pthread_mutex_unlock(&pool.life); free(r->slots); free(r); return ELPIS_EXEC_INTERNAL;
        }
        pool.attr_ready = 1;
    }
    if (pthread_cond_init(&r->changed, &pool.monotonic)) {
        pthread_mutex_unlock(&pool.life); free(r->slots); free(r); return ELPIS_EXEC_INTERNAL;
    }
    if (pthread_cond_init(&r->drained, NULL)) {
        pthread_cond_destroy(&r->changed);
        pthread_mutex_unlock(&pool.life); free(r->slots); free(r); return ELPIS_EXEC_INTERNAL;
    }
    if (c->backend) {
        r->backend = *c->backend;
        r->backend_initialized = 1;
        if (r->backend.init(r->backend.context, &r->capabilities) != ELPIS_EXEC_OK)
            r->capabilities = 0;
    }
    lock();
    r->next = pool.contexts;
    if (pool.contexts) pool.contexts->prev = r;
    pool.contexts = r;
    ++pool.live;
    unsigned have = pool.started;
    unlock();
    for (unsigned i = have; i < count; ++i) {
        worker *w = &pool.workers[i];
        if (pthread_cond_init(&w->wake, &pool.monotonic)) goto startup_failure;
        w->index = i;
        if (CREATE_THREAD(&w->thread, NULL, worker_main, w)) {
            pthread_cond_destroy(&w->wake);
            goto startup_failure;
        }
        lock();
        pool.started = i + 1;
        unlock();
    }
    pthread_mutex_unlock(&pool.life);
    *out = r;
    return ELPIS_EXEC_OK;
startup_failure:
    lock();
    unlink_context(r);
    int last = --pool.live == 0;
    unlock();
    if (last) stop_pool();
    pthread_mutex_unlock(&pool.life);
    if (r->backend_initialized) r->backend.shutdown(r->backend.context);
    free_context(r);
    return ELPIS_EXEC_INTERNAL;
}

static elpis_exec_status admit(elpis_exec_runtime *r, const elpis_exec_task *t, elpis_exec_bound_compute bound,
                                void *context, elpis_exec_buffer **input, uint64_t *sequence) {
    lock();
    elpis_exec_status code = ELPIS_EXEC_OK;
    if (r->closed || r->metrics.submitted == UINT64_MAX) code = ELPIS_EXEC_CLOSED;
    else if ((t->flags & ELPIS_EXEC_BACKEND_ONLY) &&
             (!r->backend_initialized ||
              !(r->capabilities & (UINT64_C(1) << t->operation))))
        code = ELPIS_EXEC_BACKEND_UNAVAILABLE;
    else if (r->metrics.outstanding == r->config.capacity) {
        ++r->metrics.queue_full;
        code = ELPIS_EXEC_WOULD_BLOCK;
    } else {
        uint64_t seq = r->metrics.submitted++;
        slot *s = &r->slots[seq % r->config.capacity];
        s->task = *t;
        s->bound_compute = bound;
        s->bound_context = context;
        s->input = *input;
        atomic_store(&s->input->sealed, 1);
        *input = NULL;
        s->result = (elpis_exec_result){.sequence=seq, .tag=t->tag, .operation=t->operation,
                                        .stage=t->stage, .worker=NONE};
        s->state = QUEUED;
        s->accepted_ns = now_ns();
        s->backend_ns = 0;
        s->token = NULL;
        s->polls = 0;
        s->deferred = 0;
        s->ticket = pool.ticket++;
        s->lane = t->affinity % r->config.workers;
        lane_push(r, s);
        ++r->metrics.outstanding;
        if (r->metrics.outstanding > r->metrics.high_water) r->metrics.high_water = r->metrics.outstanding;
        *sequence = seq;
        if (r->running < r->config.workers) wake_one(s->lane);
    }
    unlock();
    return code;
}
elpis_exec_status elpis_exec_submit(elpis_exec_runtime *r, const elpis_exec_task *t,
                                  elpis_exec_buffer **input, uint64_t *sequence) {
    const uint32_t allowed = ELPIS_EXEC_PURE | ELPIS_EXEC_BACKEND_ONLY;
    if (!r || !t || t->operation >= 64 || (t->flags & ~allowed) ||
        !input || !*input || !sequence || (*input)->size > r->config.max_input_bytes)
        return ELPIS_EXEC_INVALID;
    int backend_only = (t->flags & ELPIS_EXEC_BACKEND_ONLY) != 0;
    if ((backend_only && ((t->flags & ELPIS_EXEC_PURE) || t->compute)) ||
        (!backend_only && !t->compute))
        return ELPIS_EXEC_INVALID;
    return admit(r, t, NULL, NULL, input, sequence);
}
elpis_exec_status elpis_exec_submit_bound(elpis_exec_runtime *r,
                                          const elpis_exec_bound_task *t,
                                          elpis_exec_buffer **input,
                                          uint64_t *sequence) {
    if (!r || !t || !t->compute || t->operation >= 64 || (t->flags & ~ELPIS_EXEC_PURE) ||
        !input || !*input || !sequence || (*input)->size > r->config.max_input_bytes)
        return ELPIS_EXEC_INVALID;
    elpis_exec_task task = {t->operation, t->stage, t->affinity, t->flags, t->tag, NULL};
    return admit(r, &task, t->compute, t->context, input, sequence);
}

elpis_exec_status elpis_exec_take(elpis_exec_runtime *r, unsigned ms, elpis_exec_result *out) {
    if (!r || !out || ms > 60000) return ELPIS_EXEC_INVALID;
    uint64_t until = now_ns() + (uint64_t)ms * 1000000;
    struct timespec ts = deadline(until);
    lock();
    elpis_exec_status code;
    for (;;) {
        slot *s = &r->slots[r->metrics.retired % r->config.capacity];
        if (r->metrics.outstanding && s->state == DONE) {
            uint64_t now = now_ns();
            *out = s->result;
            out->retire_ns = now - s->result.completed_ns;
            r->metrics.retire_wait_ns += out->retire_ns;
            s->result.output = NULL;
            s->state = FREE;
            ++r->metrics.retired;
            --r->metrics.outstanding;
            code = ELPIS_EXEC_OK;
            break;
        }
        if (r->closed && !r->metrics.outstanding) { code = ELPIS_EXEC_CLOSED; break; }
        if (!ms || now_ns() >= until) { code = ELPIS_EXEC_WOULD_BLOCK; break; }
        int e = pthread_cond_timedwait(&r->changed, &pool.mu, &ts);
        if (e && e != ETIMEDOUT) { code = ELPIS_EXEC_INTERNAL; break; }
    }
    unlock();
    return code;
}
/* Queued work, or a deferred task waiting to run again (it has no backend token
 * and is not executing), is unlinked and completed as CANCELLED. */
static int cancellable(const slot *s) { return s->state == QUEUED || (s->state == PARKED && s->deferred); }
static void cancel_locked(elpis_exec_runtime *r, slot *s) {
    unsigned target = (unsigned)(s - r->slots), prev = NONE;
    if (s->state == PARKED) {
        unsigned at = r->parked_head;
        while (at != target) { prev = at; at = r->slots[at].next; }
        parked_unlink(r, prev, target);
    } else {
        unsigned lane = s->lane, at = r->head[lane];
        while (at != target) { prev = at; at = r->slots[at].next; }
        if (prev == NONE) r->head[lane] = s->next;
        else r->slots[prev].next = s->next;
        if (r->tail[lane] == target) r->tail[lane] = prev;
        --r->queued;
    }
    cancelled(r, s);
}
/* Completes an unlinked slot as CANCELLED. */
static void cancelled(elpis_exec_runtime *r, slot *s) {
    elpis_exec_buffer_release(s->input);
    s->input = NULL;
    uint64_t now = now_ns();
    s->result.status = ELPIS_EXEC_CANCELLED;
    s->result.queue_ns = now - s->accepted_ns;
    s->result.completed_ns = now;
    r->metrics.queue_ns += s->result.queue_ns;
    s->state = DONE;
    ++r->metrics.cancelled;
    ++r->metrics.completed;
    if (s->result.sequence == r->metrics.retired) pthread_cond_signal(&r->changed);
}
elpis_exec_status elpis_exec_cancel(elpis_exec_runtime *r, uint64_t seq) {
    if (!r) return ELPIS_EXEC_INVALID;
    lock();
    elpis_exec_status code = ELPIS_EXEC_INVALID;
    if (seq >= r->metrics.retired && seq < r->metrics.submitted) {
        slot *s = &r->slots[seq % r->config.capacity];
        code = ELPIS_EXEC_WOULD_BLOCK;
        if (cancellable(s)) {
            cancel_locked(r, s);
            code = ELPIS_EXEC_OK;
            if (r->closed && context_idle(r)) pthread_cond_broadcast(&r->drained);
        }
    }
    unlock();
    return code;
}
void elpis_exec_get_metrics(elpis_exec_runtime *r, elpis_exec_metrics *out) {
    if (!r || !out) return;
    lock();
    *out = r->metrics;
    out->queued = r->queued;
    out->running = r->running;
    out->parked = r->parked + r->polling;
    unlock();
}
void elpis_exec_get_pool_metrics(elpis_exec_pool_metrics *out) {
    if (!out) return;
    lock();
    memset(out, 0, sizeof(*out));
    out->threads = pool.started;
    out->contexts = pool.live;
    out->lock_acquisitions = pool.acquisitions;
    out->lock_contended = pool.contended;
    out->lock_wait_ns = pool.lock_wait_ns;
    for (unsigned i = 0; i < pool.started; ++i) out->worker[i] = pool.workers[i].m;
    unlock();
}
void elpis_exec_notify(elpis_exec_runtime *r) {
    if (!r) return;
    lock();
    uint64_t now = now_ns();
    for (unsigned at = r->parked_head; at != NONE; at = r->slots[at].next) r->slots[at].next_poll_ns = now;
    if (r->parked) wake_one(NONE);
    unlock();
}
elpis_exec_status elpis_exec_shutdown(elpis_exec_runtime *r, int cancel) {
    if (!r || (cancel != 0 && cancel != 1)) return ELPIS_EXEC_INVALID;
    if (r->joined) return ELPIS_EXEC_OK;
    lock();
    r->closed = 1;
    r->cancelling |= cancel;
    if (cancel) for (unsigned i = 0; i < r->config.capacity; ++i)
        if (cancellable(&r->slots[i])) cancel_locked(r, &r->slots[i]);
    pthread_cond_broadcast(&r->changed);
    wake_all();
    while (!context_idle(r)) pthread_cond_wait(&r->drained, &pool.mu);
    unlock();
    if (r->backend_initialized) r->backend.shutdown(r->backend.context);
    r->joined = 1;
    return ELPIS_EXEC_OK;
}
void elpis_exec_destroy(elpis_exec_runtime *r) {
    if (!r) return;
    elpis_exec_shutdown(r, 1);
    pthread_mutex_lock(&pool.life);
    lock();
    unlink_context(r);
    int last = --pool.live == 0;
    unlock();
    if (last) stop_pool();
    pthread_mutex_unlock(&pool.life);
    for (unsigned i = 0; i < r->config.capacity; ++i) {
        elpis_exec_buffer_release(r->slots[i].input);
        elpis_exec_buffer_release(r->slots[i].result.output);
    }
    free_context(r);
}
