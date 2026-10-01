#define _POSIX_C_SOURCE 200809L
#include "elpis/execution.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static int fail_after = -1;
int elpis_exec_test_thread_create(pthread_t *t, const pthread_attr_t *a,
                                 void *(*f)(void *), void *p) {
    if (fail_after == 0) return EAGAIN;
    if (fail_after > 0) --fail_after;
    return pthread_create(t, a, f, p);
}
static atomic_uint cpu_calls;
static elpis_exec_buffer *value(unsigned v) {
    elpis_exec_buffer *b = elpis_exec_buffer_alloc(sizeof(v));
    assert(b);
    memcpy(elpis_exec_buffer_mutable_data(b), &v, sizeof(v));
    return b;
}
static elpis_exec_status copy(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    assert(cap >= sizeof(unsigned));
    *out = elpis_exec_buffer_alloc(elpis_exec_buffer_size(b));
    assert(*out);
    memcpy(elpis_exec_buffer_mutable_data(*out), elpis_exec_buffer_data(b), elpis_exec_buffer_size(b));
    return ELPIS_EXEC_OK;
}
static elpis_exec_status echo(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    atomic_fetch_add(&cpu_calls, 1);
    return copy(b, cap, out);
}
static elpis_exec_status poison(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    copy(b, cap, out); /* Failure with allocated output must not leak. */
    return ELPIS_EXEC_INTERNAL;
}
static elpis_exec_config config(unsigned workers, unsigned capacity) {
    elpis_exec_config c = {workers, capacity, 1024, 1024, 0, NULL};
    return c;
}
static elpis_exec_runtime *create(unsigned workers, unsigned capacity) {
    elpis_exec_config c = config(workers, capacity);
    elpis_exec_runtime *r = NULL;
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_OK && r);
    return r;
}
static uint64_t submit(elpis_exec_runtime *r, unsigned n, unsigned affinity, elpis_exec_compute f) {
    elpis_exec_task t = {1, 9, affinity, 0, n, f};
    elpis_exec_buffer *b = value(n);
    uint64_t seq;
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && !b);
    return seq;
}
static elpis_exec_result take(elpis_exec_runtime *r) {
    elpis_exec_result result;
    assert(elpis_exec_take(r, 10000, &result) == ELPIS_EXEC_OK);
    return result;
}
static void check_result(elpis_exec_result *v, uint64_t seq, unsigned n) {
    unsigned got;
    assert(v->sequence == seq && v->tag == n && v->status == ELPIS_EXEC_OK);
    assert(v->stage == 9 && v->operation == 1);
    assert(elpis_exec_buffer_size(v->output) == sizeof(got));
    assert(!elpis_exec_buffer_mutable_data(v->output));
    memcpy(&got, elpis_exec_buffer_data(v->output), sizeof(got));
    assert(got == n);
    elpis_exec_buffer_release(v->output);
}
static void pause_ms(void) { struct timespec t = {0, 1000000}; nanosleep(&t, NULL); }
static void deterministic(void) {
    for (unsigned w = 1; w <= 4; ++w) {
        elpis_exec_runtime *r = create(w, 8);
        for (unsigned block = 0; block < 100; ++block) {
            for (unsigned k = 0; k < 8; ++k) {
                unsigned n = block * 8 + k;
                assert(submit(r, n, k, echo) == n);
            }
            for (unsigned k = 0; k < 8; ++k) {
                unsigned n = block * 8 + k;
                elpis_exec_result v = take(r);
                check_result(&v, n, n);
            }
        }
        elpis_exec_metrics m;
        elpis_exec_get_metrics(r, &m);
        assert(m.submitted == 800 && m.completed == 800 && m.retired == 800);
        assert(m.workers == w && m.high_water == 8 && m.outstanding == 0);
        assert(!m.backend_accepted && !m.backend_fallback);
        assert(elpis_exec_shutdown(r, 0) == ELPIS_EXEC_OK);
        elpis_exec_result v;
        assert(elpis_exec_take(r, 0, &v) == ELPIS_EXEC_CLOSED);
        elpis_exec_destroy(r);
    }
}
static pthread_mutex_t gate_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cv = PTHREAD_COND_INITIALIZER;
static int entered, released;
static void wait_gates(int n) {
    pthread_mutex_lock(&gate_mu);
    while (entered < n) pthread_cond_wait(&gate_cv, &gate_mu);
    pthread_mutex_unlock(&gate_mu);
}
static void reset_gate(void) {
    pthread_mutex_lock(&gate_mu); entered = released = 0; pthread_mutex_unlock(&gate_mu);
}
static elpis_exec_status gate(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    pthread_mutex_lock(&gate_mu);
    ++entered; pthread_cond_broadcast(&gate_cv);
    while (!released) pthread_cond_wait(&gate_cv, &gate_mu);
    pthread_mutex_unlock(&gate_mu);
    return echo(b, cap, out);
}
static void wait_gate(void) {
    pthread_mutex_lock(&gate_mu);
    while (!entered) pthread_cond_wait(&gate_cv, &gate_mu);
    pthread_mutex_unlock(&gate_mu);
}
static void open_gate(void) {
    pthread_mutex_lock(&gate_mu); released = 1; pthread_cond_broadcast(&gate_cv); pthread_mutex_unlock(&gate_mu);
}
/* Retirement is ordered; execution is not. A blocked head delays publication only:
 * later tasks, including one queued on the blocked task's own lane, still run. */
static void ordered_pressure(void) {
    elpis_exec_runtime *r = create(2, 3);
    reset_gate();
    submit(r, 0, 0, gate); wait_gate();
    submit(r, 1, 1, echo);
    submit(r, 2, 0, echo);
    elpis_exec_metrics m;
    unsigned tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (m.completed != 2 && ++tries < 10000);
    assert(m.completed == 2 && m.running == 1 && m.steals >= 1);
    elpis_exec_result result;
    assert(elpis_exec_take(r, 0, &result) == ELPIS_EXEC_WOULD_BLOCK);
    assert(elpis_exec_take(r, 2, &result) == ELPIS_EXEC_WOULD_BLOCK);
    elpis_exec_task t = {1, 9, 0, 0, 3, echo};
    elpis_exec_buffer *b = value(3);
    uint64_t seq = 999;
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_WOULD_BLOCK && b && seq == 999);
    assert(elpis_exec_buffer_mutable_data(b));
    assert(elpis_exec_cancel(r, 0) == ELPIS_EXEC_WOULD_BLOCK);
    assert(elpis_exec_cancel(r, 1) == ELPIS_EXEC_WOULD_BLOCK);
    assert(elpis_exec_cancel(r, 2) == ELPIS_EXEC_WOULD_BLOCK);
    assert(elpis_exec_cancel(r, 99) == ELPIS_EXEC_INVALID);
    open_gate();
    for (unsigned n = 0; n < 3; ++n) {
        result = take(r);
        assert(result.retire_ns <= (uint64_t)60 * 1000000000);
        if (n) assert(result.worker != UINT32_MAX);
        check_result(&result, n, n);
    }
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && seq == 3 && !b);
    result = take(r); check_result(&result, 3, 3);
    elpis_exec_get_metrics(r, &m);
    assert(m.high_water == 3 && m.queue_full == 1 && m.cancelled == 0 && m.retired == 4);
    elpis_exec_destroy(r);
}
/* Cancellation of genuinely queued work: every lane thread is blocked. */
static void queued_cancel(void) {
    elpis_exec_runtime *r = create(2, 4);
    reset_gate();
    submit(r, 0, 0, gate); submit(r, 1, 1, gate); wait_gates(2);
    submit(r, 2, 0, echo);
    assert(elpis_exec_cancel(r, 2) == ELPIS_EXEC_OK);
    assert(elpis_exec_cancel(r, 2) == ELPIS_EXEC_WOULD_BLOCK);
    open_gate();
    elpis_exec_result v = take(r); check_result(&v, 0, 0);
    v = take(r); check_result(&v, 1, 1);
    v = take(r);
    assert(v.status == ELPIS_EXEC_CANCELLED && v.sequence == 2 && !v.output && v.worker == UINT32_MAX);
    elpis_exec_destroy(r);
}
static void *close_cancel(void *arg) {
    assert(elpis_exec_shutdown(arg, 1) == ELPIS_EXEC_OK);
    return NULL;
}
static void lifecycle(void) {
    for (unsigned w = 1; w <= 4; ++w) for (int i = 0; i < 12; ++i) {
        elpis_exec_runtime *r = create(w, 4);
        assert(elpis_exec_shutdown(r, i % 2) == ELPIS_EXEC_OK);
        elpis_exec_task t = {1, 0, 0, 0, 0, echo};
        elpis_exec_buffer *b = value(0);
        uint64_t seq;
        assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_CLOSED && b);
        elpis_exec_buffer_release(b);
        elpis_exec_destroy(r);
    }
    elpis_exec_runtime *r = create(1, 4);
    reset_gate(); submit(r, 0, 0, gate); wait_gate();
    for (unsigned n = 1; n < 4; ++n) submit(r, n, 0, echo);
    pthread_t closer;
    assert(!pthread_create(&closer, NULL, close_cancel, r));
    elpis_exec_metrics m;
    unsigned tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (m.cancelled != 3 && ++tries < 10000);
    assert(m.cancelled == 3);
    open_gate(); assert(!pthread_join(closer, NULL));
    elpis_exec_result v = take(r); check_result(&v, 0, 0);
    for (unsigned n = 1; n < 4; ++n) { v = take(r); assert(v.sequence == n && v.status == ELPIS_EXEC_CANCELLED); }
    elpis_exec_destroy(r);
    r = create(3, 4);
    for (unsigned n = 0; n < 4; ++n) submit(r, n, n, n == 1 ? poison : echo);
    assert(elpis_exec_shutdown(r, 0) == ELPIS_EXEC_OK); /* Full completed queue cannot deadlock drain. */
    for (unsigned n = 0; n < 4; ++n) {
        v = take(r);
        if (n == 1) assert(v.sequence == 1 && v.status == ELPIS_EXEC_INTERNAL && !v.output);
        else check_result(&v, n, n);
    }
    elpis_exec_destroy(r);
    r = create(4, 4);
    for (unsigned n = 0; n < 4; ++n) submit(r, n, n, echo);
    elpis_exec_shutdown(r, 0);
    elpis_exec_destroy(r); /* Unconsumed outputs released. */
}
typedef struct { elpis_exec_runtime *r; unsigned producer; } producer_arg;
static void *producer(void *arg) {
    producer_arg *p = arg;
    for (unsigned i = 0; i < 150; ++i) {
        unsigned n = p->producer * 150 + i;
        elpis_exec_buffer *b = value(n);
        elpis_exec_task t = {1, 9, p->producer, 0, n, echo};
        uint64_t seq;
        elpis_exec_status code;
        do {
            code = elpis_exec_submit(p->r, &t, &b, &seq);
            if (code == ELPIS_EXEC_WOULD_BLOCK) pause_ms();
        } while (code == ELPIS_EXEC_WOULD_BLOCK);
        assert(code == ELPIS_EXEC_OK && !b);
    }
    return NULL;
}
static void contention(void) {
    elpis_exec_runtime *r = create(4, 7);
    pthread_t threads[6]; producer_arg args[6];
    unsigned seen[900] = {0};
    for (unsigned i = 0; i < 6; ++i) {
        args[i] = (producer_arg){r, i};
        assert(!pthread_create(&threads[i], NULL, producer, &args[i]));
    }
    for (unsigned seq = 0; seq < 900; ++seq) {
        elpis_exec_result v = take(r);
        assert(v.tag < 900 && seen[v.tag]++ == 0);
        check_result(&v, seq, (unsigned)v.tag);
    }
    for (unsigned i = 0; i < 6; ++i) assert(!pthread_join(threads[i], NULL));
    for (unsigned i = 0; i < 900; ++i) assert(seen[i] == 1);
    elpis_exec_destroy(r);
}
typedef struct { int mode; atomic_uint init, submit, poll, abort, shutdown; } fake;
static elpis_exec_status backend_init(void *p, uint64_t *caps) {
    fake *f = p; atomic_fetch_add(&f->init, 1); *caps = UINT64_C(1) << 1;
    return f->mode == 3 ? ELPIS_EXEC_BACKEND_UNAVAILABLE : ELPIS_EXEC_OK;
}
static elpis_exec_status backend_submit(void *p, const elpis_exec_task *t,
                                      const elpis_exec_buffer *b, size_t cap, void **token) {
    fake *f = p; atomic_fetch_add(&f->submit, 1);
    assert(t->operation == 1 && (t->flags & ELPIS_EXEC_PURE) && cap == 1024);
    if (f->mode == 1) return ELPIS_EXEC_BACKEND_REJECTED;
    *token = (void *)b;
    return ELPIS_EXEC_OK;
}
static elpis_exec_status backend_poll(void *p, void *token, elpis_exec_buffer **out) {
    fake *f = p; atomic_fetch_add(&f->poll, 1);
    if (f->mode == 2) return ELPIS_EXEC_WOULD_BLOCK;
    if (f->mode == 4) { copy(token, 1024, out); return ELPIS_EXEC_BACKEND_UNAVAILABLE; }
    return copy(token, 1024, out);
}
static void backend_abort(void *p, void *token) { assert(token); atomic_fetch_add(&((fake *)p)->abort, 1); }
static void backend_shutdown(void *p) { atomic_fetch_add(&((fake *)p)->shutdown, 1); }
static void accelerators(void) {
    for (int mode = 0; mode < 5; ++mode) {
        fake f = {.mode=mode};
        elpis_exec_backend backend = {&f, backend_init, backend_submit, backend_poll, backend_abort, backend_shutdown};
        elpis_exec_config c = config(4, 4); c.backend = &backend; c.backend_poll_limit = 3;
        elpis_exec_runtime *r;
        assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_OK);
        unsigned before = atomic_load(&cpu_calls);
        for (unsigned n = 0; n < 4; ++n) {
            /* two eligible, one unsupported kind, one not declared pure */
            elpis_exec_task t = {n == 2 ? 2u : 1u, 9, n, n == 3 ? 0u : ELPIS_EXEC_PURE, n, echo};
            elpis_exec_buffer *b = value(n); uint64_t seq;
            assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && seq == n);
        }
        elpis_exec_shutdown(r, 0);
        for (unsigned n = 0; n < 4; ++n) {
            elpis_exec_result v = take(r);
            assert(v.operation == (n == 2 ? 2u : 1u)); v.operation = 1;
            check_result(&v, n, n);
        }
        assert(atomic_load(&f.init) == 1 && atomic_load(&f.shutdown) == 1);
        assert(atomic_load(&cpu_calls) - before == (mode == 0 ? 2u : 4u));
        assert(atomic_load(&f.submit) == (mode == 3 ? 0u : 2u));
        assert(atomic_load(&f.poll) == (mode == 3 || mode == 1 ? 0u : mode == 2 ? 6u : 2u));
        assert(atomic_load(&f.abort) == (mode == 2 ? 2u : 0u));
        elpis_exec_metrics m; elpis_exec_get_metrics(r, &m);
        assert(m.backend_fallback == (mode == 0 || mode == 3 ? 0u : 2u));
        elpis_exec_destroy(r);
        assert(atomic_load(&f.shutdown) == 1);
    }
}
typedef struct {
    unsigned add;
    atomic_uint calls;
} bound_context;

static elpis_exec_status bound_add(void *context, const elpis_exec_buffer *b,
                                   size_t cap, elpis_exec_buffer **out) {
    bound_context *ctx = context;
    unsigned v = 0;
    assert(ctx && elpis_exec_buffer_size(b) == sizeof(v) && cap >= sizeof(v));
    memcpy(&v, elpis_exec_buffer_data(b), sizeof(v));
    v += ctx->add;
    *out = elpis_exec_buffer_alloc(sizeof(v));
    if (!*out) return ELPIS_EXEC_INTERNAL;
    memcpy(elpis_exec_buffer_mutable_data(*out), &v, sizeof(v));
    atomic_fetch_add(&ctx->calls, 1);
    return ELPIS_EXEC_OK;
}

static void bound_tasks(void) {
    elpis_exec_runtime *r = create(2, 4);
    bound_context ctx = {.add = 17};
    atomic_init(&ctx.calls, 0);
    for (unsigned n = 0; n < 4; ++n) {
        elpis_exec_bound_task t = {
            .operation = 2, .stage = 11, .affinity = n,
            .flags = ELPIS_EXEC_PURE, .tag = n,
            .compute = bound_add, .context = &ctx,
        };
        elpis_exec_buffer *b = value(n);
        uint64_t seq = UINT64_MAX;
        assert(elpis_exec_submit_bound(r, &t, &b, &seq) == ELPIS_EXEC_OK);
        assert(!b && seq == n);
    }
    for (unsigned n = 0; n < 4; ++n) {
        elpis_exec_result v = take(r);
        unsigned got = 0;
        assert(v.sequence == n && v.tag == n && v.operation == 2 && v.stage == 11);
        assert(v.status == ELPIS_EXEC_OK && v.output);
        memcpy(&got, elpis_exec_buffer_data(v.output), sizeof(got));
        assert(got == n + 17);
        elpis_exec_buffer_release(v.output);
    }
    assert(atomic_load(&ctx.calls) == 4);
    elpis_exec_destroy(r);
}

static void validation(void) {
    assert(elpis_exec_default_workers(0) == 1 && elpis_exec_default_workers(1) == 1);
    assert(elpis_exec_default_workers(2) == 1 && elpis_exec_default_workers(3) == 2);
    assert(elpis_exec_default_workers(4) == 3 && elpis_exec_default_workers(128) == 3);
    elpis_exec_config c = config(5, 4); elpis_exec_runtime *r = NULL;
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_INVALID && !r);
    c.workers = 1; c.capacity = 0;
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_INVALID);
    c.capacity = 4097; assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_INVALID);
    c.capacity = 4096; c.max_input_bytes = 64 * 1024 * 1024;
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_INVALID);
    assert(!elpis_exec_buffer_alloc((size_t)65 * 1024 * 1024));
    for (int fail = 0; fail < 4; ++fail) {
        fail_after = fail; c = config(4, 4);
        assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_INTERNAL && !r);
    }
    fail_after = -1;
    elpis_exec_pool_metrics pm;
    elpis_exec_get_pool_metrics(&pm);
    assert(pm.threads == 0 && pm.contexts == 0); /* failed creates left no threads */
    /* Contexts share one pool: any number may be live, never more than four threads. */
    elpis_exec_runtime *one = create(4, 1), *two = create(4, 1), *three = create(1, 1);
    elpis_exec_get_pool_metrics(&pm);
    assert(pm.threads == 4 && pm.contexts == 3);
    elpis_exec_shutdown(one, 0);
    elpis_exec_destroy(one); /* Repeated shutdown is harmless. */
    elpis_exec_get_pool_metrics(&pm);
    assert(pm.threads == 4 && pm.contexts == 2);
    elpis_exec_destroy(two); elpis_exec_destroy(three);
    elpis_exec_get_pool_metrics(&pm);
    assert(pm.threads == 0 && pm.contexts == 0); /* the last destroy joins the pool */
    r = create(0, 4);
    elpis_exec_metrics m; elpis_exec_get_metrics(r, &m); assert(m.workers >= 1 && m.workers <= 3);
    elpis_exec_buffer *b = value(55), *shared = b;
    elpis_exec_buffer_retain(shared);
    elpis_exec_task t = {64, 9, 0, 0, 55, echo}; uint64_t seq;
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_INVALID && b);
    t.operation = 1;
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && !b);
    assert(!elpis_exec_buffer_mutable_data(shared));
    elpis_exec_buffer_release(shared); /* Runtime alone now keeps input alive. */
    elpis_exec_result v = take(r);
    elpis_exec_destroy(r); /* Output outlives runtime. */
    check_result(&v, 0, 55);
    elpis_exec_destroy(NULL);
}
/* ---- scheduling invariants --------------------------------------------------- */
static atomic_int in_flight, peak_flight;
static void spin_ms(unsigned ms) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    do clock_gettime(CLOCK_MONOTONIC, &b);
    while ((b.tv_sec - a.tv_sec) * 1000 + (b.tv_nsec - a.tv_nsec) / 1000000 < (long)ms);
}
static elpis_exec_status tracked(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out, unsigned ms) {
    int now = atomic_fetch_add(&in_flight, 1) + 1, peak = atomic_load(&peak_flight);
    while (now > peak && !atomic_compare_exchange_weak(&peak_flight, &peak, now)) {}
    spin_ms(ms);
    atomic_fetch_sub(&in_flight, 1);
    return echo(b, cap, out);
}
static elpis_exec_status heavy(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    return tracked(b, cap, out, 40);
}
static elpis_exec_status light(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    return tracked(b, cap, out, 2);
}
static void drain_in_order(elpis_exec_runtime *r, unsigned first, unsigned count) {
    for (unsigned n = first; n < first + count; ++n) { elpis_exec_result v = take(r); check_result(&v, n, n); }
}
static void scheduling(void) {
    /* Skew: a blocked task and 40 more on the same lane. Every later task completes
     * while the blocked one holds its lane thread (no stranding, no HOL execution). */
    elpis_exec_runtime *r = create(4, 64);
    reset_gate();
    submit(r, 0, 0, gate); wait_gate();
    for (unsigned n = 1; n <= 40; ++n) submit(r, n, 0, echo);
    elpis_exec_metrics m;
    unsigned tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (m.completed != 40 && ++tries < 20000);
    assert(m.completed == 40 && m.running == 1 && m.queued == 0);
    open_gate();
    drain_in_order(r, 0, 41);
    elpis_exec_destroy(r);

    /* Utilisation: eight heavy tasks all on lane 0 run four at a time. */
    r = create(4, 16);
    atomic_store(&peak_flight, 0);
    for (unsigned n = 0; n < 8; ++n) submit(r, n, 0, heavy);
    unsigned used = 0;
    for (unsigned n = 0; n < 8; ++n) {
        elpis_exec_result v = take(r);
        used |= 1u << v.worker;
        check_result(&v, n, n);
    }
    assert(atomic_load(&peak_flight) == 4 && used == 0xfu);
    elpis_exec_destroy(r);

    /* Per-context cap: workers=1 keeps a context serial even when the pool is
     * wider, while another context uses the rest of the pool at the same time. */
    elpis_exec_runtime *wide = create(4, 16), *serial = create(1, 16);
    atomic_store(&peak_flight, 0);
    for (unsigned n = 0; n < 6; ++n) submit(serial, n, n, light);
    drain_in_order(serial, 0, 6);
    assert(atomic_load(&peak_flight) == 1);
    atomic_store(&peak_flight, 0);
    for (unsigned n = 0; n < 12; ++n) { submit(wide, n, n, light); submit(serial, 6 + n, n, light); }
    drain_in_order(wide, 0, 12); drain_in_order(serial, 6, 12);
    assert(atomic_load(&peak_flight) <= 4 && atomic_load(&peak_flight) >= 2);
    elpis_exec_pool_metrics pm; elpis_exec_get_pool_metrics(&pm);
    assert(pm.threads == 4 && pm.contexts == 2);
    uint64_t tasks = 0;
    for (unsigned i = 0; i < pm.threads; ++i) tasks += pm.worker[i].tasks;
    assert(tasks >= 30);
    elpis_exec_destroy(serial); elpis_exec_destroy(wide);
}
/* Nonblocking accelerator completion: an undecided token never occupies a worker. */
static atomic_int backend_ready;
static atomic_uint slow_polls;
static elpis_exec_status slow_init(void *p, uint64_t *caps) { (void)p; *caps = UINT64_C(1) << 1; return ELPIS_EXEC_OK; }
static elpis_exec_status slow_submit(void *p, const elpis_exec_task *t, const elpis_exec_buffer *b,
                                     size_t cap, void **token) {
    (void)p; (void)t; (void)cap; *token = (void *)b; return ELPIS_EXEC_OK;
}
static elpis_exec_status slow_poll(void *p, void *token, elpis_exec_buffer **out) {
    (void)p; atomic_fetch_add(&slow_polls, 1);
    if (!atomic_load(&backend_ready)) return ELPIS_EXEC_WOULD_BLOCK;
    return copy(token, 1024, out);
}
static void slow_abort(void *p, void *token) { (void)p; (void)token; assert(0 && "no abort expected"); }
static void slow_shutdown(void *p) { (void)p; }
static void backend_nonblocking(void) {
    elpis_exec_backend backend = {NULL, slow_init, slow_submit, slow_poll, slow_abort, slow_shutdown};
    elpis_exec_config c = config(2, 16); c.backend = &backend; c.backend_poll_limit = 1000;
    elpis_exec_runtime *r;
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_OK);
    atomic_store(&backend_ready, 0); atomic_store(&slow_polls, 0);
    elpis_exec_task t = {1, 9, 0, ELPIS_EXEC_PURE, 0, echo};
    elpis_exec_buffer *b = value(0); uint64_t seq;
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && seq == 0);
    for (unsigned n = 1; n <= 10; ++n) submit(r, n, 0, light); /* same lane as the token */
    elpis_exec_metrics m;
    unsigned tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (m.completed != 10 && ++tries < 20000);
    assert(m.completed == 10 && m.parked == 1 && m.running == 0); /* CPU work ran; token parked */
    unsigned polls = atomic_load(&slow_polls);
    assert(polls >= 1 && polls < 1000);
    atomic_store(&backend_ready, 1);
    elpis_exec_notify(r);
    drain_in_order(r, 0, 11);
    elpis_exec_get_metrics(r, &m);
    assert(m.backend_accepted == 1 && m.backend_fallback == 0 && m.parked == 0);
    elpis_exec_destroy(r);
}
/* Deferral: a CPU task whose resource is unavailable returns DEFER instead of
 * sleeping. It never occupies a pool thread or its context's cap while parked. */
static atomic_int resource;
static atomic_uint defer_attempts;
static elpis_exec_status waiter(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    atomic_fetch_add(&defer_attempts, 1);
    int now = atomic_fetch_add(&in_flight, 1) + 1, peak = atomic_load(&peak_flight);
    while (now > peak && !atomic_compare_exchange_weak(&peak_flight, &peak, now)) {}
    elpis_exec_status code = atomic_load(&resource) ? copy(b, cap, out) : ELPIS_EXEC_DEFER;
    atomic_fetch_sub(&in_flight, 1);
    return code;
}
static uint64_t mono_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}
static void deferral(void) {
    /* A serial context: the deferred head does not hold the only unit of its cap. */
    elpis_exec_runtime *r = create(1, 32);
    atomic_store(&resource, 0); atomic_store(&defer_attempts, 0);
    atomic_store(&in_flight, 0); atomic_store(&peak_flight, 0);
    uint64_t begin = mono_ms();
    assert(submit(r, 0, 0, waiter) == 0);
    for (unsigned n = 1; n <= 10; ++n) submit(r, n, 0, light);
    elpis_exec_metrics m;
    unsigned tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (m.completed != 10 && ++tries < 20000);
    assert(m.completed == 10 && m.queued == 0 && m.deferred >= 1);
    elpis_exec_result early;
    assert(elpis_exec_take(r, 0, &early) == ELPIS_EXEC_WOULD_BLOCK); /* order kept */
    atomic_store(&resource, 1);
    elpis_exec_notify(r);
    drain_in_order(r, 0, 11);
    uint64_t elapsed = mono_ms() - begin;
    unsigned attempts = atomic_load(&defer_attempts);
    elpis_exec_get_metrics(r, &m);
    /* Re-runs are >=1 ms apart (no spinning) and the cap held throughout. */
    assert(attempts >= 2 && m.deferred == attempts - 1 && attempts <= elapsed + 2);
    assert(atomic_load(&peak_flight) == 1 && m.parked == 0);
    elpis_exec_destroy(r);

    /* A deferred task is cancelled like queued work: explicitly or by shutdown. */
    r = create(2, 8);
    atomic_store(&resource, 0);
    uint64_t seq = submit(r, 0, 0, waiter), seq2 = submit(r, 1, 1, waiter);
    elpis_exec_status code;
    tries = 0;
    do { code = elpis_exec_cancel(r, seq); if (code != ELPIS_EXEC_OK) pause_ms(); }
    while (code != ELPIS_EXEC_OK && ++tries < 20000);
    assert(code == ELPIS_EXEC_OK);
    elpis_exec_result v = take(r);
    assert(v.sequence == seq && v.status == ELPIS_EXEC_CANCELLED && !v.output);
    tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (m.deferred < 3 && ++tries < 20000);
    assert(elpis_exec_shutdown(r, 1) == ELPIS_EXEC_OK);
    v = take(r);
    assert(v.sequence == seq2 && v.status == ELPIS_EXEC_CANCELLED && !v.output);
    elpis_exec_get_metrics(r, &m);
    assert(m.cancelled == 2 && m.parked == 0 && m.running == 0);
    elpis_exec_destroy(r);

    /* Shutdown racing re-runs: a task that defers after the cancel sweep is
     * cancelled rather than parked, so shutdown never waits on its resource. */
    for (unsigned round = 0; round < 100; ++round) {
        r = create(2, 8);
        for (unsigned n = 0; n < 4; ++n) submit(r, n, n, waiter);
        for (unsigned k = 0; k < round % 3; ++k) pause_ms();
        assert(elpis_exec_shutdown(r, 1) == ELPIS_EXEC_OK);
        for (unsigned n = 0; n < 4; ++n) {
            v = take(r);
            assert(v.sequence == n && v.status == ELPIS_EXEC_CANCELLED && !v.output);
        }
        elpis_exec_destroy(r);
    }
}
/* BACKEND_ONLY is the substrate-neutral hook for provider-owned recurrent state.
 * It deliberately has no CPU callback: accepted provider state is never replayed
 * implicitly on the host after a terminal backend failure. */
typedef struct {
    unsigned state;
    int mode; /* 0 normal, 1 submit reject, 2 poll timeout */
    int live;
    atomic_uint init, submit, poll, abort, shutdown;
} stream_fake;

static elpis_exec_status stream_init(void *p, uint64_t *caps) {
    stream_fake *f = p;
    atomic_fetch_add(&f->init, 1);
    *caps = UINT64_C(1) << 5;
    return ELPIS_EXEC_OK;
}
static elpis_exec_status stream_submit(void *p, const elpis_exec_task *t,
                                       const elpis_exec_buffer *b, size_t cap, void **token) {
    stream_fake *f = p;
    unsigned delta = 0;
    atomic_fetch_add(&f->submit, 1);
    assert(t->operation == 5 && t->flags == ELPIS_EXEC_BACKEND_ONLY && !t->compute);
    assert(cap >= sizeof(unsigned) && elpis_exec_buffer_size(b) == sizeof(unsigned));
    assert(!f->live);
    if (f->mode == 1) return ELPIS_EXEC_BACKEND_REJECTED;
    memcpy(&delta, elpis_exec_buffer_data(b), sizeof(delta));
    f->state += delta; /* provider-owned recurrent state */
    f->live = 1;
    *token = f;
    return ELPIS_EXEC_OK;
}
static elpis_exec_status stream_poll(void *p, void *token, elpis_exec_buffer **out) {
    stream_fake *f = p;
    atomic_fetch_add(&f->poll, 1);
    assert(token == f && f->live);
    if (f->mode == 2) return ELPIS_EXEC_WOULD_BLOCK;
    *out = elpis_exec_buffer_alloc(sizeof(unsigned));
    assert(*out);
    memcpy(elpis_exec_buffer_mutable_data(*out), &f->state, sizeof(f->state));
    f->live = 0;
    return ELPIS_EXEC_OK;
}
static void stream_abort(void *p, void *token) {
    stream_fake *f = p;
    assert(token == f && f->live);
    f->live = 0;
    atomic_fetch_add(&f->abort, 1);
}
static void stream_shutdown(void *p) {
    stream_fake *f = p;
    assert(!f->live);
    atomic_fetch_add(&f->shutdown, 1);
}
static elpis_exec_runtime *stream_runtime(stream_fake *f, unsigned polls) {
    elpis_exec_backend backend = {f, stream_init, stream_submit, stream_poll, stream_abort, stream_shutdown};
    elpis_exec_config c = config(1, 1);
    c.backend = &backend;
    c.backend_poll_limit = polls;
    elpis_exec_runtime *r = NULL;
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_OK && r);
    return r;
}
static void backend_only_stream(void) {
    unsigned before = atomic_load(&cpu_calls);

    /* Persistent provider state across sequential submissions; capacity=1 is the
     * adapter's recurrence fence. No CPU callback exists or executes. */
    stream_fake f = {0};
    elpis_exec_runtime *r = stream_runtime(&f, 8);
    unsigned deltas[] = {2, 3, 5}, expected = 0;
    for (unsigned n = 0; n < 3; ++n) {
        expected += deltas[n];
        elpis_exec_task t = {5, 17, 0, ELPIS_EXEC_BACKEND_ONLY, n, NULL};
        elpis_exec_buffer *b = value(deltas[n]);
        uint64_t seq = UINT64_MAX;
        assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && !b && seq == n);
        elpis_exec_result v = take(r);
        unsigned got = 0;
        assert(v.sequence == n && v.status == ELPIS_EXEC_OK && v.output);
        memcpy(&got, elpis_exec_buffer_data(v.output), sizeof(got));
        assert(got == expected);
        elpis_exec_buffer_release(v.output);
    }
    assert(f.state == 10 && atomic_load(&cpu_calls) == before);
    elpis_exec_metrics m;
    elpis_exec_get_metrics(r, &m);
    assert(m.backend_accepted == 3 && m.backend_fallback == 0 && m.submitted == 3);

    /* Unsupported capability is refused before admission: no input consumption,
     * no sequence allocation, so the caller may choose CPU before a provider
     * stream begins. */
    elpis_exec_task unsupported = {6, 17, 0, ELPIS_EXEC_BACKEND_ONLY, 99, NULL};
    elpis_exec_buffer *b = value(1);
    uint64_t seq = UINT64_MAX;
    assert(elpis_exec_submit(r, &unsupported, &b, &seq) == ELPIS_EXEC_BACKEND_UNAVAILABLE);
    assert(b && seq == UINT64_MAX);
    elpis_exec_buffer_release(b);

    /* Modes are intentionally disjoint, and BACKEND_ONLY cannot smuggle a CPU
     * callback that would become an accidental retry path. */
    elpis_exec_task mixed = {5, 17, 0, ELPIS_EXEC_PURE | ELPIS_EXEC_BACKEND_ONLY, 0, NULL};
    b = value(1); seq = UINT64_MAX;
    assert(elpis_exec_submit(r, &mixed, &b, &seq) == ELPIS_EXEC_INVALID && b);
    elpis_exec_buffer_release(b);
    elpis_exec_task callback = {5, 17, 0, ELPIS_EXEC_BACKEND_ONLY, 0, echo};
    b = value(1); seq = UINT64_MAX;
    assert(elpis_exec_submit(r, &callback, &b, &seq) == ELPIS_EXEC_INVALID && b);
    elpis_exec_buffer_release(b);
    elpis_exec_destroy(r);
    assert(atomic_load(&f.shutdown) == 1);

    /* No provider: synchronous unavailable and ownership retained. */
    r = create(1, 1);
    elpis_exec_task only = {5, 17, 0, ELPIS_EXEC_BACKEND_ONLY, 0, NULL};
    b = value(1); seq = UINT64_MAX;
    assert(elpis_exec_submit(r, &only, &b, &seq) == ELPIS_EXEC_BACKEND_UNAVAILABLE);
    assert(b && seq == UINT64_MAX);
    elpis_exec_buffer_release(b);
    elpis_exec_destroy(r);

    /* Provider rejection is terminal and never becomes a CPU call. */
    stream_fake reject = {.mode = 1};
    r = stream_runtime(&reject, 4);
    b = value(7); seq = UINT64_MAX;
    assert(elpis_exec_submit(r, &only, &b, &seq) == ELPIS_EXEC_OK && !b && seq == 0);
    elpis_exec_result v = take(r);
    assert(v.status == ELPIS_EXEC_BACKEND_REJECTED && !v.output);
    elpis_exec_get_metrics(r, &m);
    assert(m.backend_accepted == 0 && m.backend_fallback == 0 && reject.state == 0);
    assert(atomic_load(&cpu_calls) == before);
    elpis_exec_destroy(r);

    /* Timeout aborts/quiesces but does not pretend provider state rolled back.
     * The inference adapter must discard this stream rather than retry the token
     * on CPU after provider state has advanced. */
    stream_fake timeout = {.mode = 2};
    r = stream_runtime(&timeout, 3);
    b = value(7); seq = UINT64_MAX;
    assert(elpis_exec_submit(r, &only, &b, &seq) == ELPIS_EXEC_OK && !b && seq == 0);
    v = take(r);
    assert(v.status == ELPIS_EXEC_BACKEND_UNAVAILABLE && !v.output);
    elpis_exec_get_metrics(r, &m);
    assert(m.backend_accepted == 1 && m.backend_fallback == 0);
    assert(timeout.state == 7 && atomic_load(&timeout.abort) == 1);
    assert(atomic_load(&timeout.poll) == 3 && atomic_load(&cpu_calls) == before);
    elpis_exec_destroy(r);
}
int main(void) {
    validation(); deterministic(); ordered_pressure(); queued_cancel(); lifecycle(); contention(); accelerators();
    bound_tasks(); scheduling(); backend_nonblocking(); deferral(); backend_only_stream();
    puts("PASS execution: workers 1..4, order, pressure, cancellation, lifecycle, creation failure, contention, backend, bound tasks, shared pool, stealing, utilisation, caps, nonblocking backend, deferral, backend-only stream");
    return 0;
}
