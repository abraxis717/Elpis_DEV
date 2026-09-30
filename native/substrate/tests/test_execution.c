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
static void reset_gate(void) {
    pthread_mutex_lock(&gate_mu); entered = released = 0; pthread_mutex_unlock(&gate_mu);
}
static elpis_exec_status gate(const elpis_exec_buffer *b, size_t cap, elpis_exec_buffer **out) {
    pthread_mutex_lock(&gate_mu);
    entered = 1; pthread_cond_broadcast(&gate_cv);
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
static void ordered_pressure(void) {
    elpis_exec_runtime *r = create(2, 3);
    reset_gate();
    submit(r, 0, 0, gate); wait_gate();
    submit(r, 1, 1, echo);
    submit(r, 2, 0, echo);
    elpis_exec_metrics m;
    unsigned tries = 0;
    do { elpis_exec_get_metrics(r, &m); pause_ms(); } while (!m.completed && ++tries < 10000);
    assert(m.completed == 1); /* Later task finished, first still blocked. */
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
    assert(elpis_exec_cancel(r, 99) == ELPIS_EXEC_INVALID);
    assert(elpis_exec_cancel(r, 2) == ELPIS_EXEC_OK);
    open_gate();
    for (unsigned n = 0; n < 2; ++n) { result = take(r); check_result(&result, n, n); }
    result = take(r);
    assert(result.status == ELPIS_EXEC_CANCELLED && result.sequence == 2 && !result.output);
    assert(elpis_exec_submit(r, &t, &b, &seq) == ELPIS_EXEC_OK && seq == 3 && !b);
    result = take(r); check_result(&result, 3, 3);
    elpis_exec_get_metrics(r, &m);
    assert(m.high_water == 3 && m.queue_full == 1 && m.cancelled == 1 && m.retired == 4);
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
    elpis_exec_runtime *one = create(2, 1), *two = create(2, 1);
    c = config(1, 1);
    assert(elpis_exec_create(&c, &r) == ELPIS_EXEC_WOULD_BLOCK && !r);
    elpis_exec_shutdown(one, 0);
    r = create(2, 1);
    elpis_exec_destroy(one); /* Repeated shutdown must not release twice. */
    elpis_exec_runtime *extra = NULL;
    assert(elpis_exec_create(&c, &extra) == ELPIS_EXEC_WOULD_BLOCK && !extra);
    elpis_exec_destroy(two); elpis_exec_destroy(r);
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
int main(void) {
    validation(); deterministic(); ordered_pressure(); lifecycle(); contention(); accelerators(); bound_tasks();
    puts("PASS execution: workers 1..4, order, pressure, cancellation, lifecycle, creation failure, contention, backend, bound tasks");
    return 0;
}
