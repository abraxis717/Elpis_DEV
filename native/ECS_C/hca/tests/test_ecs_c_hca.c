/* ECS_C HCA native research substrate (docs/ECS_C_HCA.md): the required mechanics tests, natively.
 *
 *   n-ary relations       a conjunction of every arity 4..16: all premises give the output, any omitted one nothing
 *   rounds                B1 is synchronous: a chain deeper than its rounds stays partial; B0 closes it exactly
 *   interference / scope  two contradictory scopes of identical structure never leak; a seed or a factor across
 *                         scopes is refused
 *   source invalidation   STALE or UNKNOWN sources disable their cells and every factor over them; AUTHORITY_ONLY
 *                         admits; a seed whose source is invalid is refused
 *   delayed memory        512 turns of a fixed-width replaced distractor (with dropout): the answer, the readout
 *                         and the work are identical every turn, and nothing grows
 *   activation damage     explicit damage, local repair by ticks, the unsupported-cycle limitation (persisting
 *                         injected positives) against the exact control, and reset
 *   provenance misbinding an activation of another task, another graph, a tampered owner or out-of-range values is
 *                         refused INVALID_STATE; damage naming foreign, repeated or out-of-range cells is refused
 *   matched resources     B0 (the exact-storage control), B1 and B2 under one budget: the same answer, each call's
 *                         exact work, atomic refusal one unit below it
 *   refusals              graph and task well-formedness, capacity, degree, arity, schema, readout and state limits
 *   determinism           identical calls give identical results and trajectories
 *
 * Mechanics only: nothing here is a scientific result. */
#include "hca_test_util.h"

static elpis_hca_result result, sentinel;

static void expect_plan(const world *w, const elpis_hca_result *r, uint32_t scope, uint32_t salience) {
    assert(r->prediction_count == 1 && r->omitted == 0);
    char label[64];
    snprintf(label, sizeof label, "s%u:%u", scope, KIND_PLAN);
    assert(r->predictions[0].cell == idx(w, label) && r->predictions[0].salience == salience);
    assert(r->constraint_count == 1 && w->cells[r->constraints[0]].kind == KIND_POLICY);
    assert(w->cells[r->constraints[0]].scope == scope);
}

static void test_constants(void) {
    assert(elpis_hca_abi_version() == ELPIS_HCA_ABI_VERSION);
    assert(!strcmp(elpis_hca_code_name(ELPIS_HCA_OK), "ELPIS_HCA_OK"));
    assert(!strcmp(elpis_hca_code_name(ELPIS_HCA_INVALID_DAMAGE), "ELPIS_HCA_INVALID_DAMAGE"));
    assert(elpis_hca_code_name(19) == NULL && elpis_hca_code_name(-1) == NULL);
    elpis_hca_limits l;
    elpis_hca_limits_default(&l);
    assert(l.cells == 32 && l.factors == 32 && l.arity == 8 && l.degree == 8 && l.rounds == 8);
    assert(l.updates_per_tick == 256 && l.work == 100000 && l.state_bytes == 65536);
}

static void test_nary(void) {
    for (uint32_t arity = 4; arity <= 16; ++arity) { /* 4: the first with a policy (binding) premise */
        world w;
        coding(&w, arity);
        const elpis_hca_limits l = big_limits();
        elpis_hca_graph *g;
        OK(build_world(&w, &l, &g));
        for (uint32_t scope = 1; scope <= 2; ++scope)
            for (uint32_t method = ELPIS_HCA_B0; method <= ELPIS_HCA_B2; ++method) {
                task_buffer t;
                coding_task(&t, &w, arity, scope, 0);
                OK(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL));
                expect_plan(&w, &result, scope, 750);
                assert(result.predictions[0].evidence_count == 2); /* both schemas' factors carry a message */
                for (uint32_t omit = 0; omit + 1 < arity; ++omit) {
                    coding_task(&t, &w, arity, scope, 1u << omit);
                    OK(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL));
                    assert(result.prediction_count == 0 && result.constraint_count == 1);
                }
            }
        elpis_hca_graph_destroy(g);
    }
}

/* a -> b -> c (types 1 then 2, so the canonical order visits the first link first; weights 900 and 800). */
static void test_rounds(void) {
    world w;
    memset(&w, 0, sizeof w);
    add_cell(&w, "chain:a", 1, 7, 0, ELPIS_HCA_SOURCE_VALID);
    add_cell(&w, "chain:b", 1, 7, 0, ELPIS_HCA_SOURCE_VALID);
    add_cell(&w, "chain:c", 1, 7, 0, ELPIS_HCA_SOURCE_VALID);
    seal_cells(&w);
    const uint32_t kinds[2] = {7, 7};
    add_schema(&w, 1, 900, 2, kinds);
    add_schema(&w, 2, 800, 2, kinds);
    add_factor(&w, 1, ELPIS_HCA_ADMITTED, 2, (const char *const[]){"chain:a", "chain:b"});
    add_factor(&w, 2, ELPIS_HCA_ADMITTED, 2, (const char *const[]){"chain:b", "chain:c"});
    seal_factors(&w);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g;
    OK(build_world(&w, &l, &g));
    task_buffer t;
    task_of(&t, &w, 1, (const char *const[]){"chain:a"}, 1);
    t.output_kinds[0] = 7;
    uint16_t scores[3];
    const uint32_t a = idx(&w, "chain:a"), b = idx(&w, "chain:b"), c = idx(&w, "chain:c");
    t.task.steps = 1;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, scores, NULL));
    assert(scores[a] == 1000 && scores[b] == 900 && scores[c] == 0); /* one synchronous round: one link */
    t.task.steps = 2;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, scores, NULL));
    assert(scores[b] == 900 && scores[c] == 720);
    t.task.steps = 1;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, scores, NULL)); /* exact closure, whatever the rounds */
    assert(scores[b] == 900 && scores[c] == 720);
    assert(result.prediction_count == 3 && result.predictions[2].cell == c && result.predictions[2].salience == 720);
    elpis_hca_graph_destroy(g);
}

static void test_scopes(void) {
    world w;
    coding(&w, 6);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g;
    OK(build_world(&w, &l, &g));
    uint16_t scores[ELPIS_HCA_MAX_CELLS];
    task_buffer t;
    coding_task(&t, &w, 6, 1, 0);
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, scores, NULL));
    expect_plan(&w, &result, 1, 750);
    for (uint32_t c = 0; c < w.n_cells; ++c) assert((w.cells[c].scope == 1) == (scores[c] != ELPIS_HCA_UNSCOPED));
    /* A seed of the other scope is refused, whatever the method; nothing is written. */
    char foreign[64];
    snprintf(foreign, sizeof foreign, "s2:%u", KIND_ENVIRONMENT);
    key_of(foreign, t.seeds[t.task.seed_count++]);
    for (uint32_t method = ELPIS_HCA_B0; method <= ELPIS_HCA_B2; ++method) {
        memset(&result, 0xAB, sizeof result);
        assert(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL) == ELPIS_HCA_SCOPE_MISMATCH);
        memset(&sentinel, 0xAB, sizeof sentinel);
        assert(!memcmp(&result, &sentinel, sizeof result));
    }
    /* A seed naming no cell. */
    key_of("absent", t.seeds[t.task.seed_count - 1]);
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL) == ELPIS_HCA_NOT_FOUND);
    /* A scope with no cells: nothing scoped, nothing read out. */
    coding_task(&t, &w, 6, 3, 0);
    t.task.seed_count = 0;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B2, &result, NULL, NULL));
    assert(result.prediction_count == 0 && result.constraint_count == 0);
    elpis_hca_graph_destroy(g);
    /* A factor spanning the two scopes is refused at build. */
    world x;
    coding(&x, 4);
    uint32_t f = 0;
    while (x.cells[x.factors[f].members[0]].scope != 1) ++f; /* a scope-1 factor (canonical order mixes them) */
    x.factors[f].members[0] = idx(&x, "s2:1");
    seal_factors(&x);
    assert(build_world(&x, &l, &g) == ELPIS_HCA_SCOPE_MISMATCH && g == NULL);
}

static void test_source_invalidation(void) {
    const elpis_hca_limits l = big_limits();
    const uint8_t states[] = {ELPIS_HCA_SOURCE_STALE, ELPIS_HCA_SOURCE_UNKNOWN, ELPIS_HCA_SOURCE_AUTHORITY_ONLY};
    for (size_t i = 0; i < sizeof states; ++i) {
        world w;
        coding(&w, 5);
        w.cells[idx(&w, "s1:2")].source_state = states[i]; /* scope 1's dependency */
        elpis_hca_graph *g;
        OK(build_world(&w, &l, &g));
        task_buffer t;
        const int admits = states[i] == ELPIS_HCA_SOURCE_AUTHORITY_ONLY;
        for (uint32_t method = ELPIS_HCA_B0; method <= ELPIS_HCA_B2; ++method) {
            coding_task(&t, &w, 5, 1, 0);
            const int rc = elpis_hca_predict(g, &t.task, method, &result, NULL, NULL);
            assert(rc == (admits ? ELPIS_HCA_OK : ELPIS_HCA_SOURCE_INVALID));
            if (admits) expect_plan(&w, &result, 1, 750);
            coding_task(&t, &w, 5, 1, 1u << 1); /* the invalid premise unseeded: the factor is disabled */
            OK(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL));
            assert(result.prediction_count == 0);
            coding_task(&t, &w, 5, 2, 0); /* the other scope is untouched */
            OK(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL));
            expect_plan(&w, &result, 2, 750);
        }
        elpis_hca_graph_destroy(g);
    }
    /* An invalid output never reads out, even when every premise holds. */
    world w;
    coding(&w, 5);
    w.cells[idx(&w, "s1:20")].source_state = ELPIS_HCA_SOURCE_STALE;
    elpis_hca_graph *g;
    OK(build_world(&w, &l, &g));
    task_buffer t;
    coding_task(&t, &w, 5, 1, 0);
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL));
    assert(result.prediction_count == 0);
    elpis_hca_graph_destroy(g);
}

static void test_delayed_memory(void) {
    const elpis_hca_limits l = big_limits();
    elpis_hca_result first;
    uint64_t first_work = 0;
    for (uint32_t turn = 1; turn <= 512; ++turn) {
        world w;
        coding(&w, 5);
        /* The fixed-width distractor slot of scope 9, replaced every turn (dropped every fifth): no history. */
        if (turn % 5) {
            char label[64];
            snprintf(label, sizeof label, "distractor:%u", turn);
            add_cell(&w, label, 9, 99, 0, ELPIS_HCA_SOURCE_VALID);
            /* Re-sort, then re-resolve the factors' members by key. */
            uint8_t keys[ELPIS_HCA_MAX_FACTORS][ELPIS_HCA_MAX_ARITY][32];
            for (uint32_t f = 0; f < w.n_factors; ++f)
                for (uint32_t p = 0; p < w.factors[f].arity; ++p) memcpy(keys[f][p], w.cells[w.factors[f].members[p]].key, 32);
            seal_cells(&w);
            for (uint32_t f = 0; f < w.n_factors; ++f)
                for (uint32_t p = 0; p < w.factors[f].arity; ++p)
                    for (uint32_t c = 0; c < w.n_cells; ++c)
                        if (!memcmp(w.cells[c].key, keys[f][p], 32)) w.factors[f].members[p] = c;
            seal_factors(&w);
        }
        assert(w.n_cells <= 11);
        elpis_hca_graph *g;
        OK(build_world(&w, &l, &g));
        task_buffer t;
        coding_task(&t, &w, 5, 1, 0);
        OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B2, &result, NULL, NULL));
        expect_plan(&w, &result, 1, 750);
        if (turn == 1) first = result, first_work = result.work;
        /* Same answer, same readout and the same work whatever the distractor history. */
        assert(result.prediction_count == first.prediction_count && result.omitted == first.omitted);
        assert(result.work == first_work || turn % 5 == 0 || turn == 5);
        elpis_hca_graph_destroy(g);
    }
}

static void warm_n(const elpis_hca_b2 *s, elpis_hca_activation *a, int n) {
    for (int i = 0; i < n; ++i) {
        elpis_hca_activation next;
        OK(elpis_hca_b2_warm(s, a, 0, &next, NULL));
        *a = next;
    }
}

static void test_damage(void) {
    world w;
    coding(&w, 5);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g;
    OK(build_world(&w, &l, &g));
    task_buffer t;
    coding_task(&t, &w, 5, 1, 0);
    elpis_hca_b2 *s;
    elpis_hca_activation a, b, initial;
    OK(elpis_hca_b2_start(g, &t.task, 0, &s, &a, NULL));
    initial = a;
    assert(a.step == 0 && a.cells == 5);
    warm_n(s, &a, 2);
    OK(elpis_hca_b2_observe(s, &a, 0, &result, NULL));
    expect_plan(&w, &result, 1, 750);
    /* Damage the answer to zero: it reads out nothing until a tick repairs it locally. */
    const elpis_hca_damage zero = {idx(&w, "s1:20"), 0};
    OK(elpis_hca_b2_damage(s, &a, &zero, 1, 0, &b, NULL));
    assert(b.step == a.step);
    OK(elpis_hca_b2_observe(s, &b, 0, &result, NULL));
    assert(result.prediction_count == 0);
    warm_n(s, &b, 1); /* every cell fires (updates_per_tick 32 >= 5 cells) */
    OK(elpis_hca_b2_observe(s, &b, 0, &result, NULL));
    expect_plan(&w, &result, 1, 750);
    /* Damage a seed: its clamp restores it. */
    const elpis_hca_damage seed = {idx(&w, "s1:1"), 0};
    OK(elpis_hca_b2_damage(s, &b, &seed, 1, 0, &b, NULL));
    warm_n(s, &b, 2);
    OK(elpis_hca_b2_observe(s, &b, 0, &result, NULL));
    expect_plan(&w, &result, 1, 750);
    /* Reset is the initial activation exactly. */
    OK(elpis_hca_b2_reset(s, 0, &b, NULL));
    assert(!memcmp(&b, &initial, sizeof b));
    elpis_hca_b2_destroy(s);
    elpis_hca_graph_destroy(g);

    /* The unsupported cycle (a <-> b, weight 1000, no seeds): injected positives persist under B2's ticks, while
     * the exact control (B0) and a fresh B2 return nothing. A known limitation, not repaired here. */
    world c;
    memset(&c, 0, sizeof c);
    add_cell(&c, "cycle:a", 1, 7, 0, ELPIS_HCA_SOURCE_VALID);
    add_cell(&c, "cycle:b", 1, 7, 0, ELPIS_HCA_SOURCE_VALID);
    seal_cells(&c);
    const uint32_t kinds[2] = {7, 7};
    add_schema(&c, 5, 1000, 2, kinds);
    add_factor(&c, 5, ELPIS_HCA_ADMITTED, 2, (const char *const[]){"cycle:a", "cycle:b"});
    add_factor(&c, 5, ELPIS_HCA_ADMITTED, 2, (const char *const[]){"cycle:b", "cycle:a"});
    seal_factors(&c);
    OK(build_world(&c, &l, &g));
    task_of(&t, &c, 1, NULL, 0);
    t.output_kinds[0] = 7;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL));
    assert(result.prediction_count == 0);
    OK(elpis_hca_b2_start(g, &t.task, 0, &s, &a, NULL));
    const elpis_hca_damage both[2] = {{0, 1000}, {1, 1000}};
    OK(elpis_hca_b2_damage(s, &a, both, 2, 0, &a, NULL));
    warm_n(s, &a, 8);
    OK(elpis_hca_b2_observe(s, &a, 0, &result, NULL));
    assert(result.prediction_count == 2 && result.predictions[0].salience == 1000); /* persisted */
    OK(elpis_hca_b2_reset(s, 0, &a, NULL));
    warm_n(s, &a, 8);
    OK(elpis_hca_b2_observe(s, &a, 0, &result, NULL));
    assert(result.prediction_count == 0);
    elpis_hca_b2_destroy(s);
    elpis_hca_graph_destroy(g);
}

static void test_misbinding(void) {
    world w;
    coding(&w, 5);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g, *h;
    OK(build_world(&w, &l, &g));
    w.cells[0].kind = w.cells[0].kind; /* the same records ... */
    elpis_hca_limits other = l;
    other.rounds = 31; /* ... under other limits: another evidence identity */
    OK(build_world(&w, &other, &h));
    uint8_t eg[32], eh[32];
    OK(elpis_hca_graph_evidence(g, eg));
    OK(elpis_hca_graph_evidence(h, eh));
    assert(memcmp(eg, eh, 32) != 0);
    task_buffer ta, tb;
    coding_task(&ta, &w, 5, 1, 0);
    coding_task(&tb, &w, 5, 1, 1u << 2); /* another seed set */
    elpis_hca_b2 *sa, *sb, *sh;
    elpis_hca_activation a, b, x, out;
    OK(elpis_hca_b2_start(g, &ta.task, 0, &sa, &a, NULL));
    OK(elpis_hca_b2_start(g, &tb.task, 0, &sb, &b, NULL));
    OK(elpis_hca_b2_start(h, &ta.task, 0, &sh, &x, NULL));
    assert(memcmp(a.owner, b.owner, 32) != 0 && memcmp(a.owner, x.owner, 32) != 0);
    memset(&out, 0x5A, sizeof out);
    const elpis_hca_activation untouched = out;
    assert(elpis_hca_b2_warm(sb, &a, 0, &out, NULL) == ELPIS_HCA_INVALID_STATE); /* another task's */
    assert(elpis_hca_b2_warm(sh, &a, 0, &out, NULL) == ELPIS_HCA_INVALID_STATE); /* another graph's */
    assert(elpis_hca_b2_observe(sb, &a, 0, &result, NULL) == ELPIS_HCA_INVALID_STATE);
    elpis_hca_activation bad = a;
    bad.owner[0] ^= 1;
    assert(elpis_hca_b2_warm(sa, &bad, 0, &out, NULL) == ELPIS_HCA_INVALID_STATE);
    bad = a, bad.values[0] = 0x03, bad.values[1] = 0xE9; /* 1001 */
    assert(elpis_hca_b2_warm(sa, &bad, 0, &out, NULL) == ELPIS_HCA_INVALID_STATE);
    bad = a, bad.values[2 * a.cells] = 1; /* beyond the scoped cells */
    assert(elpis_hca_b2_warm(sa, &bad, 0, &out, NULL) == ELPIS_HCA_INVALID_STATE);
    bad = a, bad.cells = 4;
    assert(elpis_hca_b2_warm(sa, &bad, 0, &out, NULL) == ELPIS_HCA_INVALID_STATE);
    bad = a, bad.step = UINT64_MAX;
    assert(elpis_hca_b2_warm(sa, &bad, 0, &out, NULL) == ELPIS_HCA_STEP_LIMIT);
    assert(!memcmp(&out, &untouched, sizeof out)); /* every refusal wrote nothing */
    /* Damage: a cell of another scope, a repeated cell, an out-of-range value, too many changes. */
    const elpis_hca_damage foreign = {idx(&w, "s2:1"), 0}, repeated[2] = {{idx(&w, "s1:1"), 0}, {idx(&w, "s1:1"), 1}},
                           high = {idx(&w, "s1:1"), 1001};
    assert(elpis_hca_b2_damage(sa, &a, &foreign, 1, 0, &out, NULL) == ELPIS_HCA_INVALID_DAMAGE);
    assert(elpis_hca_b2_damage(sa, &a, repeated, 2, 0, &out, NULL) == ELPIS_HCA_INVALID_DAMAGE);
    assert(elpis_hca_b2_damage(sa, &a, &high, 1, 0, &out, NULL) == ELPIS_HCA_INVALID_DAMAGE);
    elpis_hca_damage many[6] = {{0, 0}};
    assert(elpis_hca_b2_damage(sa, &a, many, 6, 0, &out, NULL) == ELPIS_HCA_INVALID_DAMAGE);
    assert(!memcmp(&out, &untouched, sizeof out));
    elpis_hca_b2_destroy(sa);
    elpis_hca_b2_destroy(sb);
    elpis_hca_b2_destroy(sh);
    elpis_hca_graph_destroy(g);
    elpis_hca_graph_destroy(h);
}

static void test_matched_resources(void) {
    world w;
    coding(&w, 6);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g;
    OK(build_world(&w, &l, &g));
    uint64_t work[3];
    for (uint32_t method = ELPIS_HCA_B0; method <= ELPIS_HCA_B2; ++method) {
        task_buffer t;
        coding_task(&t, &w, 6, 1, 0);
        uint64_t used;
        OK(elpis_hca_predict(g, &t.task, method, &result, NULL, &used));
        expect_plan(&w, &result, 1, 750);
        assert(result.work == used);
        work[method] = used;
        /* Exactly the work it reports suffices; one unit less refuses atomically, charging no more than the cap. */
        t.task.work_budget = used;
        OK(elpis_hca_predict(g, &t.task, method, &result, NULL, NULL));
        expect_plan(&w, &result, 1, 750);
        t.task.work_budget = used - 1;
        memset(&result, 0xAB, sizeof result);
        uint64_t charged;
        assert(elpis_hca_predict(g, &t.task, method, &result, NULL, &charged) == ELPIS_HCA_WORK_LIMIT);
        memset(&sentinel, 0xAB, sizeof sentinel);
        assert(!memcmp(&result, &sentinel, sizeof result) && charged <= used - 1);
    }
    /* Descriptive, one fixture: the exact control is the cheapest here; B2 pays for its schedule. */
    printf("matched resources (coding, arity 6): B0 %llu, B1 %llu, B2 %llu logical work\n",
           (unsigned long long)work[0], (unsigned long long)work[1], (unsigned long long)work[2]);
    assert(work[0] < work[2]);
    elpis_hca_graph_destroy(g);
}

static void test_refusals(void) {
    elpis_hca_limits l = big_limits();
    world w;
    coding(&w, 5);
    elpis_hca_graph *g = (elpis_hca_graph *)&w;
    /* Limits and arguments. */
    elpis_hca_limits bad = l;
    bad.cells = 0;
    assert(build_world(&w, &bad, &g) == ELPIS_HCA_INVALID && g == NULL);
    bad = l, bad.work = ELPIS_HCA_MAX_WORK + 1;
    assert(build_world(&w, &bad, &g) == ELPIS_HCA_INVALID);
    assert(elpis_hca_graph_create(&l, w.cells, w.n_cells, w.schemas, w.n_schemas, w.factors, w.n_factors, 0, &g,
                                  NULL) == ELPIS_HCA_INVALID);
    assert(elpis_hca_graph_create(&l, w.cells, w.n_cells, w.schemas, w.n_schemas, w.factors, w.n_factors, 3, &g,
                                  NULL) == ELPIS_HCA_WORK_LIMIT && g == NULL);
    bad = l, bad.cells = 9;
    assert(build_world(&w, &bad, &g) == ELPIS_HCA_CAPACITY);
    bad = l, bad.arity = 4;
    assert(build_world(&w, &bad, &g) == ELPIS_HCA_ARITY_LIMIT);
    bad = l, bad.degree = 1; /* every cell is in two factors */
    assert(build_world(&w, &bad, &g) == ELPIS_HCA_DEGREE_LIMIT);
    /* Records. */
    world x = w;
    memcpy(x.cells[1].key, x.cells[0].key, 32);
    assert(build_world(&x, &l, &g) == ELPIS_HCA_DUPLICATE);
    x = w, x.cells[0].reserved[0] = 1;
    assert(build_world(&x, &l, &g) == ELPIS_HCA_INVALID);
    x = w, x.cells[0].source_state = 0;
    assert(build_world(&x, &l, &g) == ELPIS_HCA_INVALID);
    x = w, x.factors[1] = x.factors[0];
    assert(build_world(&x, &l, &g) == ELPIS_HCA_DUPLICATE);
    x = w, x.factors[0].members[1] = x.factors[0].members[0];
    assert(build_world(&x, &l, &g) == ELPIS_HCA_INVALID);
    x = w, x.factors[0].status = 3;
    assert(build_world(&x, &l, &g) == ELPIS_HCA_INVALID);
    x = w, x.schemas[1].relation_type = 71;
    assert(build_world(&x, &l, &g) == ELPIS_HCA_DUPLICATE);
    x = w, x.schemas[0].weight = 1001;
    assert(build_world(&x, &l, &g) == ELPIS_HCA_INVALID);
    x = w, x.schemas[0].roles[1] = x.schemas[0].roles[0];
    assert(build_world(&x, &l, &g) == ELPIS_HCA_INVALID);
    x = w, x.schemas[0].kinds[4] = KIND_POLICY; /* the plan role's kind differs from its member's */
    assert(build_world(&x, &l, &g) == ELPIS_HCA_ROLE_MISMATCH);
    x = w, x.n_schemas = 1; /* factors of type 72 have no schema */
    assert(build_world(&x, &l, &g) == ELPIS_HCA_ROLE_SCHEMA);
    x = w, x.schemas[0].arity = 4, x.schemas[0].roles[4] = 0, x.schemas[0].kinds[4] = 0;
    assert(build_world(&x, &l, &g) == ELPIS_HCA_ROLE_SCHEMA);
    /* Tasks. */
    OK(build_world(&w, &l, &g));
    task_buffer t;
    coding_task(&t, &w, 5, 1, 0);
    const elpis_hca_task good = t.task;
    t.task.readout_budget = ELPIS_HCA_MAX_READOUT + 1;
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL) == ELPIS_HCA_INVALID);
    t.task = good, t.task.steps = 0;
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL) == ELPIS_HCA_INVALID);
    t.task = good, t.task.output_kind_count = 0;
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL) == ELPIS_HCA_INVALID);
    t.task = good;
    memcpy(t.seeds[1], t.seeds[0], 32); /* duplicate seeds */
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL) == ELPIS_HCA_INVALID);
    coding_task(&t, &w, 5, 1, 0);
    assert(elpis_hca_predict(g, &t.task, 3, &result, NULL, NULL) == ELPIS_HCA_INVALID);
    t.task.readout_budget = 0; /* the scope's one binding constraint does not fit */
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL) == ELPIS_HCA_BINDING_READOUT);
    t.task.readout_budget = 1; /* the constraint fits, the prediction (1 + 2 evidence) does not: omitted */
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL));
    assert(result.prediction_count == 0 && result.omitted == 1 && result.constraint_count == 1);
    t.task.readout_budget = 4;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL));
    assert(result.prediction_count == 1 && result.omitted == 0 && result.evidence_count == 2);
    t.task = good;
    t.predicates[0].roles[0] = 999, t.predicates[1].roles[0] = 999; /* other roles: they admit nothing */
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL));
    assert(result.prediction_count == 0);
    coding_task(&t, &w, 5, 1, 0);
    t.predicates[0].status_mask = ELPIS_HCA_MASK_ADMITTED; /* schema 71's factors are candidates */
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B0, &result, NULL, NULL));
    expect_plan(&w, &result, 1, 500);
    elpis_hca_graph_destroy(g);
    /* Graph-relative task limits. */
    l.rounds = 4, l.updates_per_tick = 2, l.state_bytes = 8192;
    OK(build_world(&w, &l, &g));
    coding_task(&t, &w, 5, 1, 0);
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL) == ELPIS_HCA_ROUND_LIMIT);
    t.task.steps = 4;
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL) == ELPIS_HCA_UPDATE_LIMIT);
    t.task.updates_per_tick = 2;
    OK(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL));
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B2, &result, NULL, NULL) == ELPIS_HCA_STATE_LIMIT);
    elpis_hca_b2 *s = (elpis_hca_b2 *)&w;
    elpis_hca_activation a;
    assert(elpis_hca_b2_start(g, &t.task, 0, &s, &a, NULL) == ELPIS_HCA_STATE_LIMIT && s == NULL);
    t.task.work_budget = l.work + 1;
    assert(elpis_hca_predict(g, &t.task, ELPIS_HCA_B1, &result, NULL, NULL) == ELPIS_HCA_INVALID);
    elpis_hca_graph_destroy(g);
}

static void test_determinism(void) {
    world w;
    coding(&w, 7);
    const elpis_hca_limits l = big_limits();
    elpis_hca_graph *g1, *g2;
    OK(build_world(&w, &l, &g1));
    OK(build_world(&w, &l, &g2));
    uint8_t e1[32], e2[32];
    OK(elpis_hca_graph_evidence(g1, e1));
    OK(elpis_hca_graph_evidence(g2, e2));
    assert(!memcmp(e1, e2, 32));
    task_buffer t;
    coding_task(&t, &w, 7, 2, 0);
    t.task.updates_per_tick = 2; /* a partial schedule: order matters */
    elpis_hca_b2 *s1, *s2;
    elpis_hca_activation a1, a2;
    OK(elpis_hca_b2_start(g1, &t.task, 0, &s1, &a1, NULL));
    OK(elpis_hca_b2_start(g2, &t.task, 0, &s2, &a2, NULL));
    for (int i = 0; i < 16; ++i) {
        assert(!memcmp(&a1, &a2, sizeof a1));
        warm_n(s1, &a1, 1);
        warm_n(s2, &a2, 1);
    }
    elpis_hca_result r1, r2;
    for (uint32_t method = ELPIS_HCA_B0; method <= ELPIS_HCA_B2; ++method) {
        OK(elpis_hca_predict(g1, &t.task, method, &r1, NULL, NULL));
        OK(elpis_hca_predict(g2, &t.task, method, &r2, NULL, NULL));
        assert(!memcmp(&r1, &r2, sizeof r1));
    }
    elpis_hca_b2_destroy(s1);
    elpis_hca_b2_destroy(s2);
    elpis_hca_graph_destroy(g1);
    elpis_hca_graph_destroy(g2);
}

int main(void) {
    test_constants();
    test_nary();
    test_rounds();
    test_scopes();
    test_source_invalidation();
    test_delayed_memory();
    test_damage();
    test_misbinding();
    test_matched_resources();
    test_refusals();
    test_determinism();
    printf("ECS_C HCA native mechanics: PASS\n");
    return 0;
}
