/*
 * Bounded active-segment runtime session: global/base/local accounting,
 * capacity disposition before mutation, and (fault build) the append failure
 * classes and poisoning rules under a segment base.
 *
 * The seed is a synthetic but structurally valid quiescent runtime-history
 * state view (the session validates structure and recomputes the root; the
 * semantic replay authority is ECS_C Python, exercised by the Python suite).
 * Pre-existing segment frames are opaque structural frames: the native owner
 * only counts complete frames when it takes ownership.
 */
#define _GNU_SOURCE

#include "elpis/ecsc_history.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

#ifdef ECSC_SESSION_FAULTS
static int inject_enabled;
static int write_calls;
static int fsync_calls;
static int ftruncate_calls;
static int write_fail_on;
static int ftruncate_fail_on;

ssize_t __real_write(int fd, const void *buf, size_t count);
int __real_fsync(int fd);
int __real_ftruncate(int fd, off_t length);

ssize_t __wrap_write(int fd, const void *buf, size_t count)
{
    if (inject_enabled) {
        ++write_calls;
        if (write_fail_on > 0 && write_calls == write_fail_on) {
            errno = EIO;
            return -1;
        }
    }
    return __real_write(fd, buf, count);
}

int __wrap_fsync(int fd)
{
    if (inject_enabled) {
        ++fsync_calls;
    }
    return __real_fsync(fd);
}

int __wrap_ftruncate(int fd, off_t length)
{
    if (inject_enabled) {
        ++ftruncate_calls;
        if (ftruncate_fail_on > 0 && ftruncate_calls == ftruncate_fail_on) {
            errno = EIO;
            return -1;
        }
    }
    return __real_ftruncate(fd, length);
}

static void inject_reset(void)
{
    inject_enabled = 0;
    write_calls = 0;
    fsync_calls = 0;
    ftruncate_calls = 0;
    write_fail_on = 0;
    ftruncate_fail_on = 0;
}
#endif

/* Two ACTIVE entities; registry keys strictly ordered. */
#define SENDER  "1111111111111111111111111111111111111111111111111111111111111111"
#define HISTORY "2222222222222222222222222222222222222222222222222222222222222222"
#define DIG_A   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#define DIG_B   "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
#define DIG_C   "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
#define DIG_D   "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
#define DIG_E   "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

typedef struct fixture {
    elpis_ecsc_state_entity_view entities[2];
    elpis_ecsc_state_root_view root;
} fixture;

static void make_state(fixture *f, uint64_t clock)
{
    memset(f, 0, sizeof *f);

    f->entities[0].registry_key = SENDER;
    f->entities[0].entity_id = SENDER;
    f->entities[0].label = "pipeline";
    f->entities[0].label_len = 8u;
    f->entities[0].founding_index = 1u;
    f->entities[0].founding_digest = SENDER;
    f->entities[0].state_entity_id = SENDER;
    f->entities[0].prev_state_digest = DIG_A;
    f->entities[0].lifecycle = 2u; /* ACTIVE */
    f->entities[0].state_version = 1u;
    f->entities[0].state_digest = DIG_B;

    f->entities[1].registry_key = HISTORY;
    f->entities[1].entity_id = HISTORY;
    f->entities[1].label = "history";
    f->entities[1].label_len = 7u;
    f->entities[1].founding_index = 0u;
    f->entities[1].founding_digest = HISTORY;
    f->entities[1].state_entity_id = HISTORY;
    f->entities[1].prev_state_digest = DIG_C;
    f->entities[1].lifecycle = 2u; /* ACTIVE */
    f->entities[1].state_version = 1u;
    f->entities[1].state_digest = DIG_D;

    f->root.genesis_digest = DIG_E;
    f->root.history_digest = DIG_A;
    f->root.logical_clock = clock;
    f->root.mailbox_capacity = 16u;
    f->root.mailbox_default_capacity = 16u;
    f->root.next_founding_index = 2u;
    f->root.entities = f->entities;
    f->root.entity_count = 2u;
}

/* One opaque structural frame: 8-byte big-endian length + "{}". */
#define FRAME_BYTES 10u

static void write_segment(const char *path, uint64_t frames)
{
    static const uint8_t frame[FRAME_BYTES] = {0, 0, 0, 0, 0, 0, 0, 2, '{', '}'};
    uint64_t i;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);

    CHECK(fd >= 0);
    for (i = 0u; i < frames; ++i) {
        CHECK(write(fd, frame, sizeof frame) == (ssize_t)sizeof frame);
    }
    CHECK(close(fd) == 0);
}

static uint64_t file_size(const char *path)
{
    struct stat st;
    CHECK(stat(path, &st) == 0);
    return (uint64_t)st.st_size;
}

static const char PAYLOAD[] = "{\"receipt\":\"bounded-segment\"}";

static int open_seg(
    const char *path,
    const fixture *f,
    uint64_t global_count,
    uint64_t base,
    uint64_t max_bytes,
    uint64_t max_frames,
    elpis_ecsc_runtime_session **out
)
{
    elpis_ecsc_segment_policy policy;
    policy.max_segment_bytes = max_bytes;
    policy.max_segment_frames = max_frames;
    return elpis_ecsc_runtime_session_open_segment(
        path, strlen(path), &f->root, DIG_B, global_count, base, &policy, out);
}

static int record(elpis_ecsc_runtime_session *s, elpis_ecsc_runtime_record_plan_result *out)
{
    return elpis_ecsc_runtime_session_record(
        s, SENDER, HISTORY, PAYLOAD, sizeof PAYLOAD - 1u, out);
}

static void test_open_accounting(void)
{
    const char *path = "seg-accounting.seg";
    fixture f;
    elpis_ecsc_runtime_session *s = NULL;
    elpis_ecsc_segment_policy policy;

    make_state(&f, 10u);
    write_segment(path, 3u);

    /* Global 10, base 7, local frames 3 = 10 - 7. */
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_OK);
    CHECK(elpis_ecsc_runtime_session_event_count(s) == 10u);
    CHECK(elpis_ecsc_runtime_session_segment_base(s) == 7u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 3u);
    CHECK(elpis_ecsc_runtime_session_segment_bytes(s) == 3u * FRAME_BYTES);
    elpis_ecsc_runtime_session_close(s);
    s = NULL;

    /* Local frames != global - base. */
    CHECK(open_seg(path, &f, 10u, 8u, 1u << 20, 1024u, &s) == ELPIS_ECSC_SEGMENT_MISMATCH);
    CHECK(s == NULL);
    CHECK(open_seg(path, &f, 10u, 6u, 1u << 20, 1024u, &s) == ELPIS_ECSC_SEGMENT_MISMATCH);
    CHECK(s == NULL);
    /* Base beyond the global count. */
    CHECK(open_seg(path, &f, 10u, 11u, 1u << 20, 1024u, &s) == ELPIS_ECSC_SEGMENT_MISMATCH);
    CHECK(s == NULL);
    /* Global count must equal the validated logical clock. */
    CHECK(open_seg(path, &f, 11u, 8u, 1u << 20, 1024u, &s) == ELPIS_ECSC_INVALID);
    CHECK(s == NULL);
    /* Policy must be present, non-zero and finite. */
    CHECK(open_seg(path, &f, 10u, 7u, 0u, 1024u, &s) == ELPIS_ECSC_INVALID);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 0u, &s) == ELPIS_ECSC_INVALID);
    CHECK(open_seg(path, &f, 10u, 7u, UINT64_MAX, 1024u, &s) == ELPIS_ECSC_INVALID);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, UINT64_MAX, &s) == ELPIS_ECSC_INVALID);
    CHECK(elpis_ecsc_runtime_session_open_segment(
              path, strlen(path), &f.root, DIG_B, 10u, 7u, NULL, &s) == ELPIS_ECSC_INVALID);
    policy.max_segment_bytes = 1u << 20;
    policy.max_segment_frames = 1024u;
    CHECK(elpis_ecsc_runtime_session_open_segment(
              path, strlen(path), &f.root, DIG_B, 10u, 7u, &policy, NULL) == ELPIS_ECSC_INVALID);
    /* A segment already over policy is refused as full (compact first). */
    CHECK(open_seg(path, &f, 10u, 7u, 3u * FRAME_BYTES - 1u, 1024u, &s) == ELPIS_ECSC_SEGMENT_FULL);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 2u, &s) == ELPIS_ECSC_SEGMENT_FULL);
    CHECK(s == NULL);
    CHECK(file_size(path) == 3u * FRAME_BYTES);

    CHECK(elpis_ecsc_runtime_session_segment_base(NULL) == UINT64_MAX);
    CHECK(elpis_ecsc_runtime_session_segment_bytes(NULL) == UINT64_MAX);
    CHECK(elpis_ecsc_runtime_session_segment_frames(NULL) == UINT64_MAX);
    unlink(path);
}

static void test_record_reports_global_indices(void)
{
    const char *path = "seg-global.seg";
    fixture f;
    elpis_ecsc_runtime_session *s = NULL;
    elpis_ecsc_runtime_record_plan_result r;
    uint64_t before;

    make_state(&f, 10u);
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_OK);
    before = elpis_ecsc_runtime_session_segment_bytes(s);

    CHECK(record(s, &r) == ELPIS_ECSC_OK);
    /* Surviving numbering is global: never renumbered from the segment. */
    CHECK(r.enqueue_event_index == 10u);
    CHECK(r.processed_event_index == 11u);
    CHECK(r.final_logical_clock == 12u);
    CHECK(elpis_ecsc_runtime_session_event_count(s) == 12u);
    CHECK(elpis_ecsc_runtime_session_segment_base(s) == 7u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 5u);
    CHECK(elpis_ecsc_runtime_session_segment_bytes(s) ==
          before + 16u + r.enqueue_event_size + r.processed_event_size);
    CHECK(file_size(path) == elpis_ecsc_runtime_session_segment_bytes(s));

    CHECK(record(s, &r) == ELPIS_ECSC_OK);
    CHECK(r.enqueue_event_index == 12u);
    CHECK(elpis_ecsc_runtime_session_event_count(s) == 14u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 7u);
    elpis_ecsc_runtime_session_close(s);
    unlink(path);
}

static void test_capacity_refused_before_mutation(void)
{
    const char *path = "seg-capacity.seg";
    fixture f;
    elpis_ecsc_runtime_session *s = NULL;
    elpis_ecsc_runtime_record_plan_result r;
    char root_before[65];
    char root_after[65];
    uint64_t pair;
    uint64_t bytes;

    /* Learn the exact planned pair size on a scratch copy. */
    make_state(&f, 10u);
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_OK);
    CHECK(record(s, &r) == ELPIS_ECSC_OK);
    pair = 16u + r.enqueue_event_size + r.processed_event_size;
    elpis_ecsc_runtime_session_close(s);
    s = NULL;

    /* Byte bound: one byte short of the planned pair. */
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 3u * FRAME_BYTES + pair - 1u, 1024u, &s) == ELPIS_ECSC_OK);
    CHECK(elpis_ecsc_runtime_session_state_root_digest(s, root_before) == ELPIS_ECSC_OK);
    bytes = file_size(path);
#ifdef ECSC_SESSION_FAULTS
    inject_reset();
    inject_enabled = 1;
#endif
    CHECK(record(s, &r) == ELPIS_ECSC_SEGMENT_FULL);
#ifdef ECSC_SESSION_FAULTS
    /* Refused before any I/O. */
    CHECK(write_calls == 0 && fsync_calls == 0 && ftruncate_calls == 0);
    inject_reset();
#endif
    CHECK(file_size(path) == bytes);
    CHECK(elpis_ecsc_runtime_session_event_count(s) == 10u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 3u);
    /* Not poisoned; state unchanged. */
    CHECK(elpis_ecsc_runtime_session_state_root_digest(s, root_after) == ELPIS_ECSC_OK);
    CHECK(memcmp(root_before, root_after, 64u) == 0);
    CHECK(record(s, &r) == ELPIS_ECSC_SEGMENT_FULL);
    elpis_ecsc_runtime_session_close(s);
    s = NULL;

    /* Exactly fitting succeeds. */
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 3u * FRAME_BYTES + pair, 1024u, &s) == ELPIS_ECSC_OK);
    CHECK(record(s, &r) == ELPIS_ECSC_OK);
    CHECK(elpis_ecsc_runtime_session_segment_bytes(s) == 3u * FRAME_BYTES + pair);
    CHECK(record(s, &r) == ELPIS_ECSC_SEGMENT_FULL);
    CHECK(elpis_ecsc_runtime_session_event_count(s) == 12u);
    elpis_ecsc_runtime_session_close(s);
    s = NULL;

    /* Frame bound: room for one frame only, never an orphan enqueue. */
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 4u, &s) == ELPIS_ECSC_OK);
    CHECK(record(s, &r) == ELPIS_ECSC_SEGMENT_FULL);
    CHECK(file_size(path) == 3u * FRAME_BYTES);
    elpis_ecsc_runtime_session_close(s);
    unlink(path);
}

static void test_legacy_open_is_base_zero(void)
{
    const char *path = "seg-legacy.log";
    fixture f;
    elpis_ecsc_runtime_session *s = NULL;

    make_state(&f, 3u);
    write_segment(path, 3u);
    CHECK(elpis_ecsc_runtime_session_open(
              path, strlen(path), &f.root, DIG_B, 3u, &s) == ELPIS_ECSC_OK);
    CHECK(elpis_ecsc_runtime_session_segment_base(s) == 0u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 3u);
    elpis_ecsc_runtime_session_close(s);
    s = NULL;
    /* The original ABI keeps its original failure class. */
    write_segment(path, 2u);
    CHECK(elpis_ecsc_runtime_session_open(
              path, strlen(path), &f.root, DIG_B, 3u, &s) == ELPIS_ECSC_CORRUPT);
    unlink(path);
}

#ifdef ECSC_SESSION_FAULTS
static void test_fault_matrix_under_segment_base(void)
{
    const char *path = "seg-faults.seg";
    fixture f;
    elpis_ecsc_runtime_session *s = NULL;
    elpis_ecsc_runtime_record_plan_result r;
    char root[65];
    uint64_t bytes;

    make_state(&f, 10u);

    /* 1. Enqueue write fails, rollback succeeds: ROLLED_BACK, usable. */
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_OK);
    bytes = file_size(path);
    inject_reset();
    inject_enabled = 1;
    write_fail_on = 2; /* header ok, payload fails */
    CHECK(record(s, &r) == ELPIS_ECSC_APPEND_ROLLED_BACK);
    inject_reset();
    CHECK(file_size(path) == bytes);
    CHECK(elpis_ecsc_runtime_session_state_root_digest(s, root) == ELPIS_ECSC_OK);
    CHECK(elpis_ecsc_runtime_session_event_count(s) == 10u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 3u);
    CHECK(record(s, &r) == ELPIS_ECSC_OK);
    CHECK(r.enqueue_event_index == 10u);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == 5u);
    elpis_ecsc_runtime_session_close(s);
    s = NULL;

    /* 2. Enqueue durable, processed rolls back: PARTIAL_COMMIT, poisoned. */
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_OK);
    bytes = file_size(path);
    inject_reset();
    inject_enabled = 1;
    write_fail_on = 4; /* second frame's payload */
    CHECK(record(s, &r) == ELPIS_ECSC_PARTIAL_COMMIT);
    inject_reset();
    CHECK(file_size(path) > bytes); /* exactly the enqueue frame remains */
    CHECK(elpis_ecsc_runtime_session_state_root_digest(s, root) == ELPIS_ECSC_NOT_READY);
    CHECK(elpis_ecsc_runtime_session_segment_bytes(s) == UINT64_MAX);
    CHECK(elpis_ecsc_runtime_session_segment_frames(s) == UINT64_MAX);
    CHECK(record(s, &r) == ELPIS_ECSC_NOT_READY);
    elpis_ecsc_runtime_session_close(s);
    s = NULL;
    /* Reopen with the pre-call counts sees a non-quiescent extra frame:
     * native refuses; canonical replay must reconcile it. */
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_SEGMENT_MISMATCH);
    CHECK(s == NULL);

    /* 3. Enqueue write fails and rollback fails: UNCERTAIN, poisoned. */
    write_segment(path, 3u);
    CHECK(open_seg(path, &f, 10u, 7u, 1u << 20, 1024u, &s) == ELPIS_ECSC_OK);
    inject_reset();
    inject_enabled = 1;
    write_fail_on = 2;
    ftruncate_fail_on = 1;
    CHECK(record(s, &r) == ELPIS_ECSC_APPEND_UNCERTAIN);
    inject_reset();
    CHECK(elpis_ecsc_runtime_session_state_root_digest(s, root) == ELPIS_ECSC_NOT_READY);
    CHECK(record(s, &r) == ELPIS_ECSC_NOT_READY);
    elpis_ecsc_runtime_session_close(s);
    unlink(path);
}
#endif

int main(void)
{
    test_open_accounting();
    test_record_reports_global_indices();
    test_capacity_refused_before_mutation();
    test_legacy_open_is_base_zero();
#ifdef ECSC_SESSION_FAULTS
    test_fault_matrix_under_segment_base();
#endif
    puts("ok");
    return 0;
}
