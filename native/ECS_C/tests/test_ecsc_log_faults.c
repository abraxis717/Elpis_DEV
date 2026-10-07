#define _GNU_SOURCE

#include "elpis/ecsc_history.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int inject_enabled;

static int write_calls;
static int fsync_calls;
static int ftruncate_calls;

static int write_eintr_once;
static int write_short_once;
static int write_fail_on;

static int fsync_eintr_once;
static int fsync_fail_remaining;

static int ftruncate_eintr_once;
static int ftruncate_fail_once;

ssize_t __real_write(
    int fd,
    const void *buf,
    size_t count
);

int __real_fsync(int fd);

int __real_ftruncate(
    int fd,
    off_t length
);

ssize_t __wrap_write(
    int fd,
    const void *buf,
    size_t count
)
{
    if (!inject_enabled) {
        return __real_write(
            fd,
            buf,
            count
        );
    }

    ++write_calls;

    if (write_eintr_once) {
        write_eintr_once = 0;
        errno = EINTR;
        return -1;
    }

    if (write_fail_on > 0 &&
        write_calls == write_fail_on) {
        errno = EIO;
        return -1;
    }

    if (write_short_once &&
        count > 1u) {
        size_t short_count =
            count > 3u ? 3u : 1u;

        write_short_once = 0;

        return __real_write(
            fd,
            buf,
            short_count
        );
    }

    return __real_write(
        fd,
        buf,
        count
    );
}

int __wrap_fsync(int fd)
{
    if (!inject_enabled) {
        return __real_fsync(fd);
    }

    ++fsync_calls;

    if (fsync_eintr_once) {
        fsync_eintr_once = 0;
        errno = EINTR;
        return -1;
    }

    if (fsync_fail_remaining > 0) {
        --fsync_fail_remaining;
        errno = EIO;
        return -1;
    }

    return __real_fsync(fd);
}

int __wrap_ftruncate(
    int fd,
    off_t length
)
{
    if (!inject_enabled) {
        return __real_ftruncate(
            fd,
            length
        );
    }

    ++ftruncate_calls;

    if (ftruncate_eintr_once) {
        ftruncate_eintr_once = 0;
        errno = EINTR;
        return -1;
    }

    if (ftruncate_fail_once) {
        ftruncate_fail_once = 0;
        errno = EIO;
        return -1;
    }

    return __real_ftruncate(
        fd,
        length
    );
}

static void reset_faults(void)
{
    inject_enabled = 0;

    write_calls = 0;
    fsync_calls = 0;
    ftruncate_calls = 0;

    write_eintr_once = 0;
    write_short_once = 0;
    write_fail_on = 0;

    fsync_eintr_once = 0;
    fsync_fail_remaining = 0;

    ftruncate_eintr_once = 0;
    ftruncate_fail_once = 0;
}

static void create_unused_temp_path(
    char *path
)
{
    int fd = mkstemp(path);

    assert(fd >= 0);
    assert(close(fd) == 0);
    assert(unlink(path) == 0);
}

static uint64_t file_size(
    const char *path
)
{
    struct stat st;

    assert(stat(path, &st) == 0);
    assert(st.st_size >= 0);

    return (uint64_t)st.st_size;
}

static elpis_ecsc_log *open_ready_empty(
    const char *path
)
{
    elpis_ecsc_log *log = NULL;
    elpis_ecsc_log_scan scan;

    assert(
        elpis_ecsc_log_open(
            path,
            strlen(path),
            &log
        ) == ELPIS_ECSC_OK
    );

    assert(log != NULL);

    assert(
        elpis_ecsc_log_recover_scan(
            log,
            &scan
        ) == ELPIS_ECSC_OK
    );

    assert(scan.total_size == 0u);
    assert(scan.complete_prefix == 0u);
    assert(scan.frame_count == 0u);
    assert(scan.has_incomplete_tail == 0u);

    assert(
        elpis_ecsc_log_finish_recovery(
            log,
            0u
        ) == ELPIS_ECSC_OK
    );

    return log;
}

static void assert_one_exact_frame(
    const char *path,
    const uint8_t *payload,
    size_t payload_size
)
{
    uint8_t header[8];
    uint8_t *got;
    int fd;
    size_t done = 0u;

    assert(
        file_size(path) ==
        (uint64_t)(8u + payload_size)
    );

    fd = open(path, O_RDONLY);
    assert(fd >= 0);

    assert(
        read(fd, header, sizeof header) ==
        (ssize_t)sizeof header
    );

    assert(
        header[0] == 0u &&
        header[1] == 0u &&
        header[2] == 0u &&
        header[3] == 0u &&
        header[4] == 0u &&
        header[5] == 0u
    );

    assert(
        (((uint64_t)header[6] << 8) |
         (uint64_t)header[7]) ==
        (uint64_t)payload_size
    );

    got = (uint8_t *)malloc(payload_size);
    assert(got != NULL);

    while (done < payload_size) {
        ssize_t n = read(
            fd,
            got + done,
            payload_size - done
        );

        assert(n > 0);
        done += (size_t)n;
    }

    assert(
        memcmp(
            got,
            payload,
            payload_size
        ) == 0
    );

    free(got);
    assert(close(fd) == 0);
}

static void test_eintr_and_short_write_are_retried(void)
{
    static const uint8_t event[] =
        "{\"case\":\"retry\"}";

    char path[] =
        "/tmp/elpis_ecsc_retry_XXXXXX";

    elpis_ecsc_log *log;

    create_unused_temp_path(path);
    log = open_ready_empty(path);

    reset_faults();

    write_eintr_once = 1;
    write_short_once = 1;
    fsync_eintr_once = 1;
    inject_enabled = 1;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_OK
    );

    inject_enabled = 0;

    assert(write_calls >= 4);
    assert(fsync_calls >= 2);

    assert(
        elpis_ecsc_log_frame_count(log)
        == 1u
    );

    assert_one_exact_frame(
        path,
        event,
        sizeof event - 1u
    );

    elpis_ecsc_log_close(log);
    assert(unlink(path) == 0);
}

static void test_write_failure_rolls_back_durably(void)
{
    static const uint8_t event[] =
        "{\"case\":\"write-failure\"}";

    char path[] =
        "/tmp/elpis_ecsc_write_rb_XXXXXX";

    elpis_ecsc_log *log;

    create_unused_temp_path(path);
    log = open_ready_empty(path);

    reset_faults();

    /*
     * write #1 commits the frame prefix.
     * write #2 (event bytes) fails.
     */
    write_fail_on = 2;
    inject_enabled = 1;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_APPEND_ROLLED_BACK
    );

    inject_enabled = 0;

    assert(file_size(path) == 0u);
    assert(elpis_ecsc_log_size(log) == 0u);
    assert(
        elpis_ecsc_log_frame_count(log)
        == 0u
    );

    /* A definitely rolled-back owner remains usable. */
    reset_faults();

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_OK
    );

    assert_one_exact_frame(
        path,
        event,
        sizeof event - 1u
    );

    elpis_ecsc_log_close(log);
    assert(unlink(path) == 0);
}

static void test_append_fsync_failure_rolls_back_durably(void)
{
    static const uint8_t event[] =
        "{\"case\":\"fsync-failure\"}";

    char path[] =
        "/tmp/elpis_ecsc_fsync_rb_XXXXXX";

    elpis_ecsc_log *log;

    create_unused_temp_path(path);
    log = open_ready_empty(path);

    reset_faults();

    /*
     * First fsync is the attempted append durability barrier.
     * It fails. Rollback ftruncate + rollback fsync then succeed.
     */
    fsync_fail_remaining = 1;
    inject_enabled = 1;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_APPEND_ROLLED_BACK
    );

    inject_enabled = 0;

    assert(file_size(path) == 0u);
    assert(elpis_ecsc_log_size(log) == 0u);
    assert(
        elpis_ecsc_log_frame_count(log)
        == 0u
    );

    reset_faults();

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_OK
    );

    elpis_ecsc_log_close(log);
    assert(unlink(path) == 0);
}

static void test_rollback_ftruncate_retries_eintr(void)
{
    static const uint8_t event[] =
        "{\"case\":\"truncate-eintr\"}";

    char path[] =
        "/tmp/elpis_ecsc_trunc_eintr_XXXXXX";

    elpis_ecsc_log *log;

    create_unused_temp_path(path);
    log = open_ready_empty(path);

    reset_faults();

    /* Trigger rollback, then interrupt the first truncate attempt. */
    fsync_fail_remaining = 1;
    ftruncate_eintr_once = 1;
    inject_enabled = 1;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_APPEND_ROLLED_BACK
    );

    inject_enabled = 0;

    assert(ftruncate_calls >= 2);
    assert(file_size(path) == 0u);

    elpis_ecsc_log_close(log);
    assert(unlink(path) == 0);
}

static void test_failed_rollback_truncate_is_uncertain(void)
{
    static const uint8_t event[] =
        "{\"case\":\"uncertain-truncate\"}";

    char path[] =
        "/tmp/elpis_ecsc_uncertain_trunc_XXXXXX";

    elpis_ecsc_log *log;

    create_unused_temp_path(path);
    log = open_ready_empty(path);

    reset_faults();

    fsync_fail_remaining = 1;
    ftruncate_fail_once = 1;
    inject_enabled = 1;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_APPEND_UNCERTAIN
    );

    inject_enabled = 0;

    /*
     * The owner is poisoned/closed. It must never continue from an
     * indeterminate durability result.
     */
    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_NOT_READY
    );

    assert(
        elpis_ecsc_log_size(log)
        == UINT64_MAX
    );

    elpis_ecsc_log_close(log);

    /*
     * Do not assert whether the attempted event is durable. That is exactly
     * the uncertainty being represented.
     */
    assert(unlink(path) == 0);
}

static void test_failed_rollback_fsync_is_uncertain(void)
{
    static const uint8_t event[] =
        "{\"case\":\"uncertain-fsync\"}";

    char path[] =
        "/tmp/elpis_ecsc_uncertain_fsync_XXXXXX";

    elpis_ecsc_log *log;

    create_unused_temp_path(path);
    log = open_ready_empty(path);

    reset_faults();

    /*
     * fsync #1: append barrier fails.
     * rollback ftruncate succeeds.
     * fsync #2: rollback durability barrier fails.
     */
    fsync_fail_remaining = 2;
    inject_enabled = 1;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_APPEND_UNCERTAIN
    );

    inject_enabled = 0;

    assert(
        elpis_ecsc_log_append_event_bytes(
            log,
            event,
            sizeof event - 1u
        ) == ELPIS_ECSC_NOT_READY
    );

    assert(
        elpis_ecsc_log_frame_count(log)
        == UINT64_MAX
    );

    elpis_ecsc_log_close(log);

    /*
     * Locally the truncate may be visible, but without a successful rollback
     * fsync it is not promoted to a definite outcome.
     */
    assert(unlink(path) == 0);
}

int main(void)
{
    test_eintr_and_short_write_are_retried();
    test_write_failure_rolls_back_durably();
    test_append_fsync_failure_rolls_back_durably();
    test_rollback_ftruncate_retries_eintr();
    test_failed_rollback_truncate_is_uncertain();
    test_failed_rollback_fsync_is_uncertain();

    puts(
        "PASS_ECSC_LOG_FAULTS: "
        "EINTR/short-write retry, durable rollback, "
        "and uncertain rollback are distinguished"
    );

    return 0;
}
