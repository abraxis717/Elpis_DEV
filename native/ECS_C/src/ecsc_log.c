#define _GNU_SOURCE
#include "elpis/ecsc_history.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ECSC_MAX_FRAME_BYTES UINT64_C(262144)
#define ECSC_MAX_PATH_BYTES 4096u

struct elpis_ecsc_log {
    int fd;

    uint64_t size;
    uint64_t complete_prefix;
    uint64_t frame_count;

    int recovering;
    int scanned;
    int poisoned;
};

static int close_fd(elpis_ecsc_log *log)
{
    int rc = 0;

    if (log != NULL && log->fd >= 0) {
        if (close(log->fd) != 0) {
            rc = ELPIS_ECSC_IO;
        }
        log->fd = -1;
    }

    return rc;
}

static int stat_size(
    int fd,
    uint64_t *out
)
{
    struct stat st;

    if (out == NULL) {
        return ELPIS_ECSC_INVALID;
    }

    if (fstat(fd, &st) != 0 ||
        st.st_size < 0) {
        return ELPIS_ECSC_IO;
    }

    *out = (uint64_t)st.st_size;
    return ELPIS_ECSC_OK;
}

static int fsync_checked(int fd)
{
    for (;;) {
        if (fsync(fd) == 0) {
            return ELPIS_ECSC_OK;
        }

        if (errno == EINTR) {
            continue;
        }

        return ELPIS_ECSC_IO;
    }
}

static int ftruncate_checked(
    int fd,
    uint64_t size
)
{
    for (;;) {
        if (ftruncate(
                fd,
                (off_t)size
            ) == 0) {
            return ELPIS_ECSC_OK;
        }

        if (errno == EINTR) {
            continue;
        }

        return ELPIS_ECSC_IO;
    }
}

static int pread_all(
    int fd,
    void *buffer,
    size_t count,
    uint64_t offset
)
{
    uint8_t *p = (uint8_t *)buffer;
    size_t done = 0u;

    if (buffer == NULL && count != 0u) {
        return ELPIS_ECSC_INVALID;
    }

    while (done < count) {
        ssize_t n = pread(
            fd,
            p + done,
            count - done,
            (off_t)(offset + done)
        );

        if (n > 0) {
            done += (size_t)n;
            continue;
        }

        if (n < 0 && errno == EINTR) {
            continue;
        }

        if (n == 0) {
            return ELPIS_ECSC_CORRUPT;
        }

        return ELPIS_ECSC_IO;
    }

    return ELPIS_ECSC_OK;
}

static int write_all(
    int fd,
    const void *buffer,
    size_t count
)
{
    const uint8_t *p =
        (const uint8_t *)buffer;

    size_t done = 0u;

    if (buffer == NULL && count != 0u) {
        return ELPIS_ECSC_INVALID;
    }

    while (done < count) {
        ssize_t n = write(
            fd,
            p + done,
            count - done
        );

        if (n > 0) {
            done += (size_t)n;
            continue;
        }

        if (n < 0 && errno == EINTR) {
            continue;
        }

        return ELPIS_ECSC_IO;
    }

    return ELPIS_ECSC_OK;
}

static uint64_t decode_be64(
    const uint8_t bytes[8]
)
{
    uint64_t value = 0u;
    size_t i;

    for (i = 0u; i < 8u; ++i) {
        value =
            (value << 8) |
            (uint64_t)bytes[i];
    }

    return value;
}

static void encode_be64(
    uint64_t value,
    uint8_t out[8]
)
{
    size_t i;

    for (i = 0u; i < 8u; ++i) {
        out[7u - i] =
            (uint8_t)(value & UINT64_C(0xff));

        value >>= 8;
    }
}

static int partial_header_possible(
    const uint8_t *header,
    size_t size
)
{
    uint64_t low = 0u;
    uint64_t high;
    unsigned missing_bits;
    size_t i;

    if (header == NULL ||
        size == 0u ||
        size >= 8u) {
        return 0;
    }

    for (i = 0u; i < size; ++i) {
        low =
            (low << 8) |
            (uint64_t)header[i];
    }

    missing_bits =
        (unsigned)(8u * (8u - size));

    low <<= missing_bits;

    if (missing_bits == 64u) {
        high = UINT64_MAX;
    } else {
        high =
            low +
            ((UINT64_C(1) << missing_bits) -
             UINT64_C(1));
    }

    if (low > ECSC_MAX_FRAME_BYTES ||
        high < UINT64_C(1)) {
        return 0;
    }

    return 1;
}

static int boundary_in_scan(
    elpis_ecsc_log *log,
    uint64_t target
)
{
    uint64_t offset = 0u;

    if (target == 0u) {
        return 1;
    }

    while (offset < log->complete_prefix) {
        uint8_t header[8];
        uint64_t length;
        int rc;

        rc = pread_all(
            log->fd,
            header,
            sizeof header,
            offset
        );

        if (rc != ELPIS_ECSC_OK) {
            return 0;
        }

        length = decode_be64(header);

        if (length == 0u ||
            length > ECSC_MAX_FRAME_BYTES) {
            return 0;
        }

        offset += UINT64_C(8) + length;

        if (offset == target) {
            return 1;
        }

        if (offset > target) {
            return 0;
        }
    }

    return 0;
}

static int poison_uncertain(
    elpis_ecsc_log *log
)
{
    if (log != NULL) {
        log->poisoned = 1;
        log->recovering = 1;
        log->scanned = 0;
        close_fd(log);
    }

    return ELPIS_ECSC_APPEND_UNCERTAIN;
}

static int rollback_append(
    elpis_ecsc_log *log,
    uint64_t start
)
{
    if (ftruncate_checked(
            log->fd,
            start
        ) != ELPIS_ECSC_OK) {
        return poison_uncertain(log);
    }

    if (fsync_checked(log->fd) !=
        ELPIS_ECSC_OK) {
        return poison_uncertain(log);
    }

    log->size = start;

    return ELPIS_ECSC_APPEND_ROLLED_BACK;
}

int elpis_ecsc_log_open(
    const char *path,
    size_t path_len,
    elpis_ecsc_log **out_log
)
{
    elpis_ecsc_log *log;
    char *path_copy;
    int fd;
    int flags;
    int saved_errno;

    if (out_log == NULL ||
        path == NULL ||
        path_len == 0u ||
        path_len > ECSC_MAX_PATH_BYTES ||
        memchr(path, '\0', path_len) != NULL) {
        return ELPIS_ECSC_INVALID;
    }

    *out_log = NULL;

    path_copy =
        (char *)malloc(path_len + 1u);

    if (path_copy == NULL) {
        return ELPIS_ECSC_IO;
    }

    memcpy(path_copy, path, path_len);
    path_copy[path_len] = '\0';

    flags =
        O_RDWR |
        O_CREAT |
        O_APPEND;

#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif

    fd = open(
        path_copy,
        flags,
        (mode_t)0600
    );

    saved_errno = errno;
    free(path_copy);

    if (fd < 0) {
        errno = saved_errno;
        return ELPIS_ECSC_IO;
    }

    if (flock(
            fd,
            LOCK_EX | LOCK_NB
        ) != 0) {
        saved_errno = errno;
        close(fd);

        if (saved_errno == EWOULDBLOCK ||
            saved_errno == EAGAIN) {
            return ELPIS_ECSC_LOCKED;
        }

        return ELPIS_ECSC_IO;
    }

    log = (elpis_ecsc_log *)calloc(
        1u,
        sizeof *log
    );

    if (log == NULL) {
        close(fd);
        return ELPIS_ECSC_IO;
    }

    log->fd = fd;
    log->recovering = 1;
    log->scanned = 0;
    log->poisoned = 0;

    if (stat_size(
            fd,
            &log->size
        ) != ELPIS_ECSC_OK) {
        close(fd);
        free(log);
        return ELPIS_ECSC_IO;
    }

    *out_log = log;
    return ELPIS_ECSC_OK;
}

void elpis_ecsc_log_close(
    elpis_ecsc_log *log
)
{
    if (log == NULL) {
        return;
    }

    close_fd(log);

    log->poisoned = 1;
    log->recovering = 1;
    log->scanned = 0;

    free(log);
}

int elpis_ecsc_log_recover_scan(
    elpis_ecsc_log *log,
    elpis_ecsc_log_scan *out_scan
)
{
    uint64_t initial_size;
    uint64_t final_size;
    uint64_t offset = 0u;
    uint64_t frames = 0u;
    uint32_t incomplete = 0u;
    int rc;

    if (log == NULL ||
        out_scan == NULL ||
        log->fd < 0 ||
        log->poisoned ||
        !log->recovering) {
        return ELPIS_ECSC_NOT_READY;
    }

    memset(out_scan, 0, sizeof *out_scan);

    rc = stat_size(
        log->fd,
        &initial_size
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    while (offset < initial_size) {
        uint64_t remaining =
            initial_size - offset;

        if (remaining < UINT64_C(8)) {
            uint8_t partial[8];

            rc = pread_all(
                log->fd,
                partial,
                (size_t)remaining,
                offset
            );

            if (rc != ELPIS_ECSC_OK) {
                return rc;
            }

            if (!partial_header_possible(
                    partial,
                    (size_t)remaining
                )) {
                return ELPIS_ECSC_CORRUPT;
            }

            incomplete = 1u;
            break;
        }

        {
            uint8_t header[8];
            uint64_t length;

            rc = pread_all(
                log->fd,
                header,
                sizeof header,
                offset
            );

            if (rc != ELPIS_ECSC_OK) {
                return rc;
            }

            length = decode_be64(header);

            if (length == 0u ||
                length > ECSC_MAX_FRAME_BYTES) {
                return ELPIS_ECSC_CORRUPT;
            }

            if (remaining -
                    UINT64_C(8) <
                length) {
                incomplete = 1u;
                break;
            }

            offset +=
                UINT64_C(8) + length;

            frames += UINT64_C(1);
        }
    }

    rc = stat_size(
        log->fd,
        &final_size
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (final_size != initial_size) {
        return ELPIS_ECSC_CORRUPT;
    }

    log->size = initial_size;
    log->complete_prefix = offset;
    log->frame_count = frames;
    log->scanned = 1;

    out_scan->total_size =
        initial_size;

    out_scan->complete_prefix =
        offset;

    out_scan->frame_count =
        frames;

    out_scan->has_incomplete_tail =
        incomplete;

    return ELPIS_ECSC_OK;
}

int elpis_ecsc_log_finish_recovery(
    elpis_ecsc_log *log,
    uint64_t validated_complete_prefix
)
{
    uint64_t current_size;
    int rc;

    if (log == NULL ||
        log->fd < 0 ||
        log->poisoned ||
        !log->recovering ||
        !log->scanned) {
        return ELPIS_ECSC_NOT_READY;
    }

    /*
     * Never reinterpret a complete frame as crash debris.
     *
     * A complete structurally framed record that fails semantic validation
     * is corruption. The caller must therefore validate the ENTIRE complete
     * structural prefix.
     */
    if (validated_complete_prefix !=
        log->complete_prefix ||
        !boundary_in_scan(
            log,
            validated_complete_prefix
        )) {
        return ELPIS_ECSC_CORRUPT;
    }

    rc = stat_size(
        log->fd,
        &current_size
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (current_size != log->size) {
        return ELPIS_ECSC_CORRUPT;
    }

    if (log->complete_prefix !=
        current_size) {
        if (ftruncate_checked(
                log->fd,
                log->complete_prefix
            ) != ELPIS_ECSC_OK) {
            return ELPIS_ECSC_IO;
        }

        if (fsync_checked(log->fd) !=
            ELPIS_ECSC_OK) {
            return poison_uncertain(log);
        }

        log->size =
            log->complete_prefix;
    }

    log->recovering = 0;
    log->scanned = 1;

    return ELPIS_ECSC_OK;
}

int elpis_ecsc_log_append_event_bytes(
    elpis_ecsc_log *log,
    const void *canonical_event,
    size_t canonical_event_size
)
{
    uint8_t header[8];
    uint64_t current_size;
    uint64_t start;
    int rc;

    if (log == NULL ||
        log->fd < 0 ||
        log->poisoned ||
        log->recovering ||
        !log->scanned) {
        return ELPIS_ECSC_NOT_READY;
    }

    if (canonical_event == NULL ||
        canonical_event_size == 0u ||
        canonical_event_size >
            ECSC_MAX_FRAME_BYTES) {
        return ELPIS_ECSC_INVALID;
    }

    rc = stat_size(
        log->fd,
        &current_size
    );

    if (rc != ELPIS_ECSC_OK) {
        return rc;
    }

    if (current_size != log->size) {
        return ELPIS_ECSC_CORRUPT;
    }

    start = current_size;

    encode_be64(
        (uint64_t)canonical_event_size,
        header
    );

    rc = write_all(
        log->fd,
        header,
        sizeof header
    );

    if (rc != ELPIS_ECSC_OK) {
        return rollback_append(
            log,
            start
        );
    }

    rc = write_all(
        log->fd,
        canonical_event,
        canonical_event_size
    );

    if (rc != ELPIS_ECSC_OK) {
        return rollback_append(
            log,
            start
        );
    }

    rc = fsync_checked(log->fd);

    if (rc != ELPIS_ECSC_OK) {
        return rollback_append(
            log,
            start
        );
    }

    log->size =
        start +
        UINT64_C(8) +
        (uint64_t)canonical_event_size;

    log->complete_prefix =
        log->size;

    log->frame_count += UINT64_C(1);

    return ELPIS_ECSC_OK;
}

uint64_t elpis_ecsc_log_size(
    const elpis_ecsc_log *log
)
{
    if (log == NULL ||
        log->fd < 0 ||
        log->poisoned) {
        return UINT64_MAX;
    }

    return log->size;
}

uint64_t elpis_ecsc_log_frame_count(
    const elpis_ecsc_log *log
)
{
    if (log == NULL ||
        log->fd < 0 ||
        log->poisoned) {
        return UINT64_MAX;
    }

    return log->frame_count;
}
