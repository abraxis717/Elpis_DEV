/* The host's own containment (host_contain.h; docs/NATIVE_HOST.md, "Containment"): the policy of
 * native/runtime/containment/elpis_autonomous_run.c, applied by the host to itself after admission, with the grant
 * being the deployment's fixed slot files. Linux, x86-64 (the seccomp table is reviewed for that architecture only). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "host_contain.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef LANDLOCK_ACCESS_FS_IOCTL_DEV
#define LANDLOCK_ACCESS_FS_IOCTL_DEV (1ULL << 15)
#endif
_Static_assert(LANDLOCK_ACCESS_FS_IOCTL_DEV == (1ULL << 15), "unexpected Landlock ioctl device access bit");

/* The checkpoint slot's header ahead of the envelope (native/runtime/src/checkpoint.rs, HEADER). */
#define CHECKPOINT_HEADER_BYTES 128u

int host_contain_stdio(host_err *e) {
    for (int fd = 0; fd <= 2; ++fd) {
        struct stat st;
        const int flags = fcntl(fd, F_GETFL);
        if (flags < 0 || fstat(fd, &st))
            return host_fail(e, "CONTAINMENT_REFUSED", "standard descriptor %d is not open", fd);
        const int acc = flags & O_ACCMODE;
        /* A writable regular file or directory on a standard descriptor would bypass path confinement. */
        if ((S_ISREG(st.st_mode) || S_ISDIR(st.st_mode)) && (acc == O_WRONLY || acc == O_RDWR))
            return host_fail(e, "CONTAINMENT_REFUSED", "standard descriptor %d is a writable file: use a pipe", fd);
    }
    /* Nothing inherited beyond them: a preopened descriptor would bypass path mediation. */
    if (syscall(SYS_close_range, 3u, ~0u, 0u))
        return host_fail(e, "CONTAINMENT_REFUSED", "inherited descriptors could not be closed: %s", strerror(errno));
    return 0;
}

static int grant(int ruleset, const char *path, uint64_t size, uint64_t *largest, host_err *e) {
    const int fd = open(path, O_PATH | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) {
        if (fd >= 0) close(fd);
        return host_fail(e, "CONTAINMENT_REFUSED", "%s: %s", path, strerror(errno));
    }
    /* The fixed slot exactly: a regular file of its size, one link, ours, private, writable by us. */
    if (!S_ISREG(st.st_mode) || (uint64_t)st.st_size != size || st.st_nlink != 1 || st.st_uid != geteuid() ||
        (st.st_mode & (S_IRWXG | S_IRWXO)) || !(st.st_mode & S_IWUSR)) {
        close(fd);
        return host_fail(e, "CONTAINMENT_REFUSED", "%s is not the fixed %llu-byte slot it must be", path,
                         (unsigned long long)size);
    }
    struct landlock_path_beneath_attr rule = {.allowed_access = LANDLOCK_ACCESS_FS_WRITE_FILE, .parent_fd = fd};
    const long rc = syscall(SYS_landlock_add_rule, ruleset, LANDLOCK_RULE_PATH_BENEATH, &rule, 0);
    close(fd);
    if (rc) return host_fail(e, "CONTAINMENT_REFUSED", "landlock rule for %s: %s", path, strerror(errno));
    if (size > *largest) *largest = size;
    return 0;
}

#if defined(__x86_64__)
/* Landlock does not mediate chmod/chown/utime/xattr or preallocation: refuse every such syscall. */
static int deny_metadata_syscalls(void) {
#define REFUSE_NR(nr) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)
    struct sock_filter insns[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        REFUSE_NR(__NR_chmod),
        REFUSE_NR(__NR_fchmod),
        REFUSE_NR(__NR_fchmodat),
#ifdef __NR_fchmodat2
        REFUSE_NR(__NR_fchmodat2),
#endif
        REFUSE_NR(__NR_chown),
        REFUSE_NR(__NR_fchown),
        REFUSE_NR(__NR_lchown),
        REFUSE_NR(__NR_fchownat),
        REFUSE_NR(__NR_utime),
        REFUSE_NR(__NR_utimes),
        REFUSE_NR(__NR_futimesat),
        REFUSE_NR(__NR_utimensat),
        REFUSE_NR(__NR_setxattr),
        REFUSE_NR(__NR_lsetxattr),
        REFUSE_NR(__NR_fsetxattr),
        REFUSE_NR(__NR_removexattr),
        REFUSE_NR(__NR_lremovexattr),
        REFUSE_NR(__NR_fremovexattr),
        REFUSE_NR(__NR_fallocate),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
#undef REFUSE_NR
    struct sock_fprog program = {(unsigned short)(sizeof insns / sizeof *insns), insns};
    return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program);
}
#else
static int deny_metadata_syscalls(void) {
    errno = ENOTSUP;
    return -1;
}
#endif

int host_contain(const host_deployment *d, host_err *e) {
    /* As the launcher: a privileged process keeps capabilities (module loading, mounts) that leave the domain. */
    if (geteuid() == 0) return host_fail(e, "CONTAINMENT_REFUSED", "a privileged process cannot be contained");
    const int abi = (int)syscall(SYS_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 5) return host_fail(e, "CONTAINMENT_REFUSED", "Landlock ABI v5 or later is required (found %d)", abi);
    struct landlock_ruleset_attr attr = {
        .handled_access_fs = LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_REMOVE_DIR |
                             LANDLOCK_ACCESS_FS_REMOVE_FILE | LANDLOCK_ACCESS_FS_MAKE_CHAR |
                             LANDLOCK_ACCESS_FS_MAKE_DIR | LANDLOCK_ACCESS_FS_MAKE_REG |
                             LANDLOCK_ACCESS_FS_MAKE_SOCK | LANDLOCK_ACCESS_FS_MAKE_FIFO |
                             LANDLOCK_ACCESS_FS_MAKE_BLOCK | LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_REFER |
                             LANDLOCK_ACCESS_FS_TRUNCATE | LANDLOCK_ACCESS_FS_IOCTL_DEV};
    const int ruleset = (int)syscall(SYS_landlock_create_ruleset, &attr, sizeof attr, 0);
    if (ruleset < 0) return host_fail(e, "CONTAINMENT_REFUSED", "landlock ruleset: %s", strerror(errno));
    /* The grant: exactly the deployment's fixed slots (the ECS_C records and witnesses, the checkpoint slots). */
    const uint64_t record = d->api.elpis_ecs_c_record_bytes(d->capacity);
    char path[HOST_PATH + 32];
    uint64_t largest = 0;
    int rc = 0;
    for (int i = 0; !rc && i < 2; ++i) {
        snprintf(path, sizeof path, "%s/authority.%c", d->state_dir, 'a' + i);
        rc = grant(ruleset, path, record, &largest, e);
        snprintf(path, sizeof path, "%s/witness.%c", d->authority_dir, 'a' + i);
        if (!rc) rc = grant(ruleset, path, ELPIS_ECS_C_WITNESS_BYTES, &largest, e);
        snprintf(path, sizeof path, "%s/k1-checkpoint.%c", d->checkpoint_dir, 'a' + i);
        if (!rc && d->has_lineage) rc = grant(ruleset, path, CHECKPOINT_HEADER_BYTES + d->envelope_bytes, &largest, e);
    }
    if (rc) {
        close(ruleset);
        return -1;
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) || syscall(SYS_landlock_restrict_self, ruleset, 0)) {
        close(ruleset);
        return host_fail(e, "CONTAINMENT_REFUSED", "landlock restriction: %s", strerror(errno));
    }
    close(ruleset);
    /* No write may extend any file past the largest fixed slot. */
    const struct rlimit limit = {largest, largest};
    if (setrlimit(RLIMIT_FSIZE, &limit))
        return host_fail(e, "CONTAINMENT_REFUSED", "file size limit: %s", strerror(errno));
    if (deny_metadata_syscalls())
        return host_fail(e, "CONTAINMENT_REFUSED", "metadata seccomp filter: %s", strerror(errno));
    return 0;
}
