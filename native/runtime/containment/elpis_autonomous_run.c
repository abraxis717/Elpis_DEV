/* Elpis autonomous execution: Linux fail-closed filesystem mutation boundary.
 * This launcher is not an authority to run arbitrary tools; the host controls
 * argv and may optionally grant the two already-initialized continuity slots.
 * No inherited writable file descriptors, no file creation, no truncation,
 * no expansion beyond a fixed-size continuity record.
 */
#define _GNU_SOURCE 1
#include <errno.h>
#include <fcntl.h>
#include <linux/landlock.h>
#include <linux/prctl.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

/* Headers shipped with CI may predate ABI v5; the kernel runtime gate below
 * remains mandatory. Linux UAPI defines IOCTL_DEV as bit 15. */
#ifndef LANDLOCK_ACCESS_FS_IOCTL_DEV
#define LANDLOCK_ACCESS_FS_IOCTL_DEV (1ULL << 15)
#endif
_Static_assert(LANDLOCK_ACCESS_FS_IOCTL_DEV == (1ULL << 15),
               "unexpected Landlock ioctl device access bit");

#define CONTINUITY_SLOT_BYTES 176u

static void report(const char *msg) {
    int e = errno;
    fprintf(stderr, "ELPIS_AUTONOMOUS_CONTAINMENT_REFUSED: %s (errno=%d)\n", msg, e);
}

static int is_safe_stdio(void) {
    for (int fd = 0; fd <= 2; ++fd) {
        struct stat st;
        int flags = fcntl(fd, F_GETFL);
        if (flags < 0 || fstat(fd, &st) != 0) return -1;
        int acc = flags & O_ACCMODE;
        /* Regular file descriptors could bypass path confinement. */
        if (S_ISREG(st.st_mode) && (acc == O_WRONLY || acc == O_RDWR)) return -1;
        /* Never inherit a directory handle for write-oriented operations. */
        if (S_ISDIR(st.st_mode) && (acc == O_WRONLY || acc == O_RDWR)) return -1;
    }
    return 0;
}

static int ruleset_create(int abi) {
    uint64_t rights = LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_REFER |
        LANDLOCK_ACCESS_FS_TRUNCATE;
    if (abi >= 5) rights |= LANDLOCK_ACCESS_FS_IOCTL_DEV;
    struct landlock_ruleset_attr attr = { .handled_access_fs = rights };
    return (int)syscall(SYS_landlock_create_ruleset, &attr, sizeof(attr), 0);
}

static int grant_continuity_slot(int rs, int dirfd, const char *name) {
    int fd = openat(dirfd, name, O_PATH | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return -1;
    struct stat st;
    int ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
        st.st_size == (off_t)CONTINUITY_SLOT_BYTES && st.st_nlink == 1 &&
        st.st_uid == geteuid() && !(st.st_mode & (S_IRWXG | S_IRWXO)) &&
        (st.st_mode & S_IWUSR);
    if (!ok) { close(fd); errno = EPERM; return -1; }
    struct landlock_path_beneath_attr rule = {
        .allowed_access = LANDLOCK_ACCESS_FS_WRITE_FILE, .parent_fd = fd
    };
    int rc = (int)syscall(SYS_landlock_add_rule, rs,
                          LANDLOCK_RULE_PATH_BENEATH, &rule, 0);
    int saved = errno;
    close(fd);
    errno = saved;
    return rc;
}

static int grant_continuity(int rs, const char *dir) {
    int dfd = open(dir, O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dfd < 0) return -1;
    struct stat st;
    int ok = fstat(dfd, &st) == 0 && S_ISDIR(st.st_mode) &&
        st.st_uid == geteuid() && !(st.st_mode & (S_IWGRP | S_IWOTH));
    if (!ok) { close(dfd); errno = EPERM; return -1; }
    int rc = grant_continuity_slot(rs, dfd, "continuity.a");
    if (rc == 0) rc = grant_continuity_slot(rs, dfd, "continuity.b");
    int saved = errno;
    close(dfd);
    errno = saved;
    return rc;
}

/* Landlock does not mediate chmod/chown/utime/xattr; deny all metadata
 * mutation syscalls independently. Constrain to x86-64 rather than silently
 * trust an unreviewed architecture-specific syscall table.
 */
static int deny_metadata_syscalls(void) {
#if !defined(__x86_64__)
    errno = ENOTSUP;
    return -1;
#else
#define REFUSE_NR(nr) \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)
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
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(insns) / sizeof(insns[0])),
        .filter = insns
    };
    return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program);
#endif
}

int main(int argc, char **argv) {
    const char *continuity = NULL;
    int i = 1;
    if (argc > 2 && strcmp(argv[i], "--continuity-dir") == 0) {
        continuity = argv[i + 1];
        i += 2;
        if (!continuity[0] || continuity[0] != '/') {
            errno = EINVAL; report("continuity directory must be absolute"); return 125;
        }
    }
    if (i >= argc || strcmp(argv[i], "--") != 0 || i + 1 >= argc) {
        fprintf(stderr, "usage: elpis_autonomous_run [--continuity-dir ABSOLUTE_DIR] -- COMMAND [ARG...]\n");
        return 125;
    }
    ++i;
    if (geteuid() == 0) { errno = EPERM; report("privileged launch refused"); return 125; }
    if (is_safe_stdio() != 0) { report("writable inherited standard descriptor"); return 125; }
    /* ABI v5 also covers device ioctl access; no weaker policy fallback. */
    int abi = (int)syscall(SYS_landlock_create_ruleset, NULL, 0,
                           LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 5) { errno = ENOTSUP; report("Landlock ABI v5+ required"); return 125; }
    int rs = ruleset_create(abi);
    if (rs < 0) { report("ruleset creation"); return 125; }
    if (continuity && grant_continuity(rs, continuity) != 0) {
        report("continuity slot preflight or rule"); close(rs); return 125;
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        report("no_new_privs"); close(rs); return 125;
    }
    if (syscall(SYS_landlock_restrict_self, rs, 0) != 0) {
        report("restrict_self"); close(rs); return 125;
    }
    close(rs);
    /* Any preopened descriptor > 2 would bypass path-opening mediation. */
    if (syscall(SYS_close_range, 3u, ~0u, 0u) != 0) {
        report("close inherited descriptors"); return 125;
    }
    if (continuity) {
        struct rlimit limit = { CONTINUITY_SLOT_BYTES, CONTINUITY_SLOT_BYTES };
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
            report("file size hard limit"); return 125;
        }
    }
    if (deny_metadata_syscalls() != 0) {
        report("metadata mutation seccomp"); return 125;
    }
    execvp(argv[i], &argv[i]);
    report("exec command");
    return 125;
}
