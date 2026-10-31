/* The host's containment (host_contain.h; docs/NATIVE_HOST.md, "Containment"). Linux, x86-64. It grants no write
 * permission to any path: the Landlock ruleset handles every mutation right and has no rule. */
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
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef LANDLOCK_ACCESS_FS_IOCTL_DEV
#define LANDLOCK_ACCESS_FS_IOCTL_DEV (1ULL << 15)
#endif
_Static_assert(LANDLOCK_ACCESS_FS_IOCTL_DEV == (1ULL << 15), "unexpected Landlock ioctl device access bit");

/* A character device that holds nothing: /dev/null, /dev/zero, /dev/full (major 1, minors 3, 5, 7). */
static int inert_device(dev_t rdev) {
    return major(rdev) == 1 && (minor(rdev) == 3 || minor(rdev) == 5 || minor(rdev) == 7);
}

static int contain_stdio(host_err *e) {
    for (int fd = 0; fd <= 2; ++fd) {
        struct stat st;
        const int flags = fcntl(fd, F_GETFL);
        if (flags < 0 || fstat(fd, &st))
            return host_fail(e, "CONTAINMENT_REFUSED", "standard descriptor %d is not open", fd);
        const int acc = flags & O_ACCMODE;
        if (acc != O_WRONLY && acc != O_RDWR) continue;
        /* Writable storage on a standard descriptor would be a persistent write path that bypasses every check. */
        if (S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) || S_ISBLK(st.st_mode))
            return host_fail(e, "CONTAINMENT_REFUSED", "standard descriptor %d is writable storage: use a pipe or a "
                                                       "terminal", fd);
        if (S_ISCHR(st.st_mode) && !inert_device(st.st_rdev) && !isatty(fd))
            return host_fail(e, "CONTAINMENT_REFUSED", "standard descriptor %d is a writable device (%u:%u): use a "
                                                       "pipe or a terminal", fd, major(st.st_rdev), minor(st.st_rdev));
    }
    /* Nothing inherited beyond them: a preopened descriptor would bypass path mediation. */
    if (syscall(SYS_close_range, 3u, ~0u, 0u))
        return host_fail(e, "CONTAINMENT_REFUSED", "inherited descriptors could not be closed: %s", strerror(errno));
    return 0;
}

/* Active swap areas: the lines of /proc/swaps after its header; -1 when it cannot be read. */
static int swap_areas(void) {
    const char *swaps = "/proc/swaps";
#ifdef ELPIS_HOST_TESTING
    /* The testing host only: a file in the format of /proc/swaps stands for it, so the swap policy is exercised
     * without ever changing the machine's swap configuration. */
    const char *simulated = getenv("ELPIS_TESTING_PROC_SWAPS");
    if (simulated && *simulated) swaps = simulated;
#endif
    const int fd = open(swaps, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[8192];
    size_t len = 0;
    for (;;) {
        const ssize_t r = read(fd, buf + len, sizeof buf - 1 - len);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) {
            close(fd);
            return -1;
        }
        if (r == 0 || (len += (size_t)r) == sizeof buf - 1) break;
    }
    close(fd);
    int lines = 0;
    for (size_t i = 0; i < len; ++i) lines += buf[i] == '\n';
    return lines > 0 ? lines - 1 : -1; /* the header line is always present */
}

int host_contain_process(host_containment *c, host_err *e) {
    memset(c, 0, sizeof *c);
    if (geteuid() == 0 || getuid() == 0)
        return host_fail(e, "CONTAINMENT_REFUSED", "a privileged process cannot be contained: run Elpis as an "
                                                   "unprivileged user");
    if (contain_stdio(e)) return -1;
    const struct rlimit none = {0, 0};
    struct rlimit core;
    if (setrlimit(RLIMIT_CORE, &none) || getrlimit(RLIMIT_CORE, &core) || core.rlim_cur || core.rlim_max)
        return host_fail(e, "CONTAINMENT_REFUSED", "core dumps could not be disabled: %s", strerror(errno));
    c->core_limit = core.rlim_cur;
    if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) || prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0)
        return host_fail(e, "CONTAINMENT_REFUSED", "the process could not be made non-dumpable: %s", strerror(errno));
    c->dumpable = 0;
    const int areas = swap_areas();
    if (areas < 0) return host_fail(e, "SWAP_POLICY", "/proc/swaps cannot be read: the swap policy is unknown");
    c->swap_areas = (uint32_t)areas;
    if (areas > 0) {
        /* Mutable state must never be paged out to storage: lock every page of the process, now and later. */
        if (mlockall(MCL_CURRENT | MCL_FUTURE))
            return host_fail(e, "SWAP_POLICY", "%d swap area(s) are active and the process memory could not be "
                                               "locked (%s): raise RLIMIT_MEMLOCK or run without swap",
                             areas, strerror(errno));
        c->memory_locked = 1;
    }
    return 0;
}

#if defined(__x86_64__)
static int install_filter(void) {
#define REFUSE_NR(nr, err) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (err))
#define DENY(nr) REFUSE_NR(nr, EPERM)
    struct sock_filter insns[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        /* The x32 ABI aliases every syscall: refused whole. */
        BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000u, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        /* Metadata Landlock does not mediate. */
        DENY(__NR_chmod), DENY(__NR_fchmod), DENY(__NR_fchmodat),
#ifdef __NR_fchmodat2
        DENY(__NR_fchmodat2),
#endif
        DENY(__NR_chown), DENY(__NR_fchown), DENY(__NR_lchown), DENY(__NR_fchownat),
        DENY(__NR_utime), DENY(__NR_utimes), DENY(__NR_futimesat), DENY(__NR_utimensat),
        DENY(__NR_setxattr), DENY(__NR_lsetxattr), DENY(__NR_fsetxattr),
        DENY(__NR_removexattr), DENY(__NR_lremovexattr), DENY(__NR_fremovexattr),
        /* Sizes and names (also refused by Landlock; twice is deliberate). */
        DENY(__NR_fallocate), DENY(__NR_truncate), DENY(__NR_ftruncate), DENY(__NR_creat),
        DENY(__NR_rename), DENY(__NR_renameat), DENY(__NR_renameat2), DENY(__NR_unlink), DENY(__NR_unlinkat),
        DENY(__NR_rmdir), DENY(__NR_mkdir), DENY(__NR_mkdirat), DENY(__NR_link), DENY(__NR_linkat),
        DENY(__NR_symlink), DENY(__NR_symlinkat), DENY(__NR_mknod), DENY(__NR_mknodat),
        DENY(__NR_name_to_handle_at), DENY(__NR_open_by_handle_at),
        /* Host-wide storage and system maintenance. */
        DENY(__NR_sync), DENY(__NR_syncfs), DENY(__NR_swapon), DENY(__NR_swapoff), DENY(__NR_mount),
        DENY(__NR_umount2), DENY(__NR_pivot_root), DENY(__NR_chroot), DENY(__NR_acct), DENY(__NR_quotactl),
        DENY(__NR_reboot), DENY(__NR_kexec_load), DENY(__NR_kexec_file_load), DENY(__NR_init_module),
        DENY(__NR_finit_module), DENY(__NR_delete_module), DENY(__NR_unshare), DENY(__NR_setns),
#ifdef __NR_fsopen
        DENY(__NR_fsopen), DENY(__NR_fsconfig), DENY(__NR_fsmount), DENY(__NR_move_mount), DENY(__NR_open_tree),
#endif
#ifdef __NR_mount_setattr
        DENY(__NR_mount_setattr),
#endif
        /* No other process may be asked to write: no sockets (no logging daemon), no io_uring, no ptrace. */
        DENY(__NR_socket), DENY(__NR_socketpair), DENY(__NR_io_uring_setup), DENY(__NR_io_uring_enter),
        DENY(__NR_io_uring_register), DENY(__NR_ptrace), DENY(__NR_process_vm_writev), DENY(__NR_bpf),
        /* No process creation: threads only. */
        DENY(__NR_fork), DENY(__NR_vfork), DENY(__NR_execve), DENY(__NR_execveat),
#ifdef __NR_clone3
        REFUSE_NR(__NR_clone3, ENOSYS),
#endif
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 0, 3),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])), /* clone flags, low word */
        BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, CLONE_THREAD, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
#undef DENY
#undef REFUSE_NR
    struct sock_fprog program = {(unsigned short)(sizeof insns / sizeof *insns), insns};
    return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program);
}
#else
static int install_filter(void) {
    errno = ENOTSUP;
    return -1;
}
#endif

int host_contain_files(host_containment *c, host_err *e) {
    const int abi = (int)syscall(SYS_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 5) return host_fail(e, "CONTAINMENT_REFUSED", "Landlock ABI v5 or later is required (found %d)", abi);
    c->landlock_abi = abi;
    struct landlock_ruleset_attr attr = {
        .handled_access_fs = LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_REMOVE_DIR |
                             LANDLOCK_ACCESS_FS_REMOVE_FILE | LANDLOCK_ACCESS_FS_MAKE_CHAR |
                             LANDLOCK_ACCESS_FS_MAKE_DIR | LANDLOCK_ACCESS_FS_MAKE_REG |
                             LANDLOCK_ACCESS_FS_MAKE_SOCK | LANDLOCK_ACCESS_FS_MAKE_FIFO |
                             LANDLOCK_ACCESS_FS_MAKE_BLOCK | LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_REFER |
                             LANDLOCK_ACCESS_FS_TRUNCATE | LANDLOCK_ACCESS_FS_IOCTL_DEV};
    const int ruleset = (int)syscall(SYS_landlock_create_ruleset, &attr, sizeof attr, 0);
    if (ruleset < 0) return host_fail(e, "CONTAINMENT_REFUSED", "landlock ruleset: %s", strerror(errno));
    /* No rule is added: the grant is empty. */
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) || syscall(SYS_landlock_restrict_self, ruleset, 0)) {
        close(ruleset);
        return host_fail(e, "CONTAINMENT_REFUSED", "landlock restriction: %s", strerror(errno));
    }
    close(ruleset);
    const struct rlimit none = {0, 0};
    struct rlimit fsize;
    if (setrlimit(RLIMIT_FSIZE, &none) || getrlimit(RLIMIT_FSIZE, &fsize) || fsize.rlim_cur || fsize.rlim_max)
        return host_fail(e, "CONTAINMENT_REFUSED", "file size limit: %s", strerror(errno));
    c->fsize_limit = fsize.rlim_cur;
    if (install_filter()) return host_fail(e, "CONTAINMENT_REFUSED", "seccomp filter: %s", strerror(errno));
    c->restricted = 1;
    return 0;
}
