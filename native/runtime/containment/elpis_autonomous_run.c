/* Elpis autonomous execution: Linux fail-closed filesystem mutation boundary with NO write grant
 * (docs/ELPIS_MISSION.md, Amendment 3). This launcher is not an authority to run arbitrary tools; the host controls
 * argv. The contained command can create, write, truncate, extend, rename, link or remove no file anywhere: Landlock
 * handles every mutation right and has no rule (no path can be opened for writing, created, truncated, renamed,
 * linked or removed), and seccomp refuses the metadata syscalls Landlock does not mediate, preallocation and
 * truncation, sync/syncfs, mount and swap control, module and kexec loading, reboot, sockets (no logging daemon is
 * reachable), io_uring and ptrace. No core dump (RLIMIT_CORE 0, inherited across exec), no inherited writable storage
 * descriptor, nothing inherited above stderr.
 *
 * No RLIMIT_FSIZE here: the contained runtime seals each library it admits into an anonymous memfd (in memory, never
 * persistent) after exec, and a zero file-size limit would refuse that seal. With no writable path and no writable
 * descriptor, the limit would protect nothing persistent; the native host applies it itself once its seals exist.
 *
 * Swap: with any swap area active the launcher refuses, since memory locks do not survive exec (the native host locks
 * its own memory instead: docs/NATIVE_HOST.md). It never changes the host's swap configuration.
 *
 * Until RuntimeCore ABI v8 the launcher could grant the four fixed files of a provisioned ECS_C authority; that grant
 * and the durable authority are retired, and `--authority` is refused. */
#define _GNU_SOURCE 1
#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/prctl.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

/* Headers shipped with CI may predate ABI v5; the kernel runtime gate below remains mandatory. Linux UAPI defines
 * IOCTL_DEV as bit 15. */
#ifndef LANDLOCK_ACCESS_FS_IOCTL_DEV
#define LANDLOCK_ACCESS_FS_IOCTL_DEV (1ULL << 15)
#endif
_Static_assert(LANDLOCK_ACCESS_FS_IOCTL_DEV == (1ULL << 15), "unexpected Landlock ioctl device access bit");

/* Writable storage: a descriptor open for writing onto a regular file, a directory, a block device, or a character
 * device other than a terminal or /dev/null, /dev/zero, /dev/full. */
static int writable_storage(int fd) {
    struct stat st;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fstat(fd, &st) != 0) return 0;
    int acc = flags & O_ACCMODE;
    if (acc != O_WRONLY && acc != O_RDWR) return 0;
    if (S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) || S_ISBLK(st.st_mode)) return 1;
    return S_ISCHR(st.st_mode) && !isatty(fd) &&
           !(major(st.st_rdev) == 1 && (minor(st.st_rdev) == 3 || minor(st.st_rdev) == 5 || minor(st.st_rdev) == 7));
}

/* The refusal goes to standard error only when that is not storage: the launcher writes nothing onto storage. */
static void report(const char *msg) {
    int e = errno;
    if (!writable_storage(STDERR_FILENO)) fprintf(stderr, "ELPIS_AUTONOMOUS_CONTAINMENT_REFUSED: %s (errno=%d)\n", msg, e);
}

static int is_safe_stdio(void) {
    for (int fd = 0; fd <= 2; ++fd) {
        struct stat st;
        if (fcntl(fd, F_GETFL) < 0 || fstat(fd, &st) != 0) return -1;
        if (writable_storage(fd)) return -1;
    }
    return 0;
}

/* Active swap areas: the lines of /proc/swaps after its header; -1 when it cannot be read. */
static int swap_areas(void) {
    int fd = open("/proc/swaps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[8192];
    size_t len = 0;
    for (;;) {
        ssize_t r = read(fd, buf + len, sizeof buf - 1 - len);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) { close(fd); return -1; }
        if (r == 0 || (len += (size_t)r) == sizeof buf - 1) break;
    }
    close(fd);
    int lines = 0;
    for (size_t i = 0; i < len; ++i) lines += buf[i] == '\n';
    return lines > 0 ? lines - 1 : -1;
}

static int ruleset_create(void) {
    uint64_t rights = LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_REFER |
        LANDLOCK_ACCESS_FS_TRUNCATE | LANDLOCK_ACCESS_FS_IOCTL_DEV;
    struct landlock_ruleset_attr attr = { .handled_access_fs = rights };
    return (int)syscall(SYS_landlock_create_ruleset, &attr, sizeof(attr), 0);
}

/* Constrain to x86-64 rather than silently trust an unreviewed architecture-specific syscall table. execve stays
 * allowed: the launcher execs its command under this filter (and the command's children inherit every restriction). */
static int install_filter(void) {
#if !defined(__x86_64__)
    errno = ENOTSUP;
    return -1;
#else
#define DENY(nr) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)
    struct sock_filter insns[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000u, 0, 1), /* x32 aliases: refused whole */
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        DENY(__NR_chmod), DENY(__NR_fchmod), DENY(__NR_fchmodat),
#ifdef __NR_fchmodat2
        DENY(__NR_fchmodat2),
#endif
        DENY(__NR_chown), DENY(__NR_fchown), DENY(__NR_lchown), DENY(__NR_fchownat),
        DENY(__NR_utime), DENY(__NR_utimes), DENY(__NR_futimesat), DENY(__NR_utimensat),
        DENY(__NR_setxattr), DENY(__NR_lsetxattr), DENY(__NR_fsetxattr),
        DENY(__NR_removexattr), DENY(__NR_lremovexattr), DENY(__NR_fremovexattr),
        DENY(__NR_fallocate), DENY(__NR_truncate), DENY(__NR_ftruncate),
        DENY(__NR_name_to_handle_at), DENY(__NR_open_by_handle_at),
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
        DENY(__NR_socket), DENY(__NR_socketpair), DENY(__NR_io_uring_setup), DENY(__NR_io_uring_enter),
        DENY(__NR_io_uring_register), DENY(__NR_ptrace), DENY(__NR_process_vm_writev), DENY(__NR_bpf),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
#undef DENY
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(insns) / sizeof(insns[0])),
        .filter = insns
    };
    return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program);
#endif
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--authority") == 0) {
        errno = EPERM;
        report("--authority is retired: the launcher grants no write permission");
        return 125;
    }
    if (argc < 3 || strcmp(argv[1], "--") != 0) {
        fprintf(stderr, "usage: elpis_autonomous_run -- COMMAND [ARG...]\n");
        return 125;
    }
    if (geteuid() == 0 || getuid() == 0) { errno = EPERM; report("privileged launch refused"); return 125; }
    if (is_safe_stdio() != 0) { report("writable storage on a standard descriptor"); return 125; }
    /* No core dump of the command (inherited across exec). */
    const struct rlimit none = { 0, 0 };
    struct rlimit got;
    if (setrlimit(RLIMIT_CORE, &none) != 0 || getrlimit(RLIMIT_CORE, &got) != 0 || got.rlim_cur || got.rlim_max) {
        report("core dump limit"); return 125;
    }
    int areas = swap_areas();
    if (areas < 0) { report("/proc/swaps unreadable: the swap policy is unknown"); return 125; }
    if (areas > 0) { errno = EPERM; report("swap is active: memory locks do not survive exec"); return 125; }
    /* ABI v5 also covers device ioctl access; no weaker policy fallback. */
    int abi = (int)syscall(SYS_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 5) { errno = ENOTSUP; report("Landlock ABI v5+ required"); return 125; }
    int rs = ruleset_create();
    if (rs < 0) { report("ruleset creation"); return 125; }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) { report("no_new_privs"); close(rs); return 125; }
    if (syscall(SYS_landlock_restrict_self, rs, 0) != 0) { report("restrict_self"); close(rs); return 125; }
    close(rs);
    /* Any preopened descriptor > 2 would bypass path-opening mediation. */
    if (syscall(SYS_close_range, 3u, ~0u, 0u) != 0) { report("close inherited descriptors"); return 125; }
    if (install_filter() != 0) { report("seccomp filter"); return 125; }
    execvp(argv[2], &argv[2]);
    report("exec command");
    return 125;
}
