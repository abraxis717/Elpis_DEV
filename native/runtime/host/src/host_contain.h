/* The host's containment (docs/NATIVE_HOST.md, "Containment"): mandatory, fail-closed, and with no persistent write
 * permission at all. Linux x86-64 only (the seccomp table is reviewed for that architecture).
 *
 *   host_contain_process  (first, before any library is read)
 *     - a privileged process is refused (it keeps capabilities, mounts and module loading, that leave the domain);
 *     - a standard descriptor that is a writable regular file, directory, block device or non-terminal character
 *       device other than /dev/null, /dev/zero or /dev/full is refused (no persistent stdout or stderr), and every
 *       inherited descriptor above them is closed;
 *     - no core dumps: RLIMIT_CORE = 0 and PR_SET_DUMPABLE = 0, both read back;
 *     - the swap policy is validated: /proc/swaps is read; with any swap area active, every page of the process is
 *       locked in RAM (mlockall, current and future) or the host refuses. Host swap configuration is never changed.
 *   host_contain_files    (after admission, before the first operation)
 *     - Landlock (ABI v5 or later) handles every filesystem mutation right (write, truncate, create of any kind,
 *       remove, rename/link, device ioctl) and grants none: no file anywhere may be written, created or removed;
 *     - RLIMIT_FSIZE = 0: no write may give any file a byte;
 *     - no new privileges, and a seccomp filter refusing the metadata syscalls Landlock does not mediate (chmod, chown,
 *       utime, xattr), preallocation and truncation, every namespace operation that can mount or remove names,
 *       sync/syncfs, swapon/swapoff, mount/umount, module and kexec loading, reboot, sockets (no logging daemon is
 *       reachable), io_uring, ptrace and process_vm_writev, and process creation (fork, vfork, execve, clone without
 *       CLONE_THREAD; clone3 reports ENOSYS so the C library falls back to clone): no hidden writer child exists. */
#ifndef ELPIS_HOST_CONTAIN_H
#define ELPIS_HOST_CONTAIN_H

#include <stdint.h>

#include "host_install.h"

/* What the containment established (reported by `status`). */
typedef struct {
    int landlock_abi;      /* the kernel's Landlock ABI (0: not yet restricted) */
    int restricted;        /* Landlock restriction, seccomp filter and no_new_privs applied */
    uint64_t core_limit;   /* RLIMIT_CORE read back (0) */
    int dumpable;          /* PR_GET_DUMPABLE read back (0) */
    uint64_t fsize_limit;  /* RLIMIT_FSIZE read back (0) */
    uint32_t swap_areas;   /* active swap areas in /proc/swaps */
    int memory_locked;     /* mlockall(MCL_CURRENT | MCL_FUTURE) applied (required when swap_areas > 0) */
} host_containment;

/* 1 when `fd` is open for writing onto storage (a regular file, a directory, a block device, or a character device
 * that is neither a terminal nor /dev/null, /dev/zero, /dev/full): the host never writes through such a descriptor,
 * not even its refusal. */
int host_writable_storage(int fd);
int host_contain_process(host_containment *c, host_err *e);
int host_contain_files(host_containment *c, host_err *e);

#endif
