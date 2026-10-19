/* The host's containment (docs/NATIVE_HOST.md, "Containment"): after admission, `elpis --contained` restricts itself
 * exactly as native/runtime/containment/elpis_autonomous_run.c restricts a child, with the deployment's fixed slots as
 * the grant: Landlock (ABI v5 or later) denies creating, removing, renaming, linking, truncating and writing any file
 * but the six fixed slot files (each checked to be exactly its slot); RLIMIT_FSIZE stops any write past the largest
 * slot; a seccomp filter refuses chmod, chown, utime, xattr and fallocate; no new privileges. Linux x86-64 only. */
#ifndef ELPIS_HOST_CONTAIN_H
#define ELPIS_HOST_CONTAIN_H

#include "host_deploy.h"

/* Before anything is opened: refuse a writable regular file or directory on a standard descriptor, and close every
 * inherited descriptor above them. */
int host_contain_stdio(host_err *e);
/* After admission and before the first operation. Refused (CONTAINMENT_REFUSED) for a privileged process (as the
 * launcher), when the kernel lacks Landlock ABI v5, when a slot is not exactly its fixed file, or when any restriction
 * cannot be applied. */
int host_contain(const host_deployment *d, host_err *e);

#endif
