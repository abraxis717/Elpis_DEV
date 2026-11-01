/* Launcher regression (ABI v8, no write grant): a hostile child can create, modify, truncate, extend, rename, link or
 * remove no file, change no metadata, reach no socket and dump no core; a retired `--authority` grant is refused; a
 * writable inherited descriptor or standard descriptor is refused or closed. With swap active the launcher refuses
 * every launch (memory locks do not survive exec), which this test then checks instead.
 * Test-only files are created in one temporary directory and removed on exit. */
#define _GNU_SOURCE 1
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <grp.h>
#include <sys/wait.h>
#include <unistd.h>

static int rw(const char *path, int flags) {
    int fd = open(path, flags, 0600);
    if (fd < 0) return -1;
    if (write(fd, "BAD", 3) != 3) { close(fd); return -1; }
    close(fd);
    return 0;
}
static void pathcat(char *out, size_t cap, const char *dir, const char *file) {
    if (snprintf(out, cap, "%s/%s", dir, file) >= (int)cap) exit(2);
}
/* 0: the attempt was refused (or, for "read", the read worked); 1: it happened. */
static int probe(const char *mode, const char *dir) {
    char a[4096], b[4096];
    pathcat(a, sizeof a, dir, "existing.txt");
    pathcat(b, sizeof b, dir, "new.txt");
    if (!strcmp(mode, "read")) {
        int fd = open(a, O_RDONLY);
        char buf[2] = {0};
        int ok = fd >= 0 && read(fd, buf, 1) == 1 && buf[0] == 'X';
        if (fd >= 0) close(fd);
        return ok ? 0 : 1;
    }
    if (!strcmp(mode, "create")) return rw(b, O_CREAT | O_EXCL | O_WRONLY) == 0 ? 1 : 0;
    if (!strcmp(mode, "overwrite")) return rw(a, O_WRONLY) == 0 ? 1 : 0;
    if (!strcmp(mode, "append")) return rw(a, O_WRONLY | O_APPEND) == 0 ? 1 : 0;
    if (!strcmp(mode, "truncate")) return truncate(a, 0) == 0 ? 1 : 0;
    if (!strcmp(mode, "rename")) return rename(a, b) == 0 ? 1 : 0;
    if (!strcmp(mode, "unlink")) return unlink(a) == 0 ? 1 : 0;
    if (!strcmp(mode, "mkdir")) return mkdir(b, 0700) == 0 ? 1 : 0;
    if (!strcmp(mode, "hardlink")) return link(a, b) == 0 ? 1 : 0;
    if (!strcmp(mode, "symlink")) return symlink(a, b) == 0 ? 1 : 0;
    if (!strcmp(mode, "chmod")) return chmod(a, 0666) == 0 ? 1 : 0;
    if (!strcmp(mode, "fd_escape")) return fcntl(8, F_GETFD) < 0 ? 0 : 1;
    if (!strcmp(mode, "sync")) return syncfs(0) == 0 ? 1 : 0;
    if (!strcmp(mode, "socket")) {
        int s = socket(AF_UNIX, SOCK_DGRAM, 0);
        if (s >= 0) close(s);
        return s >= 0 ? 1 : 0;
    }
    if (!strcmp(mode, "limits")) {
        struct rlimit core;
        if (getrlimit(RLIMIT_CORE, &core)) return 1;
        return core.rlim_cur || core.rlim_max;
    }
    /* In-memory staging stays possible (a contained runtime seals the libraries it admits into a memfd): an anonymous
     * file is written and read back; nothing on any path is touched. */
    if (!strcmp(mode, "memfd")) {
        int m = memfd_create("elpis-containment-probe", MFD_CLOEXEC);
        char back = 0;
        int ok = m >= 0 && write(m, "M", 1) == 1 && pread(m, &back, 1, 0) == 1 && back == 'M';
        if (m >= 0) close(m);
        return ok ? 0 : 1;
    }
    return 2;
}
static int child(const char *runner, const char *self, const char *dir, const char *mode) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (!strcmp(mode, "fd_escape")) {
            char file[4096];
            pathcat(file, sizeof file, dir, "existing.txt");
            int saved = open(file, O_RDONLY);
            if (saved < 0 || dup2(saved, 8) != 8) _exit(127);
            if (saved != 8) close(saved);
        }
        if (!strcmp(mode, "stdio_escape")) {
            char file[4096];
            pathcat(file, sizeof file, dir, "existing.txt");
            int saved = open(file, O_WRONLY);
            if (saved < 0 || dup2(saved, 1) != 1) _exit(127);
            close(saved);
        }
        if (!strcmp(mode, "authority_grant"))
            execl(runner, runner, "--authority", dir, dir, "--", self, "probe", "read", dir, (char *)NULL);
        else
            execl(runner, runner, "--", self, "probe", mode, dir, (char *)NULL);
        _exit(127);
    }
    int status;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
static int swap_active(void) {
    FILE *f = fopen("/proc/swaps", "r");
    if (!f) return -1;
    int lines = 0, c;
    while ((c = fgetc(f)) != EOF) lines += c == '\n';
    fclose(f);
    return lines > 1;
}
int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "probe")) return probe(argv[2], argv[3]);
    if (argc != 2) { fprintf(stderr, "usage: test_autonomous_containment LAUNCHER\n"); return 2; }
    /* The launcher refuses a privileged process: as root, the test (and its children) run as uid 65534. */
    if (geteuid() == 0 && (setgroups(0, NULL) || setresgid(65534, 65534, 65534) || setresuid(65534, 65534, 65534))) {
        perror("drop privileges");
        return 2;
    }
    char dir[] = "/tmp/elpis-autonomous-r1-XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 2; }
    char existing[4096], newfile[4096];
    pathcat(existing, sizeof existing, dir, "existing.txt");
    pathcat(newfile, sizeof newfile, dir, "new.txt");
    int fd = open(existing, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0 || write(fd, "X", 1) != 1) { perror("create fixture"); return 2; }
    close(fd);
    const int swapping = swap_active();
    const char *tests[] = {"read", "create", "overwrite", "append", "truncate", "rename", "unlink", "mkdir",
                           "hardlink", "symlink", "chmod", "fd_escape", "stdio_escape", "sync", "socket", "limits",
                           "memfd", "authority_grant"};
    int errors = 0;
    for (size_t j = 0; j < sizeof tests / sizeof tests[0]; j++) {
        int rc = child(argv[1], argv[0], dir, tests[j]);
        int expected = !strcmp(tests[j], "stdio_escape") || !strcmp(tests[j], "authority_grant") || swapping ? 125 : 0;
        if (rc != expected) {
            fprintf(stderr, "FAIL %s rc=%d expected=%d\n", tests[j], rc, expected);
            errors++;
        }
        struct stat st;
        if (access(newfile, F_OK) == 0 || stat(existing, &st) != 0 || st.st_size != 1) {
            fprintf(stderr, "FAIL fixture after %s\n", tests[j]);
            errors++;
        }
    }
    fd = open(existing, O_RDONLY);
    char byte = 0;
    if (fd < 0 || read(fd, &byte, 1) != 1 || byte != 'X') errors++;
    if (fd >= 0) close(fd);
    if (unlink(existing) || rmdir(dir)) { perror("cleanup"); errors++; }
    if (errors) { fprintf(stderr, "ELPIS_AUTONOMOUS_CONTAINMENT_FAIL=%d\n", errors); return 1; }
    puts(swapping ? "ELPIS_AUTONOMOUS_CONTAINMENT_PASS (swap active: every launch refused)"
                  : "ELPIS_AUTONOMOUS_CONTAINMENT_PASS");
    return 0;
}
