/*
 * research/volatile_runtime_v0: the peak resident set size of one command, measured from a small parent.
 *
 *     peak_rss COMMAND [ARG...]
 *
 * The kernel's ru_maxrss for a process includes the resident size of the image it was forked from (exec records the
 * old image's high-water mark), so a command forked from a large parent (the Python harness) would report the
 * parent's size. This launcher is small: it forks, executes the command, waits for it with wait4, and prints
 * "peak_rss_kib=N exit=M" as the last line on standard error. It exits with the command's status.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: peak_rss COMMAND [ARG...]\n");
        return 125;
    }
    const pid_t pid = fork();
    if (pid < 0) return 125;
    if (pid == 0) {
        execv(argv[1], &argv[1]);
        _exit(127);
    }
    int status = 0;
    struct rusage usage;
    if (wait4(pid, &status, 0, &usage) != pid) return 125;
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    fprintf(stderr, "peak_rss_kib=%ld exit=%d\n", usage.ru_maxrss, code);
    return code;
}
