// Per-fixture performance census for the Regex execution path.
//
//   census_regex_execution direct [NAME]     each fixture (or only NAME) through the V2 API
//   census_regex_execution exec WORKERS      each fixture alone through the runtime
//   census_regex_execution test WORKERS      the exact test_regex_execution submission
//                                            shape (window 4, affinity = index)
//
// One JSON object per line. Clocks: CLOCK_MONOTONIC for wall/queue/retirement,
// CLOCK_THREAD_CPUTIME_ID for the compute thread's CPU. Worker identity is the
// executing thread's kernel id, captured by a bound-task wrapper around the
// unchanged elpis_regex_execute.
#include "census_fixtures.h"
#include <atomic>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

static uint64_t mono_ns() {
    timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000u + uint64_t(t.tv_nsec);
}
static uint64_t thread_cpu_ns() {
    timespec t{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return uint64_t(t.tv_sec) * 1000000000u + uint64_t(t.tv_nsec);
}
struct Record {
    uint64_t start = 0, end = 0, cpu = 0;
    long tid = 0;
};
static elpis_exec_status traced(void *context, const elpis_exec_buffer *in, size_t cap,
                                elpis_exec_buffer **out) {
    auto *r = static_cast<Record *>(context);
    r->tid = long(syscall(SYS_gettid));
    r->start = mono_ns();
    uint64_t c0 = thread_cpu_ns();
    elpis_exec_status s = elpis_regex_execute(in, cap, out);
    r->cpu = thread_cpu_ns() - c0;
    r->end = mono_ns();
    return s;
}
static void line(const char *mode, unsigned workers, const CensusFixture &f, size_t index, double wall_ms,
                 double cpu_ms, double queue_ms, double compute_ms, double retire_ms, long tid, int status,
                 size_t out_bytes) {
    std::printf("{\"mode\":\"%s\",\"workers\":%u,\"index\":%zu,\"fixture\":\"%s\",\"bytes\":%zu,"
                "\"wall_ms\":%.3f,\"cpu_ms\":%.3f,\"queue_ms\":%.3f,\"compute_ms\":%.3f,"
                "\"retire_wait_ms\":%.3f,\"tid\":%ld,\"status\":%d,\"output_bytes\":%zu}\n",
                mode, workers, index, f.name.c_str(), f.source.size(), wall_ms, cpu_ms, queue_ms,
                compute_ms, retire_ms, tid, status, out_bytes);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    std::string mode = argc > 1 ? argv[1] : "direct";
    unsigned workers = argc > 2 && mode != "direct" ? unsigned(std::stoul(argv[2])) : 1;
    const auto fixtures = census_fixtures();
    if (mode == "direct") {
        std::string only = argc > 2 ? argv[2] : "";
        for (size_t i = 0; i < fixtures.size(); ++i) {
            if (!only.empty() && fixtures[i].name != only) continue;
            uint64_t w0 = mono_ns(), c0 = thread_cpu_ns();
            auto r = original(fixtures[i].source);
            double wall = (mono_ns() - w0) / 1e6, cpu = (thread_cpu_ns() - c0) / 1e6;
            line("direct", 0, fixtures[i], i, wall, cpu, 0, wall, 0, long(syscall(SYS_gettid)), r.first,
                 r.second.size());
        }
        return 0;
    }
    if (workers < 1 || workers > 4) return 2;
    const size_t window = mode == "test" ? 4 : 1;
    elpis_exec_config c{workers, unsigned(window), 2 * 1024 * 1024, 1024 * 1024, 0, nullptr};
    elpis_exec_runtime *r = nullptr;
    if (elpis_exec_create(&c, &r) != ELPIS_EXEC_OK) return 3;
    std::vector<Record> records(fixtures.size());
    std::vector<uint64_t> submitted_at(fixtures.size());
    size_t submitted = 0, retired = 0;
    uint64_t run0 = mono_ns();
    while (retired < fixtures.size()) {
        while (submitted < fixtures.size() && submitted - retired < window) {
            auto *b = input_buffer(fixtures[submitted].source);
            elpis_exec_bound_task t{ELPIS_EXEC_REGEX, 1, unsigned(submitted), ELPIS_EXEC_PURE, submitted,
                                    traced, &records[submitted]};
            uint64_t seq;
            submitted_at[submitted] = mono_ns();
            if (elpis_exec_submit_bound(r, &t, &b, &seq) != ELPIS_EXEC_OK) return 4;
            ++submitted;
        }
        elpis_exec_result result{};
        uint64_t take0 = mono_ns();
        elpis_exec_status s = elpis_exec_take(r, 60000, &result);
        uint64_t took = mono_ns();
        if (s != ELPIS_EXEC_OK) {
            std::printf("{\"mode\":\"%s\",\"workers\":%u,\"index\":%zu,\"fixture\":\"%s\",\"take_status\":%d,"
                        "\"take_wait_ms\":%.3f}\n", mode.c_str(), workers, retired,
                        fixtures[retired].name.c_str(), int(s), (took - take0) / 1e6);
            return 5;
        }
        const Record &rec = records[retired];
        line(mode.c_str(), workers, fixtures[retired], retired, (took - submitted_at[retired]) / 1e6,
             rec.cpu / 1e6, result.queue_ns / 1e6, result.compute_ns / 1e6,
             rec.end ? (took - rec.end) / 1e6 : 0.0, rec.tid, int(result.status),
             elpis_exec_buffer_size(result.output));
        elpis_exec_buffer_release(result.output);
        ++retired;
    }
    elpis_exec_metrics m{};
    elpis_exec_get_metrics(r, &m);
    std::printf("{\"mode\":\"%s\",\"workers\":%u,\"total_wall_ms\":%.3f,\"high_water\":%u,\"queue_full\":%llu}\n",
                mode.c_str(), workers, (mono_ns() - run0) / 1e6, m.high_water,
                (unsigned long long)m.queue_full);
    elpis_exec_destroy(r);
}
