"""Native resources of the FROZEN v4 configurations: state and configuration bytes, logical scalars, and the
latency of one observe (learner update) and one decide (terminal decision) call. A measurement, not a gate:
latency is printed and recorded, never asserted (docs/CI_POLICY.md).

    python3 research/cgc_lao/solver_lab/v4/scripts/native_latency.py [--write]

The harness below is compiled in a temporary directory with the frozen native source and the library's flags
(cc -std=c11 -O2 -ffp-contract=off); nothing is built inside the repository. --write records the result in
evidence/native/resources.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4 / "src"))

from cgc_lao_v4 import study4  # noqa: E402
from cgc_lao_v2 import env2, evaluate2, native  # noqa: E402

HARNESS = r"""
#define _POSIX_C_SOURCE 200809L
#include "cgc_lao_rmf.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { WARMUP = 200, EPISODES = 4000 };
static double obs[EPISODES * 8], dec[EPISODES * 16];
static uint64_t lcg = 88172645463325252ull;
static double unit(void) {
    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(lcg >> 11) * (1.0 / 9007199254740992.0);
}
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static int cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void stats(const char *name, double *v, long n) {
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += v[i];
    qsort(v, (size_t)n, sizeof *v, cmp);
    printf("\"%s\": {\"calls\": %ld, \"mean_us\": %.4f, \"p50_us\": %.4f, \"p99_us\": %.4f, \"max_us\": %.4f}", name, n,
           1e6 * s / (double)n, 1e6 * v[n / 2], 1e6 * v[(long)(0.99 * (double)n)], 1e6 * v[n - 1]);
}
int main(void) {
    const uint32_t nodes[CGC_LAO_RMF_MAX_HYP] = {@NODES@};
    const double q[CGC_LAO_RMF_MAX_HYP * CGC_LAO_RMF_MAX_NODES] = {@Q@};
    const double rho[CGC_LAO_RMF_MAX_HYP] = {@RHO@};
    const double prior[CGC_LAO_RMF_MAX_HYP] = {@PRIOR@};
    const uint32_t reads[8] = {@READS@};
    cgc_lao_rmf_config cfg;
    cgc_lao_rmf_state st;
    if (cgc_lao_rmf_config_init(&cfg, @M@, nodes, q, rho, prior, 1.0, 0.6, 8, CGC_LAO_RMF_PROBE_EARLY) != CGC_LAO_RMF_OK)
        return 2;
    long no = 0, nd = 0;
    for (int e = 0; e < WARMUP + EPISODES; ++e) {
        const int rec = e >= WARMUP;
        if (cgc_lao_rmf_reset(&st, &cfg) != CGC_LAO_RMF_OK) return 3;
        for (uint32_t t = 0; t < 8; ++t) {
            const double z[3] = {4.0 * unit() - 2.0, 4.0 * unit() - 2.0, 4.0 * unit() - 2.0};
            uint32_t a = 0;
            if (cgc_lao_rmf_propose(&st, &cfg, z, &a) != CGC_LAO_RMF_OK) return 4;
            const uint8_t r[2] = {(uint8_t)(unit() < 0.5), (uint8_t)(unit() < 0.5)};
            const double t0 = now();
            const int rc = cgc_lao_rmf_observe(&st, &cfg, a, z[a], r, reads[t], 1u);
            const double dt = now() - t0;
            if (rc != CGC_LAO_RMF_OK) return 5;
            if (rec) obs[no++] = dt;
        }
        for (int j = 0; j < 16; ++j) {
            const double zq[3] = {4.0 * unit() - 2.0, 4.0 * unit() - 2.0, 4.0 * unit() - 2.0};
            uint32_t d = 0;
            const double t0 = now();
            const int rc = cgc_lao_rmf_decide(&st, &cfg, zq, &d);
            const double dt = now() - t0;
            if (rc != CGC_LAO_RMF_OK) return 6;
            if (rec) dec[nd++] = dt;
        }
        cgc_lao_rmf_zeroize(&st);
    }
    printf("{\"state_bytes\": %zu, \"config_bytes\": %zu, \"logical_scalars\": %u, ", cgc_lao_rmf_state_size(),
           cgc_lao_rmf_config_size(), cgc_lao_rmf_logical_scalars(&cfg));
    stats("observe", obs, no);
    printf(", ");
    stats("decide", dec, nd);
    printf("}\n");
    return 0;
}
"""


def harness_source(config: dict, mode: str) -> str:
    hyps, m = config["hypotheses"], len(config["hypotheses"])
    nodes, q, rho = [0] * native.MAX_HYP, [0.0] * (native.MAX_HYP * native.MAX_NODES), [0.0] * native.MAX_HYP
    prior = [0.0] * native.MAX_HYP
    for i, (qv, rv) in enumerate(hyps):
        qs = [float(qv)] if isinstance(qv, (int, float)) else [float(x) for x in qv]
        nodes[i] = len(qs)
        for j, x in enumerate(qs):
            q[i * native.MAX_NODES + j] = x
        rho[i], prior[i] = float(rv), float(config["prior"][i])
    reads = list(env2.CONTRACTS["DUP2_4"].reads) if mode == "DUPLICATE" else [1] * 8
    fmt = lambda xs: ", ".join(repr(float(x)) for x in xs)  # noqa: E731
    return (HARNESS.replace("@NODES@", ", ".join(f"{n}u" for n in nodes)).replace("@Q@", fmt(q))
            .replace("@RHO@", fmt(rho)).replace("@PRIOR@", fmt(prior))
            .replace("@READS@", ", ".join(f"{r}u" for r in reads)).replace("@M@", f"{m}u"))


def measure(config: dict, mode: str) -> dict:
    cc = native.compiler()
    if cc is None:
        raise SystemExit("no C compiler")
    with tempfile.TemporaryDirectory(prefix="cgc_lao_latency_") as tmp:
        src, exe = Path(tmp) / "harness.c", Path(tmp) / "harness"
        src.write_text(harness_source(config, mode), encoding="utf-8")
        subprocess.run([cc, "-std=c11", "-O2", "-ffp-contract=off", "-I", str(native.INCLUDE), str(src),
                        str(native.SOURCE), "-lm", "-o", str(exe)], check=True, capture_output=True, timeout=300)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=600).stdout
    return json.loads(out)


def stack_usage() -> dict | None:
    """Static stack frame of every function (gcc/clang -fstack-usage). The call graph is acyclic (no recursion,
    tested), so the sum of all frames bounds any call chain."""
    cc = native.compiler()
    with tempfile.TemporaryDirectory(prefix="cgc_lao_stack_") as tmp:
        obj = Path(tmp) / "cgc_lao_rmf.o"
        res = subprocess.run([cc, "-std=c11", "-O2", "-ffp-contract=off", "-fstack-usage", "-I", str(native.INCLUDE),
                              "-c", str(native.SOURCE), "-o", str(obj)], capture_output=True, text=True, timeout=300)
        su = obj.with_suffix(".su")
        if res.returncode != 0 or not su.exists():
            return None
        frames = {}
        for line in su.read_text(encoding="utf-8").splitlines():
            where, size, kind = line.rsplit("\t", 2)
            frames[where.rsplit(":", 1)[-1]] = {"bytes": int(size), "kind": kind}
    return {"frames": frames, "all_static": all(f["kind"] == "static" for f in frames.values()),
            "sum_of_all_frames_bytes": sum(f["bytes"] for f in frames.values()),
            "max_frame_bytes": max(f["bytes"] for f in frames.values())}


def main(argv) -> None:
    frozen = json.loads((V4 / "FROZEN_V4.json").read_text(encoding="utf-8"))
    cc = native.compiler()
    version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0] if cc else None
    rec = {"compiler": version, "flags": "-std=c11 -O2 -ffp-contract=off", "machine": platform.machine(),
           "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "note": "measurement only; one observe = one learner update of the selected expert (reads per step "
                   "as the role's mode); one decide = one terminal decision over the 3 experts and STOP",
           "roles": {}}
    rec["stack"] = stack_usage()
    if rec["stack"]:
        st = rec["stack"]
        print(f"stack: all frames static={st['all_static']}, max frame {st['max_frame_bytes']} B, sum of all frames "
              f"{st['sum_of_all_frames_bytes']} B; observe {st['frames']['cgc_lao_rmf_observe']['bytes']} B, decide "
              f"{st['frames']['cgc_lao_rmf_decide']['bytes']} B")
    for role, cand in frozen["candidates"].items():
        rec["roles"][role] = {"name": cand["name"], "mode": cand["mode"], **measure(cand["config"], cand["mode"])}
        r = rec["roles"][role]
        print(f"{role} {cand['name']} ({cand['mode']}): state {r['state_bytes']} B, config {r['config_bytes']} B, "
              f"{r['logical_scalars']} logical scalars; observe mean {r['observe']['mean_us']:.3f} us "
              f"(p99 {r['observe']['p99_us']:.3f}); decide mean {r['decide']['mean_us']:.3f} us "
              f"(p99 {r['decide']['p99_us']:.3f})")
    if "--write" in argv:
        print(evaluate2.write(study4.seal(rec, "V4-native-resources", "measurement, not a gate"),
                              V4 / "evidence" / "native" / "resources.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
