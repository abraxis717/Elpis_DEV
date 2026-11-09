"""R118-A historical replay: the frozen R78 run.sh protocol, from a clean checkout.

The historical bundle (``run.sh`` input zip ``d20bccce...``) is not published;
its members are, as loose files under ``frozen_host``. This harness rebuilds
the run.sh execution environment from those authenticated files and runs the
*unmodified* historical scripts as ``__main__`` in a volatile scratch copy:

* ``verify``: ``verify_r78.py`` (stdlib-only evidence review, 12 adversarial mutations).
* ``spot``  : ``spot_r78.py`` (one graph of genuine PyTorch tangent replay + covariance).
* ``run``   : ``r78.py run`` (full frozen-inference reproduction). The result archive
  is re-created *in scratch* and compared byte-for-byte with the sealed archive,
  then ``verify_r78.py`` is run again on the replayed tree.

Nothing is written under the repository. After the replay the repository's
frozen tree is re-authenticated.

Usage (from the repository root, with an interpreter that has torch):
    python -m research.yang_mills_hca.r118_scalable_awr.historical_replay [verify|spot|run ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from . import provenance as P

HISTORICAL_ENV = {
    # Exactly the run.sh execution environment (CUDA hidden, single-threaded BLAS/OMP).
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
}


def _run(cmd, cwd, env, timeout):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)),
        "exit": proc.returncode,
        "seconds": round(time.perf_counter() - t0, 3),
        "stdout_tail": proc.stdout.strip().splitlines()[-8:],
        "stderr_tail": proc.stderr.strip().splitlines()[-8:],
    }


def replay(modes=("verify", "spot", "run"), python: str | None = None, timeout: int = 3600) -> dict:
    python = python or sys.executable
    before = P.verify_manifest()
    out = {"manifest_before": before, "modes": {}, "python": python}
    with P.volatile_scratch("ym_r118_r78_") as scratch:
        staged = P.stage_snapshot(scratch)
        out["staged_files"] = len(staged)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        env.update(HISTORICAL_ENV)
        env.update({"HOME": str(scratch), "XDG_CACHE_HOME": str(scratch), "TMPDIR": str(scratch)})
        for mode in modes:
            if mode == "verify":
                out["modes"]["verify"] = _run([python, "verify_r78.py"], scratch, env, timeout)
            elif mode == "spot":
                out["modes"]["spot"] = _run([python, "spot_r78.py"], scratch, env, timeout)
            elif mode == "run":
                sealed = P.sha256_file(scratch / P.R78_ARCHIVE)
                rec = _run([python, "-u", "r78.py", "run"], scratch, env, timeout)
                replayed = P.sha256_file(scratch / P.R78_ARCHIVE)
                rec["sealed_sha256"] = sealed
                rec["replayed_sha256"] = replayed
                rec["byte_identical"] = sealed == replayed == P.R78_ARCHIVE_SHA256
                rec["reverify"] = _run([python, "verify_r78.py"], scratch, env, timeout)
                out["modes"]["run"] = rec
            else:
                raise P.ProvenanceError("UNKNOWN_REPLAY_MODE:" + mode)
    out["manifest_after"] = P.verify_manifest()
    ok = all(m["exit"] == 0 for m in out["modes"].values())
    if "run" in out["modes"]:
        r = out["modes"]["run"]
        ok = ok and r["byte_identical"] and r["reverify"]["exit"] == 0
    out["disposition"] = "HISTORICAL_R78_REPLAY_PASS" if ok else "HISTORICAL_R78_REPLAY_FAIL"
    return out


def main(argv=None) -> int:
    modes = tuple(argv if argv else ("verify", "spot", "run"))
    result = replay(modes)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["disposition"].endswith("PASS") else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
