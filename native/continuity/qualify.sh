#!/usr/bin/env bash
# Bounded continuity qualification with live output (docs/CONTINUITY.md).
#
#   native/continuity/qualify.sh [BUILD_DIR]        (default: build)
#
# Builds only the libraries these checks load, then runs, each under a hard wall-clock limit:
#   1. Rust qualification + C ABI against the frozen v2 vectors   (ctest -L continuity)
#   2. the Python adapter and the runtime paths that publish to continuity (pytest, one line per test)
# It never runs the repository-wide suite.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
build=$(realpath -m "${1:-$repo/build}")
python=${PYTHON:-python3}
start=$SECONDS

step() { printf '\n== [%3ss] %s (limit %ss)\n' "$((SECONDS - start))" "$1" "$2"; }

step "configure ($build)" 300
timeout 300 cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release >/dev/null
step "build continuity, K1 and ingress/retrieval libraries" 900
timeout 900 cmake --build "$build" --parallel --target elpis_continuity elpis_continuity_testing \
    test_continuity_abi elpis_ecsg_k1 elpis_ecsg_k1_fms elpis_ingress_bridge elpis_retrieval_bridge

step "Rust qualification and C ABI (ctest -L continuity)" 300
timeout 300 ctest --test-dir "$build" -L continuity --output-on-failure

step "Python adapter and runtime continuity paths (pytest)" 300
cd "$repo"
ELPIS_NATIVE_BUILD="$build" ELPIS_REQUIRE_NATIVE=1 PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
    timeout 300 "$python" -m pytest -p no:cacheprovider -v --durations=5 \
    tests/continuity tests/boundary/test_one_ecs.py tests/integration/test_evolution.py \
    tests/integration/test_runtime_hot_path.py tests/integration/test_codec_ecs_turn.py

printf '\n== [%3ss] continuity qualification PASS\n' "$((SECONDS - start))"
