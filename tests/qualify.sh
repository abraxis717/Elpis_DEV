#!/usr/bin/env bash
# Bounded qualification with live output (docs/CI_POLICY.md, "Lanes"; tests/lanes.py).
#
#   tests/qualify.sh [LANE] [BUILD_DIR]      LANE: fast (default) | native | stress | scientific | historical | all
#
#   fast        warm native build + every ctest test (NATIVE) + the FAST pytest lane      (target <= 60 s warm)
#   native      warm native build + every ctest test
#   stress      the STRESS pytest lane (multi-process contention, fault matrices, scaling)
#   scientific  the SCIENTIFIC pytest lane (frozen evidence and replay; several minutes)
#   historical  the HISTORICAL pytest lane (DSV4.1 tower, ECS dynamics, retained model mechanics)
#   all         every lane above, in order
#
# Every step runs under a hard wall-clock limit and prints as it goes. It never runs a laboratory's science
# commands. Sanitizer builds (ASan+UBSan, TSan) are separate build trees: configure one with
# -DELPIS_ENABLE_ASAN=ON -DELPIS_ENABLE_UBSAN=ON (or -DELPIS_ENABLE_TSAN=ON) and run `ctest` there.
set -euo pipefail
lane=${1:-fast}
repo=$(cd "$(dirname "$0")/.." && pwd)
build=$(realpath -m "${2:-$repo/build}")
python=${PYTHON:-python3}
start=$SECONDS
jobs=$(nproc 2>/dev/null || echo 2)

step() { printf '\n== [%3ss] %s (limit %ss)\n' "$((SECONDS - start))" "$1" "$2"; }

native() {
    step "configure ($build)" 300
    timeout 300 cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    step "build (warm: no-op)" 1800
    timeout 1800 cmake --build "$build" --parallel "$jobs" | grep -E "error|warning|Linking" || true
    step "NATIVE: ctest (ECS, K1, FMS, continuity, RuntimeCore, H-ECS mechanics, ...)" 600
    timeout 600 ctest --test-dir "$build" -j "$jobs" --timeout 300 --output-on-failure | tail -n 15
}

pylane() {  # pylane LANE LIMIT
    step "pytest --lane $1" "$2"
    (cd "$repo" && ELPIS_NATIVE_BUILD="$build" ELPIS_REQUIRE_NATIVE=1 \
        PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
        timeout "$2" "$python" -m pytest -p no:cacheprovider -q --durations=5 --lane "$1")
}

case "$lane" in
    fast) native; pylane fast 300 ;;
    native) native ;;
    stress) pylane stress 900 ;;
    scientific) pylane scientific 1800 ;;
    historical) pylane historical 900 ;;
    all) native; pylane fast 300; pylane stress 900; pylane scientific 1800; pylane historical 900 ;;
    *) echo "usage: tests/qualify.sh [fast|native|stress|scientific|historical|all] [BUILD_DIR]" >&2; exit 2 ;;
esac
printf '\n== [%3ss] %s qualification PASS\n' "$((SECONDS - start))" "$lane"
