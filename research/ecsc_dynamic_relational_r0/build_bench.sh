#!/bin/sh
# Build the HCA_REVISION_R0 bench (research only) from the repository's own sources: the ECS_C relational view, the
# ECS_C memory, the native HCA research substrate and the substrate's SHA-256. Nothing is installed or linked from
# a build tree. Usage: build_bench.sh OUTPUT_BINARY
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:?usage: build_bench.sh OUTPUT_BINARY}
cc=${CC:-gcc}
"$cc" -std=c11 -O2 -UNDEBUG -Wall -Wextra -Wpedantic -Werror \
  -I"$repo/native/ECS_C/relational/include" -I"$repo/native/ECS_C/relational/tests" \
  -I"$repo/native/ECS_C/authority/include" -I"$repo/native/ECS_C/memory/include" \
  -I"$repo/native/ECS_C/hca/include" -I"$repo/native/substrate/include" \
  "$repo/research/ecsc_dynamic_relational_r0/bench_hca_revision.c" \
  "$repo/native/ECS_C/relational/src/ecs_c_relational.c" "$repo/native/ECS_C/memory/src/ecs_memory.c" \
  "$repo/native/ECS_C/hca/src/ecs_c_hca.c" "$repo/native/substrate/src/sha256.c" \
  -Wl,--wrap=calloc -o "$out"
