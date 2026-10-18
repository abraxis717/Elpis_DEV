# ECS composite retained state (docs/ECS_COMPOSITE_STATE.md): ECS_G's K1 and ECS_C's memory under one owner, one
# transaction and one identity (ELPISEC1 v1). It binds the two domains and computes no law of its own, so it belongs to
# RuntimeCore, the only composer (docs/NATIVE_SPLIT_CENSUS.md). Sources moved byte for byte from native/ECS.
elpis_add_library(runtime elpis_ecs_composite SHARED src/ecs_composite.c)
target_include_directories(elpis_ecs_composite PUBLIC include)
target_link_libraries(elpis_ecs_composite PUBLIC elpis_ecsg_k1 elpis_ecs_memory PRIVATE elpis_hash)

elpis_add_test(runtime test_ecs_composite SOURCES tests/test_ecs_composite.c
  LIBS elpis_ecs_composite elpis_ecs_memory elpis_ecsg_k1 elpis_ecsg_math elpis_hash Threads::Threads m TIMEOUT 300)

# The legacy-surface differential guard of the live-coordinate additions (K1, K1 FMS, composite): Part A's legacy
# battery is the cross-version check (its transcript, built against the pre-correction headers, is byte-identical
# over the pre-correction and the corrected libraries; docs/RUNTIME_CORE.md, "Legacy surface"); here it must be
# deterministic, and Part B proves the additive entry points against the legacy ones in the same build.
elpis_add_test(runtime test_k1_legacy_differential SOURCES tests/test_k1_legacy_differential.c
  DEFINES ELPIS_K1_LIVE_COORDINATE
  LIBS elpis_ecs_composite elpis_ecsg_k1_fms elpis_ecsg_k1 elpis_ecs_memory elpis_ecsg_math elpis_hash m
  ARGS "."
  TIMEOUT 120)
