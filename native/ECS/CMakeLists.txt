# ECS native mathematical foundation.
#
# G0 materializes only the frozen cubic microscopic/S3 mathematics. It owns no
# production transition law and has no inference/runtime authority.

elpis_add_library(ECS elpis_ecsg_math SHARED
  src/ecsg_math.c
  src/ecsg_state.c
  src/ecsg_executor.c)
target_include_directories(elpis_ecsg_math PUBLIC include)
target_link_libraries(elpis_ecsg_math PRIVATE m)

# Native K1 consolidation runtime (docs/ECS_K1_RUNTIME.md): the Retention R3 OUTCOME_A mechanism. Additive:
# its own library over the public math ABI; libelpis_ecsg_math (the measured Runtime R1 binary) is unchanged.
elpis_add_library(ECS elpis_ecsg_k1 SHARED src/ecsg_k1.c)
target_include_directories(elpis_ecsg_k1 PUBLIC include)
target_link_libraries(elpis_ecsg_k1 PUBLIC elpis_ecsg_math PRIVATE elpis_hash m)

# K1 residency over generic FMS: pin -> native K1 over the resident bytes -> unpin (no transient executor).
elpis_add_library(ECS elpis_ecsg_k1_fms SHARED
  src/ecsg_k1_fms.c
  ../substrate/src/fms_pal_posix.c)
target_include_directories(elpis_ecsg_k1_fms PUBLIC include)
target_link_libraries(elpis_ecsg_k1_fms PUBLIC elpis_ecsg_k1 elpis_fms PRIVATE elpis_hash m Threads::Threads)

elpis_add_test(ECS test_ecsg_k1_fms
  SOURCES tests/test_ecsg_k1_fms.c
  LIBS elpis_ecsg_k1_fms elpis_ecsg_k1 elpis_ecsg_math Threads::Threads m
  TIMEOUT 120)

elpis_add_test(ECS test_ecsg_k1_performance
  SOURCES tests/test_ecsg_k1_performance.c
  LIBS elpis_ecsg_k1_fms elpis_ecsg_k1 elpis_ecsg_math m
  TIMEOUT 600)

elpis_add_test(ECS test_ecsg_k1
  SOURCES tests/test_ecsg_k1.c
  LIBS elpis_ecsg_k1 elpis_ecsg_math elpis_hash Threads::Threads m
  TIMEOUT 120)

# Mutable ECS residency is additive. Keep the Runtime R1 measured library
# unchanged; this adapter links to its public ABI and to generic FMS.
elpis_add_library(ECS elpis_ecsg_fms SHARED
  src/ecsg_fms.c
  ../substrate/src/fms_pal_posix.c)
target_include_directories(elpis_ecsg_fms PUBLIC include)
target_link_libraries(elpis_ecsg_fms PUBLIC elpis_ecsg_math elpis_fms PRIVATE m Threads::Threads)

elpis_add_test(ECS test_ecsg_fms
  SOURCES tests/test_ecsg_fms.c
  LIBS elpis_ecsg_fms Threads::Threads m
  TIMEOUT 120)

elpis_add_test(ECS test_ecsg_fms_performance
  SOURCES tests/test_ecsg_fms_performance.c
  LIBS elpis_ecsg_fms m
  TIMEOUT 120)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve explicit scalar accumulation order while differential parity is
  # frozen. Optimized/vectorized kernels are a later, separately qualified ABI.
  target_compile_options(elpis_ecsg_math PRIVATE -ffp-contract=off)
  target_compile_options(elpis_ecsg_k1 PRIVATE -ffp-contract=off)
endif()

elpis_add_test(ECS test_ecsg_math
  SOURCES tests/test_ecsg_math.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS test_ecsg_state
  SOURCES tests/test_ecsg_state.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS test_ecsg_snapshot
  SOURCES tests/test_ecsg_snapshot.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS test_ecsg_executor
  SOURCES tests/test_ecsg_executor.c
  LIBS elpis_ecsg_math m
  TIMEOUT 60)

# The same parity suite against the baseline-ISA kernels only (no runtime
# dispatch), so the clone the host does not select is qualified too.
elpis_add_test(ECS test_ecsg_executor_baseline
  SOURCES tests/test_ecsg_executor.c src/ecsg_executor.c src/ecsg_math.c src/ecsg_state.c
  INCLUDES include
  DEFINES ELPIS_ECSG_NO_CLONES
  LIBS m
  TIMEOUT 60)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_ecsg_executor_baseline PRIVATE -ffp-contract=off)
endif()

elpis_add_test(ECS test_ecsg_executor_txn
  SOURCES tests/test_ecsg_executor_txn.c
  LIBS elpis_ecsg_math m Threads::Threads
  TIMEOUT 120)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Test-only allocation failure injection. The frozen executor sources are
  # compiled into this test but are not modified by Mutable FMS R0.
  elpis_add_test(ECS test_ecsg_fms_alloc
    SOURCES tests/test_ecsg_fms_alloc.c src/ecsg_fms.c src/ecsg_executor.c src/ecsg_math.c src/ecsg_state.c
            ../substrate/src/fms_pal_posix.c
    INCLUDES include ../substrate/include
    LIBS elpis_fms m Threads::Threads
    TIMEOUT 60)
  target_compile_options(test_ecsg_fms_alloc PRIVATE -ffp-contract=off)
  target_link_options(test_ecsg_fms_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=aligned_alloc")
  # No allocation after creation: the executor and kernel sources are linked
  # into the test itself so -Wl,--wrap counts every allocation they make.
  elpis_add_test(ECS test_ecsg_executor_alloc
    SOURCES tests/test_ecsg_executor_alloc.c src/ecsg_executor.c src/ecsg_math.c src/ecsg_state.c
    INCLUDES include
    LIBS m
    TIMEOUT 60)
  target_compile_options(test_ecsg_executor_alloc PRIVATE -ffp-contract=off)
  target_link_options(test_ecsg_executor_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=aligned_alloc,--wrap=posix_memalign")
  # The K1 runtime: no allocation after create/reserve; allocation failures refuse cleanly.
  elpis_add_test(ECS test_ecsg_k1_alloc
    SOURCES tests/test_ecsg_k1_alloc.c src/ecsg_k1.c src/ecsg_math.c ../substrate/src/sha256.c
    INCLUDES include src ../substrate/include
    LIBS m
    TIMEOUT 60)
  target_compile_options(test_ecsg_k1_alloc PRIVATE -ffp-contract=off)
  target_link_options(test_ecsg_k1_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=aligned_alloc,--wrap=posix_memalign")
  # K1 residency: the warm path through FMS allocates nothing; registration allocation failures refuse cleanly.
  elpis_add_test(ECS test_ecsg_k1_fms_alloc
    SOURCES tests/test_ecsg_k1_fms_alloc.c src/ecsg_k1_fms.c src/ecsg_k1.c src/ecsg_math.c
            ../substrate/src/sha256.c ../substrate/src/fms_core.c ../substrate/src/fms_pal_posix.c
    INCLUDES include src ../substrate/include
    LIBS m Threads::Threads
    TIMEOUT 60)
  target_compile_options(test_ecsg_k1_fms_alloc PRIVATE -ffp-contract=off)
  target_link_options(test_ecsg_k1_fms_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=posix_memalign")
  # K1 residency: a failed FMS release is reconciled (link-time injected failure).
  elpis_add_test(ECS test_ecsg_k1_fms_faults
    SOURCES tests/test_ecsg_k1_fms_faults.c src/ecsg_k1_fms.c src/ecsg_k1.c src/ecsg_math.c
            ../substrate/src/sha256.c ../substrate/src/fms_core.c ../substrate/src/fms_pal_posix.c
    INCLUDES include src ../substrate/include
    LIBS m Threads::Threads
    TIMEOUT 60)
  target_compile_options(test_ecsg_k1_fms_faults PRIVATE -ffp-contract=off)
  target_link_options(test_ecsg_k1_fms_faults PRIVATE "LINKER:--wrap=fms_release")
endif()

# ECS memory (docs/ECS_MEMORY.md): the bounded model-free typed relational memory of the one ECS, relocated from
# research/ecs_memory_mf_r0 (the byte-frozen independent reference; never linked here). Integer arithmetic only.
elpis_add_library(ECS elpis_ecs_memory SHARED src/ecs_memory.c)
target_include_directories(elpis_ecs_memory PUBLIC include)
target_link_libraries(elpis_ecs_memory PRIVATE elpis_hash)

elpis_add_test(ECS test_ecs_memory SOURCES tests/test_ecs_memory.c LIBS elpis_ecs_memory elpis_hash TIMEOUT 120)
elpis_add_test(ECS test_ecs_memory_capability SOURCES tests/test_ecs_memory_capability.c
  LIBS elpis_ecs_memory elpis_hash TIMEOUT 60)
elpis_add_test(ECS test_ecs_memory_snapshot SOURCES tests/test_ecs_memory_snapshot.c
  LIBS elpis_ecs_memory elpis_hash TIMEOUT 600)
elpis_add_test(ECS test_ecs_memory_concurrency SOURCES tests/test_ecs_memory_concurrency.c
  LIBS elpis_ecs_memory elpis_hash Threads::Threads TIMEOUT 300)
# One allocation per handle, none afterwards: proven by link-time allocator wrapping, so the library source is
# compiled into the test (sanitizer runtimes interpose the allocator themselves: uninstrumented builds only).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(ECS test_ecs_memory_alloc SOURCES tests/test_ecs_memory_alloc.c src/ecs_memory.c
    INCLUDES include LIBS elpis_hash TIMEOUT 300)
  target_link_options(test_ecs_memory_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=free")
endif()

# ECS composite retained state (docs/ECS_COMPOSITE_STATE.md): K1 and the ECS memory under one owner, one
# transaction and one identity (ELPISEC1 v1). Binds the components; computes no law of its own.
elpis_add_library(ECS elpis_ecs_composite SHARED src/ecs_composite.c)
target_include_directories(elpis_ecs_composite PUBLIC include)
target_link_libraries(elpis_ecs_composite PUBLIC elpis_ecsg_k1 elpis_ecs_memory PRIVATE elpis_hash)

elpis_add_test(ECS test_ecs_composite SOURCES tests/test_ecs_composite.c
  LIBS elpis_ecs_composite elpis_ecs_memory elpis_ecsg_k1 elpis_ecsg_math elpis_hash Threads::Threads m TIMEOUT 300)
