# ECS_G native mathematical foundation.
#
# G0 materializes only the frozen cubic microscopic/S3 mathematics. It owns no
# production transition law and has no inference/runtime authority.

elpis_add_library(ECS_G elpis_ecsg_math SHARED
  src/ecsg_math.c
  src/ecsg_state.c
  src/ecsg_executor.c)
target_include_directories(elpis_ecsg_math PUBLIC include)
target_link_libraries(elpis_ecsg_math PRIVATE m)

# Mutable ECS_G residency is additive. Keep the Runtime R1 measured library
# unchanged; this adapter links to its public ABI and to generic FMS.
elpis_add_library(ECS_G elpis_ecsg_fms SHARED
  src/ecsg_fms.c
  ../substrate/src/fms_pal_posix.c)
target_include_directories(elpis_ecsg_fms PUBLIC include)
target_link_libraries(elpis_ecsg_fms PUBLIC elpis_ecsg_math elpis_fms PRIVATE m Threads::Threads)

elpis_add_test(ECS_G test_ecsg_fms
  SOURCES tests/test_ecsg_fms.c
  LIBS elpis_ecsg_fms Threads::Threads m
  TIMEOUT 120)

elpis_add_test(ECS_G test_ecsg_fms_performance
  SOURCES tests/test_ecsg_fms_performance.c
  LIBS elpis_ecsg_fms m
  TIMEOUT 120)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve explicit scalar accumulation order while differential parity is
  # frozen. Optimized/vectorized kernels are a later, separately qualified ABI.
  target_compile_options(elpis_ecsg_math PRIVATE -ffp-contract=off)
endif()

elpis_add_test(ECS_G test_ecsg_math
  SOURCES tests/test_ecsg_math.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS_G test_ecsg_state
  SOURCES tests/test_ecsg_state.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS_G test_ecsg_snapshot
  SOURCES tests/test_ecsg_snapshot.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS_G test_ecsg_executor
  SOURCES tests/test_ecsg_executor.c
  LIBS elpis_ecsg_math m
  TIMEOUT 60)

# The same parity suite against the baseline-ISA kernels only (no runtime
# dispatch), so the clone the host does not select is qualified too.
elpis_add_test(ECS_G test_ecsg_executor_baseline
  SOURCES tests/test_ecsg_executor.c src/ecsg_executor.c src/ecsg_math.c src/ecsg_state.c
  INCLUDES include
  DEFINES ELPIS_ECSG_NO_CLONES
  LIBS m
  TIMEOUT 60)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_ecsg_executor_baseline PRIVATE -ffp-contract=off)
endif()

elpis_add_test(ECS_G test_ecsg_executor_txn
  SOURCES tests/test_ecsg_executor_txn.c
  LIBS elpis_ecsg_math m Threads::Threads
  TIMEOUT 120)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Test-only allocation failure injection. The frozen executor sources are
  # compiled into this test but are not modified by Mutable FMS R0.
  elpis_add_test(ECS_G test_ecsg_fms_alloc
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
  elpis_add_test(ECS_G test_ecsg_executor_alloc
    SOURCES tests/test_ecsg_executor_alloc.c src/ecsg_executor.c src/ecsg_math.c src/ecsg_state.c
    INCLUDES include
    LIBS m
    TIMEOUT 60)
  target_compile_options(test_ecsg_executor_alloc PRIVATE -ffp-contract=off)
  target_link_options(test_ecsg_executor_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=aligned_alloc,--wrap=posix_memalign")
endif()
