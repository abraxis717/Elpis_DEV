# ECS_C: continuity and memory (docs/NATIVE_SPLIT_CENSUS.md): the memory, and the authority successor of
# native/continuity (authority/, docs/ECS_C_AUTHORITY.md).

# ECS_C memory (docs/ECS_MEMORY.md): the bounded model-free typed relational memory, relocated from
# research/ecs_memory_mf_r0 (the byte-frozen independent reference; never linked here). Integer arithmetic only. The
# sources moved byte for byte from native/ECS. ECS_C links no geometric (ECS_G) or runtime code.
elpis_add_library(ECS_C elpis_ecs_memory SHARED memory/src/ecs_memory.c)
target_include_directories(elpis_ecs_memory PUBLIC memory/include)
target_link_libraries(elpis_ecs_memory PRIVATE elpis_hash)

elpis_add_test(ECS_C test_ecs_memory SOURCES memory/tests/test_ecs_memory.c LIBS elpis_ecs_memory elpis_hash TIMEOUT 120)
elpis_add_test(ECS_C test_ecs_memory_capability SOURCES memory/tests/test_ecs_memory_capability.c
  LIBS elpis_ecs_memory elpis_hash TIMEOUT 60)
elpis_add_test(ECS_C test_ecs_memory_snapshot SOURCES memory/tests/test_ecs_memory_snapshot.c
  LIBS elpis_ecs_memory elpis_hash TIMEOUT 600)
elpis_add_test(ECS_C test_ecs_memory_concurrency SOURCES memory/tests/test_ecs_memory_concurrency.c
  LIBS elpis_ecs_memory elpis_hash Threads::Threads TIMEOUT 300)
# One allocation per handle, none afterwards: proven by link-time allocator wrapping, so the library source is
# compiled into the test (sanitizer runtimes interpose the allocator themselves: uninstrumented builds only).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(ECS_C test_ecs_memory_alloc SOURCES memory/tests/test_ecs_memory_alloc.c memory/src/ecs_memory.c
    INCLUDES memory/include LIBS elpis_hash TIMEOUT 300)
  target_link_options(test_ecs_memory_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=free")
endif()

add_subdirectory(authority)

# ECS_C HCA (docs/ECS_C_HCA.md): the native research substrate of typed-hypergraph cellular memory, the B0/B1/B2
# comparators of research/ecsc_hca_r0. RESEARCH ONLY, GATED: nothing canonical links or loads it
# (tests/boundary/test_ecs_c_hca.py); its tests and the offline differential are its only consumers.
elpis_add_library(ECS_C elpis_ecs_c_hca SHARED hca/src/ecs_c_hca.c)
target_include_directories(elpis_ecs_c_hca PUBLIC hca/include)
target_link_libraries(elpis_ecs_c_hca PRIVATE elpis_hash)
elpis_add_test(ECS_C test_ecs_c_hca SOURCES hca/tests/test_ecs_c_hca.c LIBS elpis_ecs_c_hca elpis_hash TIMEOUT 120)
# One allocation per graph and per session, none per call: link-time allocator wrapping (uninstrumented builds only).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(ECS_C test_ecs_c_hca_alloc SOURCES hca/tests/test_ecs_c_hca_alloc.c hca/src/ecs_c_hca.c
    INCLUDES hca/include LIBS elpis_hash TIMEOUT 120)
  target_link_options(test_ecs_c_hca_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=free")
endif()

# ECS_C relational view (docs/ECS_C_RELATIONAL.md): ECSC_DYNAMIC_RELATIONAL_DEV_R0, the source-revision-aware bounded
# maintenance of admitted typed evidence under the authority's sources. DEV, NOT COMPOSED: nothing canonical links or
# loads it (tests/boundary/test_ecs_c_relational.py). It links the memory (relation keys) and reads the authority's
# entity layout from its header; it links no authority code.
elpis_add_library(ECS_C elpis_ecs_c_relational SHARED relational/src/ecs_c_relational.c)
target_include_directories(elpis_ecs_c_relational PUBLIC relational/include authority/include)
target_link_libraries(elpis_ecs_c_relational PUBLIC elpis_ecs_memory PRIVATE elpis_hash)
# The contract attacked, with the test-only damage hook (the library source compiled in) and the differential against
# the exact rebuild by the real memory.
elpis_add_test(ECS_C test_ecs_c_relational
  SOURCES relational/tests/test_ecs_c_relational.c relational/src/ecs_c_relational.c
  INCLUDES relational/include authority/include DEFINES ELPIS_ECS_C_REL_TESTING
  LIBS elpis_ecs_memory elpis_hash TIMEOUT 600)
# End to end over the real authority, through the production libraries.
elpis_add_test(ECS_C test_ecs_c_relational_authority
  SOURCES relational/tests/test_ecs_c_relational_authority.c
  LIBS elpis_ecs_c_relational "${CMAKE_CURRENT_BINARY_DIR}/authority/libelpis_ecs_c_authority.so"
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_ecs_c_relational_authority/scratch"
  TIMEOUT 120)
add_dependencies(test_ecs_c_relational_authority elpis_ecs_c_authority)
# One allocation per view, none per call: link-time allocator wrapping (uninstrumented builds only).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(ECS_C test_ecs_c_relational_alloc
    SOURCES relational/tests/test_ecs_c_relational_alloc.c relational/src/ecs_c_relational.c
    INCLUDES relational/include authority/include LIBS elpis_ecs_memory elpis_hash TIMEOUT 120)
  target_link_options(test_ecs_c_relational_alloc PRIVATE
    "LINKER:--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=free")
endif()
