# ECS memory, model-free R0 (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY): bounded typed relational memory mechanics
# (docs/research/ECS_MEMORY_MF_R0.md). Built and tested in every native job; never part of the cognitive path.
# Libraries are test-kind research support (elpis_add_test_library): nothing here is a declared, loadable
# Elpis library, and src/ never links it.
set(_mem ${CMAKE_CURRENT_SOURCE_DIR}/native)

# PR #47 donor-derived lattices, kept byte-identical as differential references (D31-B linear scan, D31-C
# direct index) and the adapter that drives D31-C from (context, entity, role) identifiers.
elpis_add_test_library(research elpis_ecs_mem_lattice
  ${_mem}/src/bounded_lattice.c ${_mem}/src/indexed_lattice.c ${_mem}/src/model_free_memory.c)
target_include_directories(elpis_ecs_mem_lattice PUBLIC ${_mem}/include/elpis)
target_link_libraries(elpis_ecs_mem_lattice PUBLIC m)

elpis_add_test(research ecs_mem_lattice_model_free SOURCES ${_mem}/tests/test_model_free.c
  LIBS elpis_ecs_mem_lattice TIMEOUT 60)
elpis_add_test(research ecs_mem_lattice_audit SOURCES ${_mem}/tests/test_lattice_audit.c
  LIBS elpis_ecs_mem_lattice TIMEOUT 60)
# The allocation-free law is proven by link-time allocator wrapping (GNU ld / lld on Linux). Sanitizer
# runtimes interpose the allocator themselves, so the wrapped test runs only in uninstrumented builds.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(research ecs_mem_lattice_differential SOURCES ${_mem}/tests/test_index.c
    LIBS elpis_ecs_mem_lattice TIMEOUT 300)
  target_link_options(ecs_mem_lattice_differential PRIVATE
    -Wl,--wrap=calloc -Wl,--wrap=malloc -Wl,--wrap=free)
endif()

# The MF-R0 memory mechanism (ecs_mem.h): integer arithmetic only, one allocation per handle.
elpis_add_test_library(research elpis_ecs_mem ${_mem}/src/ecs_mem.c)
target_include_directories(elpis_ecs_mem PUBLIC ${_mem}/include/elpis)
target_link_libraries(elpis_ecs_mem PUBLIC elpis_hash)
elpis_add_test(research ecs_mem_mechanics SOURCES ${_mem}/tests/test_ecs_mem.c LIBS elpis_ecs_mem TIMEOUT 120)
elpis_add_test(research ecs_mem_reference SOURCES ${_mem}/tests/test_ecs_mem_reference.c LIBS elpis_ecs_mem TIMEOUT 600)
elpis_add_test(research ecs_mem_concurrency SOURCES ${_mem}/tests/test_ecs_mem_concurrency.c
  LIBS elpis_ecs_mem Threads::Threads TIMEOUT 300)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(research ecs_mem_alloc SOURCES ${_mem}/tests/test_ecs_mem_alloc.c LIBS elpis_ecs_mem TIMEOUT 300)
  target_link_options(ecs_mem_alloc PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
    -Wl,--wrap=aligned_alloc -Wl,--wrap=free)
endif()
elpis_add_test(research ecs_mem_snapshot SOURCES ${_mem}/tests/test_ecs_mem_snapshot.c LIBS elpis_ecs_mem TIMEOUT 600)
