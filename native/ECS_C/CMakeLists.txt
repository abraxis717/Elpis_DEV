# ECS_C: continuity and memory (docs/NATIVE_SPLIT_CENSUS.md). Stage B holds the memory; the authority successor of
# native/continuity arrives in Stage C.

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
