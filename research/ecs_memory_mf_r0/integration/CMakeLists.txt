# ECS Memory MF-R0, M4 (RESEARCH_ONLY): next to native K1 and generic FMS residency with no new authority
# (docs/research/ECS_MEMORY_MF_R0.md, "M4"). Kept apart from ../CMakeLists.txt, which the laboratory freeze binds.
set(_mem ${CMAKE_CURRENT_SOURCE_DIR}/../native)
elpis_add_test_library(research elpis_ecs_mem_composite ${_mem}/src/ecs_mem_composite.c)
target_include_directories(elpis_ecs_mem_composite PUBLIC ${_mem}/include/elpis)
target_link_libraries(elpis_ecs_mem_composite PUBLIC elpis_ecs_mem elpis_hash)
elpis_add_test(research ecs_mem_composite SOURCES ${_mem}/tests/test_ecs_mem_composite.c
  LIBS elpis_ecs_mem_composite elpis_ecsg_k1 elpis_ecsg_math elpis_fms Threads::Threads m TIMEOUT 120)
