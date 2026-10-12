# ECS Binary Observation R0 (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY, UNQUALIFIED): the observer library, its mechanics
# test against the real K1 library, and the direct-native performance driver. Never part of the cognitive path:
# nothing in src/ or native/runtime links or loads elpis_ebo_r0.
elpis_add_library(research elpis_ebo_r0 SHARED ebo_r0.c)
target_include_directories(elpis_ebo_r0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(elpis_ebo_r0 PRIVATE elpis_hash)
target_link_options(elpis_ebo_r0 PRIVATE -Wl,--exclude-libs,ALL)   # the static SHA-256 stays private
target_compile_options(elpis_ebo_r0 PRIVATE -ffp-contract=off)

elpis_add_test(research test_ebo_r0 SOURCES test_ebo_r0.c
  LIBS elpis_ebo_r0 elpis_ecsg_k1 Threads::Threads m TIMEOUT 300)
target_compile_options(test_ebo_r0 PRIVATE -ffp-contract=off)

add_executable(ebo_r0_bench ebo_r0_bench.c)
target_link_libraries(ebo_r0_bench PRIVATE elpis_ebo_r0 elpis_ecsg_k1 m)
_elpis_register(research ebo_r0_bench benchmark)
