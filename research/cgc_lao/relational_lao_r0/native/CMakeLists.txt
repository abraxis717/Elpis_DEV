# Relational LAO R0 native relational adapter (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY): the native C11 port of
# research/cgc_lao/relational_lao_r0/src/relational_lao/scalar.py. Test-kind research support (elpis_add_test_library):
# never a declared, loadable Elpis library, never linked by src/ or RuntimeCore, and it links no ECS_C or HCA library
# (the research host bridges them in Python). Built and tested in every native job, sanitizers included.
set(_rlao ${CMAKE_CURRENT_SOURCE_DIR})

elpis_add_test_library(research elpis_rlao ${_rlao}/src/rlao.c)
target_include_directories(elpis_rlao PUBLIC ${_rlao}/include)
target_link_libraries(elpis_rlao PUBLIC m)
# No fused multiply-add: the adapter is bitwise identical to its Python scalar reference.
target_compile_options(elpis_rlao PRIVATE -ffp-contract=off)

elpis_add_test(research rlao_mechanics SOURCES ${_rlao}/tests/test_rlao.c LIBS elpis_rlao TIMEOUT 120)
# The allocation-free law is proven by link-time allocator wrapping (GNU ld / lld on Linux). Sanitizer runtimes
# interpose the allocator themselves, so the wrapped test runs only in uninstrumented builds.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(research rlao_alloc SOURCES ${_rlao}/tests/test_rlao.c LIBS elpis_rlao DEFINES RLAO_WRAP_ALLOC
    TIMEOUT 120)
  target_link_options(rlao_alloc PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free)
endif()
