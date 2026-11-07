# CGC-LAO bounded reliability-mixture learner (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY): the native C11 reference of
# research/cgc_lao/solver_lab/v2. Test-kind research support (elpis_add_test_library): never a declared, loadable
# Elpis library, never linked by src/ or RuntimeCore. Built and tested in every native job, sanitizers included.
set(_rmf ${CMAKE_CURRENT_SOURCE_DIR})

elpis_add_test_library(research elpis_cgc_lao_rmf ${_rmf}/src/cgc_lao_rmf.c)
target_include_directories(elpis_cgc_lao_rmf PUBLIC ${_rmf}/include)
target_link_libraries(elpis_cgc_lao_rmf PUBLIC m)
# No fused multiply-add: the learner is bitwise identical to its Python scalar reference.
target_compile_options(elpis_cgc_lao_rmf PRIVATE -ffp-contract=off)

elpis_add_test(research cgc_lao_rmf_mechanics SOURCES ${_rmf}/tests/test_cgc_lao_rmf.c
  LIBS elpis_cgc_lao_rmf TIMEOUT 120)
# The allocation-free law is proven by link-time allocator wrapping (GNU ld / lld on Linux). Sanitizer runtimes
# interpose the allocator themselves, so the wrapped test runs only in uninstrumented builds.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT (ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_UBSAN OR ELPIS_ENABLE_TSAN))
  elpis_add_test(research cgc_lao_rmf_alloc SOURCES ${_rmf}/tests/test_cgc_lao_rmf.c
    LIBS elpis_cgc_lao_rmf DEFINES CGC_LAO_RMF_WRAP_ALLOC TIMEOUT 120)
  target_link_options(cgc_lao_rmf_alloc PRIVATE
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free)
endif()
