# Pipeline: bounded Regex -> HACF query ingress.
#
#   ingress/  Streaming Regex lexer over a bounded interval-specification
#             grammar (PCRE2), Regex -> HACF read-only context proposal, and
#             the atomic query-local proposal batch published as a semantic
#             query overlay. Every authority flag stays zero.
#   bridge/   Narrow C ABI loaded by elpis.pipeline.ingress.

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

set(ING "${CMAKE_CURRENT_SOURCE_DIR}/ingress")

elpis_add_library(pipeline elpis_regex_ingress STATIC
  ${ING}/src/streaming_regex_ingress.cpp
  ${ING}/src/incremental_lexer.cpp)
target_include_directories(elpis_regex_ingress PUBLIC ${ING}/include PRIVATE ${ING}/src)
target_link_libraries(elpis_regex_ingress PUBLIC elpis_hash PRIVATE PkgConfig::PCRE2)

elpis_add_library(pipeline elpis_regex_execution STATIC ${ING}/src/regex_execution.c)
target_include_directories(elpis_regex_execution PUBLIC ${ING}/include)
target_link_libraries(elpis_regex_execution PUBLIC elpis_execution elpis_regex_ingress)
elpis_add_test(pipeline test_regex_execution SOURCES ${ING}/tests/test_regex_execution.cpp
  LIBS elpis_regex_execution TIMEOUT 600)
elpis_add_test(pipeline benchmark_regex_execution SOURCES ${ING}/tests/benchmark_regex_execution.cpp
  LIBS elpis_regex_execution ARGS 1 small 16 TIMEOUT 60)
# Per-fixture performance census (not a ctest; run by hand or by the perf report).
add_executable(census_regex_execution ${ING}/tests/census_regex_execution.cpp)
target_link_libraries(census_regex_execution PRIVATE elpis_regex_execution)
_elpis_register(pipeline census_regex_execution test)

elpis_add_library(pipeline elpis_query_ingress STATIC
  ${ING}/src/regex_hacf_query_ingress.cpp
  ${ING}/src/query_local_proposal.c
  ${ING}/src/query_local_proposal_batch.c)
target_include_directories(elpis_query_ingress PUBLIC ${ING}/include PRIVATE ${ING}/src)
target_link_libraries(elpis_query_ingress
  PUBLIC elpis_regex_ingress elpis_corpus elpis_hybrid elpis_semantic
  PRIVATE elpis_hash Threads::Threads)

foreach(lib elpis_regex_ingress elpis_query_ingress)
  set_target_properties(${lib} PROPERTIES
    C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
endforeach()

# The one library Python loads. Its version script exports only the bridge,
# the streaming Regex ABI and the query-ingress result ABI.
elpis_add_library(pipeline elpis_ingress_bridge SHARED ${ING}/bridge/ingress_bridge.c)
target_include_directories(elpis_ingress_bridge PRIVATE ${ING}/bridge)
target_link_libraries(elpis_ingress_bridge PRIVATE elpis_query_ingress elpis_regex_ingress)
target_link_options(elpis_ingress_bridge PRIVATE
  "LINKER:--version-script=${ING}/bridge/ingress_bridge.map" "LINKER:--no-undefined")
set_target_properties(elpis_ingress_bridge PROPERTIES LINK_DEPENDS ${ING}/bridge/ingress_bridge.map)

set(STATE ${CMAKE_CURRENT_BINARY_DIR})
# Frozen public-main V2 lexer, test-only: the differential oracle for the R1 engine.
elpis_add_test_library(pipeline elpis_regex_lexer_reference ${ING}/tests/reference/incremental_lexer_r0.cpp)
target_include_directories(elpis_regex_lexer_reference PUBLIC ${ING}/tests/reference)
target_link_libraries(elpis_regex_lexer_reference PUBLIC elpis_hash PRIVATE PkgConfig::PCRE2)
elpis_add_test(pipeline test_regex_lexer_equivalence SOURCES ${ING}/tests/test_regex_lexer_equivalence.cpp
  LIBS elpis_regex_ingress elpis_regex_lexer_reference INCLUDES ${ING}/src TIMEOUT 1200)
elpis_add_test(pipeline test_regex_bounded_profile SOURCES ${ING}/tests/test_regex_bounded_profile.cpp
  LIBS elpis_regex_ingress)
elpis_add_test(pipeline test_regex_stream SOURCES ${ING}/tests/test_regex_stream.cpp
  LIBS elpis_regex_ingress elpis_hash ${CMAKE_DL_LIBS})
elpis_add_test(pipeline test_regex_allocation SOURCES ${ING}/tests/test_regex_allocation.cpp
  LIBS elpis_regex_ingress)
elpis_add_test(pipeline test_regex_c_abi SOURCES ${ING}/tests/test_regex_c_abi.c
  LIBS elpis_regex_ingress)
elpis_add_test(pipeline test_regex_memory SOURCES ${ING}/tests/test_regex_memory.cpp
  LIBS elpis_regex_ingress)
elpis_add_test(pipeline test_query_local_batch SOURCES ${ING}/tests/test_atomic_batch.c
  LIBS elpis_query_ingress elpis_hash INCLUDES ${ING}/tests)
elpis_add_test(pipeline test_query_ingress_bounded SOURCES ${ING}/tests/test_query_ingress_bounded.cpp
  LIBS elpis_query_ingress ARGS ${STATE}/query-bounded-corpus)
elpis_add_test(pipeline test_query_ingress_composition SOURCES ${ING}/tests/test_query_ingress_composition.cpp
  LIBS elpis_query_ingress ARGS ${STATE}/query-composition-corpus)
