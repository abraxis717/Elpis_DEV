# DSV4.1 tower native backend (RESEARCH_ONLY; relocated out of the canonical tree,
# see docs/ELPIS_MISSION.md). Not part of the Elpis cognitive path.
#
# R0 freezes the stateless arithmetic ABI only. No sequence state, FMS,
# tokenizer, ECS/HACF access, private worker pool or parameter serialization
# crosses this boundary.

elpis_add_library(research elpis_dsv41_native SHARED
  src/dsv41_native.c
  src/dsv41_attention_state.c
  src/dsv41_local_attention.c
  src/dsv41_compressed_attention.c
  src/dsv41_moe_dispatch.c
  src/dsv41_layer_frame.c)
target_include_directories(elpis_dsv41_native PUBLIC include)
target_link_libraries(elpis_dsv41_native PRIVATE m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Keep the R0 scalar reference accumulation order explicit. Vectorized/native
  # specializations come only after native==NumPy parity is frozen.
  target_compile_options(elpis_dsv41_native PRIVATE -ffp-contract=off)
endif()

elpis_add_test(research test_dsv41_native
  SOURCES tests/test_dsv41_native.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(research test_dsv41_attention_state
  SOURCES tests/test_dsv41_attention_state.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(research test_dsv41_local_attention
  SOURCES tests/test_dsv41_local_attention.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(research test_dsv41_compressed_attention
  SOURCES tests/test_dsv41_compressed_attention.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(research test_dsv41_moe_dispatch
  SOURCES tests/test_dsv41_moe_dispatch.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(research test_dsv41_layer_frame
  SOURCES tests/test_dsv41_layer_frame.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)

# YTS-R0 (docs/research/dsv41_tower/DSV41_PROVIDER_STREAM.md): TEST-ONLY reference provider
# for the DSV4.1 Yielded Token Stream over the generic execution port. It
# compiles the same qualified kernel sources with the same floating-point policy
# and is registered as a test target, never as a production library. It does
# not link the execution runtime: the host passes the runtime's buffer/notify
# entry points at attach (dsv41_stream.h).
add_library(elpis_dsv41_stream_reference_provider SHARED
  tests/dsv41_reference_provider.c
  src/dsv41_native.c
  src/dsv41_attention_state.c
  src/dsv41_local_attention.c
  src/dsv41_compressed_attention.c
  src/dsv41_moe_dispatch.c
  src/dsv41_layer_frame.c)
target_include_directories(elpis_dsv41_stream_reference_provider PRIVATE
  include $<TARGET_PROPERTY:elpis_execution,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(elpis_dsv41_stream_reference_provider PRIVATE m Threads::Threads)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(elpis_dsv41_stream_reference_provider PRIVATE -ffp-contract=off)
endif()
_elpis_register(research elpis_dsv41_stream_reference_provider test)

elpis_add_test(research test_dsv41_stream_provider
  SOURCES tests/test_dsv41_stream_provider.c
  LIBS elpis_dsv41_stream_reference_provider elpis_execution m Threads::Threads
  INCLUDES include
  TIMEOUT 300)

# Entry points are borrowed from the sealed runtime. Never link elpis_execution
# into this DSO (including indirectly through a private static archive).
elpis_add_library(research elpis_dsv41_clock SHARED src/dsv41_clock.c)
target_include_directories(elpis_dsv41_clock PUBLIC include
  $<TARGET_PROPERTY:elpis_execution,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(elpis_dsv41_clock PRIVATE Threads::Threads)

add_library(elpis_dsv41_clock_test_materializer SHARED tests/dsv41_clock_materializer.c)
target_include_directories(elpis_dsv41_clock_test_materializer PRIVATE include
  $<TARGET_PROPERTY:elpis_execution,INTERFACE_INCLUDE_DIRECTORIES>)
_elpis_register(research elpis_dsv41_clock_test_materializer test)

elpis_add_test(research test_dsv41_clock
  SOURCES tests/test_dsv41_clock.c
  LIBS elpis_dsv41_clock elpis_dsv41_clock_test_materializer
       elpis_dsv41_stream_reference_provider elpis_execution m Threads::Threads
  TIMEOUT 300)

# Native Materializer R1 (docs/research/dsv41_tower/DSV41_NATIVE_CLOCK.md): the production
# Elpis host service behind elpis_dsv41_materializer_v1. It embeds the generic
# substrate file-asset service and FMS core; it never links elpis_execution and
# exposes no filesystem authority to the accelerator provider.
elpis_add_library(research elpis_dsv41_materializer SHARED src/dsv41_materializer.c)
target_include_directories(elpis_dsv41_materializer PUBLIC include
  $<TARGET_PROPERTY:elpis_execution,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(elpis_dsv41_materializer PRIVATE elpis_fms_file_service m Threads::Threads)
target_link_options(elpis_dsv41_materializer PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41_materializer.map" "LINKER:--no-undefined")
set_target_properties(elpis_dsv41_materializer PROPERTIES
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41_materializer.map)

elpis_add_test_library(research elpis_dsv41_materializer_test_support src/dsv41_materializer.c)
target_include_directories(elpis_dsv41_materializer_test_support PUBLIC include
  $<TARGET_PROPERTY:elpis_execution,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(elpis_dsv41_materializer_test_support PUBLIC elpis_fms_file_service_test_support m Threads::Threads)

elpis_add_test(research test_dsv41_materializer
  SOURCES tests/test_dsv41_materializer.c
  LIBS elpis_dsv41_materializer_test_support m Threads::Threads
  TIMEOUT 300)

# Native Clock + production materializer end to end over the real runtime and
# the test-only reference provider (bitwise against the R0 test materializer).
elpis_add_test(research test_dsv41_clock_production
  SOURCES tests/test_dsv41_clock_production.c
  LIBS elpis_dsv41_clock elpis_dsv41_clock_test_materializer elpis_dsv41_materializer_test_support
       elpis_dsv41_stream_reference_provider elpis_execution m Threads::Threads
  TIMEOUT 300)
