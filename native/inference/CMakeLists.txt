# DSV4.1 inference native backend.
#
# R0 freezes the stateless arithmetic ABI only. No sequence state, FMS,
# tokenizer, ECS/HACF access, private worker pool or parameter serialization
# crosses this boundary.

elpis_add_library(inference elpis_dsv41_native SHARED
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

elpis_add_test(inference test_dsv41_native
  SOURCES tests/test_dsv41_native.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(inference test_dsv41_attention_state
  SOURCES tests/test_dsv41_attention_state.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(inference test_dsv41_local_attention
  SOURCES tests/test_dsv41_local_attention.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(inference test_dsv41_compressed_attention
  SOURCES tests/test_dsv41_compressed_attention.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(inference test_dsv41_moe_dispatch
  SOURCES tests/test_dsv41_moe_dispatch.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)


elpis_add_test(inference test_dsv41_layer_frame
  SOURCES tests/test_dsv41_layer_frame.c
  LIBS elpis_dsv41_native m
  TIMEOUT 30)
