# ECS_G native mathematical foundation.
#
# G0 materializes only the frozen cubic microscopic/S3 mathematics. It owns no
# production transition law and has no inference/runtime authority.

elpis_add_library(ECS_G elpis_ecsg_math SHARED
  src/ecsg_math.c
  src/ecsg_state.c)
target_include_directories(elpis_ecsg_math PUBLIC include)
target_link_libraries(elpis_ecsg_math PRIVATE m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve explicit scalar accumulation order while differential parity is
  # frozen. Optimized/vectorized kernels are a later, separately qualified ABI.
  target_compile_options(elpis_ecsg_math PRIVATE -ffp-contract=off)
endif()

elpis_add_test(ECS_G test_ecsg_math
  SOURCES tests/test_ecsg_math.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)

elpis_add_test(ECS_G test_ecsg_state
  SOURCES tests/test_ecsg_state.c
  LIBS elpis_ecsg_math m
  TIMEOUT 30)
