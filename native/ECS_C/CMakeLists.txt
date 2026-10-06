# ECS_C native runtime-history foundation.
#
# This phase only nativeizes the exact bounded receipt protocol primitives.
# The existing Python ECS_C remains the replay/reference authority until the
# later runtime-history transition stages are qualified.

elpis_add_library(ECS_C elpis_ecsc_history SHARED
  src/ecsc_history.c
  src/ecsc_log.c
)
target_include_directories(elpis_ecsc_history
  PUBLIC include
  PRIVATE ../substrate/include
)
target_link_libraries(elpis_ecsc_history PRIVATE elpis_hash)

elpis_add_test(ECS_C test_ecsc_history
  SOURCES tests/test_ecsc_history.c
  LIBS elpis_ecsc_history
)


if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # Test-only syscall fault injection. Compile the durability source directly
  # into the test so GNU ld --wrap intercepts its write/fsync/ftruncate calls.
  # No fault hooks enter the production shared library.
  elpis_add_test(ECS_C test_ecsc_log_faults
    SOURCES tests/test_ecsc_log_faults.c src/ecsc_log.c
    INCLUDES include
    TIMEOUT 60)
  target_link_options(test_ecsc_log_faults PRIVATE
    "LINKER:--wrap=write,--wrap=fsync,--wrap=ftruncate")
endif()
