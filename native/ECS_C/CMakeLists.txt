# ECS_C native runtime-history foundation.
#
# This phase only nativeizes the exact bounded receipt protocol primitives.
# The existing Python ECS_C remains the replay/reference authority until the
# later runtime-history transition stages are qualified.

elpis_add_library(ECS_C elpis_ecsc_history SHARED
  src/ecsc_history.c
  src/ecsc_log.c
  src/ecsc_session.c
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

# Bounded active-segment session ABI (global/base/local accounting and the
# capacity disposition before mutation).
elpis_add_test(ECS_C test_ecsc_session_segment
  SOURCES tests/test_ecsc_session_segment.c
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

  # Session-level fault matrix under a segment base: the same wrap technique
  # over the session + log + planner sources (no hooks in the shared library).
  elpis_add_test(ECS_C test_ecsc_session_faults
    SOURCES tests/test_ecsc_session_segment.c src/ecsc_session.c src/ecsc_log.c src/ecsc_history.c
    INCLUDES include ../substrate/include
    LIBS elpis_hash
    DEFINES ECSC_SESSION_FAULTS
    TIMEOUT 60)
  target_link_options(test_ecsc_session_faults PRIVATE
    "LINKER:--wrap=write,--wrap=fsync,--wrap=ftruncate")
endif()
