# Substrate: content hashing, FMS residency (HOT/WARM/COLD tiers over RAM,
# device and storage domains) with the POSIX platform abstraction layer, and
# the file-asset bridge loaded by elpis.substrate.file_assets.

elpis_add_library(substrate elpis_hash STATIC src/sha256.c)
target_include_directories(elpis_hash PUBLIC include)

# One optional execution budget shared by native adapters; no global singleton.
elpis_add_library(substrate elpis_execution SHARED src/execution.c)
target_include_directories(elpis_execution PUBLIC include)
target_link_libraries(elpis_execution PUBLIC Threads::Threads)

elpis_add_test_library(substrate elpis_execution_test_support src/execution.c)
target_compile_definitions(elpis_execution_test_support PRIVATE ELPIS_EXEC_TESTING=1)
target_include_directories(elpis_execution_test_support PUBLIC include)
target_link_libraries(elpis_execution_test_support PUBLIC Threads::Threads)
elpis_add_test(substrate test_execution SOURCES tests/test_execution.c
  LIBS elpis_execution_test_support TIMEOUT 60)

elpis_add_library(substrate elpis_fms STATIC src/fms_core.c src/fms_pal_posix.c)
target_include_directories(elpis_fms PUBLIC include)
target_link_libraries(elpis_fms PUBLIC elpis_hash Threads::Threads)

# Loaded through a sealed memfd by the Python substrate; it carries its own
# copy of the FMS core so that its exported ABI is exactly FMS ABI v2 plus the
# three elpis_fms_file_* entry points.
elpis_add_library(substrate elpis_fms_file_assets SHARED
  src/fms_file_assets.c src/fms_core.c src/fms_pal_posix.c src/sha256.c)
target_include_directories(elpis_fms_file_assets PRIVATE include)
target_link_libraries(elpis_fms_file_assets PRIVATE Threads::Threads)

elpis_add_test_library(substrate elpis_fms_test_pal tests/support/fms_pal_test.c)
target_include_directories(elpis_fms_test_pal PUBLIC include tests/support/include)
target_link_libraries(elpis_fms_test_pal PUBLIC elpis_fms elpis_hash Threads::Threads)

elpis_add_test(substrate test_hash SOURCES tests/test_hash.c LIBS elpis_hash)
elpis_add_test(substrate test_fms SOURCES tests/test_fms.c
  LIBS elpis_fms_test_pal ARGS ${CMAKE_CURRENT_BINARY_DIR}/fms-state)
elpis_add_test(substrate test_fms_concurrency SOURCES tests/test_fms_concurrency.c
  LIBS elpis_fms_test_pal ARGS ${CMAKE_CURRENT_BINARY_DIR}/fms-concurrency-state)
elpis_add_test(substrate test_fms_adversarial SOURCES tests/test_fms_adversarial.c
  LIBS elpis_fms_test_pal ARGS ${CMAKE_CURRENT_BINARY_DIR}/fms-adversarial-state)
