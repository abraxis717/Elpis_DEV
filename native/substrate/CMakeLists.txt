# Substrate: content hashing, FMS residency (FMS ABI v3: HOT and WARM tiers over RAM and device domains, volatile
# only; no COLD residency and no storage domain) with the RAM-only POSIX platform abstraction layer, and the
# read-only file-asset bridge loaded by elpis.substrate.file_assets (immutable COLD backing loaded into WARM).

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
# copy of the FMS core so that its exported ABI is exactly FMS ABI v3 plus the
# elpis_fms_file_* entry points.
elpis_add_library(substrate elpis_fms_file_assets SHARED
  src/fms_file_assets.c src/fms_core.c src/fms_pal_posix.c src/sha256.c)
target_include_directories(elpis_fms_file_assets PRIVATE include)
target_link_libraries(elpis_fms_file_assets PRIVATE Threads::Threads)

# Native verified file-asset page service (fms_file_service.h): the runtime half
# of elpis.substrate.file_assets. Substrate-generic; embedded by native hosts
# (e.g. the DSV4.1 materializer) that receive already-authorized descriptors.
elpis_add_library(substrate elpis_fms_file_service STATIC src/fms_file_service.c)
target_include_directories(elpis_fms_file_service PUBLIC include)
target_link_libraries(elpis_fms_file_service PUBLIC elpis_fms elpis_hash Threads::Threads)

# Same source with test-only fault injection (pread/allocation hooks).
elpis_add_test_library(substrate elpis_fms_file_service_test_support src/fms_file_service.c)
target_compile_definitions(elpis_fms_file_service_test_support PUBLIC ELPIS_FILE_SERVICE_TESTING=1)
target_include_directories(elpis_fms_file_service_test_support PUBLIC include)
target_link_libraries(elpis_fms_file_service_test_support PUBLIC elpis_fms elpis_hash Threads::Threads)

elpis_add_test_library(substrate elpis_fms_test_pal tests/support/fms_pal_test.c)
target_include_directories(elpis_fms_test_pal PUBLIC include tests/support/include)
target_link_libraries(elpis_fms_test_pal PUBLIC elpis_fms elpis_hash Threads::Threads)

elpis_add_test(substrate test_hash SOURCES tests/test_hash.c LIBS elpis_hash)
# FMS ABI v3 is volatile: no test needs a directory (ABI v2's cold-store reclamation test retired with the store).
elpis_add_test(substrate test_fms SOURCES tests/test_fms.c LIBS elpis_fms_test_pal)
elpis_add_test(substrate test_fms_concurrency SOURCES tests/test_fms_concurrency.c LIBS elpis_fms_test_pal)
elpis_add_test(substrate test_fms_adversarial SOURCES tests/test_fms_adversarial.c LIBS elpis_fms_test_pal)
elpis_add_test(substrate test_fms_file_service SOURCES tests/test_fms_file_service.c
  LIBS elpis_fms_file_service_test_support Threads::Threads TIMEOUT 120)

# The volatile-residency law, cases 1-10 (tests/test_fms_volatile_cold.c): frozen COLD -> WARM through the read-only
# file service, WARM -> HOT on the test PAL's emulated device, volatile HOT mutation, dematerialization without a
# write, reload with the identical SHA-256, repetition, concurrent references, budget exhaustion without spill and
# the refusal of every legacy COLD writer, under a link-time write trap over every storage-mutating libc entry point.
elpis_add_test(substrate test_fms_volatile_cold SOURCES tests/test_fms_volatile_cold.c
  LIBS elpis_fms_file_service elpis_fms_test_pal ${CMAKE_DL_LIBS} Threads::Threads
  DEFINES ELPIS_FMS_SHARED_LIBRARY="$<TARGET_FILE:elpis_fms_file_assets>" TIMEOUT 120)
add_dependencies(test_fms_volatile_cold elpis_fms_file_assets)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  foreach(fn write pwrite pwrite64 writev pwritev ftruncate truncate fallocate posix_fallocate fsync fdatasync
             rename renameat unlink unlinkat mkdir mkdirat rmdir link symlink open open64 openat creat fopen)
    target_link_options(test_fms_volatile_cold PRIVATE "-Wl,--wrap=${fn}")
  endforeach()
  # Case 9: the same COLD -> WARM path over the working directory bound read-only in a private mount namespace
  # (a child process; nothing global is remounted). Reported SKIPPED (77), never passed, where no such namespace
  # can be made.
  add_test(NAME substrate.test_fms_volatile_cold_rofs COMMAND test_fms_volatile_cold --rofs)
  set(_rofs "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_fms_volatile_cold_rofs")
  file(MAKE_DIRECTORY "${_rofs}")
  set_tests_properties(substrate.test_fms_volatile_cold_rofs PROPERTIES
    LABELS substrate WORKING_DIRECTORY "${_rofs}" SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
