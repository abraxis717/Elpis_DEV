# RuntimeCore: the runtime's mutable systems authority, implemented in Rust behind a stable C ABI
# (include/elpis/runtime.h). It embeds the continuity crate (one authority, one implementation) and reaches
# native K1 only through caller-supplied K1 function tables. Two shared libraries from one crate:
#   libelpis_runtime.so          production (no test hooks)
#   libelpis_runtime_testing.so  the same authority plus continuity fault injection and I/O counters
# Both are built offline from Cargo.lock (no dependencies beyond the Rust standard library and the
# continuity crate in this repository).

# The shared libraries carry a stable SONAME (passed to the final link only, so a crate that embeds this
# one never inherits it), so C callers can link them by name.
find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)

set(_crate "${CMAKE_CURRENT_SOURCE_DIR}")
set(_continuity "${CMAKE_SOURCE_DIR}/native/continuity")
file(GLOB _rust_sources CONFIGURE_DEPENDS "${_crate}/src/*.rs" "${_continuity}/src/*.rs")
set(_crate_inputs ${_rust_sources} "${_crate}/Cargo.toml" "${_crate}/Cargo.lock"
    "${_continuity}/Cargo.toml")
set(_prod "${CMAKE_CURRENT_BINARY_DIR}/libelpis_runtime.so")
set(_testing "${CMAKE_CURRENT_BINARY_DIR}/libelpis_runtime_testing.so")

add_custom_command(OUTPUT "${_prod}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo"
          "${ELPIS_CARGO}" rustc --offline --locked --release --lib --manifest-path "${_crate}/Cargo.toml"
          -- -C "link-arg=-Wl,-soname,libelpis_runtime.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo/release/libelpis_runtime.so" "${_prod}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_runtime.so (production)"
  VERBATIM)
add_custom_command(OUTPUT "${_testing}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-testing"
          "${ELPIS_CARGO}" rustc --offline --locked --release --lib --features testing
          --manifest-path "${_crate}/Cargo.toml" -- -C "link-arg=-Wl,-soname,libelpis_runtime.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo-testing/release/libelpis_runtime.so" "${_testing}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_runtime_testing.so (fault injection)"
  VERBATIM)

elpis_add_cargo_target(runtime elpis_runtime library "${_prod}")

# The ECS composite retained state: RuntimeCore's composition of ECS_G (K1) and ECS_C (memory) as one native object
# (docs/ECS_COMPOSITE_STATE.md). RuntimeCore's crate still links neither; hosts pass the composite's own table.
add_subdirectory(composite)
elpis_add_cargo_target(runtime elpis_runtime_testing test "${_testing}")

# The C ABI over a real native K1 state, through the production library.
elpis_add_test(runtime test_runtime_abi
  SOURCES tests/test_runtime_abi.c
  INCLUDES "${_crate}/include" "${_continuity}/include"
  LIBS "${_prod}" elpis_ecsg_k1
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_runtime_abi"
  TIMEOUT 60)
add_dependencies(test_runtime_abi elpis_runtime)

# The turn lifecycle law over real native K1, standalone and FMS-resident (a real FMS context): an open native
# transaction is ended by exactly one commit or abort on every path (abort, commit, close, destroy, reopen,
# fail-stop + close), the authority is unchanged by an implicit abort and the resident WRITE pin is released.
elpis_add_test(runtime test_runtime_lifecycle
  SOURCES tests/test_runtime_lifecycle.c
  INCLUDES "${_crate}/include" "${_continuity}/include"
  LIBS "${_prod}" elpis_ecsg_k1_fms elpis_ecsg_k1
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_runtime_lifecycle"
  TIMEOUT 120)
add_dependencies(test_runtime_lifecycle elpis_runtime)

# RuntimeCore over the ECS composite retained state (docs/ECS_COMPOSITE_STATE.md): composite anchor (continuity
# v3), managed turns that learn K1 and observe the memory, checkpoint + crash injection + recovery with the memory
# transition applied exactly once, and the explicit legacy import. It needs the testing library's crash hook; that
# library's SONAME is libelpis_runtime.so, so the test links a copy under that name in its own directory.
set(_testing_dir "${CMAKE_CURRENT_BINARY_DIR}/testing-lib")
# A POST_BUILD step of the one target that builds the testing library, so no second rule (and no second, concurrent
# cargo build or copy) exists for these files in a parallel build. The test links it by name, after that target.
add_custom_command(TARGET elpis_runtime_testing POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_testing_dir}"
  COMMAND ${CMAKE_COMMAND} -E copy "${_testing}" "${_testing_dir}/libelpis_runtime.so"
  VERBATIM)
elpis_add_test(runtime test_runtime_composite
  SOURCES tests/test_runtime_composite.c
  INCLUDES "${_crate}/include" "${_continuity}/include"
  LIBS elpis_hacf_observation elpis_semantic elpis_ecs_composite elpis_ecs_memory elpis_ecsg_k1
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_runtime_composite"
  TIMEOUT 120)
target_link_options(test_runtime_composite PRIVATE "-L${_testing_dir}")
target_link_libraries(test_runtime_composite PRIVATE -lelpis_runtime)
set_target_properties(test_runtime_composite PROPERTIES BUILD_RPATH "${_testing_dir}")
add_dependencies(test_runtime_composite elpis_runtime_testing)

# Gate A at the system-call boundary: the executable interposes the C library's persistent-I/O entry points and
# counts them per thread while managed turns run with the durability worker held (testing library: the hold).
elpis_add_test(runtime test_runtime_hot_io
  SOURCES tests/test_runtime_hot_io.c
  INCLUDES "${_crate}/include" "${_continuity}/include"
  LIBS elpis_ecsg_k1_fms elpis_ecsg_k1
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_runtime_hot_io"
  TIMEOUT 120)
target_compile_definitions(test_runtime_hot_io PRIVATE _GNU_SOURCE=1)
target_link_options(test_runtime_hot_io PRIVATE "-L${_testing_dir}" "-rdynamic")
target_link_libraries(test_runtime_hot_io PRIVATE -lelpis_runtime ${CMAKE_DL_LIBS})
set_target_properties(test_runtime_hot_io PROPERTIES BUILD_RPATH "${_testing_dir}")
add_dependencies(test_runtime_hot_io elpis_runtime_testing)

# Typed HACF -> ECS memory observation path (docs/ECS_MEMORY.md, "HACF observations"): joins structure (identity and
# validation owner) and the ECS memory type at the composition layer; RuntimeCore itself still links neither.
elpis_add_library(runtime elpis_hacf_observation SHARED observation/src/ecs_observation_hacf.c)
target_include_directories(elpis_hacf_observation PUBLIC observation/include)
target_link_libraries(elpis_hacf_observation PUBLIC elpis_ecs_memory elpis_semantic)
elpis_add_test(runtime test_ecs_observation_hacf SOURCES observation/tests/test_ecs_observation_hacf.c
  INCLUDES "${CMAKE_SOURCE_DIR}/native/ECS_C/memory/tests"
  LIBS elpis_hacf_observation elpis_ecs_memory elpis_semantic elpis_hash TIMEOUT 120)

# The Rust law suite (lifecycle, lineage, turn, evolution and the continuity crash law through RuntimeCore).
add_test(NAME runtime.cargo_test
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "${ELPIS_CARGO}" test --offline --locked --release --features testing --manifest-path "${_crate}/Cargo.toml")
set_tests_properties(runtime.cargo_test PROPERTIES LABELS runtime TIMEOUT 600)

# Explicit opt-in Linux autonomous launcher. Linux Landlock+seccomp is the
# process-wide enforcement boundary; without invocation it grants no protection.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(elpis_autonomous_run containment/elpis_autonomous_run.c)
  # The native target verifier registers every target. This is a production
  # executable, not a test or a manifest-declared native library; there is no
  # elpis_add_executable() wrapper in native/elpis.cmake. Use its registry.
  _elpis_register(runtime elpis_autonomous_run executable)
  target_compile_definitions(elpis_autonomous_run PRIVATE _GNU_SOURCE=1)

  # The shared elpis_add_test() helper handles subsystem registration, CTest
  # naming, test scratch, Release assertions, warnings and timeout.
  elpis_add_test(runtime test_autonomous_containment
    SOURCES tests/test_autonomous_containment.c
    ARGS $<TARGET_FILE:elpis_autonomous_run>
    TIMEOUT 30)
  target_compile_definitions(test_autonomous_containment PRIVATE _GNU_SOURCE=1)
endif()
