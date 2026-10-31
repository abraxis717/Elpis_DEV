# ECS_C volatile session authority (docs/ECS_C_SESSION.md): ECSC_VOLATILE_SESSION_R0, the ECS_C transition law over
# bounded, RAM-only session state, in Rust (no_std) behind a stable C ABI (include/elpis/ecs_c_session.h). One shared
# library, built offline from Cargo.lock (the crate has no dependencies); it imports no file or I/O function (only the
# C library's malloc, free, abort, mem* and getrandom; tests/boundary/test_volatile_runtime.py checks the binary).
find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)

set(_crate "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB _rust_sources CONFIGURE_DEPENDS "${_crate}/src/*.rs")
set(_crate_inputs ${_rust_sources} "${_crate}/Cargo.toml" "${_crate}/Cargo.lock")
set(_lib "${CMAKE_CURRENT_BINARY_DIR}/libelpis_ecs_c_session.so")

# The crate is an rlib for embedding Rust crates; the shared library is its cdylib form with the C ABI feature.
add_custom_command(OUTPUT "${_lib}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo"
          "${ELPIS_CARGO}" rustc --offline --locked --release --lib --crate-type cdylib --features c-abi
          --manifest-path "${_crate}/Cargo.toml" -- -C "link-arg=-Wl,-soname,libelpis_ecs_c_session.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo/release/libelpis_ecs_c_session.so" "${_lib}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_ecs_c_session.so (volatile, no_std)"
  VERBATIM)
elpis_add_cargo_target(ECS_C elpis_ecs_c_session library "${_lib}")

# The C ABI end to end: lifecycle, atomic refusals, retired persistence operations, session identity, capacity,
# receipts, and a heap that does not grow. It touches no file.
elpis_add_test(ECS_C test_ecs_c_session_abi
  SOURCES tests/test_ecs_c_session_abi.c
  INCLUDES "${_crate}/include"
  LIBS "${_lib}"
  TIMEOUT 120)
add_dependencies(test_ecs_c_session_abi elpis_ecs_c_session)

# The Rust suite: the differential against the pure transition law, atomicity, allocation-free transitions, bounds.
add_test(NAME ECS_C.cargo_test_session
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS
          "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "${ELPIS_CARGO}" test --offline --locked --release --features c-abi --manifest-path "${_crate}/Cargo.toml")
set_tests_properties(ECS_C.cargo_test_session PROPERTIES LABELS ECS_C TIMEOUT 900)
