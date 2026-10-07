# Continuity: the durable runtime authority, implemented in Rust behind a stable C ABI
# (include/elpis/continuity.h). Two shared libraries are built from one crate:
#   libelpis_continuity.so          production (no test hooks)
#   libelpis_continuity_testing.so  the same authority plus fault injection and I/O counters
# Both are built offline from Cargo.lock (no dependencies beyond the Rust standard library).

find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)

set(_crate "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB _rust_sources CONFIGURE_DEPENDS "${_crate}/src/*.rs")
set(_crate_inputs ${_rust_sources} "${_crate}/Cargo.toml" "${_crate}/Cargo.lock" "${_crate}/build.rs")
set(_prod "${CMAKE_CURRENT_BINARY_DIR}/libelpis_continuity.so")
set(_testing "${CMAKE_CURRENT_BINARY_DIR}/libelpis_continuity_testing.so")

add_custom_command(OUTPUT "${_prod}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo"
          "${ELPIS_CARGO}" build --offline --locked --release --lib --manifest-path "${_crate}/Cargo.toml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo/release/libelpis_continuity.so" "${_prod}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_continuity.so (production)"
  VERBATIM)
add_custom_command(OUTPUT "${_testing}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-testing"
          "${ELPIS_CARGO}" build --offline --locked --release --lib --features testing
          --manifest-path "${_crate}/Cargo.toml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo-testing/release/libelpis_continuity.so" "${_testing}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_continuity_testing.so (fault injection)"
  VERBATIM)

elpis_add_cargo_target(continuity elpis_continuity library "${_prod}")
elpis_add_cargo_target(continuity elpis_continuity_testing test "${_testing}")

# The C ABI against the frozen v2 vectors, through the production library.
elpis_add_test(continuity test_continuity_abi
  SOURCES tests/test_continuity_abi.c
  INCLUDES "${_crate}/include"
  LIBS "${_prod}"
  ARGS "${_crate}/tests/fixtures" "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_continuity_abi/scratch"
  TIMEOUT 60)
add_dependencies(test_continuity_abi elpis_continuity)

# The Rust qualification suite (frozen parity, crash matrix, evolution law). Bounded and deterministic.
add_test(NAME continuity.cargo_test
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "${ELPIS_CARGO}" test --offline --locked --release --manifest-path "${_crate}/Cargo.toml")
set_tests_properties(continuity.cargo_test PROPERTIES LABELS continuity TIMEOUT 600)
