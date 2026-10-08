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

# The Rust law suite (lifecycle, lineage, turn, evolution and the continuity crash law through RuntimeCore).
add_test(NAME runtime.cargo_test
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "${ELPIS_CARGO}" test --offline --locked --release --features testing --manifest-path "${_crate}/Cargo.toml")
set_tests_properties(runtime.cargo_test PROPERTIES LABELS runtime TIMEOUT 600)
