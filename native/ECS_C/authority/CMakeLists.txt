# ECS_C authority (docs/ECS_C_AUTHORITY.md): the bounded, authenticated continuity and entity authority, in Rust
# behind a stable C ABI (include/elpis/ecs_c_authority.h). Two shared libraries are built from one crate:
#   libelpis_ecs_c_authority.so          production (no test hooks)
#   libelpis_ecs_c_authority_testing.so  the same authority plus fault injection and I/O counters
# Built offline from Cargo.lock; the only dependency is the continuity crate (path), which it embeds without
# re-exporting its C ABI (build.rs).
find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)

set(_crate "${CMAKE_CURRENT_SOURCE_DIR}")
set(_continuity "${CMAKE_SOURCE_DIR}/native/continuity")
file(GLOB _rust_sources CONFIGURE_DEPENDS "${_crate}/src/*.rs" "${_continuity}/src/*.rs")
set(_crate_inputs ${_rust_sources} "${_crate}/build.rs" "${_crate}/Cargo.toml" "${_crate}/Cargo.lock"
    "${_continuity}/Cargo.toml")
set(_prod "${CMAKE_CURRENT_BINARY_DIR}/libelpis_ecs_c_authority.so")
set(_testing "${CMAKE_CURRENT_BINARY_DIR}/libelpis_ecs_c_authority_testing.so")

add_custom_command(OUTPUT "${_prod}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo"
          "${ELPIS_CARGO}" rustc --offline --locked --release --lib --manifest-path "${_crate}/Cargo.toml"
          -- -C "link-arg=-Wl,-soname,libelpis_ecs_c_authority.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo/release/libelpis_ecs_c_authority.so" "${_prod}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_ecs_c_authority.so (production)"
  VERBATIM)
# The testing library has its own SONAME, so a program linked against it never resolves to the production one.
add_custom_command(OUTPUT "${_testing}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS
          "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-testing"
          "${ELPIS_CARGO}" rustc --offline --locked --release --lib --features testing
          --manifest-path "${_crate}/Cargo.toml" -- -C "link-arg=-Wl,-soname,libelpis_ecs_c_authority_testing.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo-testing/release/libelpis_ecs_c_authority.so"
          "${_testing}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_ecs_c_authority_testing.so (fault injection)"
  VERBATIM)

elpis_add_cargo_target(ECS_C elpis_ecs_c_authority library "${_prod}")
elpis_add_cargo_target(ECS_C elpis_ecs_c_authority_testing test "${_testing}")

# The C ABI end to end, through the production library, and once more through the testing library (one crash
# through the C fault hooks).
elpis_add_test(ECS_C test_ecs_c_authority_abi
  SOURCES tests/test_ecs_c_authority_abi.c
  INCLUDES "${_crate}/include"
  LIBS "${_prod}"
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_ecs_c_authority_abi/scratch"
  TIMEOUT 60)
add_dependencies(test_ecs_c_authority_abi elpis_ecs_c_authority)
elpis_add_test(ECS_C test_ecs_c_authority_abi_testing
  SOURCES tests/test_ecs_c_authority_abi.c
  INCLUDES "${_crate}/include"
  DEFINES ELPIS_ECS_C_TESTING
  LIBS "${_testing}"
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/test-state/test_ecs_c_authority_abi_testing/scratch"
  TIMEOUT 60)
add_dependencies(test_ecs_c_authority_abi_testing elpis_ecs_c_authority_testing)

# The Rust qualification suite: the transition law, the crash matrix, the T1 adversary matrix and boundedness.
add_test(NAME ECS_C.cargo_test_authority
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS
          "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "${ELPIS_CARGO}" test --offline --locked --release --manifest-path "${_crate}/Cargo.toml")
set_tests_properties(ECS_C.cargo_test_authority PROPERTIES LABELS ECS_C TIMEOUT 900)
