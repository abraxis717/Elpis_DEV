# H-ECS R0 (RESEARCH_ONLY): the FAST MECHANICS lane of the hecs_r0 crate. It runs `cargo test` against the
# built K1 library. The science commands (calibrate, run) are never part of the build or of ctest.
find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)
add_test(NAME research.hecs_r0_mechanics
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS
          "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "ELPIS_ECS_K1_LIBRARY=$<TARGET_FILE:elpis_ecsg_k1>" ELPIS_REQUIRE_NATIVE=1
          "${ELPIS_CARGO}" test --offline --locked --release --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml")
set_tests_properties(research.hecs_r0_mechanics PROPERTIES LABELS "research;hecs" TIMEOUT 600)
