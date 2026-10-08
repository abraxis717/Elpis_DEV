# H-ECS R1 (RESEARCH_ONLY): the FAST MECHANICS lane of the hecs_r1 crate. It runs `cargo test` against the
# built K1 library. The science commands (calibrate, run) are never part of the build or of ctest.
find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)

# In a sanitizer build the K1 library depends on the sanitizer runtime, which must be the first library of the
# (uninstrumented) Rust test process that dlopens it. A generated runner preloads the runtime into the test
# executable only (cargo's target runner), so cargo and rustc themselves stay unsanitized. Under TSan it also
# appends to the caller's TSAN_OPTIONS: ignore_noninstrumented_modules=1 (TSan judges only instrumented code,
# i.e. K1; the Rust test harness synchronizes through uninstrumented std atomics TSan cannot see) and tsan.supp.
set(_hecs_runner)
if(ELPIS_ENABLE_ASAN OR ELPIS_ENABLE_TSAN)
  if(ELPIS_ENABLE_TSAN)
    set(_hecs_runtime libtsan.so)
  else()
    set(_hecs_runtime libasan.so)
  endif()
  execute_process(COMMAND ${CMAKE_C_COMPILER} -print-file-name=${_hecs_runtime}
                  OUTPUT_VARIABLE _hecs_runtime_path OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT IS_ABSOLUTE "${_hecs_runtime_path}" OR NOT EXISTS "${_hecs_runtime_path}")
    message(FATAL_ERROR "hecs_r1: cannot locate ${_hecs_runtime} for the sanitizer build")
  endif()
  set(_hecs_tsan_options)
  if(ELPIS_ENABLE_TSAN)
    set(_hecs_tsan_options
        "TSAN_OPTIONS=\"\${TSAN_OPTIONS:+\$TSAN_OPTIONS:}ignore_noninstrumented_modules=1:suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp\"\n")
  endif()
  set(_hecs_runner_script "${CMAKE_CURRENT_BINARY_DIR}/sanitizer-runner.sh")
  file(WRITE "${_hecs_runner_script}"
       "#!/bin/sh\n${_hecs_tsan_options}LD_PRELOAD=\"${_hecs_runtime_path}\"\nexport LD_PRELOAD TSAN_OPTIONS\nexec \"\$@\"\n")
  set(_hecs_runner --config "target.'cfg(all())'.runner=['sh','${_hecs_runner_script}']")
endif()

add_test(NAME research.hecs_r1_mechanics
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS
          "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "ELPIS_ECS_K1_LIBRARY=$<TARGET_FILE:elpis_ecsg_k1>" ELPIS_REQUIRE_NATIVE=1
          "${ELPIS_CARGO}" test --offline --locked --release ${_hecs_runner}
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml")
set_tests_properties(research.hecs_r1_mechanics PROPERTIES LABELS "research;hecs" TIMEOUT 600)
