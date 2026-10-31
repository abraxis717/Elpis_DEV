# RuntimeCore: the runtime's mutable systems authority, implemented in Rust behind a stable C ABI
# (include/elpis/runtime.h, ABI v8). Volatile only (docs/ELPIS_MISSION.md, Amendment 3): it embeds the volatile ECS_C
# session crate (without its C ABI) as its authority and has no persistent-state write path. It reaches native K1 only
# through caller-supplied K1 function tables. One shared library, built offline from Cargo.lock (no dependencies
# beyond the Rust standard library and the ECS_C session crate in this repository):
#   libelpis_runtime.so          production
# ABI v7's testing library (durability fault injection) retired with the durability lane.

# The shared library carries a stable SONAME (passed to the final link only, so a crate that embeds this one never
# inherits it), so C callers can link it by name.
find_program(ELPIS_CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)

set(_crate "${CMAKE_CURRENT_SOURCE_DIR}")
set(_ecs_c_session "${CMAKE_SOURCE_DIR}/native/ECS_C/session")
file(GLOB _rust_sources CONFIGURE_DEPENDS "${_crate}/src/*.rs" "${_ecs_c_session}/src/*.rs")
set(_crate_inputs ${_rust_sources} "${_crate}/Cargo.toml" "${_crate}/Cargo.lock" "${_ecs_c_session}/Cargo.toml")
set(_prod "${CMAKE_CURRENT_BINARY_DIR}/libelpis_runtime.so")

add_custom_command(OUTPUT "${_prod}"
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo"
          "${ELPIS_CARGO}" rustc --offline --locked --release --lib --manifest-path "${_crate}/Cargo.toml"
          -- -C "link-arg=-Wl,-soname,libelpis_runtime.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cargo/release/libelpis_runtime.so" "${_prod}"
  DEPENDS ${_crate_inputs}
  COMMENT "cargo: libelpis_runtime.so"
  VERBATIM)

elpis_add_cargo_target(runtime elpis_runtime library "${_prod}")

# The ECS composite retained state: RuntimeCore's composition of ECS_G (K1) and ECS_C (memory) as one native object
# (docs/ECS_COMPOSITE_STATE.md). RuntimeCore's crate still links neither; hosts pass the composite's own table.
add_subdirectory(composite)

# The C ABI v8 over a real native K1 state, through the production library: no path, file or directory anywhere.
set(_runtime_test_includes "${_crate}/include" "${_ecs_c_session}/include")
elpis_add_test(runtime test_runtime_abi
  SOURCES tests/test_runtime_abi.c
  INCLUDES ${_runtime_test_includes}
  LIBS "${_prod}" elpis_ecsg_k1 ${CMAKE_DL_LIBS}
  TIMEOUT 60)
target_compile_definitions(test_runtime_abi PRIVATE _GNU_SOURCE=1)
add_dependencies(test_runtime_abi elpis_runtime)

# The turn lifecycle law over real native K1, standalone and FMS-resident (a real, RAM-only FMS context): an open native
# transaction is ended by exactly one commit or abort on every path (abort, commit, close, destroy, reopen,
# fail-stop + close), and the resident WRITE pin and residency hold are released.
elpis_add_test(runtime test_runtime_lifecycle
  SOURCES tests/test_runtime_lifecycle.c
  INCLUDES ${_runtime_test_includes}
  LIBS "${_prod}" elpis_ecsg_k1_fms elpis_ecsg_k1
  TIMEOUT 120)
add_dependencies(test_runtime_lifecycle elpis_runtime)

# RuntimeCore over the ECS composite retained state (docs/ECS_COMPOSITE_STATE.md): composite anchor (the composite
# format), managed turns that learn K1 and observe the memory, an in-memory rollback, the typed HACF path and the
# observe-only turn, all in volatile memory.
elpis_add_test(runtime test_runtime_composite
  SOURCES tests/test_runtime_composite.c
  INCLUDES ${_runtime_test_includes}
  LIBS "${_prod}" elpis_hacf_observation elpis_semantic elpis_ecs_composite elpis_ecs_memory elpis_ecsg_k1
  TIMEOUT 120)
add_dependencies(test_runtime_composite elpis_runtime)

# RuntimeCore over its session, the real composite and the ECS_C relational view (synced from the runtime's own
# session rows), composed in memory: the operator's sources, typed-evidence admission, the retired persistence-only
# operator events, capacity, the relational view's admission and LOOKUP, and the session's end.
elpis_add_test(runtime test_runtime_session
  SOURCES tests/test_runtime_session.c
  INCLUDES ${_runtime_test_includes} "${CMAKE_SOURCE_DIR}/native/ECS_C/relational/include"
  LIBS "${_prod}" elpis_ecs_c_relational elpis_ecs_composite elpis_ecs_memory elpis_ecsg_k1
  TIMEOUT 120)
add_dependencies(test_runtime_session elpis_runtime)

# Zero persistent I/O at the C library's boundary: the executable interposes the C library's persistent-I/O and
# read entry points and counts them per thread over a complete session lifecycle (and FMS-resident turns over a
# RAM-only FMS context): none at all.
elpis_add_test(runtime test_runtime_hot_io
  SOURCES tests/test_runtime_hot_io.c
  INCLUDES ${_runtime_test_includes}
  LIBS "${_prod}" elpis_ecsg_k1_fms elpis_ecsg_k1 ${CMAKE_DL_LIBS}
  TIMEOUT 120)
target_compile_definitions(test_runtime_hot_io PRIVATE _GNU_SOURCE=1)
target_link_options(test_runtime_hot_io PRIVATE "-rdynamic")
add_dependencies(test_runtime_hot_io elpis_runtime)

# The native operator host (docs/NATIVE_HOST.md): the production entry point, one executable per RuntimeCore build.
# RAM-only (docs/ELPIS_MISSION.md, Amendment 3): it has no home, no deployment state and no write path. It links no
# Elpis library: each one is admitted at run time, read-only, from an immutable library directory (sealed in-memory
# copy, SHA-256 against the catalog pin the operator passes, dlopen of the seal), so the headers here give types only.
# Its own closure is the C library and the static SHA-256.
set(_host_sources host/src/elpis_host.c host/src/host_install.c host/src/host_admit.c host/src/host_json.c
    host/src/host_contain.c)
set(_host_includes "${_crate}/include" "${_ecs_c_session}/include"
    "${CMAKE_SOURCE_DIR}/native/runtime/composite/include" "${CMAKE_SOURCE_DIR}/native/runtime/observation/include"
    "${CMAKE_SOURCE_DIR}/native/ECS_C/memory/include" "${CMAKE_SOURCE_DIR}/native/ECS_G/include"
    "${CMAKE_SOURCE_DIR}/native/structure/semantic/include" "${CMAKE_SOURCE_DIR}/native/structure/hacf/include")
# The testing host adds only the negative controls (a deliberate write attempt, a pause for inspection, process death
# at an operation boundary) to the same RAM-only host; it is test support and never installed.
add_executable(elpis_host ${_host_sources})
_elpis_register(runtime elpis_host executable)
add_executable(elpis_host_testing ${_host_sources})
_elpis_register(runtime elpis_host_testing test)
foreach(_variant IN ITEMS "" "_testing")
  set_target_properties(elpis_host${_variant} PROPERTIES OUTPUT_NAME "elpis${_variant}")
  target_include_directories(elpis_host${_variant} PRIVATE ${_host_includes})
  target_compile_definitions(elpis_host${_variant} PRIVATE _GNU_SOURCE=1)
  target_link_libraries(elpis_host${_variant} PRIVATE elpis_hash ${CMAKE_DL_LIBS})
endforeach()
target_compile_definitions(elpis_host_testing PRIVATE ELPIS_HOST_TESTING=1)
# The host end to end in every native build (sanitizer and clang jobs included); no interpreter is involved.
add_test(NAME runtime.native_host_smoke
  COMMAND ${CMAKE_COMMAND} "-DHOST=$<TARGET_FILE:elpis_host>" "-DTESTING_HOST=$<TARGET_FILE:elpis_host_testing>"
          "-DBUILD=${CMAKE_BINARY_DIR}" "-DSCRATCH=${CMAKE_CURRENT_BINARY_DIR}/test-state/native_host_smoke"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/host/tests/host_smoke.cmake")
set_tests_properties(runtime.native_host_smoke PROPERTIES LABELS runtime TIMEOUT 300)

# Typed HACF -> ECS memory observation path (docs/ECS_MEMORY.md, "HACF observations"): joins structure (identity and
# validation owner) and the ECS memory type at the composition layer; RuntimeCore itself still links neither.
elpis_add_library(runtime elpis_hacf_observation SHARED observation/src/ecs_observation_hacf.c)
target_include_directories(elpis_hacf_observation PUBLIC observation/include)
target_link_libraries(elpis_hacf_observation PUBLIC elpis_ecs_memory elpis_semantic)
# Name only the libraries it uses, whatever the toolchain's default (the native host refuses an admitted library that
# depends on one it did not admit: docs/NATIVE_HOST.md, "Admission").
target_link_options(elpis_hacf_observation PRIVATE "LINKER:--as-needed")
elpis_add_test(runtime test_ecs_observation_hacf SOURCES observation/tests/test_ecs_observation_hacf.c
  INCLUDES "${CMAKE_SOURCE_DIR}/native/ECS_C/memory/tests"
  LIBS elpis_hacf_observation elpis_ecs_memory elpis_semantic elpis_hash TIMEOUT 120)

# The Rust law suite (lifecycle, lineage, turn, evolution, admission and the session through RuntimeCore), over the
# volatile ECS_C session.
add_test(NAME runtime.cargo_test
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-test"
          "${ELPIS_CARGO}" test --offline --locked --release --manifest-path "${_crate}/Cargo.toml")
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

# The installed product (docs/NATIVE_HOST.md, "Installation"): the host in bin/ and the six libraries it admits in
# lib/elpis/, flat and read-only (`elpis --libraries <prefix>/lib/elpis catalog` prints the catalog the operator pins).
# Nothing of the runtime is ever written beneath the prefix.
install(TARGETS elpis_host RUNTIME DESTINATION bin COMPONENT runtime)
install(TARGETS elpis_ecsg_math elpis_ecsg_k1 elpis_ecs_memory elpis_ecs_composite elpis_hacf_observation
        LIBRARY DESTINATION lib/elpis COMPONENT runtime)
install(FILES "${_prod}" DESTINATION lib/elpis COMPONENT runtime
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
