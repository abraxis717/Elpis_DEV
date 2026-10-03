# ECS Runtime R1 direct-native benchmark driver (PERFORMANCE_ONLY; research,
# never part of the cognitive path). Calls the ECS_G ABI from C with no Python.
add_executable(ecsg_runtime_bench ecsg_runtime_bench.c)
target_link_libraries(ecsg_runtime_bench PRIVATE elpis_ecsg_math m)
_elpis_register(research ecsg_runtime_bench benchmark)
