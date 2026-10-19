# The native host end to end in every native build (ctest runtime.native_host_smoke; sanitizer and clang jobs
# included): init from this build tree, create, a session that founds a source, learns, records typed evidence and
# publishes; a new process resumes exactly the published state; the text interface is refused. No interpreter is
# involved. The full suite is tests/integration/test_native_host.py.
#   cmake -DHOST=<elpis> -DBUILD=<build tree> -DSCRATCH=<dir> -P host_smoke.cmake
file(REMOVE_RECURSE "${SCRATCH}")
file(MAKE_DIRECTORY "${SCRATCH}")
set(HOME "${SCRATCH}/home")

# Run the host; every line must be a JSON object; the exit status must be `expect`; stderr must be empty.
function(host expect out)
  execute_process(COMMAND "${HOST}" --home "${HOME}" ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE stdout
                  ERROR_VARIABLE stderr)
  if(NOT rc EQUAL expect OR NOT stderr STREQUAL "")
    message(FATAL_ERROR "elpis ${ARGN}: exit ${rc} (expected ${expect})\n${stdout}\n${stderr}")
  endif()
  string(STRIP "${stdout}" stdout)
  string(REPLACE "\n" ";" lines "${stdout}")
  set(${out} "${lines}" PARENT_SCOPE)
endfunction()

function(field line out)
  string(JSON value GET "${line}" ${ARGN})
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(expect_equal what a b)
  if(NOT "${a}" STREQUAL "${b}")
    message(FATAL_ERROR "${what}: '${a}' != '${b}'")
  endif()
endfunction()

host(0 init init --native-build "${BUILD}")
host(0 created create --seed 7)
field("${created}" initial identity)

string(REPEAT "1" 64 n1)
string(REPEAT "2" 64 n2)
string(REPEAT "a" 64 provenance)
set(x "[[0.1,0.2,0.3,0.4,0.5,0.6],[0.6,0.5,0.4,0.3,0.2,0.1]]")
set(edge "\"relation\": \"hyperedge:1\", \"participants\": [{\"node\": \"${n1}\", \"role\": \"incidence:1\"}, {\"node\": \"${n2}\", \"role\": \"incidence:2\"}]")
file(WRITE "${SCRATCH}/first.jsonl"
  "{\"op\": \"source_found\", \"label\": \"smoke source\", \"source_authority\": 1, \"scope_label\": \"smoke\"}\n"
  "{\"op\": \"learn\", \"x\": ${x}, \"y\": [0.1, -0.1], \"schedule\": [[2, 3]], \"rate\": 0.002, \"grant\": \"smoke\"}\n"
  "{\"op\": \"evidence\", \"grant\": \"smoke\", \"sources\": [1], \"items\": [{${edge}, \"provenance\": \"${provenance}\", \"authority\": 1, \"polarity\": \"ASSERT\", \"tick\": 1}]}\n"
  "{\"op\": \"query\", \"x\": ${x}}\n"
  "{\"op\": \"publish\"}\n")
host(0 first session --script "${SCRATCH}/first.jsonl")
list(LENGTH first n)
expect_equal("operations" "${n}" 5)
foreach(line IN LISTS first)
  field("${line}" ok ok)
  expect_equal("ok" "${ok}" "ON")
endforeach()
list(GET first 2 recorded)
list(GET first 3 answered)
list(GET first 4 published)
field("${recorded}" live identity)
field("${published}" acknowledged ticket acknowledged)
field("${published}" durable_identity durable identity)
expect_equal("acknowledged" "${acknowledged}" "ON")
expect_equal("published identity" "${durable_identity}" "${live}")
if(live STREQUAL initial)
  message(FATAL_ERROR "learning and evidence left the identity unchanged")
endif()
field("${answered}" values_sha256 values_sha256)

file(WRITE "${SCRATCH}/second.jsonl"
  "{\"op\": \"live\"}\n"
  "{\"op\": \"query\", \"x\": ${x}}\n"
  "{\"op\": \"lookup\", \"relation\": \"hyperedge:1\", \"target_role\": \"incidence:2\", \"cues\": [{\"role\": \"incidence:1\", \"node\": \"${n1}\"}]}\n")
host(0 second session --script "${SCRATCH}/second.jsonl")
list(GET second 0 resumed)
list(GET second 1 again)
list(GET second 2 found)
field("${resumed}" a live identity)
field("${resumed}" b durable identity)
expect_equal("resumed live identity" "${a}" "${durable_identity}")
expect_equal("resumed durable identity" "${b}" "${durable_identity}")
field("${again}" again_sha values_sha256)
expect_equal("QUERY after restart" "${again_sha}" "${values_sha256}")
field("${found}" answer answer)
expect_equal("LOOKUP after restart" "${answer}" "${n2}")

host(3 text text --input hello)
field("${text}" code code)
expect_equal("text refusal" "${code}" "ECS_CODEC_UNQUALIFIED")
file(REMOVE_RECURSE "${SCRATCH}")
file(MAKE_DIRECTORY "${SCRATCH}")
message(STATUS "native host smoke: PASS")
