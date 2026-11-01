"""research/volatile_runtime_v1: the Python reference's FMS, RuntimeCore and HACF paths, composed, in one process.

    python3 -I -B research/volatile_runtime_v1/python_fms_hacf.py REPO BUILD

Run by the harness (zero_write_composed.py) as an unprivileged, traced process. It writes nothing, not even a
report: one JSON object on standard output (a pipe), exit 0 when every check holds. It exercises:

  * the process gate (elpis.runtime.environment) and two RuntimeCore ABI v8 sessions in a row: the second starts empty;
  * an FMS-resident K1 lineage on a RAM-only FMS context: managed numeric LEARN and QUERY through RuntimeCore,
    in-memory transaction refusals (fuel, capacity), a newcomer beyond the WARM budget refused CAPACITY;
  * a volatile HACF retrieval epoch (corpus, vector index, retrieval) and query ingress over it, whose state root is
    never created;
  * the retired persistent interfaces: absent operations, a retired FMS parameter, a refused legacy event.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from array import array
from pathlib import Path
import random
import sys

REPO, BUILD = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO / "src"))

from elpis.ECS.k1 import K1Error, K1FMSRuntime, K1Library  # noqa: E402
from elpis.pipeline.ingress import IngressLibrary, QueryIngress  # noqa: E402
from elpis.runtime import Runtime, RuntimeConfig, SessionConfig  # noqa: E402
from elpis.runtime.authority import SourceGrant  # noqa: E402
from elpis.runtime.cognition import LearnAuthority, QueryStimulus, Stimulus  # noqa: E402
from elpis.runtime.composition import CompositionError  # noqa: E402
from elpis.structure.retrieval.hacf import RetrievalLibrary, build_corpus_and_index  # noqa: E402
from elpis.substrate.authority import PinnedAuthority  # noqa: E402
from elpis.substrate.native_admission import admit_libraries  # noqa: E402
from elpis.substrate.residency import Context  # noqa: E402

DIM, WIDTH, ROWS = 6, 36, 8
PINNED = {"elpis_runtime": "runtime/libelpis_runtime.so", "elpis_ecsg_math": "ECS_G/libelpis_ecsg_math.so",
          "elpis_ecsg_k1": "ECS_G/libelpis_ecsg_k1.so", "elpis_ecsg_k1_fms": "ECS_G/libelpis_ecsg_k1_fms.so"}
DOCS = [("spec", "Intervals whose touching endpoints may merge keep the maximum end.", "elpis.docs", "canonical"),
        ("note", "Keep at least one merged range for every overlap.", "elpis.docs", "reference")]
checks: dict[str, bool] = {}


def check(name: str, ok) -> None:
    checks[name] = bool(ok)


def catalog(entries: dict[str, Path]) -> PinnedAuthority:
    """A catalog pinning these files (built in memory from their read-only bytes; the harness is its author)."""
    libraries = [{"library_id": lid, "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for lid, p in sorted(entries.items())]
    spec = json.dumps({"schema": "elpis.inference-authority.v1", "source": "qualification harness",
                       "provenance": "deployment", "assets": [], "libraries": libraries},
                      sort_keys=True, separators=(",", ":")).encode()
    return PinnedAuthority(spec, expected_sha256=hashlib.sha256(spec).hexdigest())


def rows(seed: int, n: int) -> tuple[array, array]:
    r = random.Random(seed)
    return (array("d", (r.uniform(-0.4, 0.4) for _ in range(n * DIM))), array("d", (r.uniform(-0.5, 0.5) for _ in range(n))))


native = BUILD / "native"
authority = catalog({lid: native / rel for lid, rel in PINNED.items()})
config = RuntimeConfig(native / PINNED["elpis_runtime"], None, authority, "elpis_runtime",
                       session=SessionConfig(capacity=8, sources=(SourceGrant("qualification source", 1, 8, "scope"),)))
_, k1_admitted, fms_admitted = admit_libraries(native, authority, (
    ("elpis_ecsg_math", PINNED["elpis_ecsg_math"]), ("elpis_ecsg_k1", PINNED["elpis_ecsg_k1"]),
    ("elpis_ecsg_k1_fms", PINNED["elpis_ecsg_k1_fms"])))
k1 = K1Library(k1_admitted.lib)
image = k1.envelope_bytes(DIM, WIDTH) - 32
w0 = array("d", (random.Random(1).uniform(-0.18, 0.18) for _ in range(DIM * WIDTH)))
grant = LearnAuthority(0.002, "qualification: numeric experience")

# ---- FMS-resident K1 under RuntimeCore, two sessions --------------------------------------------------------------
with Context(fms_admitted.lib, warm_bytes=2 * image + image // 2, max_objects=4) as ctx, \
        K1FMSRuntime(k1, ctx, fms_admitted.lib, max_states=4) as fms:
    state = fms.state(fms.register(b"\x01" * 32, DIM, WIDTH, w0, max_rows=ROWS))
    other = fms.state(fms.register(b"\x02" * 32, DIM, WIDTH, w0, max_rows=ROWS))
    try:                                                   # WARM holds two images: a third is refused, never spilled
        fms.register(b"\x03" * 32, DIM, WIDTH, w0, max_rows=ROWS)
        check("warm_budget_refuses_a_newcomer", False)
    except K1Error as exc:
        check("warm_budget_refuses_a_newcomer", exc.code == "CAPACITY")
    answers = []
    with Runtime(config) as runtime:
        check("process_gate", runtime.containment.core_limit == (0, 0) and runtime.containment.dumpable == 0)
        opened = runtime.continuity.snapshot().generation
        runtime.anchor_cognition(state)
        initial = runtime.live_identity(state)
        for t in range(12):
            x, y = rows(10 + t, 4)
            runtime.run_numeric_learn(state, Stimulus(x, y, array("Q", [2, 3, 2, 5]), dim=DIM), grant)
            answers.append(list(runtime.run_numeric_query(state, QueryStimulus(x, dim=DIM)).values))
        learned = runtime.live_identity(state)
        check("learning_moved_the_live_state", learned != initial)
        x, y = rows(99, 4)
        try:                                               # refused in memory before anything native: fuel
            runtime.run_numeric_learn(state, Stimulus(x, y, array("Q", [4, 1 << 20]), dim=DIM), grant)
            check("fuel_refusal", False)
        except CompositionError as exc:
            check("fuel_refusal", exc.code == "COGNITION_FUEL_EXCEEDED" and runtime.live_identity(state) == learned)
        for retired in ("publish_recoverable", "publish_identity", "recover_k1", "durability", "publish_canonical",
                        "import_composite", "discard_k1_candidate", "adopt_k1_candidate"):
            check(f"retired_operation_absent:{retired}", not hasattr(runtime, retired))
        check("residency_never_moved", all(fms.metrics()["residency"][k] == 0
                                           for k in ("demotions", "cold_reads", "cold_writes")))
        ended = runtime.continuity.snapshot().generation
    with Runtime(config) as again:                        # a new session: genesis again, nothing resumed
        check("new_session_starts_at_genesis", again.continuity.snapshot().generation == opened < ended)
        again.anchor_cognition(other)
    fms.close_state(state.id)
    fms.close_state(other.id)
try:
    Context(fms_admitted.lib, warm_bytes=4096, max_objects=4, cold_root="/elpis-cold")
    check("retired_cold_root_refused", False)
except TypeError:
    check("retired_cold_root_refused", True)

# ---- the volatile HACF epoch and query ingress -------------------------------------------------------------------
bridges = {"elpis_retrieval_bridge": native / "structure/libelpis_retrieval_bridge.so",
           "elpis_ingress_bridge": native / "pipeline/libelpis_ingress_bridge.so"}
retrieval = RetrievalLibrary(bridges["elpis_retrieval_bridge"], root=bridges["elpis_retrieval_bridge"].parent,
                             authority=catalog({"elpis_retrieval_bridge": bridges["elpis_retrieval_bridge"]}))
ingress_library = IngressLibrary(bridges["elpis_ingress_bridge"], root=bridges["elpis_ingress_bridge"].parent,
                                 authority=catalog({"elpis_ingress_bridge": bridges["elpis_ingress_bridge"]}))
state_root = Path("/nonexistent-elpis-epoch")              # never created: a retrieval epoch lives in RAM
for epoch in range(3):
    handle = build_corpus_and_index(retrieval, state_root, DOCS)
    try:
        with QueryIngress(ingress_library, handle) as ingress:
            result = ingress.run(b"touching endpoints may merge; maximum end.")
            check(f"ingress_ran:{epoch}", result is not None)
    finally:
        handle.destroy()
check("epoch_state_root_never_created", not state_root.exists())

ok = all(checks.values())
print(json.dumps({"ok": ok, "checks": checks, "answers": len(answers)}, sort_keys=True))
sys.exit(0 if ok else 1)
