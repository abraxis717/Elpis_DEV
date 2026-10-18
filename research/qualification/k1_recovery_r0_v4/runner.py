"""K1 Recovery R0 under RuntimeCore ABI v4 (the v5 library): the qualification runner (docs/qualification/K1_RECOVERY_R0_V4.md).

    python -I -B runner.py freeze --native-build BUILD --work-root ROOT --out FREEZE.json
    python -I -B runner.py run --native-build BUILD --work ROOT/qual --inputs qual --freeze FREEZE.json --out RECORDS
    python -I -B runner.py seal --freeze FREEZE.json --records RECORDS --verdict VERDICT --out EVIDENCE_DIR

``freeze`` measures the executing checkout, interpreter, imported package, libraries (and a trial deployment of
them), toolchain, platform and filesystem (``freeze.py``) and writes the freeze record once. ``run`` measures them
again and refuses before it creates or opens anything unless every one equals the record; after deploying the testing
RuntimeCore build it verifies the deployed copies and their pins; at the end it measures the checkout again. Then, for
every trial of the frozen ``plan.json`` (its coordinates: family, lineage, point, run index; the seed and fault
argument the plan derives from them), a child process drives a real lineage through the public runtime API up to the
point, parks there (the durability worker, or the control operation's caller) and reports what it holds; the parent
sends it ``SIGKILL`` and records the wait status. A fresh process (``restart``) records the raw slot and continuity
bytes, what restart decides, performs the point's pre-registered operator action, resumes and continues. The parent
writes one JSON record per trial; it decides nothing (``check.py`` evaluates the gates from the plan and the records).
``seal`` assembles the write-once evidence directory and its ``SHA256SUMS``.

Nothing here is a model or a fixture of the runtime: the stimuli are deterministic numbers, the observations typed
fixture evidence (TRAINING=NONE SEMANTICS=NONE); only process death (``SIGKILL``) is injected, plus the in-process
lost-sync faults of the uncertain family. ``SIGKILL`` keeps the page cache: no power-loss claim follows from it.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import random
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

RECORD_SCHEMA = "elpis.k1-recovery-r0-v4.record.v2"
RUN_SCHEMA = "elpis.k1-recovery-r0-v4.run.v2"
HERE = Path(__file__).resolve().parent


def _load_freeze():
    """The instrument's own ``freeze.py``, loaded by its path (``-I`` puts no script directory on ``sys.path``, and
    the instrument never adds one: nothing here can shadow a module ``elpis`` imports)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("k1_recovery_r0_v4_freeze", HERE / "freeze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = _load_freeze()

GRANT = "k1-recovery-r0-v4: qualification stimulus (TRAINING=NONE SEMANTICS=NONE)"
OBSERVE_GRANT = "k1-recovery-r0-v4: fixture typed evidence (TRAINING=NONE SEMANTICS=NONE)"
# Control operations of the testing build (elpis_runtime_testing_control_fault): op and the parked report.
CONTROL_OP = {"anchor": 1, "import": 2, "discard": 3, "adopt": 4}
CONTINUITY_FILES = ("continuity.a", "continuity.b")
WINDOW = 16   # raw bytes recorded on each side of a slot tear


def _say(**kw) -> None:
    sys.stdout.write(json.dumps(kw, sort_keys=True) + "\n")
    sys.stdout.flush()


def _hex(b) -> str | None:
    return None if b is None else bytes(b).hex()


def plan_load() -> dict:
    return json.loads((HERE / "plan.json").read_text())


def trials(plan: dict, runs_override: int | None = None) -> list:
    """The plan's trials in its registered enumeration order (plan.coordinates)."""
    out = []
    for family, fam in plan["families"].items():
        points = [p["point"] for p in fam.get("points", [{"point": "random"}])]
        runs = runs_override or fam.get("runs_per_lineage", plan["runs_per_point"])
        for lineage in fam["lineages"]:
            for point in points:
                for run_index in range(runs):
                    out.append((family, lineage, point, run_index))
    return out


# plan.json "torn_offsets.slot", in order: bytes of the slot (header + envelope) written before the death.
_SLOT_TORN = (lambda c: 17, lambda c: 127, lambda c: 128 + c // 2, lambda c: 128 + c - 32)


def fault_arg(plan: dict, lineage: str, point: str, run_index: int) -> int:
    if point == "step:3":
        return _SLOT_TORN[run_index % 4](plan["lineages"][lineage]["envelope_bytes"])
    if point in ("step:6", "control:4"):
        return plan["torn_offsets"]["record"][run_index % 4]
    return 0


# =====================================================================================================================
# The lineage, through the public API only (children and restarts)
# =====================================================================================================================

class Lineage:
    """One trial's lineage: its directories, its kind and its deterministic inputs."""

    def __init__(self, spec: dict):
        _child_environment(spec)
        from elpis.runtime.deployment import Deployment

        self.spec = spec
        self.plan = plan_load()
        self.kind = spec["lineage"]
        self.shape = self.plan["lineages"][self.kind]
        self.run = Path(spec["run_dir"])
        self.seed = spec["seed"]
        self.deployment = Deployment(Path(spec["home"]))
        self.libs = self.deployment.libraries

    def config(self, store: Path | None):
        from elpis.runtime import RuntimeConfig

        d = self.deployment
        return RuntimeConfig(self.run / "continuity", d.home / "lib" / d.config["runtime_library"],
                             native_authority=d.authority, runtime_library_id=d.config["runtime_library_id"],
                             k1_checkpoint_dir=store)

    def runtime(self, store: Path | None):
        from elpis.runtime import Runtime

        return Runtime(self.config(store)).open()

    def composite_config(self):
        from elpis.ECS.composite import CompositeConfig

        s = self.plan["lineages"]["composite"]
        return CompositeConfig(s["dim"], s["width"], max_rows=s["max_rows"], memory_dim=s["memory_dim"],
                               banks=s["banks"], slot_capacity=s["slot_capacity"],
                               codebook_capacity=s["codebook_capacity"])

    def envelope_bytes(self, kind=None) -> int:
        return int(self.plan["lineages"][kind or self.kind]["envelope_bytes"])

    def create(self, kind=None):
        from elpis.ECS.composite import CompositeState
        from elpis.ECS.k1 import K1State
        from elpis.runtime.deployment import initial_weights

        w = initial_weights(self.shape["dim"], self.shape["width"], self.seed)
        if (kind or self.kind) == "k1":
            return K1State.create(self.libs.k1, self.shape["dim"], self.shape["width"], w,
                                  max_rows=self.shape["max_rows"])
        return CompositeState.create(self.libs.composite, self.composite_config(), memoryview(w))

    def restore(self, envelope: bytes, kind=None):
        from elpis.ECS.composite import CompositeState
        from elpis.ECS.k1 import K1State

        if (kind or self.kind) == "k1":
            return K1State.restore(self.libs.k1, envelope, max_rows=self.shape["max_rows"])
        return CompositeState.restore(self.libs.composite, self.composite_config(), envelope)

    def stimulus(self, i: int):
        from elpis.runtime.cognition import Stimulus

        st, dim = self.plan["stimulus"], self.shape["dim"]
        rows = st["rows"]
        digest = lambda tag, j: int.from_bytes(hashlib.sha256(
            b"k1-recovery-r0-v4.stimulus" + tag + self.seed.to_bytes(8, "little") + i.to_bytes(4, "little")
            + j.to_bytes(4, "little")).digest()[:8], "little") / float(1 << 64) - 0.5
        x = array("d", (digest(b"x", j) for j in range(rows * dim)))
        y = array("d", (digest(b"y", j) for j in range(rows)))
        return Stimulus(x, y, array("Q", [rows, st["steps"]]), dim=dim)

    def observation(self, i: int):
        from elpis.ECS.composite import MemoryObservation

        o = self.plan["stimulus"]["observation"]
        entity = lambda n: hashlib.sha256(b"k1-recovery-r0-v4.entity" + self.seed.to_bytes(8, "little")
                                          + n.to_bytes(4, "little")).digest()
        provenance = hashlib.sha256(b"k1-recovery-r0-v4.provenance" + self.seed.to_bytes(8, "little")
                                    + i.to_bytes(4, "little")).digest()
        return MemoryObservation(o["relation_type"], o["polarity"], ((1, 0, entity(2 * i)), (2, 0, entity(2 * i + 1))),
                                 i + 1, o["source_authority"], provenance)

    def learn(self, runtime, state, i: int):
        """LEARN i (volatile); a composite LEARN carries one typed observation. Returns (identity, relation key)."""
        from elpis.runtime.cognition import LearnAuthority, ObservationAuthority

        authority = LearnAuthority(self.plan["stimulus"]["learning_rate"], GRANT)
        if type(state).__name__ == "CompositeState":
            observation = self.observation(i)
            result = runtime.run_numeric_learn(
                state, self.stimulus(i), authority, observations=(observation,),
                observation_authority=ObservationAuthority(OBSERVE_GRANT, frozenset({observation.source_authority})))
            key = result.observed[0]
        else:
            runtime.run_numeric_learn(state, self.stimulus(i), authority)
            key = None
        return runtime.live_identity(state)[0], key


def _child_environment(spec: dict) -> None:
    """A child runs only in the parent's verified environment: isolated, the verified package."""
    import elpis

    root = str(Path(elpis.__file__).resolve().parent)
    if sys.flags.isolated != 1 or root != spec["package_root"]:
        raise SystemExit(f"child environment refused: isolated={sys.flags.isolated} package={root}")


# =====================================================================================================================
# Raw evidence: what the slot and continuity files hold on disk (read directly; slots never authorize anything)
# =====================================================================================================================

_SLOT_MAGIC, _SLOT_DOMAIN = b"ELPK1CK\x01", b"elpis.k1-checkpoint.slot.v1\0"


def raw_slots(store: Path) -> dict:
    """Each slot's header identity and generation and whether it verifies (header checksum and envelope trailer)."""
    out = {}
    for p in sorted(store.iterdir()):
        raw = p.read_bytes()
        h, envelope = raw[:128], raw[128:]
        if h[:8] != _SLOT_MAGIC:
            out[p.name] = None
            continue
        identity = h[32:64]
        verified = (hashlib.sha256(_SLOT_DOMAIN + h[:96] + envelope).digest() == h[96:128]
                    and hashlib.sha256(envelope[:-32]).digest() == envelope[-32:] == identity)
        out[p.name] = {"identity": identity.hex(), "generation": int.from_bytes(h[16:24], "little"),
                       "verified": verified}
    return out


def snapshot_files(run: Path, stores: tuple, tag: str) -> dict:
    """The raw bytes before an interruption: both continuity records in full (hex) and every checkpoint slot's
    SHA-256, with a copy of each slot kept beside the run for the restart's byte comparison."""
    out = {"continuity": {n: (run / "continuity" / n).read_bytes().hex() for n in CONTINUITY_FILES
                          if (run / "continuity" / n).exists()}, "slots": {}}
    for store in stores:
        keep = run / f"raw-{tag}" / store.name
        keep.mkdir(parents=True, exist_ok=True)
        for p in sorted(store.iterdir()):
            shutil.copyfile(p, keep / p.name)
            out["slots"][f"{store.name}/{p.name}"] = frozen.sha256(p)
    return out


def compare_files(run: Path, stores: tuple, tag: str, arg: int) -> dict:
    """The raw bytes after the interruption, against the ``tag`` snapshot: the continuity records in full (hex);
    for every slot its SHA-256 and the differing byte range, whether every difference lies below ``arg`` (a write of
    exactly the first ``arg`` bytes), and the bytes around ``arg`` before and after."""
    out = {"continuity": {n: (run / "continuity" / n).read_bytes().hex() for n in CONTINUITY_FILES
                          if (run / "continuity" / n).exists()}, "slots": {}}
    for store in stores:
        for p in sorted(store.iterdir()):
            before_path = run / f"raw-{tag}" / store.name / p.name
            before, after = before_path.read_bytes() if before_path.exists() else b"", p.read_bytes()
            diff = [i for i in range(min(len(before), len(after))) if before[i] != after[i]] if before != after else []
            entry = {"sha256": frozen.sha256(p), "changed": before != after, "size": len(after),
                     "first_diff": diff[0] if diff else None, "last_diff": diff[-1] if diff else None,
                     "diff_bytes": len(diff)}
            if arg:
                lo, hi = max(0, arg - WINDOW), arg + WINDOW
                entry.update(within_arg=bool(diff) and diff[-1] < arg, window=[lo, hi],
                             window_before=before[lo:hi].hex(), window_after=after[lo:hi].hex())
            out["slots"][f"{store.name}/{p.name}"] = entry
    return out


def _ticket(t) -> dict:
    return {"kind": t.kind, "status": t.status, "identity": _hex(t.identity), "recoverable": bool(t.recoverable),
            "code": t.code}


def _wait_parked(runtime, value: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while runtime.durability().parked != value:
        if time.monotonic() > deadline:
            raise TimeoutError(f"never parked at {value}")
        time.sleep(0.002)


def _refusal(call) -> str | None:
    """The refusal code of ``call()``, or None if it succeeded."""
    from elpis.runtime.errors import CompositionError

    try:
        call()
    except CompositionError as exc:
        return exc.code
    except Exception as exc:   # ContinuityError and the like carry a code too
        return getattr(exc, "code", type(exc).__name__)
    return None


# =====================================================================================================================
# Children: drive the lineage to the point, report, and wait to be killed
# =====================================================================================================================

def _caller_keep(lin: Lineage, runtime, state, name: str) -> None:
    """The caller keeps a copy of its state's bytes: release (the volatile work stays live), snapshot, write. The next
    operation rebinds and verifies it."""
    runtime.release(state)
    (lin.run / "caller").mkdir(exist_ok=True)
    (lin.run / "caller" / name).write_bytes(state.snapshot())


def _stores(lin: Lineage) -> tuple:
    return tuple(p for p in (lin.run / "checkpoint", lin.run / "checkpoint-composite") if p.exists())


def _prepare(lin: Lineage, runtime, state, store: bool) -> dict:
    """Anchor; LEARN 0; publish (acknowledged, or identity-published without a store) = D1; the caller keeps D1's
    bytes; LEARN 1 = D2 (the candidate)."""
    out = {"D0": _hex(state.state_digest())}
    out["anchor"] = runtime.anchor_cognition(state).generation
    keys = []
    d, k = lin.learn(runtime, state, 0)
    keys.append(_hex(k))
    t = runtime.publish_recoverable(state) if store else runtime.publish_identity(state)
    out["first_ticket"] = _ticket(t)
    out["D1"] = _hex(d)
    _caller_keep(lin, runtime, state, "D1")
    d, k = lin.learn(runtime, state, 1)
    keys.append(_hex(k))
    out["D2"] = _hex(d)
    if not store:
        _caller_keep(lin, runtime, state, "D2")
    out["keys"] = keys
    return out


def child_publish(lin: Lineage, point: str, arg: int, store: bool) -> None:
    """Families publish / identity / the reconciliation families' first phase: park the worker at a step of the
    D2 publication (or deliver its acknowledgment), then LEARN 2 = D3 (volatile, never requested)."""
    runtime = lin.runtime(lin.run / "checkpoint" if store else None)
    state = lin.create()
    out = _prepare(lin, runtime, state, store)
    publish = runtime.publish_recoverable if store else runtime.publish_identity
    out["files_before"] = snapshot_files(lin.run, _stores(lin), "before")
    if point.startswith("step:"):
        step = int(point[5:])
        runtime._core.testing_durable_fault(2, step, arg)
        pending = publish(state, wait=False)
        out["ticket"] = _ticket(pending)        # captured at once: a request never waits for the worker
        d3, k3 = lin.learn(runtime, state, 2)       # the HOT lane is not blocked by the park
        _wait_parked(runtime, step)
    else:                                            # delivered: the acknowledgment reached the caller
        ticket = publish(state)
        out["ticket"] = _ticket(ticket)
        d3, k3 = lin.learn(runtime, state, 2)
    out["D3"], out["keys"] = _hex(d3), out["keys"] + [_hex(k3)]
    _say(ready=True, **out)
    time.sleep(3600)                                 # killed here


def child_uncertain(lin: Lineage, action: int) -> None:
    """Family uncertain (in-process lost-sync faults, no kill): the D2 publication's record sync fails with the bytes
    lost (8) or although durable (9). The lane faults until reopen; the HOT lane keeps running."""
    runtime = lin.runtime(lin.run / "checkpoint")
    state = lin.create()
    out = _prepare(lin, runtime, state, True)
    out["files_before"] = snapshot_files(lin.run, _stores(lin), "before")
    runtime._core.testing_durable_fault(action)
    ticket = runtime.publish_recoverable(state)
    out["ticket"] = _ticket(ticket)
    out["lane_fault"] = runtime.durability().fault
    d3, k3 = lin.learn(runtime, state, 2)            # HOT continues under the lane fault
    out["D3"], out["keys"] = _hex(d3), out["keys"] + [_hex(k3)]
    out["hot_after_fault"] = d3 not in (None, out["D2"])
    out["request_under_fault"] = _refusal(lambda: runtime.publish_recoverable(state))
    runtime.close()
    _say(ready=True, **out)


def child_identity_refused(lin: Lineage) -> None:
    """A composite lineage without a store: anchoring is identity-only; LEARN; an identity-only publication is refused
    (its envelope size is stated only by a provisioned store); a recoverable one has no store; nothing durable
    changes."""
    runtime = lin.runtime(None)
    state = lin.create()
    out = {"D0": _hex(state.state_digest()), "anchor": runtime.anchor_cognition(state).generation}
    d, k = lin.learn(runtime, state, 0)
    out["D1"], out["keys"] = _hex(d), [_hex(k)]
    out["files_before"] = snapshot_files(lin.run, (), "before")
    out["refusal"] = _refusal(lambda: runtime.publish_identity(state))
    out["recoverable_refusal"] = _refusal(lambda: runtime.publish_recoverable(state))
    snap = runtime.continuity.snapshot()
    out["durable_after"] = {"authorized": _hex(snap.k1_state_digest), "generation": snap.generation,
                            "format": snap.state_format}
    out["last_acknowledged"] = _ticket(runtime.durability().last_acknowledged)
    runtime.close()
    _say(ready=True, **out)


def _control(runtime, op: str, point: int, arg: int, call) -> dict:
    """Arm the control operation's fault (park at ``point``), run it on a thread, wait until it is parked."""
    runtime._core.testing_control_fault(CONTROL_OP[op], 2, point, arg)
    failure = {}

    def run():
        try:
            call()
            failure["returned"] = True
        except BaseException as exc:   # a park ends only with the process; anything else is recorded
            failure["raised"] = repr(exc)

    threading.Thread(target=run, daemon=True).start()
    _wait_parked(runtime, 100 * CONTROL_OP[op] + point)
    return failure


def child_anchor(lin: Lineage, point: int, arg: int) -> None:
    runtime = lin.runtime(lin.run / "checkpoint")
    state = lin.create()
    out = {"D0": _hex(state.state_digest()), "files_before": snapshot_files(lin.run, _stores(lin), "before")}
    out["control"] = _control(runtime, "anchor", point, arg, lambda: runtime.anchor_cognition(state))
    _say(ready=True, **out)
    time.sleep(3600)


def child_reconcile(lin: Lineage, op: str, point: int, arg: int) -> None:
    """Second phase of the discard / adopt families: a fresh process finds the candidate and parks inside the
    reconciliation of exactly it."""
    runtime = lin.runtime(lin.run / "checkpoint")
    r = runtime.recover_k1()
    out = {"found": {"disposition": r.disposition, "authorized": _hex(r.authorized), "candidate": _hex(r.candidate)},
           "files_before": snapshot_files(lin.run, _stores(lin), "before")}
    call = (lambda: runtime.discard_k1_candidate(r.candidate)) if op == "discard" else \
        (lambda: runtime.adopt_k1_candidate(r.candidate))
    out["control"] = _control(runtime, op, point, arg, call)
    _say(ready=True, **out)
    time.sleep(3600)


def child_import(lin: Lineage, point: int, arg: int) -> None:
    """A K1 lineage (its own K1-sized store) publishes K; the operator provisions a composite store and imports K as
    a composite with an empty memory; the import parks at ``point``."""
    from elpis.ECS.composite import CompositeState

    runtime = lin.runtime(lin.run / "checkpoint")
    state = lin.create()
    out = {"D0": _hex(state.state_digest())}
    out["anchor"] = runtime.anchor_cognition(state).generation
    k, _ = lin.learn(runtime, state, 0)
    out["first_ticket"] = _ticket(runtime.publish_recoverable(state))
    runtime.close()
    out["K"] = _hex(k)
    composite = CompositeState.import_k1(lin.libs.composite, lin.composite_config(), state.snapshot())
    out["C"] = _hex(composite.state_digest())
    runtime = lin.runtime(lin.run / "checkpoint-composite")
    out["files_before"] = snapshot_files(lin.run, _stores(lin), "before")
    out["control"] = _control(runtime, "import", point, arg, lambda: runtime.import_composite(composite))
    _say(ready=True, **out)
    time.sleep(3600)


def child_random(lin: Lineage, rng_seed: int) -> None:
    """Family random: LEARN repeatedly and request publications at random (some awaited); report every requested
    identity, every delivered ticket and every resolution of an unawaited one as it happens. Killed at a random
    moment."""
    fam = lin.plan["families"]["random"]
    rng = random.Random(rng_seed)
    runtime = lin.runtime(lin.run / "checkpoint")
    state = lin.create()
    _say(event="anchor", identity=_hex(state.state_digest()), generation=runtime.anchor_cognition(state).generation)
    _say(event="ready")
    pending = []
    for i in range(fam["max_learns"][lin.kind]):
        d, k = lin.learn(runtime, state, i)
        _say(event="learn", i=i, identity=_hex(d), key=_hex(k))
        if rng.random() < fam["publish_probability"]:
            wait = rng.random() < fam["wait_probability"]
            _say(event="request", identity=_hex(d), wait=wait)
            t = runtime.publish_recoverable(state, wait=wait)
            if wait:
                _say(event="ticket", **_ticket(t))
            else:
                pending.append((d, t))
        for requested, t in list(pending):
            if t.status != "PENDING":
                _say(event="resolved", requested=_hex(requested), **_ticket(t))
                pending.remove((requested, t))
    _say(event="done")
    time.sleep(3600)


# =====================================================================================================================
# Restart: what a fresh process decides, the operator action, resumption
# =====================================================================================================================

def _observe(lin: Lineage, store: Path | None) -> dict:
    runtime, out = lin.runtime(store), {}
    try:
        snap = runtime.continuity.snapshot()
        out.update(format=snap.state_format, authorized=_hex(snap.k1_state_digest), generation=snap.generation)
        if store is not None:
            r = runtime.recover_k1()
            out.update(disposition=r.disposition, recovered_authorized=_hex(r.authorized),
                       authorized_present=bool(r.authorized_present), candidate=_hex(r.candidate),
                       envelope_identity=_hex(r.envelope[-32:]) if r.envelope is not None else None)
        return out
    finally:
        runtime.close()
        if store is not None:
            out["slots"] = raw_slots(store)


def _resume(lin: Lineage, store: Path, keys: list, kind=None) -> dict:
    """Restore exactly the RESUMABLE envelope, verify it against continuity, report the resident relation keys (and
    the memory's occupancy), then LEARN once and publish recoverably once."""
    runtime = lin.runtime(store)
    try:
        r = runtime.recover_k1()
        out = {"disposition": r.disposition}
        if r.disposition != "RESUMABLE":
            return out
        state = lin.restore(r.envelope, kind)
        out["restored_identity"] = _hex(state.state_digest())
        out["continuity_identity"] = _hex(runtime.continuity.snapshot().k1_state_digest)
        composite = type(state).__name__ == "CompositeState"
        if composite:
            out["resident"] = {k: (None if state.slot(bytes.fromhex(k)) is None else
                                   state.slot(bytes.fromhex(k)).assert_count) for k in keys if k is not None}
            out["memory_slots"] = state.inspect()["memory_slots_used"]
        d, _ = lin.learn(runtime, state, 50)
        t = runtime.publish_recoverable(state)
        out["continued"] = {"identity": _hex(d), "ticket": _ticket(t)}
        if composite:
            out["memory_slots_after"] = state.inspect()["memory_slots_used"]
        return out
    finally:
        runtime.close()


def _cross_format(lin: Lineage, store: Path) -> dict:
    """S4: a state of the other format, from the same seed, offered to this lineage: refused at bind, and continuity
    is exactly as before (a digest or a shape that looks plausible authorizes nothing across formats)."""
    other = "composite" if lin.kind == "k1" else "k1"
    before = _observe(lin, store)
    runtime = lin.runtime(store)
    try:
        state = lin.create(other)
        refusal = _refusal(lambda: lin.learn(runtime, state, 80))
    finally:
        runtime.close()
    after = _observe(lin, store)
    keys = ("format", "authorized", "generation", "disposition", "candidate")
    return {"offered": other, "refusal": refusal, "unchanged": all(before.get(k) == after.get(k) for k in keys)}


def _publish_over_candidate(lin: Lineage, store: Path) -> dict:
    """G3: with a candidate unresolved, the caller rebinds the authorized state from its own bytes, LEARNs and asks
    for a recoverable publication: it must fail CHECKPOINT_UNRESOLVED."""
    path = lin.run / "caller" / "D1"
    if not path.exists():
        return {"attempted": False}
    runtime = lin.runtime(store)
    try:
        state = lin.restore(path.read_bytes())
        lin.learn(runtime, state, 60)
        return {"attempted": True, "ticket": _ticket(runtime.publish_recoverable(state))}
    finally:
        runtime.close()


def restart(spec: dict) -> dict:
    lin = Lineage(spec)
    family, run_index, arg = spec["family"], spec["run_index"], spec.get("arg", 0)
    store = lin.run / "checkpoint"
    keys = spec.get("child", {}).get("keys", [])
    # The raw continuity records exactly as restart finds them, before anything here opens or changes them.
    out = {"continuity_raw": {n: (lin.run / "continuity" / n).read_bytes().hex() for n in CONTINUITY_FILES
                              if (lin.run / "continuity" / n).exists()}}
    if family == "identity_refused":
        from elpis.runtime.recovery import provision_k1_checkpoint

        out["files_after"] = compare_files(lin.run, (), "before", 0)
        out["before"] = _observe(lin, None)
        provision_k1_checkpoint(lin.libs.runtime, store, lin.envelope_bytes())   # a store attached afterwards
        out["observed"] = _observe(lin, store)
        return out
    if family == "identity":
        # No store ever held these bytes: restart attaches a freshly provisioned one and finds the authority missing.
        from elpis.runtime.recovery import provision_k1_checkpoint

        out["files_after"] = compare_files(lin.run, (), "before", arg)
        out["before"] = _observe(lin, None)
        provision_k1_checkpoint(lin.libs.runtime, store, lin.envelope_bytes())
        out["provisioned"] = {p.name: {"size": p.stat().st_size, "inode": p.stat().st_ino}
                              for p in sorted(store.iterdir())}
        out["observed"] = _observe(lin, store)
        authorized = out["observed"]["authorized"]
        caller = {name: (lin.run / "caller" / name).read_bytes() for name in ("D1", "D2")}
        wrong = [n for n, e in caller.items() if lin.restore(e).state_digest().hex() != authorized]
        right = [n for n, e in caller.items() if lin.restore(e).state_digest().hex() == authorized]
        out["caller_bytes"] = {"authorized": right, "other": wrong}
        if wrong:   # other caller bytes are refused at bind (never adopted): a fresh runtime per attempt
            runtime = lin.runtime(store)
            try:
                state = lin.restore(caller[wrong[0]])
                out["wrong_bytes_refusal"] = _refusal(lambda: lin.learn(runtime, state, 70))
            finally:
                runtime.close()
        runtime = lin.runtime(store)
        try:
            state = lin.restore(caller[right[0]])
            d, _ = lin.learn(runtime, state, 50)
            out["continued"] = {"identity": _hex(d), "ticket": _ticket(runtime.publish_recoverable(state))}
        finally:
            runtime.close()
        out["after"] = _observe(lin, store)
        out["final_footprint"] = footprint(store)
        return out
    stores = _stores(lin)
    out["files_after"] = compare_files(lin.run, stores, "before", arg) if (lin.run / "raw-before").exists() else None
    if family == "import":
        composite_store = lin.run / "checkpoint-composite"
        out["action"] = None
        out["observed_composite_store"] = _observe(lin, composite_store)
        out["observed_k1_store"] = _observe(lin, store)
        if out["observed_composite_store"]["format"] == "K1":
            out["action"] = "import"   # the operator re-runs the import from the authorized K1 state
            from elpis.ECS.composite import CompositeState

            runtime = lin.runtime(store)
            try:
                k1_envelope = runtime.recover_k1().envelope
            finally:
                runtime.close()
            composite = CompositeState.import_k1(lin.libs.composite, lin.composite_config(), k1_envelope)
            runtime = lin.runtime(composite_store)
            try:
                out["reimported"] = {"identity": _hex(composite.state_digest()),
                                     "format": runtime.import_composite(composite).state_format}
            finally:
                runtime.close()
        out["after_action"] = _observe(lin, composite_store)
        out["resumed"] = _resume(lin, composite_store, [], kind="composite")
        out["final"] = _observe(lin, composite_store)
        out["final_footprint"] = footprint(store)
        out["final_footprint_composite"] = footprint(composite_store)
        return out
    out["observed"] = _observe(lin, store)
    disposition = out["observed"]["disposition"]
    action = None
    if disposition == "CANDIDATE_UNRESOLVED":
        out["publish_over_candidate"] = _publish_over_candidate(lin, store)
        # discard / adopt: the family's own reconciliation, again; elsewhere the plan's alternation.
        action = family if family in ("discard", "adopt") else ("adopt" if run_index % 2 == 0 else "discard")
        runtime = lin.runtime(store)
        try:
            candidate = bytes.fromhex(out["observed"]["candidate"])
            if action == "adopt":
                runtime.adopt_k1_candidate(candidate)
            else:
                runtime.discard_k1_candidate(candidate)
        finally:
            runtime.close()
    elif disposition == "NOTHING_TO_RECOVER" and family == "anchor":
        action = "anchor"   # the operator anchors again, explicitly, with the caller's (recreated) state
        runtime = lin.runtime(store)
        state = lin.create()
        try:
            out["reanchored"] = {"identity": _hex(state.state_digest()),
                                 "generation": runtime.anchor_cognition(state).generation}
        finally:
            runtime.close()
    out["action"] = action
    out["after_action"] = _observe(lin, store)
    if family in ("publish", "anchor") and out["after_action"]["disposition"] == "RESUMABLE":
        out["cross_format"] = _cross_format(lin, store)
    out["resumed"] = _resume(lin, store, keys)
    out["final"] = _observe(lin, store)
    out["final_footprint"] = footprint(store)
    return out


def footprint(store: Path) -> dict:
    if not store.exists():
        return {}
    return {p.name: {"size": p.stat().st_size, "inode": p.stat().st_ino} for p in sorted(store.iterdir())}


# =====================================================================================================================
# The parent: verify, deploy once, run every pre-registered trial, write one record per trial
# =====================================================================================================================

def _spawn(role: str, spec: dict) -> subprocess.Popen:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}   # -I ignores every PYTHON* variable
    return subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), role, json.dumps(spec)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=spec["run_dir"])


class _Output:
    """A child's JSON-line stdout, read through one buffer for the child's whole life: the lines a chunk carries past
    the one a caller waits for stay buffered for the next read, and after the kill ``rest`` returns every line the
    child wrote (a pipe keeps what a killed writer flushed). Nothing the child reported is dropped."""

    def __init__(self, proc):
        self.proc, self.fd, self.buf = proc, proc.stdout.fileno(), b""

    def _take(self, until):
        lines = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if line.strip():
                lines.append(json.loads(line))
                if until is not None and until(lines[-1]):
                    return lines, True
        return lines, False

    def read(self, timeout: float, until=None) -> tuple[list, bool]:
        """Lines until ``until(line)`` holds, the process exits, or ``timeout``. Returns (lines, satisfied)."""
        lines, deadline = [], time.monotonic() + timeout
        while True:
            more, done = self._take(until)
            lines += more
            if done:
                return lines, True
            left = deadline - time.monotonic()
            if left <= 0:
                return lines, False
            ready, _, _ = select.select([self.fd], [], [], min(left, 0.5))
            if not ready:
                if self.proc.poll() is not None:
                    return lines, False
                continue
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return lines, False
            self.buf += chunk

    def rest(self) -> list:
        """Every line not yet returned, to the end of the (dead) child's output; a torn last line raises."""
        while chunk := os.read(self.fd, 65536):
            self.buf += chunk
        lines, _ = self._take(None)
        if self.buf.strip():
            raise ValueError(f"the child's output ends in a torn line: {self.buf[-200:]!r}")
        return lines


def _not_ready(proc) -> str:
    """Why a child never reported its point: its exit and stderr, or (still running) it is killed."""
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _kill(proc)
        return "never ready (killed)"
    return f"exit {proc.returncode}: {proc.stderr.read()[-4000:]}"


def _kill(proc) -> dict:
    """SIGKILL the child and reap it: the wait status (``waitpid``) and the child's stderr are the evidence."""
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=60)
    err = proc.stderr.read()
    return {"returncode": proc.returncode, "signaled": proc.returncode == -signal.SIGKILL,
            "stderr_tail": err[-2000:], "stderr_sha256": hashlib.sha256(err.encode()).hexdigest()}


def _finish(proc, timeout: float) -> tuple[dict | None, dict]:
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, dict(_kill(proc), error="timeout")
    lines = [json.loads(line) for line in out.splitlines() if line.strip()]
    status = {"returncode": proc.returncode, "signaled": False, "stderr_tail": err[-2000:],
              "stderr_sha256": hashlib.sha256(err.encode()).hexdigest()}
    return (lines[-1] if lines and proc.returncode == 0 else None), status


def _drive(family: str, spec: dict, timeout: float) -> dict:
    """Run the child (and, for a reconciliation, its second phase); kill it at its point."""
    out = {"child": None, "child_error": None, "child_exit": None}
    if family in ("uncertain", "identity_refused"):
        proc = _spawn("child", spec)
        result, status = _finish(proc, timeout)
        out["child"], out["child_exit"] = result, status
        out["child_error"] = None if result is not None else status["stderr_tail"] or "no report"
        return out
    if family == "random":
        proc = _spawn("child", spec)
        stdout = _Output(proc)
        lines, ready = stdout.read(timeout, until=lambda line: line.get("event") == "ready")
        if not ready:
            out["child_error"] = _not_ready(proc)
            return out
        target = spec["kill_after"]           # kill as soon as that many LEARNs were reported
        more, _ = stdout.read(timeout, until=lambda line: line.get("event") == "learn" and line["i"] + 1 >= target)
        out["child_exit"] = _kill(proc)
        out["events"] = lines + more + stdout.rest()   # what it did between the last LEARN read and its death
        out["keys"] = [e.get("key") for e in out["events"] if e.get("event") == "learn"]
        return out
    proc = _spawn("child", spec)
    lines, ready = _Output(proc).read(timeout, until=lambda line: line.get("ready"))
    if not ready:
        out["child_error"] = _not_ready(proc)
        return out
    out["child"] = lines[-1]
    out["child_exit"] = _kill(proc)
    if family in ("discard", "adopt"):
        proc = _spawn("child", dict(spec, phase="reconcile"))
        lines, ready = _Output(proc).read(timeout, until=lambda line: line.get("ready"))
        if not ready:
            out["reconcile_error"] = _not_ready(proc)
            return out
        out["reconcile"] = lines[-1]
        out["reconcile_exit"] = _kill(proc)
        out["reconcile_error"] = None
    return out


def _deployment_capacity(deployment, plan: dict, kind: str) -> int:
    """The envelope size the admitted library states for the lineage's shape (it must be the plan's)."""
    from elpis.ECS.composite import CompositeConfig

    if kind == "k1":
        return int(deployment.libraries.k1.envelope_bytes(plan["lineages"]["k1"]["dim"], plan["lineages"]["k1"]["width"]))
    s = plan["lineages"]["composite"]
    return int(deployment.libraries.composite.envelope_bytes(CompositeConfig(
        s["dim"], s["width"], max_rows=s["max_rows"], memory_dim=s["memory_dim"], banks=s["banks"],
        slot_capacity=s["slot_capacity"], codebook_capacity=s["codebook_capacity"])))


def run(args) -> int:
    native_build, work = Path(args.native_build).absolute(), Path(args.work).absolute()
    if args.inputs == "qual" and (args.freeze is None or args.only is not None or args.dev_runs is not None):
        raise SystemExit("a QUAL run runs every trial of the plan, only under a verified freeze record (--freeze)")
    # 1. Measure the executing environment and refuse before anything is created, opened or read for the run.
    observed = frozen.observe(native_build=native_build, work_root=work.parent)
    freeze = None
    if args.freeze:
        freeze = json.loads(Path(args.freeze).read_text())
        problems = frozen.compare(freeze, observed)
        if problems:
            raise SystemExit("freeze verification refused the run: " + "; ".join(problems))
    from elpis.runtime.deployment import Deployment, init_deployment
    from elpis.runtime.recovery import provision_k1_checkpoint

    plan = plan_load()
    inputs = plan["inputs"][args.inputs]
    work.mkdir(mode=0o700, parents=False, exist_ok=False)
    home = work / "home"
    init_deployment(home, native_build, testing=True, source_commit=observed["source"]["commit"])
    # 2. The deployed copies, their pins and the ABI the admitted libraries report.
    deployed = frozen.deployed(home)
    if freeze is not None:
        problems = frozen.compare_deployed(freeze, deployed)
        if problems:
            raise SystemExit("deployment verification refused the run: " + "; ".join(problems))
    deployment = Deployment(home)
    for kind in plan["lineages"]:
        stated = _deployment_capacity(deployment, plan, kind)
        if stated != plan["lineages"][kind]["envelope_bytes"]:
            raise SystemExit(f"the {kind} envelope is {stated} bytes, not the plan's")
    timeout = plan["bounds"]["child_timeout_seconds"]
    records = open(args.out, "x")
    started = time.monotonic()
    rng = random.Random(inputs["random_seed"])
    only = None if args.only is None else set(args.only.split(","))
    count = 0
    for index, (family, kind, point, run_index) in enumerate(trials(plan, args.dev_runs)):
        fam = plan["families"][family]
        extra = {}
        if family == "random":   # consumed for every random trial in order, run or not (plan.coordinates.random)
            extra = {"rng_seed": rng.randrange(1 << 63), "kill_after": rng.randint(1, fam["max_learns"][kind] - 1)}
        if only is not None and family not in only:
            continue
        rid = f"{family}/{kind}/{point.replace(':', '-')}/{run_index}"
        run_dir = work / "runs" / f"{index:04d}"
        run_dir.mkdir(parents=True)
        seed = inputs["lineage_seed_base"] + index
        arg = fault_arg(plan, kind, point, run_index)
        spec = {"home": str(home), "run_dir": str(run_dir), "lineage": kind, "seed": seed, "family": family,
                "point": point, "run_index": run_index, "arg": arg, "package_root": observed["python"]["package_root"],
                **extra}
        record = {"schema": RECORD_SCHEMA, "index": index, "id": rid, "family": family, "lineage": kind,
                  "point": point, "run_index": run_index, "seed": seed, "arg": arg, "inputs": args.inputs, **extra}
        if fam["store"]:
            provision_k1_checkpoint(deployment.libraries.runtime, run_dir / "checkpoint", plan["lineages"][kind]
                                    ["envelope_bytes"])
            record["provisioned"] = footprint(run_dir / "checkpoint")
        if family == "import":
            provision_k1_checkpoint(deployment.libraries.runtime, run_dir / "checkpoint-composite",
                                    plan["lineages"]["composite"]["envelope_bytes"])
            record["provisioned_composite"] = footprint(run_dir / "checkpoint-composite")
        record.update(_drive(family, spec, timeout))
        if record.get("child_error") is None and record.get("reconcile_error") is None:
            keys = record["keys"] if family == "random" else (record["child"] or {}).get("keys", [])
            proc = _spawn("restart", dict(spec, child={"keys": keys}))
            result, status = _finish(proc, timeout)
            record["restart"], record["restart_exit"] = result, status
            record["restart_error"] = None if result is not None else status["stderr_tail"] or "no report"
        records.write(json.dumps(record, sort_keys=True) + "\n")
        records.flush()
        count += 1
    records.close()
    # 3. The checkout after the run: still exactly the frozen one.
    after = frozen.observe(native_build=native_build, work_root=work.parent)
    post = frozen.compare(freeze, after) if freeze is not None else frozen._preconditions(after)
    summary = {"schema": RUN_SCHEMA, "records": count, "seconds": round(time.monotonic() - started, 1),
               "inputs": args.inputs, "dev_runs": args.dev_runs, "only": args.only,
               "records_sha256": frozen.sha256(Path(args.out)),
               "freeze_sha256": frozen.sha256(Path(args.freeze)) if args.freeze else None,
               "observed": observed, "deployed": deployed, "post_run_problems": post}
    with open(args.out + ".run.json", "x") as f:
        f.write(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"records": count, "seconds": summary["seconds"], "out": args.out, "post_run_problems": post}))
    return 0 if not post else 1


def freeze_command(args) -> int:
    native_build, work_root = Path(args.native_build).absolute(), Path(args.work_root).absolute()
    if not work_root.is_dir():
        raise SystemExit(f"{work_root}: the work root must exist (its filesystem is part of the freeze)")
    plan = plan_load()
    runs = {family: fam.get("runs_per_lineage", plan["runs_per_point"]) for family, fam in plan["families"].items()}
    scratch = work_root / f"freeze-trial-{os.getpid()}"
    record = frozen.record(native_build=native_build, work_root=work_root, plan_runs=runs, scratch=scratch)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"freeze": args.out, "sha256": frozen.sha256(Path(args.out)),
                      "source_commit": record["source"]["commit"]}))
    return 0


def seal_command(args) -> int:
    """Assemble the write-once evidence directory: the freeze record, the records, the run summary and the verdict,
    read-only, with SHA256SUMS over exactly those files. The directory must not exist."""
    out = Path(args.out)
    out.mkdir(mode=0o755, parents=False, exist_ok=False)
    names = {"freeze.json": args.freeze, "records.jsonl": args.records, "run.json": args.records + ".run.json",
             "verdict.json": args.verdict}
    for name, source in names.items():
        shutil.copyfile(source, out / name)
        os.chmod(out / name, 0o444)
    sums = "".join(f"{frozen.sha256(out / name)}  {name}\n" for name in sorted(names))
    fd = os.open(out / "SHA256SUMS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as f:
        f.write(sums)
    print(json.dumps({"evidence": str(out), "files": sorted(names) + ["SHA256SUMS"]}))
    return 0


def child(spec: dict) -> None:
    lin = Lineage(spec)
    family, point, arg = spec["family"], spec["point"], spec.get("arg", 0)
    if family in ("publish", "identity"):
        child_publish(lin, point, arg, family == "publish")
    elif family in ("discard", "adopt"):
        if spec.get("phase") == "reconcile":
            child_reconcile(lin, family, int(point.split(":")[1]), arg)
        else:
            child_publish(lin, "step:5", 0, True)
    elif family == "anchor":
        child_anchor(lin, int(point.split(":")[1]), arg)
    elif family == "import":
        child_import(lin, int(point.split(":")[1]), arg)
    elif family == "uncertain":
        child_uncertain(lin, int(point.split(":")[1]))
    elif family == "identity_refused":
        child_identity_refused(lin)
    elif family == "random":
        child_random(lin, spec["rng_seed"])
    else:
        raise SystemExit(f"unknown family {family}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("child", "restart"):
        spec = json.loads(argv[1])
        if argv[0] == "child":
            child(spec)
        else:
            _say(**restart(spec))
        return 0
    p = argparse.ArgumentParser(prog="runner.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--native-build", required=True)
    f.add_argument("--work-root", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--native-build", required=True)
    r.add_argument("--work", required=True, help="a new directory inside the frozen work root")
    r.add_argument("--inputs", choices=("dev", "qual"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--freeze", default=None, help="the freeze record; required for QUAL inputs")
    r.add_argument("--only", default=None, help="comma-separated families (DEV only)")
    r.add_argument("--dev-runs", type=int, default=None, help="runs per point and lineage (DEV only)")
    s = sub.add_parser("seal")
    s.add_argument("--freeze", required=True)
    s.add_argument("--records", required=True)
    s.add_argument("--verdict", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"freeze": freeze_command, "run": run, "seal": seal_command}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
