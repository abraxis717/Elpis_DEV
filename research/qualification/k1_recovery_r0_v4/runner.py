"""K1 Recovery R0 under RuntimeCore ABI v4 (the v5 library): the qualification runner (docs/qualification/K1_RECOVERY_R0_V4.md).

    python -I runner.py run --native-build BUILD --work WORK --source-commit SHA --inputs {dev,qual} --out RECORDS

``run`` deploys the testing RuntimeCore build of one operator build once (``elpis.runtime.deployment``: every library
pinned and admitted from sealed bytes), then, for every family, point, lineage and run of the frozen ``plan.json``,
starts a child process (``child``) that drives a real lineage through the public runtime API up to the point, parks
there (the durability worker, or the control operation's caller) and reports what it holds; the parent sends it
``SIGKILL``. A fresh process (``restart``) then observes what restart decides over the same directories, performs the
point's pre-registered operator action, resumes and continues. The parent writes one JSON record per run; it decides
nothing (``check.py`` evaluates the pre-registered gates from the records alone).

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
import signal
import subprocess
import sys
import threading
import time

RECORD_SCHEMA = "elpis.k1-recovery-r0-v4.record.v1"
HERE = Path(__file__).resolve().parent
GRANT = "k1-recovery-r0-v4: qualification stimulus (TRAINING=NONE SEMANTICS=NONE)"
OBSERVE_GRANT = "k1-recovery-r0-v4: fixture typed evidence (TRAINING=NONE SEMANTICS=NONE)"
# Control operations of the testing build (elpis_runtime_testing_control_fault): op and the parked report.
CONTROL_OP = {"anchor": 1, "import": 2, "discard": 3, "adopt": 4}


def _say(**kw) -> None:
    sys.stdout.write(json.dumps(kw, sort_keys=True) + "\n")
    sys.stdout.flush()


def _hex(b) -> str | None:
    return None if b is None else bytes(b).hex()


# =====================================================================================================================
# The lineage, through the public API only (children and restarts)
# =====================================================================================================================

class Lineage:
    """One run's lineage: its directories, its kind and its deterministic inputs."""

    def __init__(self, spec: dict):
        from elpis.runtime.deployment import Deployment

        self.spec = spec
        self.plan = json.loads(Path(spec["plan"]).read_text())
        self.kind = spec["lineage"]
        self.shape = self.plan["lineages"][self.kind]
        self.run = Path(spec["run_dir"])
        self.seed = spec["seed"]
        self.deployment = Deployment(Path(spec["home"]))
        self.libs = self.deployment.libraries

    # -- configuration -------------------------------------------------------------------------------------------
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
        kind = kind or self.kind
        if kind == "k1":
            return int(self.libs.k1.envelope_bytes(self.shape["dim"], self.shape["width"]))
        return int(self.libs.composite.envelope_bytes(self.composite_config()))

    # -- states --------------------------------------------------------------------------------------------------
    def create(self):
        from elpis.ECS.composite import CompositeState
        from elpis.ECS.k1 import K1State
        from elpis.runtime.deployment import initial_weights

        w = initial_weights(self.shape["dim"], self.shape["width"], self.seed)
        if self.kind == "k1":
            return K1State.create(self.libs.k1, self.shape["dim"], self.shape["width"], w,
                                  max_rows=self.shape["max_rows"])
        return CompositeState.create(self.libs.composite, self.composite_config(), memoryview(w))

    def restore(self, envelope: bytes, kind=None):
        from elpis.ECS.composite import CompositeState
        from elpis.ECS.k1 import K1State

        if (kind or self.kind) == "k1":
            return K1State.restore(self.libs.k1, envelope, max_rows=self.shape["max_rows"])
        return CompositeState.restore(self.libs.composite, self.composite_config(), envelope)

    # -- LEARN i: deterministic numbers; a composite LEARN carries one typed observation ---------------------------
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
        """LEARN i (volatile); returns (identity after, relation key or None)."""
        from elpis.runtime.cognition import LearnAuthority, ObservationAuthority

        authority = LearnAuthority(self.plan["stimulus"]["learning_rate"], GRANT)
        if self.kind == "composite" or type(state).__name__ == "CompositeState":
            observation = self.observation(i)
            result = runtime.run_numeric_learn(
                state, self.stimulus(i), authority, observations=(observation,),
                observation_authority=ObservationAuthority(OBSERVE_GRANT, frozenset({observation.source_authority})))
            key = result.observed[0]
        else:
            runtime.run_numeric_learn(state, self.stimulus(i), authority)
            key = None
        return runtime.live_identity(state)[0], key

    # -- the checkpoint directory's fixed footprint (G6) ---------------------------------------------------------
    @staticmethod
    def footprint(store: Path) -> dict:
        if not store.exists():
            return {}
        return {p.name: {"size": p.stat().st_size, "inode": p.stat().st_ino} for p in sorted(store.iterdir())}


_SLOT_MAGIC, _SLOT_DOMAIN = b"ELPK1CK\x01", b"elpis.k1-checkpoint.slot.v1\0"


def raw_slots(store: Path) -> dict:
    """What the slot files hold on disk, read directly (evidence only: slots never authorize anything): the header's
    identity and generation, and whether the slot verifies (its header checksum and the envelope's own trailer)."""
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

def _caller_keep(lin: Lineage, runtime, state, name: str) -> str:
    """The caller keeps a copy of its state's bytes: release (the volatile work stays live), snapshot, write. The next
    operation rebinds and verifies it."""
    runtime.release(state)
    envelope = state.snapshot()
    (lin.run / "caller").mkdir(exist_ok=True)
    (lin.run / "caller" / name).write_bytes(envelope)
    return hashlib.sha256(envelope).hexdigest()


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
    if point.startswith("step:"):
        step = int(point[5:])
        runtime._core.testing_durable_fault(2, step, arg)
        pending = publish(state, wait=False)
        d3, k3 = lin.learn(runtime, state, 2)       # the HOT lane is not blocked by the park
        _wait_parked(runtime, step)
        out["ticket"] = _ticket(pending)
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
    runtime._core.testing_durable_fault(action)
    ticket = runtime.publish_recoverable(state)
    out["ticket"] = _ticket(ticket)
    out["lane_fault"] = runtime.durability().fault
    d3, k3 = lin.learn(runtime, state, 2)            # HOT continues under the lane fault
    out["D3"], out["keys"] = _hex(d3), out["keys"] + [_hex(k3)]
    out["hot_after_fault"] = True
    out["request_under_fault"] = _refusal(lambda: runtime.publish_recoverable(state))
    runtime.close()
    _say(ready=True, **out)


def child_identity_refused(lin: Lineage) -> None:
    """A composite lineage without a store: anchoring is identity-only; LEARN; an identity-only publication is refused
    (its envelope size is stated only by a provisioned store), and nothing durable changes."""
    runtime = lin.runtime(None)
    state = lin.create()
    out = {"D0": _hex(state.state_digest()), "anchor": runtime.anchor_cognition(state).generation}
    d, k = lin.learn(runtime, state, 0)
    out["D1"], out["keys"] = _hex(d), [_hex(k)]
    out["refusal"] = _refusal(lambda: runtime.publish_identity(state))
    out["recoverable_refusal"] = _refusal(lambda: runtime.publish_recoverable(state))
    out["durable_after"] = _hex(runtime.continuity.snapshot().k1_state_digest)
    runtime.close()
    _say(ready=True, **out)


def _control(lin: Lineage, runtime, op: str, point: int, arg: int, call) -> dict:
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
    out = {"D0": _hex(state.state_digest())}
    out["control"] = _control(lin, runtime, "anchor", point, arg, lambda: runtime.anchor_cognition(state))
    _say(ready=True, **out)
    time.sleep(3600)


def child_reconcile(lin: Lineage, op: str, point: int, arg: int) -> None:
    """Second phase of the discard / adopt families: a fresh process finds the candidate and parks inside the
    reconciliation of exactly it."""
    runtime = lin.runtime(lin.run / "checkpoint")
    r = runtime.recover_k1()
    out = {"found": {"disposition": r.disposition, "authorized": _hex(r.authorized), "candidate": _hex(r.candidate)}}
    call = (lambda: runtime.discard_k1_candidate(r.candidate)) if op == "discard" else \
        (lambda: runtime.adopt_k1_candidate(r.candidate))
    out["control"] = _control(lin, runtime, op, point, arg, call)
    _say(ready=True, **out)
    time.sleep(3600)


def child_import(lin: Lineage, point: int, arg: int) -> None:
    """A K1 lineage (its own K1-sized store) publishes K; the operator provisions a composite store and imports K as
    a composite with an empty memory; the import parks at ``point``."""
    from elpis.ECS.composite import CompositeState

    runtime = lin.runtime(lin.run / "checkpoint")
    state = lin.create()
    out = {"D0": _hex(state.state_digest())}
    runtime.anchor_cognition(state)
    k, _ = lin.learn(runtime, state, 0)
    out["first_ticket"] = _ticket(runtime.publish_recoverable(state))
    runtime.close()
    out["K"] = _hex(k)
    k1_envelope = state.snapshot()
    composite = CompositeState.import_k1(lin.libs.composite, lin.composite_config(), k1_envelope)
    out["C"] = _hex(composite.state_digest())
    runtime = lin.runtime(lin.run / "checkpoint-composite")
    out["control"] = _control(lin, runtime, "import", point, arg, lambda: runtime.import_composite(composite))
    _say(ready=True, **out)
    time.sleep(3600)


def child_random(lin: Lineage, rng_seed: int) -> None:
    """Family random: LEARN repeatedly and request publications at random (some awaited); report every requested
    identity and every acknowledgment as it happens. Killed at a random moment."""
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
                pending.append(t)
        for t in list(pending):
            if t.status != "PENDING":
                pending.remove(t)
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
                       envelope_sha256=None)
            if r.envelope is not None:
                out["envelope_sha256"] = hashlib.sha256(r.envelope).hexdigest()
        return out
    finally:
        runtime.close()
        if store is not None:
            out["slots"] = raw_slots(store)


def _resume(lin: Lineage, store: Path, keys: list, kind=None) -> dict:
    """Restore exactly the RESUMABLE envelope, verify it against continuity, report the resident relation keys, then
    LEARN once and publish recoverably once."""
    runtime = lin.runtime(store)
    try:
        r = runtime.recover_k1()
        out = {"disposition": r.disposition}
        if r.disposition != "RESUMABLE":
            return out
        state = lin.restore(r.envelope, kind)
        out["restored_identity"] = _hex(state.state_digest())
        out["continuity_identity"] = _hex(runtime.continuity.snapshot().k1_state_digest)
        if type(state).__name__ == "CompositeState":
            out["resident"] = {}
            for key in keys:
                if key is not None:
                    slot = state.slot(bytes.fromhex(key))
                    out["resident"][key] = None if slot is None else slot.assert_count
        d, _ = lin.learn(runtime, state, 50)
        t = runtime.publish_recoverable(state)
        out["continued"] = {"identity": _hex(d), "ticket": _ticket(t)}
        return out
    finally:
        runtime.close()


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
    family, run_index = spec["family"], spec["run_index"]
    store = lin.run / "checkpoint"
    child = spec.get("child", {})
    keys = child.get("keys", [])
    out = {}
    if family == "identity_refused":
        out["observed"] = _observe(lin, None)
        return out
    if family == "identity":
        # No store ever held these bytes: restart attaches a freshly provisioned one and finds the authority missing.
        from elpis.runtime.recovery import provision_k1_checkpoint

        out["before"] = _observe(lin, None)
        provision_k1_checkpoint(lin.libs.runtime, store, lin.envelope_bytes())
        out["provisioned"] = Lineage.footprint(store)
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
        out["final_footprint"] = Lineage.footprint(store)
        return out
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
                r = runtime.recover_k1()
                k1_envelope = r.envelope
            finally:
                runtime.close()
            composite = CompositeState.import_k1(lin.libs.composite, lin.composite_config(), k1_envelope)
            runtime = lin.runtime(composite_store)
            try:
                out["reimported"] = {"identity": _hex(composite.state_digest()),
                                     "format": runtime.import_composite(composite).state_format}
            finally:
                runtime.close()
        out["resumed"] = _resume(lin, composite_store, [], kind="composite")
        out["final_footprint"] = Lineage.footprint(store)
        out["final_footprint_composite"] = Lineage.footprint(composite_store)
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
    out["after_action"] = _observe(lin, store) if action else None
    out["resumed"] = _resume(lin, store, keys)
    out["final_footprint"] = Lineage.footprint(store)
    return out


# =====================================================================================================================
# The parent: deploy once, run every pre-registered point, write one record per run
# =====================================================================================================================

def _spawn(role: str, spec: dict) -> subprocess.Popen:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": "0"}
    return subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), role, json.dumps(spec)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=spec["run_dir"])


def _read_lines(proc, timeout: float, until=None) -> tuple[list, bool]:
    """Read JSON lines until ``until(line)`` holds, the process exits, or ``timeout``. Returns (lines, satisfied)."""
    lines, deadline, buf = [], time.monotonic() + timeout, ""
    fd = proc.stdout.fileno()
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return lines, False
        ready, _, _ = select.select([fd], [], [], min(left, 0.5))
        if not ready:
            if proc.poll() is not None:
                return lines, False
            continue
        chunk = os.read(fd, 65536).decode()
        if not chunk:
            return lines, False
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if line.strip():
                lines.append(json.loads(line))
                if until is not None and until(lines[-1]):
                    return lines, True


def _not_ready(proc) -> str:
    """Why a child never reported its point: its exit and stderr, or (still running) it is killed."""
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _kill(proc)
        return "never ready (killed)"
    return f"exit {proc.returncode}: {proc.stderr.read()[-4000:]}"


def _kill(proc) -> int:
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=60)
    return proc.returncode


def _finish(proc, timeout: float) -> tuple[dict | None, str]:
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        return None, "timeout"
    lines = [json.loads(line) for line in out.splitlines() if line.strip()]
    return (lines[-1] if lines and proc.returncode == 0 else None), err[-4000:]


# plan.json "torn_offsets.slot", in order: bytes of the slot (128-byte header + the envelope) written before the death.
_SLOT_TORN = (lambda c: 1, lambda c: 127, lambda c: 128 + c // 2, lambda c: 128 + c - 1)


def _torn(plan: dict, point: str, run_index: int, capacity: int) -> int:
    if point == "step:3":
        return _SLOT_TORN[run_index % 4](capacity)
    if point in ("step:6", "control:4"):
        return plan["torn_offsets"]["record"][run_index % 4]
    return 0


FREEZE_SCHEMA = "elpis.k1-recovery-r0-v4.freeze.v1"
# The deployable libraries a run admits (elpis.runtime.deployment, testing build), relative to the build tree.
LIBRARIES = ("native/runtime/libelpis_runtime_testing.so", "native/ECS/libelpis_ecsg_math.so",
             "native/ECS/libelpis_ecsg_k1.so", "native/ECS/libelpis_ecsg_k1_fms.so", "native/ECS/libelpis_ecs_memory.so",
             "native/ECS/libelpis_ecs_composite.so", "native/runtime/libelpis_hacf_observation.so")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument() -> dict:
    """The digests of the instrument: the plan, this runner, the checker and the specification they implement."""
    spec = HERE.parents[2] / "docs" / "qualification" / "K1_RECOVERY_R0_V4.md"
    return {"plan.json": _sha(HERE / "plan.json"), "runner.py": _sha(HERE / "runner.py"),
            "check.py": _sha(HERE / "check.py"), "specification": _sha(spec) if spec.exists() else None}


def verify_freeze(freeze_path: Path, native_build: Path, source_commit: str) -> dict:
    """Refuse to run anything unless the instrument, the libraries and the source commit are exactly the frozen ones."""
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise SystemExit(f"{freeze_path}: not a {FREEZE_SCHEMA} record")
    problems = []
    if freeze["source_commit"] != source_commit:
        problems.append(f"source commit {source_commit} is not the frozen {freeze['source_commit']}")
    for name, digest in instrument().items():
        if freeze["instrument"].get(name) != digest:
            problems.append(f"{name} differs from the frozen instrument")
    for rel in LIBRARIES:
        if freeze["libraries"].get(rel) != _sha(native_build / rel):
            problems.append(f"{rel} differs from the frozen library")
    if problems:
        raise SystemExit("freeze verification refused the run: " + "; ".join(problems))
    return freeze


def environment(work: Path) -> dict:
    import platform

    def tool(argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            return "unavailable"

    return {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "kernel": os.uname().release, "cpus": os.cpu_count(),
            "work_filesystem": tool(["stat", "-f", "-c", "%T", str(work)]),
            "elpis_package": str(Path(__import__("elpis").__file__).resolve().parent)}


def run(args) -> int:
    from elpis.runtime.deployment import Deployment, init_deployment
    from elpis.runtime.recovery import provision_k1_checkpoint

    native_build = Path(args.native_build).absolute()
    freeze = verify_freeze(Path(args.freeze), native_build, args.source_commit) if args.freeze else None
    if args.inputs == "qual" and freeze is None:
        raise SystemExit("QUAL inputs run only under a verified freeze record (--freeze)")
    plan_path = HERE / "plan.json"
    plan = json.loads(plan_path.read_text())
    inputs = plan["inputs"][args.inputs]
    work = Path(args.work).absolute()
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    home = work / "home"
    init_deployment(home, native_build, testing=True, source_commit=args.source_commit)
    deployment = Deployment(home)
    timeout = plan["bounds"]["child_timeout_seconds"]
    records = open(args.out, "x")
    count = 0
    started = time.monotonic()
    rng = random.Random(inputs["random_seed"])

    def emit(record):
        nonlocal count
        record["schema"] = RECORD_SCHEMA
        records.write(json.dumps(record, sort_keys=True) + "\n")
        records.flush()
        count += 1

    if args.only is not None and args.inputs == "qual":
        raise SystemExit("a QUAL run runs every family")
    families = [f for f in plan["families"] if args.only is None or f in args.only.split(",")]
    for family in families:
        fam = plan["families"][family]
        points = fam.get("points", [{"point": "random", "expect": {}}])
        runs = fam.get("runs_per_lineage", plan["runs_per_point"])
        for kind in fam["lineages"]:
            for entry in points:
                for run_index in range(runs):
                    point = entry["point"]
                    rid = f"{family}/{kind}/{point.replace(':', '-')}/{run_index}"
                    run_dir = work / "runs" / rid.replace("/", "_")
                    run_dir.mkdir(parents=True)
                    seed = inputs["lineage_seed_base"] + count
                    spec = {"plan": str(plan_path), "home": str(home), "run_dir": str(run_dir), "lineage": kind,
                            "seed": seed, "family": family, "point": point, "run_index": run_index}
                    record = {"id": rid, "family": family, "lineage": kind, "point": point, "run_index": run_index,
                              "seed": seed, "expect": entry["expect"]}
                    if fam["store"]:
                        capacity = (deployment.libraries.k1.envelope_bytes(plan["lineages"]["k1"]["dim"],
                                                                           plan["lineages"]["k1"]["width"])
                                    if kind == "k1" else
                                    deployment.libraries.composite.envelope_bytes(_composite_config(plan)))
                        provision_k1_checkpoint(deployment.libraries.runtime, run_dir / "checkpoint", int(capacity))
                        record["provisioned"] = Lineage.footprint(run_dir / "checkpoint")
                    else:
                        capacity = 0
                    if family == "import":
                        cap = deployment.libraries.composite.envelope_bytes(_composite_config(plan))
                        provision_k1_checkpoint(deployment.libraries.runtime, run_dir / "checkpoint-composite",
                                                int(cap))
                        record["provisioned_composite"] = Lineage.footprint(run_dir / "checkpoint-composite")
                    arg = _torn(plan, point, run_index, int(capacity))
                    spec["arg"] = record["arg"] = arg
                    if family == "random":
                        spec["rng_seed"] = record["rng_seed"] = rng.randrange(1 << 63)
                        spec["kill_after"] = record["kill_after"] = rng.randint(1, fam["max_learns"][kind] - 1)
                    record.update(_drive(family, spec, timeout))
                    if record.get("child_error") is None and record.get("reconcile_error") is None:
                        if family == "random" and not record.get("events"):
                            record["restart_error"] = "no events"
                        else:
                            keys = record["keys"] if family == "random" else (record["child"] or {}).get("keys", [])
                            spec["child"] = {"keys": keys}
                            proc = _spawn("restart", spec)
                            result, err = _finish(proc, timeout)
                            record["restart"] = result
                            record["restart_error"] = None if result is not None else err
                    emit(record)
    records.close()
    summary = {"records": count, "seconds": round(time.monotonic() - started, 1), "out": args.out,
               "inputs": args.inputs, "source_commit": args.source_commit, "instrument": instrument(),
               "freeze_sha256": _sha(Path(args.freeze)) if args.freeze else None, "environment": environment(work),
               "catalog_sha256": json.loads((home / "elpis.json").read_text())["catalog_sha256"]}
    with open(args.out + ".run.json", "x") as f:
        f.write(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"records": count, "seconds": summary["seconds"], "out": args.out}))
    return 0


def _composite_config(plan):
    from elpis.ECS.composite import CompositeConfig

    s = plan["lineages"]["composite"]
    return CompositeConfig(s["dim"], s["width"], max_rows=s["max_rows"], memory_dim=s["memory_dim"], banks=s["banks"],
                           slot_capacity=s["slot_capacity"], codebook_capacity=s["codebook_capacity"])


def _drive(family: str, spec: dict, timeout: float) -> dict:
    """Run the child (and, for a reconciliation, its second phase); kill it at its point."""
    out = {"child": None, "child_error": None, "child_signal": None}
    if family in ("uncertain", "identity_refused"):
        proc = _spawn("child", spec)
        result, err = _finish(proc, timeout)
        out["child"], out["child_error"] = result, (None if result is not None else err)
        out["child_signal"] = proc.returncode
        return out
    if family == "random":
        proc = _spawn("child", spec)
        lines, ready = _read_lines(proc, timeout, until=lambda l: l.get("event") == "ready")
        if not ready:
            out["child_error"] = _not_ready(proc)
            return out
        target = spec["kill_after"]           # kill as soon as that many LEARNs were reported
        more, _ = _read_lines(proc, timeout, until=lambda l: l.get("event") == "learn" and l["i"] + 1 >= target)
        out["child_signal"] = _kill(proc) if proc.poll() is None else proc.returncode
        tail, _ = _read_lines(proc, 1.0)
        out["events"] = lines + more + tail
        out["keys"] = [e.get("key") for e in out["events"] if e.get("event") == "learn"]
        return out
    proc = _spawn("child", spec)
    lines, ready = _read_lines(proc, timeout, until=lambda l: l.get("ready"))
    if not ready:
        out["child_error"] = _not_ready(proc)
        return out
    out["child"] = lines[-1]
    out["child_signal"] = _kill(proc)
    if family in ("discard", "adopt"):
        second = dict(spec, phase="reconcile")
        proc = _spawn("child", second)
        lines, ready = _read_lines(proc, timeout, until=lambda l: l.get("ready"))
        if not ready:
            out["reconcile_error"] = _not_ready(proc)
            return out
        out["reconcile"] = lines[-1]
        out["reconcile_signal"] = _kill(proc)
        out["reconcile_error"] = None
    return out


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
    r = sub.add_parser("run")
    r.add_argument("--native-build", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--source-commit", required=True)
    r.add_argument("--inputs", choices=("dev", "qual"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--only", default=None, help="comma-separated families (DEV runs only)")
    r.add_argument("--freeze", default=None, help="the freeze record; required for QUAL inputs")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
