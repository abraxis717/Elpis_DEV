"""ECS_C Recovery Q0: the runner. It drives the installed testing host through every pre-registered trial and records
what each process printed and what each fresh process observed. It decides nothing: the checker (check.py) derives
every expectation from plan.json and judges the records.

    python -I -B runner.py freeze --prefix PREFIX --work-root ROOT --out FREEZE.json
    python -I -B runner.py run --prefix PREFIX --work ROOT/run --inputs dev|qual [--freeze FREEZE.json] \\
        --out records.jsonl [--families publish,attack] [--runs N]
    python -I -B runner.py seal --freeze FREEZE.json --records records.jsonl --verdict verdict.json --out evidence

The host runs as separate processes with a minimal environment (no PYTHONPATH, no LD_*), exactly as an operator runs
it; nothing here imports Elpis. A QUAL run refuses to start, before it creates anything, unless freeze.py finds the
executing object equal to the freeze record. DEV inputs exist to develop the instrument; their records are not
evidence. SIGKILL keeps the page cache: nothing here bears on power loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freeze as FREEZE  # noqa: E402  (the sibling module; -I keeps the script directory off sys.path)

PLAN = json.loads((HERE / "plan.json").read_text())
RECORD_SCHEMA = "elpis.ecs-c-recovery-q0.record.v1"
RUN_SCHEMA = "elpis.ecs-c-recovery-q0.run.v1"
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
WRONG_DIGEST = "e" * 64


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- the stimulus (plan.json "stimulus") ------------------------------------------------------------------------

def _node(name: str) -> str:
    return sha256_hex(b"elpis.ecs-c-recovery-q0.node:" + name.encode())


def learn(turn: int) -> dict:
    s = PLAN["stimulus"]["learn"]
    x = [[((i * 7 + j * 3) % 11 - 5) / 16.0 for j in range(6)] for i in range(4)]
    return {"op": "learn", "x": x, "y": s["y"], "schedule": s["schedule"], "rate": s["rate"], "grant": s["grant"]}


def evidence(turn: int) -> dict:
    s = PLAN["stimulus"]["evidence"]
    participants = [{"node": _node(name), "role": role} for name, role in zip(("subject", "object", "role"), s["roles"])]
    item = {"relation": s["relation"], "participants": participants, "authority": s["authority"],
            "polarity": s["polarity"], "tick": turn + 1,
            "provenance": sha256_hex(b"elpis.ecs-c-recovery-q0.provenance:" + str(turn).encode())}
    return {"op": "evidence", "items": [item], "grant": s["grant"], "sources": s["sources"]}


def turn_ops(turn: int) -> list:
    return [learn(turn), evidence(turn)]


# ---- processes --------------------------------------------------------------------------------------------------

class Host:
    """The installed testing host of one prefix."""

    def __init__(self, prefix: Path):
        self.binary = prefix / "bin" / "elpis_testing"
        self.libraries = prefix / "lib" / "elpis"
        self.timeout = PLAN["bounds"]["process_timeout_seconds"]

    def argv(self, home: Path, *args) -> list:
        return [str(self.binary), "--home", str(home), *map(str, args)]

    def run(self, home: Path, *args, script=None) -> dict:
        """One command (or one session over ``script``): its exit status and every line it printed."""
        cmd = self.argv(home, *args)
        text = None
        if script is not None:
            cmd += ["session", "--script", "-"]
            text = "".join(json.dumps(op) + "\n" for op in script)
        p = subprocess.run(cmd, input=text, capture_output=True, text=True, env=ENV, cwd="/", timeout=self.timeout)
        return {"argv": [Path(cmd[0]).name, *cmd[1:]], "rc": p.returncode, "stderr": p.stderr[-2000:],
                "lines": _lines(p.stdout)}

    def killed(self, home: Path, args: list, *, script=None, until=None, after_lines=None, work: Path) -> dict:
        """A child killed with SIGKILL: when it prints a line satisfying ``until`` (a pause or a park), or as soon as
        ``after_lines`` lines have been read. Lines printed before the death are read to the end and kept."""
        cmd = self.argv(home, *args)
        if script is not None:
            path = work / "child.jsonl"
            path.write_text("".join(json.dumps(op) + "\n" for op in script))
            cmd += ["session", "--script", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=ENV, cwd="/")
        lines, reached = [], False
        try:
            for raw in proc.stdout:
                lines.append(_line(raw))
                if (until and until(lines[-1])) or (after_lines and len(lines) >= after_lines):
                    reached = True
                    break
        finally:
            proc.send_signal(signal.SIGKILL)
            rest, err = proc.communicate(timeout=PLAN["bounds"]["child_timeout_seconds"])
        lines += _lines(rest)
        return {"argv": [Path(cmd[0]).name, *cmd[1:]], "rc": proc.returncode, "reached": reached,
                "signal": -proc.returncode if proc.returncode < 0 else 0, "stderr": err[-2000:], "lines": lines}


def _line(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"unparsed": raw[:2000]}


def _lines(text: str) -> list:
    return [_line(raw) for raw in text.splitlines() if raw.strip()]


def files(home: Path) -> dict:
    """Every file of the home: its size and inode (the persistence footprint)."""
    return {p.relative_to(home).as_posix(): [p.stat().st_size, p.stat().st_ino]
            for p in sorted(home.rglob("*")) if p.is_file()}


# ---- one trial's pieces -----------------------------------------------------------------------------------------

def deploy(host: Host, work: Path, seed: int, *, create=True, source=True) -> dict:
    home = work / "home"
    out = {"init": host.run(home, "init", "--libraries", host.libraries, *PLAN["deployment"]["init"])}
    if create:
        out["create"] = host.run(home, "create", "--seed", seed)
    if source:
        out["source"] = host.run(home, script=[PLAN["source"]])
        out["baseline"] = files(home)                      # the provisioned file set, before any trial
    return out


def observe(host: Host, home: Path) -> dict:
    """What a fresh process finds: the authority alone first (its own open's report), then the runtime's status."""
    return {"authority": host.run(home, "authority"), "status": host.run(home, "status"), "files": files(home)}


def candidate_action(host: Host, home: Path, seen: dict, action: str) -> dict:
    """The pre-registered checkpoint reconciliation of the candidate the status reports (if any)."""
    status = _first(seen["status"])
    recovery = (status or {}).get("recovery") or {}
    if recovery.get("disposition") != "CANDIDATE_UNRESOLVED":
        return {"taken": None}
    # A recoverable publication over the unresolved candidate is attempted first (it must be refused).
    probe = host.run(home, script=[{"op": "publish"}])
    out = host.run(home, script=[{"op": "reconcile", "action": action, "candidate": recovery["candidate"]},
                                 {"op": "live"}])
    return {"taken": action, "candidate": recovery["candidate"], "probe": probe, "result": out}


def continuation(host: Host, home: Path, turn: int) -> dict:
    published = host.run(home, script=[*turn_ops(turn), {"op": "publish"}])
    authority = host.run(home, "authority")                # alone: the authority's own open and counts
    resumed = host.run(home, script=[{"op": "live"}])
    return {"published": published, "authority": authority, "resumed": resumed, "files": files(home)}


def _first(run: dict):
    lines = run.get("lines") or []
    return lines[0] if lines else None


def parity(run: int) -> str:
    return "adopt" if run % 2 == 0 else "discard"


# ---- families ----------------------------------------------------------------------------------------------------

def _publish_child(host, home, work, step):
    script = [*turn_ops(0), {"op": "publish"}, *turn_ops(1),
              {"op": "testing_durable_fault", "action": 2, "step": step, "arg": 100},
              {"op": "publish", "wait": False}, {"op": "testing_wait_parked", "parked": step}, {"op": "testing_pause"}]
    return host.killed(home, [], script=script, until=lambda op: op.get("op") == "testing_pause", work=work)


def trial_publish(host, work, t):
    home = work / "home"
    step = int(t["point"].split(":")[1])
    out = {"deploy": deploy(host, work, t["seed"]), "child": _publish_child(host, home, work, step)}
    out["observe"] = observe(host, home)
    out["action"] = candidate_action(host, home, out["observe"], parity(t["run"]))
    out["continuation"] = continuation(host, home, 2)
    return out


def trial_failure(host, work, t):
    home = work / "home"
    plan = int(t["point"].split(":")[1])
    out = {"deploy": deploy(host, work, t["seed"])}
    out["child"] = host.run(home, script=[*turn_ops(0), {"op": "publish"}, *turn_ops(1),
                                          {"op": "testing_durable_fault", "action": plan, "step": 0, "arg": 64},
                                          {"op": "publish"}])
    out["observe"] = observe(host, home)
    out["action"] = candidate_action(host, home, out["observe"], parity(t["run"]))
    out["continuation"] = continuation(host, home, 2)
    return out


def trial_anchor(host, work, t):
    home = work / "home"
    point = int(t["point"].split(":")[1])
    out = {"deploy": deploy(host, work, t["seed"], create=False, source=False)}
    out["child"] = host.killed(home, ["create", "--seed", t["seed"], "--testing-anchor-park", point,
                                      "--testing-fault-arg", 100],
                               until=lambda op: op.get("op") == "testing_parked", work=work)
    out["observe"] = observe(host, home)
    out["action"] = {"taken": "create", "result": host.run(home, "create", "--seed", t["seed"]),
                     "other_seed": host.run(home, "create", "--seed", t["seed"] + 1)}
    out["source"] = host.run(home, script=[PLAN["source"]])
    out["baseline"] = files(home)
    out["continuation"] = continuation(host, home, 0)
    return out


def _control_trial(host, work, t, kind, action):
    home = work / "home"
    point = int(t["point"].split(":")[1])
    out = {"deploy": deploy(host, work, t["seed"]), "child": _publish_child(host, home, work, 5)}
    out["observe_candidate"] = observe(host, home)
    candidate = ((_first(out["observe_candidate"]["status"]) or {}).get("recovery") or {}).get("candidate")
    out["control"] = host.killed(home, [], script=[
        {"op": "testing_control_fault", "kind": kind, "action": 2, "point": point, "arg": 100},
        {"op": "reconcile", "action": action, "candidate": candidate}],
        until=lambda op: op.get("op") == "testing_parked", work=work)
    out["observe"] = observe(host, home)
    out["action"] = candidate_action(host, home, out["observe"], action)
    out["continuation"] = continuation(host, home, 2)
    return out


def trial_discard(host, work, t):
    return _control_trial(host, work, t, 3, "discard")


def trial_adopt(host, work, t):
    return _control_trial(host, work, t, 4, "adopt")


def trial_random(host, work, t):
    home = work / "home"
    out = {"deploy": deploy(host, work, t["seed"]), "random": t["random"]}
    script = []
    for turn, publish in enumerate(t["random"]["publish"]):
        script += turn_ops(turn)
        if publish:
            script.append({"op": "publish"})
    out["child"] = host.killed(home, [], script=script, after_lines=t["random"]["kill_after"], work=work)
    out["observe"] = observe(host, home)
    out["action"] = candidate_action(host, home, out["observe"], parity(t["run"]))
    out["continuation"] = continuation(host, home, len(t["random"]["publish"]))
    return out


# -- attacks: the bytes of the state directory, rewritten in place with every process closed --------------------

def _slots(home: Path) -> list:
    return [home / "state" / f"authority.{c}" for c in "ab"]


def _rewrite(path: Path, data: bytes) -> None:
    """In place, under the same inode: the T1 adversary's write (no rename, no new file)."""
    with open(path, "r+b") as f:
        f.write(data)
        f.truncate(len(data))


def _gen(data: bytes) -> int:
    return int.from_bytes(data[32:40], "little") if len(data) >= 40 else 0


def _reseal(data: bytearray) -> None:
    body = len(data) - 32
    data[body:] = hashlib.sha256(b"elpis.ecs_c.authority.record.v1\0" + bytes(data[:body])).digest()


def _restate(data: bytearray) -> None:
    capacity = int.from_bytes(data[16:20], "little")
    state = bytes(data[256:len(data) - 576])
    data[176:208] = hashlib.sha256(b"elpis.ecs_c.authority.state.v1\0" + bytes(data[48:80]) + bytes(data[32:40]) +
                                   capacity.to_bytes(4, "little") + state).digest()


def attack(home: Path, spec: dict, saved: dict) -> dict:
    """Apply one registered attack (plan.json "attack_layout"); returns what was done, by digests."""
    paths = _slots(home)
    raw = [p.read_bytes() for p in paths]
    newest = 0 if _gen(raw[0]) >= _gen(raw[1]) else 1
    other = 1 - newest
    kind = spec["kind"]
    done = {"kind": kind, "newest": "ab"[newest], "before": [sha256_hex(r) for r in raw]}
    if kind == "copy_back":
        for p, data in zip(paths, saved[spec["saved_after"]]):
            _rewrite(p, data)
    elif kind == "flip":
        data = bytearray(raw[newest])
        data[spec["offset"]] ^= 0xFF
        _rewrite(paths[newest], bytes(data))
    elif kind == "shorten":
        _rewrite(paths[newest], raw[newest][:spec["length"]])
    elif kind == "shorten_other":
        _rewrite(paths[other], raw[other][:spec["length"]])
    elif kind in ("forge_generation", "substitute"):
        data = bytearray(raw[newest])
        data[32:40] = (_gen(raw[newest]) + spec["add"]).to_bytes(8, "little")
        _reseal(data)
        _rewrite(paths[newest if kind == "forge_generation" else other], bytes(data))
    elif kind == "forge_evolution":
        data = bytearray(raw[newest])
        data[256 + 56:256 + 64] = spec["revision"].to_bytes(8, "little")
        data[256 + 64:256 + 96] = hashlib.sha256(spec["head_label"].encode()).digest()
        _restate(data)
        _reseal(data)
        _rewrite(paths[newest], bytes(data))
    elif kind == "swap":
        _rewrite(paths[0], raw[1])
        _rewrite(paths[1], raw[0])
    elif kind == "duplicate":
        _rewrite(paths[other], raw[newest])
    elif kind == "witness_lost":
        for c in "ab":
            _rewrite(home / "authority" / f"witness.{c}", b"")
    else:
        raise ValueError(kind)
    done["after"] = [sha256_hex(p.read_bytes()) for p in paths]
    return done


def _history(host, home, out):
    """Six turns, each with its own process's publication (D1..D6); both record slots saved after each."""
    saved, published = {}, []
    for k in range(1, 7):
        published.append(host.run(home, script=[*turn_ops(k - 1), {"op": "publish"}]))
        saved[k] = [p.read_bytes() for p in _slots(home)]
    out["history"] = published
    out["before_attack"] = host.run(home, "authority")
    return saved


def trial_attack(host, work, t):
    home = work / "home"
    out = {"deploy": deploy(host, work, t["seed"])}
    saved = _history(host, home, out)
    out["attack"] = attack(home, t["spec"]["attack"], saved)
    out["observe"] = observe(host, home)
    reconciliation = (_first(out["observe"]["authority"]) or {}).get("reconciliation") or {}
    steps = []
    if t["run"] % 2 == 1:
        steps.append(host.run(home, script=[{"op": "authority_reconcile", "kind": reconciliation.get("kind"),
                                             "digest": WRONG_DIGEST}]))
        steps.append(host.run(home, "authority"))          # the wrong one changed nothing
    steps.append(host.run(home, script=[{"op": "authority_reconcile", "kind": reconciliation.get("kind"),
                                         "digest": reconciliation.get("digest")}]))
    out["reconcile"] = steps
    out["after_reconcile"] = observe(host, home)
    out["action"] = candidate_action(host, home, out["after_reconcile"], parity(t["run"]))
    recovery = ((_first(out["after_reconcile"]["status"]) or {}).get("recovery") or {}).get("disposition")
    if recovery == "CHECKPOINT_MISSING":
        out["continuation"] = None                          # no operator action exists: recorded as it is
        out["final"] = {"authority": host.run(home, "authority"), "status": host.run(home, "status"),
                        "learn": host.run(home, script=[learn(6)]), "files": files(home)}
    else:
        out["continuation"] = continuation(host, home, 6)
    return out


def trial_rearrangement(host, work, t):
    home = work / "home"
    out = {"deploy": deploy(host, work, t["seed"])}
    saved = _history(host, home, out)
    out["attack"] = attack(home, t["spec"]["attack"], saved)
    out["observe"] = observe(host, home)
    out["continuation"] = continuation(host, home, 6)
    return out


trial_witness = trial_attack


def trial_bounded(host, work, t):
    home = work / "home"
    out = {"deploy": deploy(host, work, t["seed"]), "marks": []}
    block = PLAN["families"]["bounded"]["block"]
    turn = 0
    for mark in PLAN["families"]["bounded"]["publications"]:
        while turn < mark:
            n = min(block, mark - turn)
            ops = []
            for k in range(turn, turn + n):
                ops += [*turn_ops(k), {"op": "publish"}]
            run = host.run(home, script=ops)
            failed = [o for o in run["lines"] if not o.get("ok")]
            if run["rc"] or failed:
                out["marks"].append({"publications": turn, "failed": run})
                return out
            turn += n
        out["marks"].append({"publications": mark, "authority": host.run(home, "authority"), "files": files(home)})
    return out


TRIALS = {"publish": trial_publish, "failure": trial_failure, "anchor": trial_anchor, "discard": trial_discard,
          "adopt": trial_adopt, "random": trial_random, "attack": trial_attack,
          "rearrangement": trial_rearrangement, "witness": trial_witness, "bounded": trial_bounded}


# ---- the enumeration (plan.json "coordinates") -------------------------------------------------------------------

def enumerate_trials(inputs: str, families=None, runs=None) -> list:
    """Every trial in plan order with its index, seed and (random family) draws. Restricting families or runs keeps
    each trial's index, seed and draws: a subset is a subset of the same enumeration."""
    base = PLAN["inputs"][inputs]["lineage_seed_base"]
    rng = random.Random(PLAN["inputs"][inputs]["random_seed"])
    out, index = [], 0
    for family, f in PLAN["families"].items():
        if family == "random":
            points = [{"point": "random"}]
            n = f["runs"]
        elif family == "bounded":
            points = [{"point": "bounded"}]
            n = f["runs"]
        else:
            points = f["points"]
            n = PLAN["runs_per_point"]
        for p in points:
            for run in range(n):
                t = {"index": index, "family": family, "point": p["point"], "run": run, "seed": base + index,
                     "spec": p}
                if family == "random":
                    turns = rng.randint(1, f["max_turns"])
                    publish = [rng.random() < f["publish_probability"] for _ in range(turns)]
                    lines = 2 * turns + sum(publish)
                    t["random"] = {"publish": publish, "kill_after": rng.randint(1, lines - 1) if lines > 1 else 1}
                index += 1
                if (families is None or family in families) and (runs is None or run < runs):
                    out.append(t)
    return out


# ---- commands ----------------------------------------------------------------------------------------------------

def cmd_run(args) -> int:
    prefix, work = Path(args.prefix).resolve(), Path(args.work).resolve()
    if args.inputs == "qual":
        if not args.freeze:
            raise SystemExit("a QUAL run needs --freeze")
        frozen = json.loads(Path(args.freeze).read_text())
        problems = FREEZE.compare(frozen, FREEZE.observe(prefix=prefix, work_root=work.parent))
        if problems:
            raise SystemExit("refused, nothing was created: " + "; ".join(problems))
        if args.families or args.runs is not None:
            raise SystemExit("a QUAL run runs the whole plan")
    work.mkdir(mode=0o700)
    host = Host(prefix)
    families = set(args.families.split(",")) if args.families else None
    with open(args.out, "x") as sink:
        sink.write(json.dumps({"schema": RUN_SCHEMA, "inputs": args.inputs, "prefix": str(prefix),
                               "families": sorted(families) if families else None, "runs": args.runs}) + "\n")
        for t in enumerate_trials(args.inputs, families, args.runs):
            trial_dir = work / f"t{t['index']:04d}"
            trial_dir.mkdir(mode=0o700)
            try:
                observed = TRIALS[t["family"]](host, trial_dir, t)
                error = None
            except (subprocess.TimeoutExpired, OSError, KeyError, ValueError, TypeError) as exc:
                observed, error = None, f"{type(exc).__name__}: {exc}"
            record = {"schema": RECORD_SCHEMA, **{k: t[k] for k in ("index", "family", "point", "run", "seed")},
                      "random": t.get("random"), "observed": observed, "instrument_error": error}
            sink.write(json.dumps(record, sort_keys=True) + "\n")
            sink.flush()
            shutil.rmtree(trial_dir)                     # each trial's home is removed once recorded
            print(f"{t['index']:4d} {t['family']:13s} {t['point']:18s} run {t['run']}"
                  f"{'  INSTRUMENT ERROR ' + error if error else ''}", flush=True)
    return 0


def cmd_freeze(args) -> int:
    record = FREEZE.record(prefix=Path(args.prefix).resolve(), work_root=Path(args.work_root).resolve())
    with open(args.out, "x") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    return 0


def cmd_seal(args) -> int:
    out = Path(args.out)
    out.mkdir()
    for name, src in (("FREEZE.json", args.freeze), ("records.jsonl", args.records), ("verdict.json", args.verdict)):
        shutil.copyfile(src, out / name)
    sums = "".join(f"{sha256_hex((out / n).read_bytes())}  {n}\n"
                   for n in ("FREEZE.json", "records.jsonl", "verdict.json"))
    (out / "SHA256SUMS").write_text(sums)
    for p in out.iterdir():
        p.chmod(0o444)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--prefix", required=True)
    f.add_argument("--work-root", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prefix", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--inputs", choices=("dev", "qual"), required=True)
    r.add_argument("--freeze")
    r.add_argument("--out", required=True)
    r.add_argument("--families")
    r.add_argument("--runs", type=int)
    s = sub.add_parser("seal")
    for name in ("--freeze", "--records", "--verdict", "--out"):
        s.add_argument(name, required=True)
    args = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "run": cmd_run, "seal": cmd_seal}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
