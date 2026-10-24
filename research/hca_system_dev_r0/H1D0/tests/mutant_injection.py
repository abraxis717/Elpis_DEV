"""Implementation-level mutant injection.

For each forbidden semantic change, copy the real source into a disposable
mutant tree, apply the change, and prove that an independent instrument
DETECTS it.  A mutant that is not detected is a real finding: the instrument
is blind to that failure mode.

Detectors are chosen to match the invariant the mutant attacks:
  DISAGREE          oracle vs independent checker must diverge
  ENGINE_DISAGREE   engine output vs oracle gold must diverge
  DAMAGE_DETECTOR   an integrity-checked engine must notice and repair damage
  RX_FIXTURE        a chain that is complete except for an RX first link
  IDENTIFIABILITY   a genuinely underdetermined decision must stay underdetermined
"""
import json, shutil, subprocess, sys, tempfile, os

SRC = os.path.abspath("src")
PY = sys.executable

DISAGREE = """
import sys; sys.path.insert(0, "src")
from generator import build_split
from oracle import gold_for_case
from checker import checker_local_labels
TR, TRT = build_split("TRAIN", 12345, 2)
bad = tot = 0
for w in TR["worlds"]:
    for c in w["cases"]:
        g = gold_for_case(c, TRT[c["case_id"]])
        cl = checker_local_labels(c["events"], c["factors"], c["queries"])
        n = len(c["events"])
        for q in c["queries"]:
            tot += 1
            if g[n][q["id"]] != cl[n][q["id"]]:
                bad += 1
print("DISAGREE", bad, tot)
"""

ENGINE_DISAGREE = """
import sys; sys.path.insert(0, "src")
from generator import build_split
from oracle import gold_for_case
from engines import Incremental, HcaNativeSemantics
TR, TRT = build_split("TRAIN", 12345, 2)
bad = tot = 0
for w in TR["worlds"]:
    for c in w["cases"]:
        g = gold_for_case(c, TRT[c["case_id"]])
        n = len(c["events"])
        eng = Incremental(c)
        b0 = HcaNativeSemantics(c, method="B0")
        for e in c["events"]:
            eng.apply(e)
            b0.apply(e)
        si, sb = eng.snapshot(), b0.snapshot()
        for q in c["queries"]:
            tot += 1
            bad += (si[q["id"]] != g[n][q["id"]]) + (sb[q["id"]] != g[n][q["id"]])
print("DISAGREE", bad, tot)
"""

LEARNED_DISAGREE = """
import sys; sys.path.insert(0, "src")
import numpy as np
from generator import build_split
from oracle import gold_for_case
from learned import FactorNet, edge_arrays
TR, TRT = build_split("TRAIN", 12345, 2)
net = FactorNet(rounds=3, seed=7)
net.t = np.zeros((3, 6)); net.theta = 0.0
bad = tot = 0
for w in TR["worlds"]:
    for c in w["cases"]:
        toks = sorted({v for f in c["factors"] for v in f["roles"].values()}
                      | {q["from"] for q in c["queries"]} | {q["to"] for q in c["queries"]})
        ni = {t: i for i, t in enumerate(toks)}
        g = gold_for_case(c, TRT[c["case_id"]])
        n = len(c["events"])
        for q in c["queries"]:
            e = edge_arrays(c, n, ni, q)
            tot += 1
            bad += (net.predict(len(ni), ni[q["from"]], ni[q["to"]], e)[0] != g[n][q["id"]])
print("DISAGREE", bad, tot)
"""

DAMAGE_DETECTOR = """
import sys, random; sys.path.insert(0, "src")
from generator import build_split
from oracle import gold_for_case
from engines import Incremental
TR, TRT = build_split("TRAIN", 12345, 2)
detected = 0; cases = 0
for w in TR["worlds"]:
    for c in w["cases"]:
        cases += 1
        k = c["damage_prefix"]
        eng = Incremental(c, integrity=True)
        for e in c["events"][:k]:
            eng.apply(e)
        d = eng.damage(random.Random(c["case_id"]), 4)
        for e in c["events"][k:]:
            eng.apply(e)
        if d > 0 and eng.rebuilds > 0:
            detected += 1
print("DISAGREE", detected, cases)
"""

RX_FIXTURE = """
import sys; sys.path.insert(0, "src")
from oracle import answer
factors = [
  {"id":"f0","rel":"RX","roles":{"from":"a","to":"b","scope":"s","kind":"k"},"source":"s1","version":0},
  {"id":"f1","rel":"R1","roles":{"from":"b","to":"c","scope":"s","kind":"k"},"source":"s2","version":0},
  {"id":"f2","rel":"R2","roles":{"from":"c","to":"d","scope":"s","kind":"k"},"source":"s3","version":0}]
events = [{"type":"SOURCE","source":"s1","status":"VALID","epoch":0},
          {"type":"SOURCE","source":"s2","status":"VALID","epoch":0},
          {"type":"SOURCE","source":"s3","status":"VALID","epoch":0},
          {"type":"EMIT","factor":"f0"},{"type":"EMIT","factor":"f1"},{"type":"EMIT","factor":"f2"}]
query = {"id":"q","from":"a","to":"d","scope":"s","kind":"k"}
print("DISAGREE", 1 if answer(events, factors, query, len(events)) != 0 else 0, 1)
"""

IDENTIFIABILITY_FIXTURE = """
import sys; sys.path.insert(0, "src")
from oracle import possible_final_states, answer_from_state
factors = [
  {"id":"g0","rel":"R0","roles":{"from":"a","to":"b","scope":"s","kind":"k"},"source":"s1","version":1},
  {"id":"g1","rel":"R1","roles":{"from":"b","to":"c","scope":"s","kind":"k"},"source":"s2","version":0},
  {"id":"g2","rel":"R2","roles":{"from":"c","to":"d","scope":"s","kind":"k"},"source":"s3","version":0}]
visible = [{"type":"SOURCE","source":"s1","status":"STALE","epoch":0},
           {"type":"SOURCE","source":"s2","status":"VALID","epoch":0},
           {"type":"SOURCE","source":"s3","status":"VALID","epoch":0},
           {"type":"EMIT","factor":"g0"},{"type":"EMIT","factor":"g1"},{"type":"EMIT","factor":"g2"}]
pool = [{"type":"SOURCE","source":"s1","status":"VALID","epoch":1}]
q = {"id":"q","from":"a","to":"d","scope":"s","kind":"k"}
states = possible_final_states(visible, pool)
vals = {answer_from_state(s, factors, q) for s in states}
print("DISAGREE", 1 if len(vals) > 1 else 0, 1)
"""

FIXTURE_SUITE = """
import sys; sys.path.insert(0, "src")
sys.argv = ["fixtures"]
exec(open("tests/test_adversarial_fixtures.py").read())
"""


def fixture_detected(stdout):
    return "FIXTURE VERDICT: FAIL" in stdout


MUTANTS = [
    ("stale_source_accepted", "oracle.py",
     'st[0] == "VALID"', 'st[0] in ("VALID", "STALE")', DISAGREE),
    ("epoch_ignored", "oracle.py",
     'st[1] == f["version"]', 'True', DISAGREE),
    ("reversed_causal_order", "oracle.py",
     'for e in events[:n]:', 'for e in reversed(events[:n]):', DISAGREE),
    ("checker_stale_accepted", "checker.py",
     'status.get(src.get(f["id"], f["source"])) == ("VALID", f["version"])',
     'status.get(src.get(f["id"], f["source"])) is not None', DISAGREE),
    ("incremental_ignores_provenance", "engines.py",
     'self.status.get(self.src[fid]) == ("VALID", f["version"])', 'True',
     ENGINE_DISAGREE),
    ("native_seeds_missing_scope_kind", "engines.py",
     'self.index[(q["scope"], "S")],\n                self.index[(q["kind"], "K")]}',
     'self.index[(q["scope"], "S")]}', ENGINE_DISAGREE),
    ("native_live_ignores_epoch", "engines.py",
     'and self.status.get(self.src[f["id"]]) == ("VALID", f["version"]))',
     'and self.status.get(self.src[f["id"]]) is not None)', ENGINE_DISAGREE),
    ("learned_stale_treated_as_valid", "learned.py",
     'return "STALE" if st[0] == "STALE" else "UNKNOWN"',
     'return "VALID_MATCH"', LEARNED_DISAGREE),
    ("learned_dropped_treated_as_live", "learned.py",
     'return "DROPPED"', 'return "VALID_MATCH"', LEARNED_DISAGREE),
    ("rx_decoy_admitted_in_oracle", "oracle.py",
     'if f0["rel"] != "R0":', 'if f0["rel"] not in ("R0", "RX"):', RX_FIXTURE),
    ("rx_decoy_admitted_in_checker", "checker.py",
     'if f["rel"] == "R0" and f["roles"]["from"] == q["from"]',
     'if f["rel"] in ("R0", "RX") and f["roles"]["from"] == q["from"]', FIXTURE_SUITE),
    ("identifiability_ignores_pending_pool", "oracle.py",
     'for e in pool:\n        by_key.setdefault(_key(e), []).append(e)',
     'for e in pool[:0]:\n        by_key.setdefault(_key(e), []).append(e)',
     FIXTURE_SUITE),
    ("damage_is_a_no_op", "engines.py",
     'def damage(self, rng, units):\n        n = 0',
     'def damage(self, rng, units):\n        return 0\n        n = 0',
     FIXTURE_SUITE),
    ("integrity_check_never_rebuilds", "engines.py",
     'if self._checksum() != self.check:', 'if False:', FIXTURE_SUITE),
]

results = []
for name, fname, old, new, detector in MUTANTS:
    tmp = tempfile.mkdtemp(prefix="mutant_")
    shutil.copytree(SRC, os.path.join(tmp, "src"))
    shutil.copytree(os.path.dirname(SRC) + "/tests", os.path.join(tmp, "tests"))
    path = os.path.join(tmp, "src", fname)
    text = open(path).read()
    if old not in text:
        results.append({"mutant": name, "file": fname, "status": "PATCH_NOT_APPLIED",
                        "note": "anchor not found"})
        shutil.rmtree(tmp)
        continue
    open(path, "w").write(text.replace(old, new))
    try:
        out = subprocess.run([PY, "-c", detector], capture_output=True, text=True,
                             cwd=tmp, timeout=400)
        detail = out.stdout.strip() or "ERROR: " + out.stderr.strip()[:160]
        if detector is FIXTURE_SUITE:
            detected = fixture_detected(out.stdout)
        else:
            detected = detail.startswith("DISAGREE") and int(detail.split()[1]) > 0
    except Exception as e:
        detail = "CRASH: " + str(e)
        detected = True
    results.append({"mutant": name, "file": fname, "occurrences": text.count(old),
                    "detected": detected, "evidence": detail[:120]})
    shutil.rmtree(tmp)

print(json.dumps({"mutants": results,
                  "n_mutants": len(results),
                  "detected": sum(r["detected"] for r in results),
                  "undetected": [r["mutant"] for r in results if not r["detected"]],
                  "verdict": "MUTANT_INJECTION_PASS"
                  if all(r["detected"] for r in results) else "INSTRUMENT_BLIND"},
                 indent=1))
