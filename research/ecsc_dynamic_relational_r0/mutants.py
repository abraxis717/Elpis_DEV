"""Mutation sensitivity of the ECS_C relational view (ECSC_DYNAMIC_RELATIONAL_DEV_R0).

Each mutant is one targeted change of the law in native/ECS_C/relational/src/ecs_c_relational.c, applied to a copy
outside the repository (the repository is never written). Each is compiled into both suites and judged:

  unit  native/ECS_C/relational/tests/test_ecs_c_relational.c  (contract attacks + the differential against the
        exact rebuild by a fresh real ECS_C memory)
  e2e   native/ECS_C/relational/tests/test_ecs_c_relational_authority.c  (the real ECS_C authority)

A mutant is KILLED when either suite exits nonzero (an assertion, never a caught exception); SURVIVED otherwise.
The unmutated source must pass both (the control). Usage:

  python3 -I mutants.py REPO BUILD_DIR WORK_DIR OUT_JSON
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

SRC = "native/ECS_C/relational/src/ecs_c_relational.c"

# (id, what it breaks, exact text, replacement). Each text occurs exactly once in the source.
MUTANTS = [
    ("M01", "liveness ignores the binding's version",
     "s->lifecycle == ELPIS_ECS_C_ACTIVE && s->version == c->version &&",
     "s->lifecycle == ELPIS_ECS_C_ACTIVE && 1 &&"),
    ("M02", "liveness ignores the binding's scope",
     "s->version == c->version &&\n           !memcmp(s->scope, c->scope, 32);",
     "s->version == c->version &&\n           1;"),
    ("M03", "a DORMANT source counts as live",
     "s->lifecycle == ELPIS_ECS_C_ACTIVE && s->version == c->version &&",
     "s->lifecycle != ELPIS_ECS_C_TERMINATED && s->version == c->version &&"),
    ("M04", "a superseded version is never retired",
     "|| s->version > c->version;", "|| 0;"),
    ("M05", "TERMINATED does not retire",
     "return !s->used || s->lifecycle == ELPIS_ECS_C_TERMINATED ||", "return !s->used || 0 ||"),
    ("M06", "admission skips the version (SOURCE_STALE)",
     "if (version != e->version) return ELPIS_ECS_C_REL_SOURCE_STALE;", ""),
    ("M07", "admission skips the scope (SCOPE_CONFLICT)",
     "if (memcmp(scope, e->scope, 32)) return ELPIS_ECS_C_REL_SCOPE_CONFLICT;", ""),
    ("M08", "admission skips the lifecycle",
     "if (e->lifecycle != ELPIS_ECS_C_ACTIVE) return ELPIS_ECS_C_REL_LIFECYCLE;", ""),
    ("M09", "admission skips the 32-bit source_authority binding",
     "if (o->source_authority != v->ents[s].source_authority) return refuse(v, ELPIS_ECS_C_REL_BINDING_MISMATCH);", ""),
    ("M10", "admission skips the clock",
     "if (o->tick < v->clock || v->seq", "if (0 || v->seq"),
    ("M11", "sync accepts a decreasing source version",
     "if (e->source_version < s->version) return 0;", ""),
    ("M12", "sync accepts TERMINATED -> live again",
     "if (s->lifecycle == ELPIS_ECS_C_TERMINATED && e->lifecycle != ELPIS_ECS_C_TERMINATED) return 0;", ""),
    ("M13", "sync accepts a vanished row without a reclamation",
     "if (vanished > t->reclaimed_count - v->reclaimed_count) return refuse(v, ELPIS_ECS_C_REL_REGRESSION);", ""),
    ("M14", "sync accepts another lineage",
     "if (memcmp(t->lineage, v->lineage, 32)) return refuse(v, ELPIS_ECS_C_REL_LINEAGE);", ""),
    ("M15", "sync does not see a version change as a binding change",
     "e->version != rows[i].source_version || memcmp", "0 || memcmp"),
    ("M16", "an equal generation is a no-op without comparing",
     "int same = !vanished && !inserts && !memcmp(t->state_root, v->state_root, 32) &&",
     "int same = 1 || (!vanished && !inserts && !memcmp(t->state_root, v->state_root, 32)) ||"),
    ("M17", "the aggregate's latest observation chosen wrongly",
     "c->last_seq > out->last_seq", "c->last_seq < out->last_seq"),
    ("M18", "the aggregate's first observation chosen wrongly",
     "c->first_seq < out->first_seq", "c->first_seq > out->first_seq"),
    ("M19", "scoped reads ignore the scope",
     "if (!live || (scope && memcmp(c->scope, scope, 32))) continue;", "if (!live) continue;"),
    ("M20", "aggregation counts suspended contributions",
     "const int live = by_table ? live_under(&v->ents[c->source], c) : (int)c->live;",
     "const int live = by_table ? live_under(&v->ents[c->source], c) : 1;"),
    ("M21", "LOOKUP answers a relation with zero net evidence",
     "if (net(&g) > 0 && (best == NONE", "if (net(&g) >= 0 && (best == NONE"),
    ("M22", "LOOKUP filler tie order reversed",
     "if (c) return c < 0;", "if (c) return c > 0;"),
    ("M23", "LOOKUP counts relations with no live evidence",
     "if (!g.live_count) continue; /* not in the live view */", ""),
    ("M24", "export is not ordered by key",
     "heapsort_u32(v->scratch, n, rel_key_less, v->rels, &w.sorted);", ""),
    ("M25", "damage is sealed into a recomputed aggregate",
     "v->rels[r].check = ~rel_check(r, &v->rels[r]);\n        return;", "(void)0;"),
    ("M26", "a damaged contribution is resealed by a flip",
     "if (sound) seal_con(v, c);", "(void)sound; seal_con(v, c);"),
    ("M27", "admission merges another version's contribution",
     "v->cons[i].source == s && v->cons[i].version == version", "v->cons[i].source == s"),
    ("M28", "a coordinate check ignores the state root",
     "c->generation == v->generation && !memcmp(c->state_root, v->state_root, 32);",
     "c->generation == v->generation;"),
    ("M29", "pinned reads ignore the caller's coordinate",
     "if (expected && !coordinate_ok(v, expected)) return ELPIS_ECS_C_REL_STALE_VIEW;", ""),
    ("M30", "the relation check does not cover the aggregate",
     "return agg_hash(h, &r->live);", "return h;"),
    ("M31", "verify does not compare aggregates",
     "memcmp(&a, &rr->live, sizeof(a)) ||", ""),
    ("M32", "the identity omits the assert counts",
     "put32(&h, k->assert_count);", ""),
    ("M33", "a flipped contribution's relation is not recomputed",
     "if (sound) seal_con(v, c);\n                    mark_dirty(v, k->relation, dirty, &nd);",
     "if (sound) seal_con(v, c);"),
    ("M34", "retirement does not recompute the relation",
     "    mark_dirty(v, c->relation, dirty, nd);\n    memset(c, 0, sizeof(*c));",
     "    memset(c, 0, sizeof(*c));"),
    ("M35", "participants are not canonicalized",
     "canonical(o, nr->participants);",
     "memcpy(nr->participants, o->participants, sizeof(nr->participants));"),
    ("M36", "a non-LOOKUP request is accepted",
     "q->channel != ELPIS_ECS_MEM_LOOKUP) return 0;", "q->channel > ELPIS_ECS_MEM_SLOW) return 0;"),
    ("M37", "a founder may vanish",
     "if (v->ents[s].founding_index < 2) return refuse(v, ELPIS_ECS_C_REL_REGRESSION);", ""),
    ("M38", "rebuild ignores the binding invariant",
     "if (!con_sound(v, c) || !binding_held(&v->ents[v->cons[c].source], &v->cons[c]))",
     "if (!con_sound(v, c))"),
    ("M39", "verify ignores the binding invariant",
     "(uint32_t)live_under(e, k) != k->live || !binding_held(e, k)) ++bad;",
     "(uint32_t)live_under(e, k) != k->live) ++bad;"),
]

# Predicted equivalent (stated before the run): the binding invariant (binding_held) holds for every retained
# contribution in every reachable state (sync releases RETIRED evidence; admission is at the current binding only),
# so these terms are redundant there. verify counts and rebuild refuses any state that violates it.
EQUIVALENT = {
    "M01": "a retained contribution's version is its source's current version (older versions are retired at sync)",
    "M02": "scope changes only with a version, so a retained contribution's scope is its source's current scope",
    "M27": "a source's retained contributions are all at its current version, the only version admission accepts",
}


def run(cmd, cwd=None, timeout=900):
    t = time.monotonic()
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    return p.returncode, p.stdout, time.monotonic() - t


def build_and_judge(repo, build, work, source_path, tag):
    inc = [f"-I{repo}/native/ECS_C/relational/include", f"-I{repo}/native/ECS_C/authority/include",
           f"-I{repo}/native/ECS_C/memory/include", f"-I{repo}/native/substrate/include"]
    common = ["gcc", "-std=c11", "-O2", "-UNDEBUG", "-w"] + inc
    libs = [f"{repo}/native/ECS_C/memory/src/ecs_memory.c", f"{repo}/native/substrate/src/sha256.c"]
    unit = os.path.join(work, f"unit_{tag}")
    e2e = os.path.join(work, f"e2e_{tag}")
    rc, out, _ = run(common + ["-DELPIS_ECS_C_REL_TESTING", f"{repo}/native/ECS_C/relational/tests/test_ecs_c_relational.c",
                               source_path] + libs + ["-o", unit])
    if rc:
        return {"compiled": False, "log": out[-400:]}
    auth = f"{build}/native/ECS_C/authority/libelpis_ecs_c_authority.so"
    rc, out, _ = run(common + [f"{repo}/native/ECS_C/relational/tests/test_ecs_c_relational_authority.c", source_path]
                     + libs + [auth, f"-Wl,-rpath,{os.path.dirname(auth)}", "-o", e2e])
    if rc:
        return {"compiled": False, "log": out[-400:]}
    result = {"compiled": True}
    for suite, cmd in (("unit", [unit]), ("e2e", [e2e, os.path.join(work, f"scratch_{tag}")])):
        try:
            rc, out, secs = run(cmd, cwd=work, timeout=600)
        except subprocess.TimeoutExpired:
            rc, out, secs = -9, "TIMEOUT", 600.0
        lines = [line for line in out.splitlines() if "CHECK failed" in line or "mismatch" in line or "ERROR" in line]
        result[suite] = {"rc": rc, "seconds": round(secs, 2), "first_failure": lines[0] if lines else (
            out.strip().splitlines()[-1] if rc and out.strip() else "")}
    return result


def main():
    repo, build, work, out_json = (os.path.abspath(a) for a in sys.argv[1:5])
    os.makedirs(work, exist_ok=True)
    original = open(os.path.join(repo, SRC), encoding="utf-8").read()
    report = {"schema": "elpis.ecsc-dynamic-relational-r0.mutants.v1",
              "source": SRC, "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
              "judges": ["test_ecs_c_relational (unit + differential)", "test_ecs_c_relational_authority (e2e)"],
              "mutants": []}
    control_path = os.path.join(work, "control.c")
    open(control_path, "w", encoding="utf-8").write(original)
    control = build_and_judge(repo, build, work, control_path, "control")
    report["control"] = control
    assert control["compiled"] and control["unit"]["rc"] == 0 and control["e2e"]["rc"] == 0, control
    for mid, what, old, new in MUTANTS:
        assert original.count(old) == 1, (mid, original.count(old))
        path = os.path.join(work, f"{mid}.c")
        open(path, "w", encoding="utf-8").write(original.replace(old, new))
        r = build_and_judge(repo, build, work, path, mid)
        killed_by = [s for s in ("unit", "e2e") if r.get(s, {}).get("rc", 0) != 0]
        r.update({"id": mid, "breaks": what, "killed": bool(killed_by) or not r["compiled"], "killed_by": killed_by,
                  "equivalent_predicted": EQUIVALENT.get(mid)})
        report["mutants"].append(r)
        print(f"{mid} {'KILLED' if r['killed'] else 'SURVIVED'} by {killed_by or '-'}: {what}", flush=True)
    report["killed"] = sum(m["killed"] for m in report["mutants"])
    report["total"] = len(report["mutants"])
    report["survived_not_predicted_equivalent"] = [m["id"] for m in report["mutants"]
                                                   if not m["killed"] and not m["equivalent_predicted"]]
    report["predicted_equivalent_killed"] = [m["id"] for m in report["mutants"]
                                             if m["killed"] and m["equivalent_predicted"]]
    open(out_json, "w", encoding="utf-8").write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{report['killed']}/{report['total']} killed; survivors not predicted equivalent: "
          f"{report['survived_not_predicted_equivalent']}")


if __name__ == "__main__":
    main()
