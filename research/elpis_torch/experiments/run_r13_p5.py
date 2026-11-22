"""ELPIS_TORCH_R13 P5: representation and distributed competence of the Geodome substrate. RESEARCH_ONLY.

The protocol is frozen in ``results/r13_p5/preregistration.json`` (``PREREGISTRATION_R13_P5.md``). Every command but
``plan`` refuses a plan file that differs from ``PLAN`` below, and every record carries the plan's digest and the
digests of the sources that produced it.

Commands (each writes one JSON record with ``--out``):
    plan        write the frozen plan (the freeze; refused if a different plan exists)
    preflight   RAM, swap, scratch and threads, a short timing probe, and the projected resource envelope of the pilot
                and of the confirmation; refuses any run projected above the RAM limit
    pilot       DEVELOPMENT ONLY: the development seed, at most 200 updates per trainable arm, smaller evaluation sets;
                it decides nothing
    confirm     one confirmation seed; refused without ``--authorization`` (the owner's further review)
    combine     the frozen statistics and reading over the five confirmation records (development records refused)
    manifest    every P5 record with its digest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import composition_tasks as ct
from ..config import ModelConfig
from ..ersa_model import ErsaConfig
from ..evidence_model import EvidenceConfig, EvidenceLM
from ..model import parameter_digest
from ..sparse_channel import SparseConfig
from ..geodome_cognitive import arms as am
from ..geodome_cognitive import controls as cg
from ..geodome_cognitive import regions as rg
from ..geodome_cognitive.binding import CognitiveSession
from ..geodome_cognitive.kernels import CognitiveConfig
from ..geodome_cognitive.plan import ExecutionPlan, compile_plan
from ..geodome_cognitive.state import TensorWorkingState
from ..geodome_cognitive.topology import substrate

PHASE = "ELPIS_TORCH_R13_P5_DISTRIBUTED_COMPETENCE"
REPO = Path(__file__).resolve().parents[3]
TORCH_DIR = REPO / "research" / "elpis_torch"
RESULTS = TORCH_DIR / "results" / "r13_p5"
PLAN_FILE = RESULTS / "preregistration.json"
R13_SUPPLEMENT = TORCH_DIR / "results" / "r13_hardening" / "MANIFEST.json"
SOURCES = {"runner": Path(__file__).resolve(), "tasks": TORCH_DIR / "composition_tasks.py",
           **{name: TORCH_DIR / "geodome_cognitive" / f"{name}.py"
              for name in ("arms", "controls", "regions", "binding", "kernels", "activation", "plan", "state",
                           "topology", "source")},
           **{name: TORCH_DIR / f"{name}.py" for name in ("evidence_model", "evidence_ring", "ersa_model",
                                                         "sparse_channel", "model", "config", "lastblock")}}
GIB = 1 << 30

# -- the frozen protocol ------------------------------------------------------------------------------------------

BACKBONE = {
    "model": dict(vocab_size=ct.VOCAB, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, channel="cgc",
                  cgc_bottleneck=8, cgc_channels=32, cgc_units=4, dtype="float32"),
    "ersa": dict(ersa_heads=4, key_dim=16, value_dim=16, ersa_kernel="chunked", scan_block=24, conv_kernel=4, chunk=72),
    "sparse": dict(experts=4, top_k=2, layers=[0, 1], shared=True),
    "evidence": dict(block=16, capacity=512, per_block=2, key_dim=16, coarse_top=17, evidence=3, fine_top=8,
                     overlap=4, encoder_dim=32, fine_dim=8, read_heads=2, read_head_dim=16, query_tile=72,
                     supersession="off", fine_keys="reread", mode="learned", rerank="learned"),
    "training": dict(recipe="R5 dense loss: the answer's cross-entropy + 0.05 x the mean next-token cross-entropy of "
                            "every position + R5's training-only coarse_source_loss and fine_evidence_loss",
                     steps=1200, batch=8, lr=3e-3, betas=[0.9, 0.95], weight_decay=0.01, warmup=50,
                     min_lr_fraction=0.1, grad_clip=1.0, background_weight=0.05, teacher=True,
                     evidence_label="the hop-1 fact block (the only block whose subject is the query's start entity), "
                                    "at the ASK position only; never a hop-2 or hop-3 fact",
                     episodes="split train, indices from 1,000,000 (disjoint from the arms' training episodes)",
                     time_limit_minutes=60),
}
PLAN = {
    "phase": PHASE,
    "status": "FROZEN",
    "question": "Does a learned, bounded, distributed computation over admitted evidence on Geodome's physical "
                "topology, injected at the EvidenceLM final block, improve the final answer of multi-hop record "
                "composition over the native backbone and over parameter-, compute-, recurrence- and graph-matched "
                "controls, and does the gain survive the causal interventions that should remove it?",
    "base": {"elpis_base": "d73cdde3a5050de50cce098472ee966eec103000",
             "r13_supplemental_receipts": "research/elpis_torch/results/r13_hardening (mechanics on the corrected "
                                          "sources; their manifest digest is recorded with every P5 record)"},
    "task": {"module": "research/elpis_torch/composition_tasks.py", "vocab": ct.VOCAB, "entities": ct.ENTITIES,
             "block": ct.BLOCK, "blocks": ct.BLOCKS, "length": ct.LENGTH, "fact_blocks": ct.FACT_BLOCKS,
             "note_blocks": ct.NOTE_BLOCKS, "heldout_fraction": ct.HELDOUT_FRACTION, "heldout_seed": ct.HELDOUT_SEED,
             "heldout_pairs": len(ct.HELDOUT_PAIRS),
             "train_structures": {str(k): [list(s) for s in v] for k, v in ct.TRAIN_STRUCTURES.items()},
             "heldout_structures": {str(k): [list(s) for s in v] for k, v in ct.HELDOUT_STRUCTURES.items()},
             "query": "QUERY start HOPk ASK; the answer is the next token (never in the stream as a query input)",
             "invariants": "every required edge in its own earlier FACT block; one outgoing edge per entity "
                           "(ambiguous paths rejected); no shortcut edge; hard negatives: an overshoot edge out of "
                           "the answer, converging edges into path entities, unrelated edges, NOTE blocks with "
                           "entity mentions; no held-out pair in any training fact; every entity in training facts"},
    "backbone": BACKBONE,
    "kernels": dict(width=32, hidden=64, capacity=64, max_rounds=16, max_arcs=1024, max_visits=16384, max_evidence=8,
                    evidence_heads=2, gate_init=0.0,
                    seed="the study seed: the same initialization in every trainable arm of a seed"),
    "substrate": dict(region_size=48, rounds=8, evidence_bindings=8, evidence_ring=2, query="center",
                      readout="center", schedule="synchronous", training_regions=3, heldout_regions=1,
                      centers="uniform and independent over all vertices, regions disjoint, generator seeded by the "
                              "study seed; bindings fixed by the region alone (no episode, route or answer)",
                      assignment="training and primary episode i on training region i mod 3; the held-out-region "
                                 "test evaluates every primary episode on the held-out region"),
    "evidence": {"deployable": "the frozen backbone's own top-8 fine-ranked live blocks at the query, captured from the "
                               "same forward (CognitiveSession capture_candidates=8; no extra token, no label) and "
                               "admitted through the exact reread; binding k receives rank k",
                 "oracle": "the episode's 8 FACT blocks in a seeded order (DIAGNOSTIC_ORACLE only; never deployable)",
                 "complete_set_recall": "every required fact block among the deployable candidates"},
    "arms": list(am.ARMS),
    "controls": {
        "EQUAL_BUDGET": "DirectReader (attention read of the evidence rows by the query row, then an MLP of hidden 72): "
                        "9,128 parameters in place of the kernels' 9,120 message, port and update parameters; the "
                        "same ingress, readout and injection",
        "COMPUTE_MATCHED": "a fully mixed random graph with the region's degree sequence (Havel-Hakimi, then 10 |E| "
                           "successful degree-preserving swaps), simple and connected; bindings placed to match the "
                           "evidence-to-readout distance 2; same kernels and rounds",
        "REWIRED": "the region rewired by |E| / 2 successful degree-preserving double-edge swaps (simple, "
                   "connected); bindings placed to match distance 2; same kernels and rounds",
        "FLAT_RECURRENT": "a hub (query and readout) reading 8 evidence leaves (arcs leaf to hub, port 0) for 8 rounds",
        "RELABEL_ISOMORPHIC": "the region under a declared seeded vertex permutation; arc order and port ordinals kept, "
                              "so the kernels receive the same inputs, permuted",
        "port_permutation": "evaluation-only diagnostic on the trained GEODOME arm: ports remapped by (1,2,3,4,5,0)",
        "mismatches": "every control's vertex, arc, degree, connectivity and distance statistics are recorded"},
    "training": dict(steps=2000, batch=1, optimizer="AdamW", lr=1e-3, betas=[0.9, 0.999], weight_decay=0.0,
                     grad_clip=1.0, supervision="final answer only: cross-entropy of the refined logits at the query",
                     episodes="split train, indices 0 .. steps - 1, hop counts cycling 1, 2, 3",
                     isolation="every episode observed from a reset session; every arm starts each episode from rows "
                               "of zero; no state or graph crosses episodes; one frozen backbone per seed for every arm",
                     log_steps=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1999]),
    "evaluation": {"primary": {"heldout_combination": {"2": 150, "3": 150}, "heldout_structure": {"2": 150, "3": 150}},
                   "iid": {"1": 100, "2": 100, "3": 100},
                   "metric": "accuracy: the argmax of the refined logits (NATIVE: the backbone's logits) is the answer"},
    "interventions": {
        "arm": "the trained GEODOME arm, evaluation only, on the primary set",
        "evidence_removal": "no evidence ingress (no evidence binding)",
        "evidence_swap": "the deployable evidence of the next primary episode (cyclic)",
        "reduced_depth": "4 rounds instead of 8 (decisive); 2 rounds (reported)",
        "interrupted_communication": "every arc between the readout's 1-hop disc and the rest of the region removed",
        "intermediate_state_erasure": "after 4 of 8 rounds every row but the query and evidence rows set to zero "
                                      "(reported, not decisive)",
        "disabled_injection": "the backbone's own tail: must equal NATIVE bit for bit (decisive)",
        "port_permutation": "reported, not decisive"},
    "statistics": {
        "primary_metric": "accuracy over the primary set (held-out combinations and held-out structures, 2 and 3 "
                          "hops; 600 episodes per seed)",
        "paired": "per seed and episode: correct(GEODOME) - correct(X) on the same episodes, the same frozen backbone",
        "estimate": "the mean over seeds of the per-seed mean paired difference",
        "bootstrap": {"kind": "hierarchical, seed-aware", "draws": 10000, "generator_seed": 13013,
                      "procedure": "draw 5 seeds with replacement; within each drawn seed draw its episodes with "
                                   "replacement; the statistic is the mean over the drawn seeds of the mean paired "
                                   "difference"},
        "bound": "one-sided lower bound: the (alpha / m) quantile of the bootstrap distribution (Bonferroni over the "
                 "m comparisons of the family)",
        "alpha": 0.05, "margin": 0.05, "positive_seeds": 4,
        "families": {"distributed": ["NATIVE", "EQUAL_BUDGET", "FLAT_RECURRENT"],
                     "geometry": ["COMPUTE_MATCHED", "REWIRED"],
                     "interventions": ["evidence_removal", "evidence_swap", "reduced_depth",
                                       "interrupted_communication"],
                     "substrate_agnostic": "each of COMPUTE_MATCHED and REWIRED against the distributed family"},
        "criteria": {"comparison": "estimate >= margin AND positive in >= 4 of 5 seeds AND lower bound > 0",
                     "intervention": "drop estimate > 0 AND positive in >= 4 of 5 seeds AND lower bound > 0",
                     "disabled_injection": "bit-exact NATIVE logits in every primary episode of every seed",
                     "relabel": "|estimate(GEODOME - RELABEL_ISOMORPHIC)| <= 0.02"},
        "recall_threshold": 0.90},
    "readings": [
        "GEOMETRY_SPECIFIC_GAIN: GEODOME passes the distributed family, the decisive interventions and the relabel "
        "check, AND the geometry family",
        "DISTRIBUTED_COMPETENCE_ESTABLISHED: GEODOME passes the distributed family, the decisive interventions and "
        "the relabel check; the geometry family does not pass and no nongeometric graph control passes the "
        "distributed family",
        "SUBSTRATE_AGNOSTIC_GAIN: distributed processing helps (GEODOME as above, or COMPUTE_MATCHED or REWIRED "
        "passes the distributed family) but GEODOME does not pass the geometry family, and a nongeometric graph "
        "control passes the distributed family",
        "RETRIEVAL_LIMITED: none of the above, and the pooled complete-set recall of the deployable evidence on the "
        "primary set is below the recall threshold",
        "NOT_ESTABLISHED: every other case"],
    "seeds": {"confirmation": [1301, 1302, 1303, 1304, 1305], "development": [9001]},
    "compute": {"device": "cpu", "threads": 2, "time_limit_minutes_per_arm_seed": 60, "ram_limit_bytes": 5 * GIB,
                "on_exhaustion": "stop and report; never retried silently"},
    "pilot": {"label": "DEVELOPMENT ONLY", "seed": 9001, "max_updates_per_trainable_arm": 200,
              "backbone_steps": 1200,
              "evaluation": {"primary": {"heldout_combination": {"2": 60, "3": 60},
                                         "heldout_structure": {"2": 60, "3": 60}},
                             "iid": {"1": 40, "2": 40, "3": 40}},
              "records": "retrieval recall, gate opening and per-group gradients, loss movement, latency, RSS and the "
                         "projected resource envelope"},
}
LIMITS = {
    "confirmation": {"steps": PLAN["training"]["steps"], "backbone_steps": BACKBONE["training"]["steps"],
                     "evaluation": PLAN["evaluation"], "label": "CONFIRMATION"},
    "pilot": {"steps": PLAN["pilot"]["max_updates_per_trainable_arm"], "backbone_steps": PLAN["pilot"]["backbone_steps"],
              "evaluation": PLAN["pilot"]["evaluation"], "label": "DEVELOPMENT ONLY"},
}


class Exhausted(RuntimeError):
    """A run hit a frozen limit (time or memory): stopped and reported, never retried silently."""


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args) -> str:
    run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else ""


def plan_text(plan: dict = PLAN) -> str:
    return json.dumps(plan, indent=1, sort_keys=True) + "\n"


def frozen() -> str:
    """The frozen plan's digest; refuses a missing plan or one that differs from ``PLAN`` (drift)."""
    if not PLAN_FILE.is_file():
        raise RuntimeError("the P5 plan is not frozen (run `plan` first)")
    if PLAN_FILE.read_text(encoding="utf-8") != plan_text():
        raise RuntimeError("the frozen P5 plan differs from the runner's PLAN: refused")
    return sha(PLAN_FILE)


def provenance() -> dict:
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)research/elpis_torch/results"))
    return {**{f"{k}_sha256": sha(v) for k, v in SOURCES.items()},
            "plan_sha256": sha(PLAN_FILE) if PLAN_FILE.is_file() else None,
            "r13_supplement_manifest_sha256": sha(R13_SUPPLEMENT),
            "git_revision": _git("rev-parse", "HEAD") + ("+dirty" if dirty else ""), "torch": torch.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads()}


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def guard(started: float, limit_minutes: float, what: str) -> None:
    if time.perf_counter() - started > 60 * limit_minutes:
        raise Exhausted(f"TIME_LIMIT: {what} exceeded {limit_minutes} minutes")
    if rss_bytes() > PLAN["compute"]["ram_limit_bytes"]:
        raise Exhausted(f"RAM_LIMIT: {what} exceeded {PLAN['compute']['ram_limit_bytes']} bytes")


# -- the backbone ---------------------------------------------------------------------------------------------------

def build_backbone(seed: int) -> EvidenceLM:
    widths = ModelConfig(seed=seed, **BACKBONE["model"])
    sparse = dict(BACKBONE["sparse"], layers=tuple(BACKBONE["sparse"]["layers"]))
    ersa = ErsaConfig(widths, sparse=SparseConfig(**sparse), **BACKBONE["ersa"])
    return EvidenceLM(ersa, EvidenceConfig(**BACKBONE["evidence"]))


def train_backbone(model, seed: int, steps: int, *, batch: int = None, started: float | None = None) -> dict:
    """R5's dense recipe on the task's training split (labels used by the training-only evidence losses only)."""
    cfg = BACKBONE["training"]
    batch = batch or cfg["batch"]
    started = started or time.perf_counter()
    torch.manual_seed(seed)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg["weight_decay"]},
                                   {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                                  lr=cfg["lr"], betas=tuple(cfg["betas"]))
    chunk, history, status = BACKBONE["ersa"]["chunk"], [], "COMPLETED"
    for step in range(steps):
        scale = (step + 1) / cfg["warmup"] if step < cfg["warmup"] else \
            cfg["min_lr_fraction"] + (1 - cfg["min_lr_fraction"]) * 0.5 * (
                1 + math.cos(math.pi * (step - cfg["warmup"]) / max(1, steps - cfg["warmup"])))
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * scale
        episodes = ct.episodes(seed, "train", batch, offset=1_000_000 + step * batch)
        tokens = torch.tensor([e.tokens for e in episodes])
        targets = torch.cat((tokens[:, 1:], torch.tensor([[e.answer] for e in episodes])), dim=1)
        labels = torch.full_like(tokens, -1)
        labels[:, -1] = torch.tensor([e.required_blocks[0] * ct.BLOCK + 3 for e in episodes])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state, total, parts = model.new_state(batch), 0.0, {"answer": 0.0, "coarse": 0.0, "fine": 0.0}
        for at in range(0, ct.LENGTH, chunk):
            logits, diag = model(tokens[:, at:at + chunk], state, evidence=labels[:, at:at + chunk],
                                 teacher=cfg["teacher"], diagnostics="losses")
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets[:, at:at + chunk].reshape(-1),
                                 reduction="none").view(batch, -1)
            loss = cfg["background_weight"] * ce.mean() * chunk / ct.LENGTH
            if at + chunk == ct.LENGTH:
                loss = loss + ce[:, -1].mean()
                parts["answer"] = float(ce[:, -1].mean().detach())
            for name, key in (("coarse", "coarse_source_loss"), ("fine", "fine_evidence_loss")):
                if key in diag.get("losses", {}):
                    loss = loss + diag["losses"][key]
                    parts[name] += float(diag["losses"][key].detach())
            total = total + loss
        if not torch.isfinite(total):
            status = "DIVERGED"
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(total.detach()), **{k: round(v, 4) for k, v in parts.items()}})
        guard(started, cfg["time_limit_minutes"], "backbone training")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return {"status": status, "steps": step + 1, "history": history, "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()), "digest": parameter_digest(model)}


# -- regions, plans and controls --------------------------------------------------------------------------------------

def geodome_plans(sub, region: rg.Region, rounds: int, *, cut: bool = False) -> dict:
    """Geodome-validated plans of one region for every evidence count 0..8 (compile_plan on a working state)."""
    working = TensorWorkingState(sub, capacity=len(region.vertices), width=1)
    working.acquire(region.vertices)
    arcs = region.arcs
    if cut:
        inside = set(region.vertices)
        dist = rg.distances(lambda v: [w for w in sub.neighbors(v) if w in inside], inside, region.readout)
        arcs = tuple(a for a in arcs if (dist[sub.endpoints(a)[0]] <= 1) == (dist[sub.endpoints(a)[1]] <= 1))
    k = PLAN["kernels"]
    out = {}
    for count in range(len(region.evidence) + 1):
        plan = ExecutionPlan(vertices=region.vertices, arcs=arcs, rounds=rounds, query=(region.query,),
                             evidence=region.evidence[:count], readout=(region.readout,))
        out[count] = compile_plan(plan, working, max_rounds=k["max_rounds"], max_arcs=k["max_arcs"],
                                  max_visits=k["max_visits"], max_evidence=k["max_evidence"], evidence_available=count)
    return out


def control_plans(compiled) -> dict:
    return {count: cg.with_evidence(compiled, count) for count in range(compiled.evidence.numel() + 1)}


def build_plans(sub, regions, seed: int) -> tuple:
    """Every arm's plans: variant -> region -> evidence count -> plan; and their statistics."""
    s = PLAN["substrate"]
    rounds, ring = s["rounds"], s["evidence_ring"]
    physical = [geodome_plans(sub, r, rounds) for r in regions]
    plans = {"GEODOME": {"main": physical, "depth4": [geodome_plans(sub, r, 4) for r in regions],
                         "depth2": [geodome_plans(sub, r, 2) for r in regions],
                         "half": [geodome_plans(sub, r, rounds // 2) for r in regions],
                         "cut": [geodome_plans(sub, r, rounds, cut=True) for r in regions],
                         "ports": [control_plans(cg.port_permuted(p[8], (1, 2, 3, 4, 5, 0))) for p in physical]}}
    plans["DIAGNOSTIC_ORACLE"] = {"main": physical}
    controls = {"COMPUTE_MATCHED": [cg.randomized(p[8], seed * 10 + i, rounds, ring) for i, p in enumerate(physical)],
                "REWIRED": [cg.rewired(p[8], seed * 10 + i, rounds, ring) for i, p in enumerate(physical)],
                "RELABEL_ISOMORPHIC": [cg.relabeled(p[8], seed * 10 + i, rounds) for i, p in enumerate(physical)],
                "FLAT_RECURRENT": [cg.star(s["evidence_bindings"], rounds) for _ in physical]}
    for arm, graphs in controls.items():
        plans[arm] = {"main": [control_plans(g) for g in graphs]}
    plans["EQUAL_BUDGET"] = {"main": [None for _ in physical]}
    stats = {"regions": [{"center": r.center, **r.stats} for r in regions],
             "controls": {arm: [g.stats for g in graphs] for arm, graphs in controls.items()},
             "cut_arcs_removed": [len(r.arcs) - plans["GEODOME"]["cut"][i][8].arc_ids.numel()
                                  for i, r in enumerate(regions)]}
    return plans, stats


# -- observing the episodes once ----------------------------------------------------------------------------------------

def view_of(session: CognitiveSession, episode: ct.Episode, order_seed: str) -> am.View:
    """One episode as the frozen backbone sees it: a reset session, the stream in chunks, the captured candidates,
    the deployable evidence (exact reread) and the oracle evidence (the FACT blocks, seeded order)."""
    session.reset()
    chunk = session.model.ersa.chunk
    tokens = torch.tensor([episode.tokens])
    for at in range(0, ct.LENGTH, chunk):
        session.observe(tokens[:, at:at + chunk])
    observation = session._observation
    blocks = tuple(c.block for c in observation.candidates)
    deployable = None
    if observation.candidates:
        session.admit_evidence(observation.candidates)
        deployable = session._evidence
    order = list(episode.fact_blocks)
    random.Random(order_seed).shuffle(order)
    session.admit_evidence([session.state.ring.address(b) for b in order])
    oracle = session._evidence
    return am.View(observation, deployable, oracle, episode.answer, episode.hops, blocks,
                   set(episode.required_blocks) <= set(blocks))


def episode_sets(seed: int, evaluation: dict, steps: int) -> dict:
    train = ct.episodes(seed, "train", steps)
    primary = []
    for split, cells in evaluation["primary"].items():
        for hops, count in cells.items():                  # disjoint indices per stratum: independent generators
            primary += ct.episodes(seed, split, count, hops=(int(hops),), offset=10_000 * int(hops))
    iid = []
    for hops, count in evaluation["iid"].items():
        iid += ct.episodes(seed, "iid", count, hops=(int(hops),), offset=10_000 * int(hops))
    return {"train": train, "primary": primary, "iid": iid}


# -- training and evaluation --------------------------------------------------------------------------------------------

def train_arm(arm: am.Arm, views, steps: int, *, started: float, log_steps) -> dict:
    cfg = PLAN["training"]
    params = arm.trainable()
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    source = "oracle" if arm.arm == "DIAGNOSTIC_ORACLE" else "deployable"
    losses, log, status = [], [], "COMPLETED"
    for step in range(steps):
        view = views[step]
        optimizer.zero_grad(set_to_none=True)
        logits = arm(view, step % PLAN["substrate"]["training_regions"], source=source)
        loss = F.cross_entropy(logits, torch.tensor([view.answer]))
        if not torch.isfinite(loss):
            status = "DIVERGED"
            break
        loss.backward()
        if step in log_steps:
            log.append({"step": step, "loss": float(loss.detach()), "gate": float(arm.kernels.gate.detach()),
                        "grad_norm": am.gradient_norms(arm)})
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
        guard(started, PLAN["compute"]["time_limit_minutes_per_arm_seed"], f"{arm.arm} training")
    window = max(1, len(losses) // 5)
    return {"status": status, "steps": len(losses), "log": log, "seconds": time.perf_counter() - started,
            "loss_first_fifth": sum(losses[:window]) / window if losses else None,
            "loss_last_fifth": sum(losses[-window:]) / window if losses else None,
            "final_gate": float(arm.kernels.gate.detach()), "trainable_parameters": am.parameter_count(params)}


@torch.no_grad()
def evaluate(fn, views, regions_for) -> dict:
    correct, nll = [], 0.0
    for i, view in enumerate(views):
        logits = fn(view, regions_for(i), i)
        correct.append(int(int(logits.argmax()) == view.answer))
        nll += float(F.cross_entropy(logits, torch.tensor([view.answer])))
    return {"accuracy": sum(correct) / max(1, len(correct)), "nll": nll / max(1, len(views)),
            "correct": "".join(map(str, correct))}


def run_seed(seed: int, limits: dict, *, plan_digest: str | None) -> dict:
    """Everything one seed needs: backbone, regions and controls, cached observations, every arm, interventions."""
    sub = substrate()
    started = time.perf_counter()
    timing, record = {}, {"phase": PHASE, "seed": seed, "label": limits["label"], "plan_sha256": plan_digest}
    model = build_backbone(seed)
    record["backbone"] = train_backbone(model, seed, limits["backbone_steps"])
    timing["backbone_seconds"] = record["backbone"]["seconds"]
    t = time.perf_counter()
    regions = rg.draw_regions(sub, seed, PLAN["substrate"]["training_regions"] + PLAN["substrate"]["heldout_regions"],
                              size=PLAN["substrate"]["region_size"], evidence=PLAN["substrate"]["evidence_bindings"],
                              ring=PLAN["substrate"]["evidence_ring"])
    plans, record["topology"] = build_plans(sub, regions, seed)
    timing["plans_seconds"] = time.perf_counter() - t
    sets = episode_sets(seed, limits["evaluation"], limits["steps"])
    observer = CognitiveSession(model, am.CognitiveKernels(CognitiveConfig(**_kernel_config(seed)),
                                                           model.config.d_model, model.encoder.width),
                                capture_candidates=PLAN["kernels"]["max_evidence"])
    t = time.perf_counter()
    views = {name: [view_of(observer, e, f"r13-p5-oracle:{seed}:{name}:{i}") for i, e in enumerate(eps)]
             for name, eps in sets.items()}
    timing["observe_seconds_per_episode"] = (time.perf_counter() - t) / sum(len(v) for v in sets.values())
    observer.close()
    record["recall"] = {name: _recall(views[name], sets[name]) for name in ("primary", "iid", "train")}
    record["native"] = {name: evaluate(lambda v, r, i: v.observation.logits, views[name], lambda i: 0)
                        for name in ("primary", "iid")}
    log_steps = [s for s in PLAN["training"]["log_steps"] if s < limits["steps"]] + [limits["steps"] - 1]
    record["arms"], interventions = {}, None
    for name in am.TRAINABLE:
        t = time.perf_counter()
        arm = am.Arm(name, model, CognitiveConfig(**_kernel_config(seed)), plans[name])
        entry = {"training": train_arm(arm, views["train"], limits["steps"], started=t, log_steps=sorted(set(log_steps)))}
        source = "oracle" if name == "DIAGNOSTIC_ORACLE" else "deployable"
        primary_region = lambda i: i % PLAN["substrate"]["training_regions"]   # noqa: E731
        heldout = PLAN["substrate"]["training_regions"]
        entry["primary"] = evaluate(lambda v, r, i: arm(v, r, source=source), views["primary"], primary_region)
        entry["iid"] = evaluate(lambda v, r, i: arm(v, r, source=source), views["iid"], primary_region)
        entry["heldout_region"] = evaluate(lambda v, r, i: arm(v, r, source=source), views["primary"],
                                           lambda i: heldout)
        if name == "GEODOME":
            interventions = _interventions(arm, views["primary"], primary_region)
        entry["seconds"] = time.perf_counter() - t
        record["arms"][name] = entry
        guard(t, PLAN["compute"]["time_limit_minutes_per_arm_seed"], f"{name} evaluation")
        arm.session.close()
    record["interventions"] = interventions
    record["backbone"]["ok_unchanged_after_every_arm"] = parameter_digest(model) == record["backbone"]["digest"]
    timing["total_seconds"] = time.perf_counter() - started
    record["timing"] = timing
    record["peak_rss_bytes"] = rss_bytes()
    record["episodes"] = {name: len(v) for name, v in sets.items()}
    return record


def _kernel_config(seed: int) -> dict:
    k = {key: value for key, value in PLAN["kernels"].items() if key != "seed"}
    return dict(k, seed=seed)


def _recall(views, episodes) -> dict:
    by_hops = {}
    for view in views:
        by_hops.setdefault(str(view.hops), []).append(view.complete)
    facts = [len(set(v.candidates) & set(e.fact_blocks)) / ct.FACT_BLOCKS for v, e in zip(views, episodes)]
    first = [v.candidates[:1] == (e.required_blocks[0],) for v, e in zip(views, episodes)]
    return {"complete_set": sum(v.complete for v in views) / max(1, len(views)),
            "complete_set_by_hops": {k: sum(x) / len(x) for k, x in sorted(by_hops.items())},
            "fact_blocks_in_candidates": sum(facts) / max(1, len(facts)),
            "hop1_ranked_first": sum(first) / max(1, len(first)),
            "candidates_per_episode": sum(len(v.candidates) for v in views) / max(1, len(views))}


def _interventions(arm: am.Arm, views, region_for) -> dict:
    nxt = lambda i: views[(i + 1) % len(views)].deployable                    # noqa: E731
    out = {
        "full": evaluate(lambda v, r, i: arm(v, r), views, region_for),
        "evidence_removal": evaluate(lambda v, r, i: arm(v, r, source=None), views, region_for),
        "evidence_swap": evaluate(lambda v, r, i: arm(v, r, source=nxt(i)), views, region_for),
        "reduced_depth": evaluate(lambda v, r, i: arm(v, r, variant="depth4"), views, region_for),
        "reduced_depth_2": evaluate(lambda v, r, i: arm(v, r, variant="depth2"), views, region_for),
        "interrupted_communication": evaluate(lambda v, r, i: arm(v, r, variant="cut"), views, region_for),
        "intermediate_state_erasure": evaluate(lambda v, r, i: arm.erased(v, r), views, region_for),
        "port_permutation": evaluate(lambda v, r, i: arm(v, r, variant="ports"), views, region_for),
    }
    with torch.no_grad():
        exact = [torch.equal(arm.disabled(v), v.observation.logits) for v in views]
    out["disabled_injection"] = {"bit_exact_native": all(exact), "episodes": len(exact)}
    return out


# -- statistics (the frozen procedure) ------------------------------------------------------------------------------

def bits(text: str) -> list:
    return [int(c) for c in text]


def hierarchical_bootstrap(per_seed: list, draws: int, seed: int, *, block: int = 500) -> torch.Tensor:
    """The seed-aware bootstrap distribution of the mean over seeds of per-seed mean paired differences."""
    data = torch.tensor(per_seed, dtype=torch.float64)                       # [S, n] (equal n per seed)
    seeds, n = data.shape
    generator = torch.Generator().manual_seed(seed)
    out = []
    for at in range(0, draws, block):
        size = min(block, draws - at)
        which = torch.randint(seeds, (size, seeds), generator=generator)
        picks = torch.randint(n, (size, seeds, n), generator=generator)
        values = data[which[..., None], picks]                               # [size, S, n]
        out.append(values.mean(-1).mean(-1))
    return torch.cat(out)


def compare(per_seed: list, *, m: int, margin: float, stats: dict) -> dict:
    """``per_seed``: per-seed lists of paired differences. The frozen comparison criterion."""
    means = [sum(d) / len(d) for d in per_seed]
    estimate = sum(means) / len(means)
    boot = hierarchical_bootstrap(per_seed, stats["bootstrap"]["draws"], stats["bootstrap"]["generator_seed"])
    lower = float(torch.quantile(boot, stats["alpha"] / m))
    positive = sum(x > 0 for x in means)
    passed = estimate >= margin and positive >= stats["positive_seeds"] and lower > 0
    return {"estimate": estimate, "per_seed": means, "positive_seeds": positive, "lower_bound": lower,
            "level": 1 - stats["alpha"] / m, "margin": margin, "pass": passed}


def reading(records: list) -> dict:
    """The frozen statistics and reading over the confirmation records (one per seed, in seed order)."""
    stats = PLAN["statistics"]
    fam = stats["families"]
    primary = {name: [bits(r["arms"][name]["primary"]["correct"]) for r in records] for name in am.TRAINABLE}
    primary["NATIVE"] = [bits(r["native"]["primary"]["correct"]) for r in records]

    def diffs(a, b):
        return [[x - y for x, y in zip(pa, pb)] for pa, pb in zip(primary[a], primary[b])]
    distributed = {x: compare(diffs("GEODOME", x), m=len(fam["distributed"]), margin=stats["margin"], stats=stats)
                   for x in fam["distributed"]}
    geometry = {x: compare(diffs("GEODOME", x), m=len(fam["geometry"]), margin=stats["margin"], stats=stats)
                for x in fam["geometry"]}
    agnostic = {c: {x: compare(diffs(c, x), m=len(fam["distributed"]), margin=stats["margin"], stats=stats)
                    for x in fam["distributed"]} for c in fam["geometry"]}
    full = [bits(r["interventions"]["full"]["correct"]) for r in records]
    interventions = {}
    for name in fam["interventions"]:
        cut = [bits(r["interventions"][name]["correct"]) for r in records]
        drop = [[a - b for a, b in zip(f, c)] for f, c in zip(full, cut)]
        interventions[name] = compare(drop, m=len(fam["interventions"]), margin=1e-12, stats=stats)
    exact = all(r["interventions"]["disabled_injection"]["bit_exact_native"] for r in records)
    relabel = compare(diffs("GEODOME", "RELABEL_ISOMORPHIC"), m=1, margin=0.0, stats=stats)
    relabel_ok = abs(relabel["estimate"]) <= 0.02
    recall = sum(r["recall"]["primary"]["complete_set"] for r in records) / len(records)
    geodome_d = all(c["pass"] for c in distributed.values()) and all(c["pass"] for c in interventions.values()) \
        and exact and relabel_ok
    geometry_pass = all(c["pass"] for c in geometry.values())
    control_d = any(all(c["pass"] for c in agnostic[ctrl].values()) for ctrl in agnostic)
    if geodome_d and geometry_pass:
        label = "GEOMETRY_SPECIFIC_GAIN"
    elif geodome_d and not control_d:
        label = "DISTRIBUTED_COMPETENCE_ESTABLISHED"
    elif (geodome_d or control_d) and not geometry_pass and control_d:
        label = "SUBSTRATE_AGNOSTIC_GAIN"
    elif recall < stats["recall_threshold"]:
        label = "RETRIEVAL_LIMITED"
    else:
        label = "NOT_ESTABLISHED"
    return {"reading": label, "distributed": distributed, "geometry": geometry, "substrate_agnostic": agnostic,
            "interventions": interventions, "disabled_injection_exact": exact, "relabel": relabel,
            "relabel_ok": relabel_ok, "recall_primary_pooled": recall,
            "seeds": [r["seed"] for r in records]}


# -- commands -----------------------------------------------------------------------------------------------------

def command_plan(args) -> dict:
    if PLAN_FILE.is_file() and PLAN_FILE.read_text(encoding="utf-8") != plan_text():
        raise RuntimeError("a different P5 plan is already frozen: refused")
    return PLAN


def meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return out


def command_preflight(args) -> dict:
    digest = frozen()
    torch.set_num_threads(PLAN["compute"]["threads"])
    started, sub = time.perf_counter(), substrate()
    model = build_backbone(PLAN["pilot"]["seed"])
    t = time.perf_counter()
    train_backbone(model, PLAN["pilot"]["seed"], 3)
    backbone_step = (time.perf_counter() - t) / 3
    episodes = ct.episodes(PLAN["pilot"]["seed"], "iid", 6)
    observer = CognitiveSession(model, am.CognitiveKernels(CognitiveConfig(**_kernel_config(1)), model.config.d_model,
                                                           model.encoder.width), capture_candidates=8)
    t = time.perf_counter()
    views = [view_of(observer, e, f"preflight:{i}") for i, e in enumerate(episodes)]
    observe = (time.perf_counter() - t) / len(views)
    regions = rg.draw_regions(sub, PLAN["pilot"]["seed"], 4)
    plans, _ = build_plans(sub, regions, PLAN["pilot"]["seed"])
    arm = am.Arm("GEODOME", model, CognitiveConfig(**_kernel_config(1)), plans["GEODOME"])
    t = time.perf_counter()
    train_arm(arm, views, len(views), started=time.perf_counter(), log_steps=[])
    arm_step = (time.perf_counter() - t) / len(views)
    t = time.perf_counter()
    with torch.no_grad():
        for v in views:
            arm(v, 0)
    arm_eval = (time.perf_counter() - t) / len(views)
    view_bytes = sum(x.numel() * x.element_size() for x in (views[0].observation.rows, views[0].observation.pre,
                                                             views[0].observation.logits, views[0].deployable.encoded,
                                                             views[0].oracle.encoded))

    def envelope(limits, seeds) -> dict:
        ev = limits["evaluation"]
        primary = sum(sum(c.values()) for c in ev["primary"].values())
        iid = sum(ev["iid"].values())
        cached = limits["steps"] + primary + iid
        arms = len(am.TRAINABLE)
        seconds = (limits["backbone_steps"] * backbone_step + cached * observe + arms * limits["steps"] * arm_step
                   + arms * (2 * primary + iid) * arm_eval + 9 * primary * arm_eval)
        return {"seeds": seeds, "cached_episodes_per_seed": cached, "view_cache_bytes": cached * view_bytes,
                "projected_seconds_per_seed": seconds, "projected_hours_total": seeds * seconds / 3600,
                "projected_peak_rss_bytes": rss_bytes() + cached * view_bytes,
                "within_ram_limit": rss_bytes() + cached * view_bytes < PLAN["compute"]["ram_limit_bytes"]}
    mem = meminfo()
    disk = {str(p): shutil.disk_usage(p).free for p in (RESULTS.parent, Path("/tmp"))}
    pilot = envelope(LIMITS["pilot"], 1)
    confirmation = envelope(LIMITS["confirmation"], len(PLAN["seeds"]["confirmation"]))
    checks = {"ok_plan_frozen": bool(digest), "ok_threads": torch.get_num_threads() == PLAN["compute"]["threads"],
              "ok_ram_available": mem.get("MemAvailable", 0) > max(pilot["projected_peak_rss_bytes"],
                                                                   confirmation["projected_peak_rss_bytes"]),
              "ok_pilot_within_ram_limit": pilot["within_ram_limit"],
              "ok_confirmation_within_ram_limit": confirmation["within_ram_limit"],
              "ok_scratch": min(disk.values()) > GIB}
    return {"phase": PHASE, "command": "preflight", "plan_sha256": digest, "memory": mem, "scratch_free_bytes": disk,
            "cpu_count": os.cpu_count(), "threads": torch.get_num_threads(),
            "probe_seconds": {"backbone_step": backbone_step, "observe_episode": observe, "arm_training_step": arm_step,
                              "arm_evaluation_episode": arm_eval},
            "view_bytes": view_bytes, "pilot": pilot, "confirmation": confirmation, "checks": checks,
            "go": all(checks.values()), "seconds": time.perf_counter() - started, "provenance": provenance()}


def command_pilot(args) -> dict:
    digest = frozen()
    torch.set_num_threads(PLAN["compute"]["threads"])
    try:
        record = run_seed(PLAN["pilot"]["seed"], LIMITS["pilot"], plan_digest=digest)
        record["status"] = "COMPLETED"
        record["confirmation_projection"] = projection(record)
    except Exhausted as error:
        record = {"phase": PHASE, "seed": PLAN["pilot"]["seed"], "label": "DEVELOPMENT ONLY", "status": str(error)}
    record.update({"command": "pilot", "decides": "nothing (DEVELOPMENT ONLY)", "provenance": provenance()})
    return record


def projection(pilot: dict) -> dict:
    """The confirmation's resource envelope, scaled from the pilot's MEASURED times (same backbone budget)."""
    steps, conf = pilot["episodes"]["train"], LIMITS["confirmation"]
    primary = sum(sum(c.values()) for c in conf["evaluation"]["primary"].values())
    iid = sum(conf["evaluation"]["iid"].values())
    pilot_eval = 2 * pilot["episodes"]["primary"] + pilot["episodes"]["iid"]
    per_arm = {}
    for name, a in pilot["arms"].items():
        train = a["training"]["seconds"] / max(1, steps) * conf["steps"]
        evaluation = (a["seconds"] - a["training"]["seconds"]) / max(1, pilot_eval) * (2 * primary + iid)
        per_arm[name] = {"train_seconds": train, "evaluation_seconds": evaluation,
                         "within_time_limit": train + evaluation < 60 * PLAN["compute"]["time_limit_minutes_per_arm_seed"]}
    cached = conf["steps"] + primary + iid
    per_seed = (pilot["timing"]["backbone_seconds"] + pilot["timing"]["observe_seconds_per_episode"] * cached
                + sum(x["train_seconds"] + x["evaluation_seconds"] for x in per_arm.values()))
    return {"per_arm": per_arm, "seconds_per_seed": per_seed,
            "hours_for_five_seeds": per_seed * len(PLAN["seeds"]["confirmation"]) / 3600,
            "backbone_within_time_limit": pilot["timing"]["backbone_seconds"] < 60 * BACKBONE["training"][
                "time_limit_minutes"],
            "basis": "pilot measurements scaled linearly by steps and episodes; the pilot's backbone budget equals "
                     "the confirmation's"}


def command_confirm(args) -> dict:
    digest = frozen()
    if not args.authorization:
        raise RuntimeError("confirmation requires the owner's further authorization (--authorization): refused")
    if args.seed not in PLAN["seeds"]["confirmation"]:
        raise RuntimeError("not a confirmation seed of the frozen plan")
    torch.set_num_threads(PLAN["compute"]["threads"])
    try:
        record = run_seed(args.seed, LIMITS["confirmation"], plan_digest=digest)
        record["status"] = "COMPLETED"
    except Exhausted as error:
        record = {"phase": PHASE, "seed": args.seed, "label": "CONFIRMATION", "status": str(error)}
    record.update({"command": "confirm", "authorization": args.authorization, "provenance": provenance()})
    return record


def command_combine(args) -> dict:
    digest = frozen()
    records = [json.loads((RESULTS / f"confirm_{s}.json").read_text(encoding="utf-8"))
               for s in PLAN["seeds"]["confirmation"]]
    for r in records:
        if r.get("label") != "CONFIRMATION" or r.get("status") != "COMPLETED" or r.get("plan_sha256") != digest:
            raise RuntimeError(f"seed {r.get('seed')}: not a completed confirmation record of the frozen plan")
    return {"phase": PHASE, "command": "combine", "plan_sha256": digest, **reading(records),
            "inputs": {f"confirm_{s}.json": sha(RESULTS / f"confirm_{s}.json") for s in PLAN["seeds"]["confirmation"]},
            "provenance": provenance()}


def command_manifest(args) -> dict:
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json")
    return {"phase": PHASE, "command": "manifest", "files": {p.name: sha(p) for p in files},
            "sources": {k: sha(v) for k, v in SOURCES.items()}}


COMMANDS = {"plan": command_plan, "preflight": command_preflight, "pilot": command_pilot, "confirm": command_confirm,
            "combine": command_combine, "manifest": command_manifest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--authorization", default="")
    parser.add_argument("--out", help="the JSON record to write")
    args = parser.parse_args(argv)
    record = COMMANDS[args.command](args)
    text = plan_text(record) if args.command == "plan" else json.dumps(record, indent=1, sort_keys=True,
                                                                         default=str) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
