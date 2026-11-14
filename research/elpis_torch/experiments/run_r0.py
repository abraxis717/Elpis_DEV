#!/usr/bin/env python3
"""Elpis Torch R0 feasibility experiment: FFN versus CGC versus FFN+CGC, then volatile adaptation.

For every (variant, seed): build the model, train it offline on the finite-state transduction task, evaluate it
on held-out data, and run the session adaptation protocol on the frozen result. Every variant sees the same
training set, the same batches (the batch sampler depends on the seed only), the same number of tokens and the
same optimizer settings.

    python -m research.elpis_torch.experiments.run_r0                    # full protocol, JSON on stdout
    python -m research.elpis_torch.experiments.run_r0 --quick            # one seed, 300 steps: a smoke run
    python -m research.elpis_torch.experiments.run_r0 --out results.json # also write the record (explicit)

Nothing is written unless ``--out`` is given. No model weights are ever written. Wall-clock numbers are printed
for information (PERFORMANCE_ONLY, one machine) and are never part of a pass criterion.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import torch

from .. import PHASE, accounting
from ..config import ModelConfig
from ..data import SINGLE, build_splits, generate, make_world, rewire
from ..model import ElpisTorchLM, parameter_digest
from ..session import SessionConfig, SupportSet, VolatileSession
from ..training import TrainConfig, evaluate, freeze, rollout, train
from ..upstream import UPSTREAM_COMMIT

BASE = dict(vocab_size=259, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=48)
FFN = 172                     # SwiGLU hidden width of the baseline: 3 * 64 * 172 = 33,024 parameters per block
VARIANTS = {
    # name: (configuration, relation to A's parameter count)
    "A_ffn": (dict(channel="ffn", ffn_hidden=FFN), "baseline"),
    "B_cgc_bottleneck": (dict(channel="cgc", cgc_bottleneck=32, cgc_channels=121, cgc_units=6), "matched exactly"),
    "B_cgc_dense": (dict(channel="cgc", cgc_channels=86, cgc_units=5), "matched exactly"),
    "B_cgc_bottleneck_unbounded": (dict(channel="cgc", cgc_bottleneck=32, cgc_channels=121, cgc_units=6,
                                        cgc_preact_bound=None), "matched exactly; faithful E0 pre-activations"),
    "C_matched_gate0": (dict(channel="ffn+cgc", ffn_hidden=128, cgc_bottleneck=16, cgc_channels=58, cgc_units=4),
                        "matched to within the scalar gates (+1 per block)"),
    "C_matched_gate0.1": (dict(channel="ffn+cgc", ffn_hidden=128, cgc_bottleneck=16, cgc_channels=58, cgc_units=4,
                               cgc_gate_init=0.1), "matched to within the scalar gates (+1 per block)"),
    "C_augmented_gate0": (dict(channel="ffn+cgc", ffn_hidden=FFN, cgc_bottleneck=16, cgc_channels=58, cgc_units=4),
                          "A plus a CGC path: more parameters and more compute than A"),
}
ADAPT = (dict(lr=0.05, steps=16), dict(lr=0.2, steps=16))


def cgc_diagnostics(model: ElpisTorchLM, tokens: torch.Tensor) -> list:
    """Pre-activation scale and saturation of every block's CGC core on real inputs."""
    captured, handles = [], []
    for block in model.blocks:
        if block.cgc is not None:
            handles.append(block.cgc.core.register_forward_pre_hook(lambda m, args: captured.append((m, args[0]))))
    with torch.no_grad():
        model(tokens)
    for handle in handles:
        handle.remove()
    return [core.diagnostics(x) for core, x in captured]


def throughput(model: ElpisTorchLM, prompt: torch.Tensor) -> float:
    """Cached greedy generation, tokens per second (PERFORMANCE_ONLY)."""
    new = model.config.max_seq_len - prompt.shape[1]
    model.generate(prompt, new)
    started = time.perf_counter()
    for _ in range(3):
        model.generate(prompt, new)
    return 3 * new / (time.perf_counter() - started)


def masked_metrics(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    nll = -log_probs.gather(-1, tokens[:, 1:, None])[..., 0]
    right = logits.argmax(dim=-1) == tokens[:, 1:]
    return {"loss": float(nll[mask].mean()), "accuracy": float(right[mask].double().mean()), "targets": int(mask.sum())}


def adaptation(model: ElpisTorchLM, world, seed: int) -> dict:
    """A session in a world whose operation 'e' was rewired after the checkpoint was trained."""
    shifted = rewire(world, op=world.ops - 1, seed=seed)
    changed = shifted.changed_from(world)
    evidence = generate(shifted, 16, BASE["max_seq_len"], seed=7000 + seed, p_double=0.0)
    held = generate(shifted, 256, BASE["max_seq_len"], seed=8000 + seed, p_double=0.0)
    hit = torch.zeros_like(held.kind, dtype=torch.bool)
    for state, op in changed:
        hit |= (held.state == state) & (held.op == op)
    hit &= held.kind == SINGLE
    mask_changed, mask_same = hit[:, 1:], ((held.kind == SINGLE) & ~hit)[:, 1:]
    support = SupportSet.observed(evidence.tokens, mask=evidence.target_mask(), source="rewired-world evidence")
    digest, probe = parameter_digest(model), held.tokens[:8]
    reference = model(probe[:, :-1])
    out = {"changed_transitions": len(changed), "evidence_sequences": len(evidence), "held_out_sequences": len(held),
           "runs": []}
    with VolatileSession(model, SessionConfig(learning_enabled=False)) as frozen:
        refused = frozen.adapt(support)[0].status.name
        logits = frozen.forward(held.tokens[:, :-1])
        out["learning_disabled"] = {"adapt_status": refused,
                                    "changed": masked_metrics(logits, held.tokens, mask_changed),
                                    "unchanged": masked_metrics(logits, held.tokens, mask_same)}
    for setting in ADAPT:
        with VolatileSession(model, SessionConfig(lr=setting["lr"], max_steps=setting["steps"])) as session:
            results = session.adapt(support, steps=setting["steps"])
            logits = session.forward(held.tokens[:, :-1])
            record = {**setting, "statuses": sorted({r.status.name for r in results}),
                      "applied": sum(r.applied for r in results),
                      "support_loss": [results[0].loss_before, results[-1].loss_after],
                      "clipped_steps": sum(r.clipped for r in results),
                      "projected_steps": sum(r.projected for r in results), "delta_norm": session.delta_norm(),
                      "changed": masked_metrics(logits, held.tokens, mask_changed) if results[0].applied else None,
                      "unchanged": masked_metrics(logits, held.tokens, mask_same) if results[0].applied else None}
            session.reset()
            record["reset_restores_logits_exactly"] = bool(torch.equal(session.forward(probe[:, :-1]), reference))
            out["runs"].append(record)
    out["base_parameters_unchanged"] = parameter_digest(model) == digest
    return out


def one_run(name: str, seed: int, splits, world, steps: int) -> dict:
    config = ModelConfig(**BASE, **VARIANTS[name][0], seed=seed)
    model = ElpisTorchLM(config)
    settings = TrainConfig(steps=steps, batch_size=64, lr=3e-3, warmup_steps=50, weight_decay=0.01, grad_clip=1.0,
                           seed=seed, eval_every=100)
    before = evaluate(model, splits.val)
    report = train(model, splits.train, settings, val=splits.val)
    freeze(model)
    reached = next((h["step"] for h in report.history
                    if h["val"].get("double", {}).get("accuracy", 0.0) >= 0.99), None)
    record = {
        "variant": name, "seed": seed, "status": report.status, "anomalies": report.anomalies,
        "parameters": accounting.parameter_counts(model),
        "train_loss": {"first": report.first_loss, "final": report.final_loss},
        "val_before_training": before, "val": report.history[-1]["val"] if report.history else None,
        "compositional": evaluate(model, splits.comp), "rollout_val": rollout(model, splits.val.select(slice(0, 256))),
        "steps_to_val_double_accuracy_0.99": reached,
        "history": [[h["step"], round(h["val"]["loss"], 4), round(h["val"]["single"]["accuracy"], 4),
                     round(h["val"]["double"]["accuracy"], 4)] for h in report.history],
        "grad_norm": report.grad_norm, "update_norm": report.update_norm,
        "cgc_diagnostics": cgc_diagnostics(model, splits.val.tokens[:64]),
        "train_seconds": report.seconds, "train_steps_per_second": report.steps_done / report.seconds,
        "generation_tokens_per_second": throughput(model, splits.val.tokens[:1, :2]),
    }
    record["adaptation"] = adaptation(model, world, seed) if report.status == "COMPLETED" else None
    return record


def summarize(runs: list) -> dict:
    out = {}
    for name in VARIANTS:
        mine = [r for r in runs if r["variant"] == name]
        if not mine:
            continue

        def spread(values):
            values = [v for v in values if v is not None]
            return None if not values else {"mean": statistics.mean(values), "min": min(values), "max": max(values)}

        done = [r for r in mine if r["status"] == "COMPLETED"]
        row = {"seeds": len(mine), "completed": len(done), "parameters": mine[0]["parameters"]["total"],
               "final_train_loss": spread(r["train_loss"]["final"] for r in done),
               "val_loss": spread(r["val"]["loss"] for r in done),
               "val_single_accuracy": spread(r["val"]["single"]["accuracy"] for r in done),
               "val_double_accuracy": spread(r["val"]["double"]["accuracy"] for r in done),
               "compositional_heldout_accuracy": spread(r["compositional"]["heldout"]["accuracy"] for r in done),
               "rollout_exact_trajectories": spread(r["rollout_val"]["exact_trajectories"] for r in done),
               "steps_to_val_double_accuracy_0.99": [r["steps_to_val_double_accuracy_0.99"] for r in done],
               "train_steps_per_second": spread(r["train_steps_per_second"] for r in done),
               "generation_tokens_per_second": spread(r["generation_tokens_per_second"] for r in done),
               "cgc_saturated_fraction_max": spread(max((d["saturated_fraction"] for d in r["cgc_diagnostics"]),
                                                        default=None) for r in done),
               "cgc_preact_abs_max": spread(max((d["preact_abs_max"] for d in r["cgc_diagnostics"]), default=None)
                                            for r in done)}
        for index, setting in enumerate(ADAPT):
            picked = [r["adaptation"]["runs"][index] for r in done if r["adaptation"]]
            base = [r["adaptation"]["learning_disabled"] for r in done if r["adaptation"]]
            row[f"adapt_lr{setting['lr']}"] = {
                "statuses": sorted({s for p in picked for s in p["statuses"]}),
                "changed_loss_disabled": spread(b["changed"]["loss"] for b in base),
                "changed_loss_adapted": spread(p["changed"]["loss"] if p["changed"] else None for p in picked),
                "changed_accuracy_adapted": spread(p["changed"]["accuracy"] if p["changed"] else None for p in picked),
                "unchanged_loss_disabled": spread(b["unchanged"]["loss"] for b in base),
                "unchanged_loss_adapted": spread(p["unchanged"]["loss"] if p["unchanged"] else None for p in picked),
                "unchanged_accuracy_adapted": spread(p["unchanged"]["accuracy"] if p["unchanged"] else None
                                                     for p in picked),
                "reset_exact_in_all": all(p["reset_restores_logits_exactly"] for p in picked),
                "base_unchanged_in_all": all(r["adaptation"]["base_parameters_unchanged"] for r in done
                                             if r["adaptation"]),
            }
        out[name] = row
    return out


def _finite(value):
    """Strict JSON: a non-finite number (a refused update has no loss) is recorded as null."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="one seed, 300 steps (smoke run, not the protocol)")
    parser.add_argument("--out", default=None, help="write the JSON record here (explicit offline artifact)")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.steps = [0], 300
    torch.set_num_threads(args.threads)
    world = make_world(10, 5, seed=0)
    splits = build_splits(world, train=4096, val=512, comp=512, length=BASE["max_seq_len"], seed=0)
    runs = []
    for name in args.variants:
        for seed in args.seeds:
            started = time.perf_counter()
            runs.append(one_run(name, seed, splits, world, args.steps))
            last = runs[-1]
            print(f"[{name} seed {seed}] {last['status']} val_loss="
                  f"{last['val']['loss'] if last['val'] else float('nan'):.4f} "
                  f"({time.perf_counter() - started:.0f}s)", file=sys.stderr, flush=True)
    configs = {name: ModelConfig(**BASE, **VARIANTS[name][0]) for name in args.variants}
    record = {
        "phase": PHASE, "claim": "FEASIBILITY_ONLY SEMANTICS=NONE NO_LANGUAGE_CLAIM",
        "protocol": {"seeds": args.seeds, "steps": args.steps, "batch_size": 64, "sequence_length": BASE["max_seq_len"],
                     "tokens_per_run": args.steps * 64 * BASE["max_seq_len"], "optimizer": "AdamW lr 3e-3, betas "
                     "(0.9, 0.95), weight decay 0.01 on matrices, warmup 50, cosine to 10%, gradient clip 1.0",
                     "quick": args.quick, "adaptation": list(ADAPT)},
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "machine": platform.machine(), "threads": torch.get_num_threads(),
                        "cgc_upstream_commit": UPSTREAM_COMMIT},
        "data": splits.report,
        "variants": {name: {"config": VARIANTS[name][0], "parameter_relation_to_A": VARIANTS[name][1],
                            "channel_parameters_per_block": accounting.channel_parameters(configs[name]),
                            "compute_per_token_at_full_context": accounting.compute_per_token(
                                configs[name], BASE["max_seq_len"]),
                            "training_memory_estimate": accounting.memory_estimate(
                                configs[name], batch=64, length=BASE["max_seq_len"] - 1),
                            "session_memory": accounting.session_memory(configs[name])} for name in args.variants},
        "summary": summarize(runs), "runs": runs,
    }
    text = json.dumps(_finite(record), indent=1, sort_keys=True, allow_nan=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:     # explicit offline artifact, requested by --out
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
