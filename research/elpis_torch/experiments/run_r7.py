#!/usr/bin/env python3
"""Elpis Torch R7: does a PLE supply useful information, and what does it cost? Records go only where --out says.

    study       matched training on this repository's own source and documentation bytes (held-out files chosen by
                a hash of their path). Variants:
                  r6            routed FFN + CGC experts (R6), no PLE
                  ple           the same plus a trainable PLE
                  ple_random    the same with content-free rows, redrawn on every call (CONTROL)
                  more_experts  R6 with more experts and the PLE model's total parameters (CAPACITY CONTROL)
                Each trained PLE model is also evaluated with its contribution removed. Outputs: held-out bits/byte
                on code and on prose, the gates, the rows used and the gradients
    local       positive control: a synthetic stream of fixed byte words, whose bytes the last two predict
    revision    negative control: R5 record streams, whose answers depend on the latest of several revisions
    collisions  hash statistics on the training corpus: distinct n-grams, rows used, n-grams sharing rows
    latency     decode and prefill time and rows read, with PLE off / resident / file-backed and R5 off / on
    accounting  formulas for the R6 500M / 1B configurations with a PLE (nothing is allocated)
    combine     one study record from the records of several study processes (--inputs a.json,b.json)
    smoke       every command once, tiny

Wall-clock numbers are PERFORMANCE_ONLY (one shared CPU, untrained weights where stated). A file-backed store
written by this process is read back from the page cache: no physical device read is measured.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import io
import statistics
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import evidence_tasks as et
from .. import lm_training as lmt
from ..config import ModelConfig
from ..ersa_model import ErsaConfig, ErsaLM
from ..evidence_model import EvidenceConfig, EvidenceLM
from ..ple import PLEConfig, ple_parameters
from ..ple_store import FORMAT_VERSION, MAGIC, FileStore, TensorStore
from ..sparse_channel import SparseConfig, active_parameters
from ..tokenizer import ByteTokenizer
from ..training import TrainConfig

PHASE = "ELPIS_TORCH_R7_PLE_ADMISSION"
REPO = Path(__file__).resolve().parents[3]
TOKENIZER = ByteTokenizer()
# Sized before training from the training corpus alone (``collisions`` at CORPUS_REVISION: 44,079 distinct trigrams,
# 4,918 bigrams): 65,536 rows per head keep the trigram load near 0.7, so ~6% of trigrams are expected to share a row
# in all four heads (measured: 5.7%); 8-wide rows make the concatenation (8 rows) as wide as the model. Fused before block 1, after one mixing
# round (Engram's single-layer optimum was the second layer).
STUDY_PLE = PLEConfig(orders=(2, 3), heads=4, rows=65536, head_dim=8, layer=1, boundary=ByteTokenizer.BOS)
STUDY_SPARSE = SparseConfig(experts=8, top_k=2, layers=(0, 1), ffn_hidden=86, cgc_channels=48, cgc_bottleneck=16)
VARIANTS = ("r6", "ple", "ple_random", "more_experts")
# The corpus is read from this revision (R7's base, the R6 head), so it never changes with the working tree and holds
# none of R7's own files. The smoke test reads the working tree (shallow CI checkouts lack the history).
CORPUS_REVISION = "c05adffafbd5d8a94ee40c285031d1cd219f1c4c"


# -- the explicit offline store writer ---------------------------------------------------------------------------

def export_store(table: torch.Tensor, path, *, addressing_digest: str, dtype: str = "float32") -> dict:
    """EXPLICIT OFFLINE ARTIFACT: write ``table [rows, dim]`` in the ``ple_store`` format at the stored precision."""
    values = table.detach().to(device="cpu", dtype=getattr(torch, dtype)).contiguous()
    rows, dim = values.shape
    data = ctypes.string_at(values.data_ptr(), values.numel() * values.element_size())   # the row-major bytes
    header = {"version": FORMAT_VERSION, "dtype": dtype, "rows": rows, "dim": dim, "row_bytes": dim * values.element_size(),
              "data_offset": 0, "addressing_digest": addressing_digest, "content_sha256": hashlib.sha256(data).hexdigest()}
    probe = json.dumps(header).encode()
    offset = -(-(12 + len(probe) + 32) // 64) * 64                      # room for the offset's own digits
    header["data_offset"] = offset
    text = json.dumps(header).encode()
    if 12 + len(text) > offset:
        raise ValueError("header too long for its reserved space")
    blob = MAGIC + len(text).to_bytes(4, "little") + text
    with open(path, "wb") as handle:                                     # explicit offline artifact
        handle.write(blob + bytes(offset - len(blob)) + data)
    return header


# -- corpus ------------------------------------------------------------------------------------------------------

def _repository_files(revision: str | None, root: Path = REPO) -> dict:
    """``{relative path: bytes}`` of ``src/``, ``research/elpis_torch/`` and ``docs/``: at a git revision (read with
    ``git archive``, nothing written) or, with ``revision=None``, from the working tree."""
    tops = ("src", "research/elpis_torch", "docs")
    if revision is None:
        return {p.relative_to(root).as_posix(): p.read_bytes() for top in tops for p in (root / top).rglob("*")
                if p.is_file()}
    blob = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", revision, *tops], capture_output=True,
                          check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def corpus(revision: str | None = CORPUS_REVISION, root: Path = REPO) -> dict:
    """Project-owned source (``src/``, ``research/elpis_torch/`` without ``upstream``) and documentation
    (``docs/``) bytes, one document per file, at ``revision``. A file is held out when SHA-256 of its relative path
    is 0 mod 10."""
    files = _repository_files(revision, root)
    code = sorted(p for p in files if p.endswith(".py") and "/__pycache__/" not in p and "/upstream/" not in p
                  and (p.startswith("src/") or p.startswith("research/elpis_torch/")))
    prose = sorted(p for p in files if p.endswith(".md") and p.startswith("docs/"))
    out = {"train": [], "val_code": [], "val_prose": []}
    digest = hashlib.sha256()
    for kind, paths in (("code", code), ("prose", prose)):
        for rel in paths:
            data = files[rel]
            digest.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
            held = int(hashlib.sha256(rel.encode()).hexdigest(), 16) % 10 == 0
            out[f"val_{kind}" if held else "train"].append(data)
    out["digest"], out["revision"] = digest.hexdigest(), revision
    out["summary"] = {k: {"documents": len(out[k]), "bytes": sum(map(len, out[k]))}
                      for k in ("train", "val_code", "val_prose")}
    return out


def _val_batches(documents, lm, count: int, seed: int = 99):
    rows = list(lmt.document_rows(documents, TOKENIZER, lm.seq))
    return list(lmt.document_batches(rows, 16, seed=seed, ignore_pad=TOKENIZER.PAD))[:count]


# -- models ------------------------------------------------------------------------------------------------------

def _widths(seed: int, vocab: int = 259) -> ModelConfig:
    return ModelConfig(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
                       channel="ffn+cgc", ffn_hidden=171, cgc_bottleneck=32, cgc_channels=96, cgc_units=4,
                       cgc_gate_init=0.1, seed=seed)


def study_model(variant: str, seed: int, *, experts: int | None = None, ple: PLEConfig = STUDY_PLE,
                vocab: int = 259):
    sparse, table = STUDY_SPARSE, None
    if variant == "ple":
        table = ple
    elif variant == "ple_random":
        table = ple.replace(addressing="random", seed=seed)
    elif variant == "more_experts":
        sparse = dataclasses.replace(STUDY_SPARSE, experts=experts or capacity_control_size(ple, vocab))
    elif variant != "r6":
        raise ValueError(variant)
    return ErsaLM(ErsaConfig(_widths(seed, vocab), conv_kernel=4, chunk=64, ersa_kernel="chunked", scan_block=64,
                             sparse=sparse, ple=table))


def capacity_control_size(ple: PLEConfig = STUDY_PLE, vocab: int = 259) -> int:
    """The expert count per block whose total parameters come closest to the PLE model's (meta device)."""
    with torch.device("meta"):
        target = active_parameters(study_model("ple", 0, ple=ple, vocab=vocab))["total"]
        best = None
        for experts in range(STUDY_SPARSE.experts, 129):
            sparse = dataclasses.replace(STUDY_SPARSE, experts=experts)
            total = active_parameters(ErsaLM(ErsaConfig(_widths(0, vocab), conv_kernel=4, chunk=64,
                                                        ersa_kernel="chunked", scan_block=64,
                                                        sparse=sparse)))["total"]
            if best is None or abs(total - target) < abs(best[1] - target):
                best = (experts, total)
            if total > target:
                break
    return best[0]


# -- diagnostics -------------------------------------------------------------------------------------------------

class _RowsUsed:
    """Counts the table rows addressed while attached (a forward hook on the table; detached by ``close``)."""

    def __init__(self, model):
        self.touched = torch.zeros(model.ple.table.num_embeddings, dtype=torch.bool)
        self.hook = model.ple.table.register_forward_hook(self._seen)

    def _seen(self, module, inputs, output):
        self.touched[inputs[0].reshape(-1).cpu()] = True

    def close(self) -> float:
        self.hook.remove()
        return float(self.touched.float().mean())


@torch.no_grad()
def _gates(model, batches, lm) -> dict:
    values = []
    for inputs, _ in batches:
        state = model.new_state(inputs.shape[0])
        for at in range(0, inputs.shape[1], lm.chunk):
            model(inputs[:, at:at + lm.chunk], state)
            values.append(model.ple.last_gate.flatten().float())
    g = torch.cat(values)
    q = torch.quantile(g, torch.tensor([0.05, 0.5, 0.95]))
    return {"mean": float(g.mean()), "std": float(g.std()), "p05": float(q[0]), "p50": float(q[1]),
            "p95": float(q[2]), "below_0.05": float((g < 0.05).float().mean()),
            "above_0.95": float((g > 0.95).float().mean()), "positions": g.numel()}


def _gradient_flow(model, batch, lm) -> dict:
    model.zero_grad(set_to_none=True)
    model.train()
    lmt.batch_loss(model, *batch, lm, backward=True)
    model.eval()
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = ("ple." + name.split("ple.", 1)[1].split(".")[0] if name.startswith("ple.") else
               "routed_experts" if ".experts." in name else "ersa" if ".ersa." in name else
               "embeddings" if name.startswith("embed.") else "other")
        value = lmt._values(p.grad).double()
        out.setdefault(key, 0.0)
        out[key] += float(value.pow(2).sum())
        if name == "ple.table.weight":
            out["ple.table_rows_with_gradient"] = int(p.grad.coalesce().indices().unique().numel())
    model.zero_grad(set_to_none=True)
    return {k: (math.sqrt(v) if not k.endswith("_gradient") else v) for k, v in out.items()}


@torch.no_grad()
def _position_losses(model, inputs, targets, chunk: int) -> torch.Tensor:
    """Per-position cross-entropy (nats) in float32 or wider, IGNORE positions as NaN."""
    model.eval()
    state, parts = model.new_state(inputs.shape[0]), []
    for at in range(0, inputs.shape[1], chunk):
        logits = model(inputs[:, at:at + chunk], state)
        t = targets[:, at:at + chunk]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).double(), t.reshape(-1).clamp(min=0),
                             reduction="none").view(t.shape)
        parts.append(torch.where(t == lmt.IGNORE, torch.full_like(ce, float("nan")), ce))
    return torch.cat(parts, dim=1)


# -- the study ---------------------------------------------------------------------------------------------------

def _train(model, rows, steps: int, seed: int, lm) -> dict:
    started = time.perf_counter()
    result = lmt.train_lm(model, lmt.document_batches(rows, 16, seed=seed, ignore_pad=TOKENIZER.PAD),
                          TrainConfig(steps=steps, batch_size=16, lr=3e-3, warmup_steps=min(50, steps), seed=seed), lm)
    return {"status": result["status"], "steps": result["steps"], "seconds": round(time.perf_counter() - started, 1)}


def command_study(args) -> dict:
    lm = lmt.LMConfig(seq=128, chunk=64)
    data = corpus(None if args.smoke else CORPUS_REVISION)
    steps = 8 if args.smoke else args.steps
    count = 2 if args.smoke else args.val_batches
    val = {"code": _val_batches(data["val_code"], lm, count), "prose": _val_batches(data["val_prose"], lm, count)}
    train_rows = list(lmt.document_rows(data["train"], TOKENIZER, lm.seq))
    variants = args.variants.split(",") if args.variants else VARIANTS
    seeds = (0,) if args.smoke else tuple(int(s) for s in args.seed_list.split(","))
    out = {"phase": PHASE, "corpus": {"digest": data["digest"], "revision": data["revision"], **data["summary"],
                                      "source": "src/ and research/elpis_torch/ (*.py, without upstream/) and docs/ "
                                                "(*.md) of this repository; held out: SHA-256(path) % 10 == 0"},
           "steps": steps, "tokens_per_step": 16 * lm.seq, "ple": STUDY_PLE.to_dict(),
           "matched_experts": capacity_control_size(), "rows": []}
    for variant in variants:
        for seed in seeds:
            model = study_model(variant, seed)
            counts = active_parameters(model)
            row = {"variant": variant, "seed": seed, "parameters": counts}
            used = _RowsUsed(model) if model.ple is not None else None
            row["train"] = _train(model, train_rows, steps, seed, lm)
            if used is not None:
                row["rows_used_in_training"] = used.close()
            model.eval()
            for name, batches in val.items():
                row[f"val_{name}_bits_per_byte"] = lmt.evaluate_lm(model, batches, lm)["bits_per_byte"]
            if model.ple is not None:
                held = _RowsUsed(model)
                row["gates_val_code"] = _gates(model, val["code"], lm)
                row["rows_used_val_code"] = held.close()
                row["gradient_norms"] = _gradient_flow(model, val["code"][0], lm)
                model.ple.enabled = False                                    # removal: the same weights
                for name, batches in val.items():
                    row[f"val_{name}_bits_per_byte_ple_removed"] = lmt.evaluate_lm(model, batches, lm)["bits_per_byte"]
                model.ple.enabled = True
            out["rows"].append(row)
            print(f"[r7 study] {variant} s{seed}: code {row['val_code_bits_per_byte']:.4f} prose "
                  f"{row['val_prose_bits_per_byte']:.4f} bits/byte; total {counts['total']}, active "
                  f"{counts['active_per_token']} ({row['train']['seconds']} s)", flush=True)
            if args.out:
                _write(args.out, out)
    return out


# -- controls ----------------------------------------------------------------------------------------------------

def _word_documents(count: int, seed: int, words: list, per_document: int = 160) -> list:
    rng = random.Random(f"elpis-torch-r7-local:{seed}")
    return [b" ".join(rng.choice(words) for _ in range(per_document)) for _ in range(count)]


def command_local(args) -> dict:
    """Positive control: 512 fixed lowercase words of 3-7 letters; documents are uniform word sequences. Inside a
    word the last two bytes nearly fix the next one; a word's first letter carries ~log2(512)/word entropy."""
    rng = random.Random("elpis-torch-r7-local:words")
    words = set()
    while len(words) < 512:
        words.add(bytes(rng.choice(b"abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 7))))
    words = sorted(words)
    lm = lmt.LMConfig(seq=128, chunk=64)
    train = _word_documents(16 if args.smoke else 400, 1, words)
    val_docs = _word_documents(4 if args.smoke else 40, 2, words)
    val_rows = list(lmt.document_rows(val_docs, TOKENIZER, lm.seq))
    val = list(lmt.document_batches(val_rows, 16, seed=99, ignore_pad=TOKENIZER.PAD))[:2 if args.smoke else 24]
    rows = list(lmt.document_rows(train, TOKENIZER, lm.seq))
    steps = 6 if args.smoke else args.control_steps
    seeds = (0,) if args.smoke else (0, 1)
    out = {"phase": PHASE, "task": "512 fixed words, uniform word sequences", "steps": steps, "rows": []}
    for variant in ("r6", "ple", "ple_random"):
        for seed in seeds:
            model = study_model(variant, seed)
            row = {"variant": variant, "seed": seed, "train": _train(model, rows, steps, seed, lm)}
            losses = torch.cat([_position_losses(model, i, t, lm.chunk) for i, t in val])
            inputs = torch.cat([i for i, _ in val])
            inside = (inputs != ord(" ")) & (inputs != TOKENIZER.BOS)            # the next byte after a letter
            valid = ~torch.isnan(losses)
            to_bits = 1 / math.log(2)
            row["val_bits_per_byte"] = float(losses[valid].mean()) * to_bits
            row["inside_word_bits_per_byte"] = float(losses[valid & inside].mean()) * to_bits
            if model.ple is not None:
                model.ple.enabled = False
                removed = torch.cat([_position_losses(model, i, t, lm.chunk) for i, t in val])
                row["val_bits_per_byte_ple_removed"] = float(removed[valid].mean()) * to_bits
                model.ple.enabled = True
            out["rows"].append(row)
            print(f"[r7 local] {variant} s{seed}: {row['val_bits_per_byte']:.3f} bits/byte "
                  f"(inside words {row['inside_word_bits_per_byte']:.3f})", flush=True)
    return out


def command_revision(args) -> dict:
    """Negative control: R5 record streams (revised binds, hard negatives). Answers depend on the latest bind of a
    key, possibly hundreds of tokens back: next-token loss at answer positions, overall and for revised keys."""
    task = et.RecordTask(keys=64, queries=12, revisions=(1, 3), background=8.0)
    length = 512 if args.smoke else 2048
    train = et.generate_records(task, 8 if args.smoke else 96, length, seed=31)
    val = et.generate_records(task, 4 if args.smoke else 16, length, seed=32)
    lm = lmt.LMConfig(seq=128, chunk=64)
    rows = []
    for stream in train.tokens.tolist():
        for at in range(0, length - 1, lm.seq):
            window = stream[at:at + lm.seq + 1]
            rows.append((window + [0] * (lm.seq + 1 - len(window)), len(window) - 1))
    steps = 6 if args.smoke else args.control_steps
    seeds = (0,) if args.smoke else (0, 1)
    ple = STUDY_PLE.replace(boundary=None)
    out = {"phase": PHASE, "task": "R5 records, revisions 1-3, 2,048-token streams", "steps": steps, "rows": []}
    inputs, targets = val.tokens[:, :-1], val.tokens[:, 1:]
    answer_rows, answer_pos = val.probes["row"], val.probes["answer"] - 1        # predicting the answer token
    revised = val.probes["revised"]
    for variant in ("r6", "ple"):
        for seed in seeds:
            model = study_model(variant, seed, ple=ple, vocab=et.VOCAB)
            batches = lmt.document_batches(rows, 16, seed=seed, ignore_pad=-1)
            started = time.perf_counter()
            result = lmt.train_lm(model, batches, TrainConfig(steps=steps, batch_size=16, lr=3e-3,
                                                              warmup_steps=min(50, steps), seed=seed), lm)
            row = {"variant": variant, "seed": seed, "train": {"status": result["status"],
                                                               "seconds": round(time.perf_counter() - started, 1)}}
            for label, enabled in (("", True), ("_ple_removed", False)):
                if model.ple is None and not enabled:
                    continue
                if model.ple is not None:
                    model.ple.enabled = enabled
                losses = _position_losses(model, inputs, targets, lm.chunk)
                at = losses[answer_rows, answer_pos]
                row[f"all_nats{label}"] = float(losses.nanmean())
                row[f"answer_nats{label}"] = float(at.mean())
                row[f"answer_nats_revised{label}"] = float(at[revised].mean()) if bool(revised.any()) else None
            if model.ple is not None:
                model.ple.enabled = True
            out["rows"].append(row)
            print(f"[r7 revision] {variant} s{seed}: all {row['all_nats']:.3f} answers {row['answer_nats']:.3f} "
                  f"nats", flush=True)
    return out


# -- collisions --------------------------------------------------------------------------------------------------

def command_collisions(args) -> dict:
    """Distinct n-grams of the training corpus (exact, BOS-bounded as the model sees them), the rows they map to
    per head, and how many share a row with another n-gram in one head and in every head of their order."""
    from ..ple import NgramAddresser
    data = corpus(None if args.smoke else CORPUS_REVISION)
    docs = data["train"][:8] if args.smoke else data["train"]
    addresser = NgramAddresser(STUDY_PLE, 259)
    grams = {n: set() for n in STUDY_PLE.orders}
    for document in docs:
        ids = torch.tensor([TOKENIZER.encode_bytes(document, bos=True, eos=True)], dtype=torch.long)
        found = addresser.grams(ids, addresser.new_history(1))
        for n in STUDY_PLE.orders:
            grams[n].update(map(tuple, found[n][0].tolist()))
    out = {"phase": PHASE, "ple": STUDY_PLE.to_dict(), "table_sizes": addresser.sizes, "orders": {},
           "corpus": {"digest": data["digest"], "revision": data["revision"], **data["summary"]}}
    for i, n in enumerate(STUDY_PLE.orders):
        table = torch.tensor(sorted(grams[n]), dtype=torch.long)
        rows = addresser.rows_of(i, table)                                       # [N, heads]
        shared_any = torch.zeros(len(table), dtype=torch.bool)
        shared_all = torch.ones(len(table), dtype=torch.bool)
        heads = []
        for h in range(STUDY_PLE.heads):
            _, inverse, counts = torch.unique(rows[:, h], return_inverse=True, return_counts=True)
            shared = counts[inverse] > 1
            shared_any |= shared
            shared_all &= shared
            heads.append({"rows_used": int(counts.numel()), "table_rows": addresser.sizes[i * STUDY_PLE.heads + h],
                          "ngrams_sharing_a_row": float(shared.float().mean())})
        out["orders"][str(n)] = {"distinct_ngrams": len(table), "heads": heads,
                                 "shared_in_some_head": float(shared_any.float().mean()),
                                 "shared_in_every_head": float(shared_all.float().mean())}
        print(f"[r7 collisions] order {n}: {len(table)} distinct; sharing in every head "
              f"{float(shared_all.float().mean()):.4f}", flush=True)
    return out


# -- latency and bytes -------------------------------------------------------------------------------------------

def latency_model(ple: bool, evidence: bool, *, seed: int = 0):
    m = ModelConfig(vocab_size=259, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=64,
                    channel="ffn+cgc", ffn_hidden=683, cgc_bottleneck=64, cgc_channels=256, cgc_units=4,
                    cgc_gate_init=0.1, seed=seed)
    sparse = SparseConfig(experts=8, top_k=2, layers=(1, 3), ffn_hidden=341, cgc_channels=128, cgc_bottleneck=32)
    table = PLEConfig(orders=(2, 3), heads=4, rows=32768, head_dim=32, layer=1, boundary=ByteTokenizer.BOS) if ple \
        else None
    ersa = ErsaConfig(m, conv_kernel=4, chunk=512, ersa_kernel="chunked", scan_block=64, sparse=sparse, ple=table)
    model = EvidenceLM(ersa, EvidenceConfig(capacity=16384)) if evidence else ErsaLM(ersa)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _percentiles(samples: list) -> dict:
    ordered = sorted(samples)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]   # noqa: E731
    return {"p50_ms": round(1000 * pick(0.5), 3), "p95_ms": round(1000 * pick(0.95), 3),
            "mean_ms": round(1000 * statistics.fmean(ordered), 3), "samples": len(ordered)}


LATENCY_MODES = ("off", "resident", "file")


def _timed(fn, *args) -> float:
    started = time.perf_counter()
    fn(*args)
    return time.perf_counter() - started


@torch.no_grad()
def command_latency(args) -> dict:
    """Paired, interleaved timing. For R5 off and on, ONE model serves three identical streams (PLE off; resident
    rows; file-backed rows), fed the same fixed tokens mode after mode in a rotating order, so drift and noise
    touch all three alike; per-step differences against PLE off are reported beside each mode's own percentiles.
    Then the PLE's parts alone: addressing, row gather, fusion."""
    steps, prefills = (8, 2) if args.smoke else (args.decode_steps, args.prefill_runs)
    generator = torch.Generator().manual_seed(0)
    warm = torch.randint(0, 256, (1, 1024), generator=generator)
    chunks = torch.randint(0, 256, (prefills, 1, 512), generator=generator)
    feed = torch.randint(0, 256, (steps, 1, 1), generator=generator)              # the same tokens for every mode
    out = {"phase": PHASE, "threads": torch.get_num_threads(), "decode_steps": steps, "prefill_runs": prefills,
           "model": "d 256, 4 blocks, 8 experts top-2 in blocks 1 and 3, PLE 2 orders x 4 heads x 32771 rows x 32 "
                    "(before block 1), UNTRAINED (timing only)", "rows": []}
    folder = tempfile.mkdtemp(prefix="elpis-r7-store-")
    path = os.path.join(folder, "table.ple")
    try:
        for evidence in (False, True):
            model = latency_model(True, evidence)
            export_store(model.ple.table.weight, path, addressing_digest=model.ple.addresser.digest())
            store = FileStore(path)
            states = {mode: model.new_state(1) for mode in LATENCY_MODES}

            def run(mode, tokens):
                model.ple.enabled = mode != "off"
                model.ple.attach_store(store if mode == "file" else None)
                return _timed(model, tokens, states[mode])

            for at in range(0, warm.shape[1], 512):
                for mode in LATENCY_MODES:
                    run(mode, warm[:, at:at + 512])
            prefill = {mode: [] for mode in LATENCY_MODES}
            decode = {mode: [] for mode in LATENCY_MODES}
            for i in range(prefills):
                for k in range(3):
                    mode = LATENCY_MODES[(i + k) % 3]
                    prefill[mode].append(run(mode, chunks[i]))
            store.reset_counters()
            for i in range(steps):
                for k in range(3):
                    mode = LATENCY_MODES[(i + k) % 3]
                    decode[mode].append(run(mode, feed[i]))
            counters = dict(store.counters)
            model.ple.attach_store(None)
            store.close()
            for mode in LATENCY_MODES:
                row = {"r5": evidence, "ple": mode, "prefill_512": _percentiles(prefill[mode]),
                       "decode_per_token": _percentiles(decode[mode]),
                       "rows_read_per_token": model.ple.config.lookups if mode != "off" else 0,
                       "logical_bytes_read_per_token": model.ple.lookup_bytes_per_token() if mode != "off" else 0}
                if mode != "off":
                    row["decode_minus_off"] = _percentiles([a - b for a, b in zip(decode[mode], decode["off"])])
                    row["prefill_minus_off"] = _percentiles([a - b for a, b in zip(prefill[mode], prefill["off"])])
                if mode == "file":
                    row["store_counters_during_decode"] = counters
                out["rows"].append(row)
                print(f"[r7 latency] r5={evidence} ple={mode}: decode p50 {row['decode_per_token']['p50_ms']} ms"
                      + (f" (paired difference p50 {row['decode_minus_off']['p50_ms']} ms)" if mode != "off" else "")
                      + f", prefill p50 {row['prefill_512']['p50_ms']} ms", flush=True)
        model = latency_model(True, False)
        repeats = 2 if args.smoke else 200
        parts = {}
        for length in (1, 512):
            tokens = chunks[0][:, :length]
            history = model.ple.addresser.new_history(1)
            addresses = model.ple.addresser.addresses(tokens, history)
            x = torch.randn(1, length, model.config.d_model, generator=generator)
            parts[f"addressing_{length}_tokens"] = _percentiles(
                [_timed(model.ple.addresser.addresses, tokens, history) for _ in range(repeats)])
            parts[f"row_gather_{length}_tokens"] = _percentiles([_timed(model.ple.table, addresses)
                                                                for _ in range(repeats)])
            parts[f"fusion_with_gather_{length}_tokens"] = _percentiles([_timed(model.ple, x, addresses)
                                                                        for _ in range(repeats)])
        out["ple_parts_resident"] = parts
        out["cold_device_reads"] = "NOT_RUN: the store is read back from the page cache in this container"
    finally:
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        os.rmdir(folder)
    return out


# -- accounting --------------------------------------------------------------------------------------------------

def command_accounting(args) -> dict:
    """PROJECTED: a PLE beside the R6 scale configurations, from formulas (nothing is allocated)."""
    from .run_r6 import SCALES, per_token_macs, scale_model
    out = {"phase": PHASE, "note": "formulas; PLE before block 1; orders (2, 3); 8 heads per order", "rows": []}
    for name in SCALES:
        with torch.device("meta"):
            model = scale_model(name)
        base = per_token_macs(model, 1 << 20)
        d = model.config.d_model
        for rows in (1 << 16, 1 << 20, 1 << 24):
            config = PLEConfig(orders=(2, 3), heads=8, rows=rows, head_dim=64, layer=1)
            sizes = ple_parameters(config, d, model.config.vocab_size)
            out["rows"].append({
                "configuration": name, "rows_per_head": rows, "table_parameters": sizes["table"],
                "table_bytes": {"fp32": 4 * sizes["table"], "bf16": 2 * sizes["table"]},
                "rows_read_per_token": config.lookups,
                "bytes_read_per_token": {"fp32": 4 * config.width, "bf16": 2 * config.width},
                "fusion_parameters": sizes["fusion"], "fusion_macs_per_token": 2 * config.width * d + 2 * d,
                "fusion_macs_over_active_weight_macs": (2 * config.width * d) / base["active_weights"],
                "row_values_over_active_weights_per_token": config.width / base["active_weights"],
                "addressing_integer_ops_per_token": sum(2 * n for n in config.orders) * config.heads})
    return out


def command_combine(args) -> dict:
    """Merge study records written by parallel processes (same settings, disjoint variants)."""
    parts = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs.split(",")]
    keep = ("corpus", "steps", "tokens_per_step", "ple", "matched_experts", "phase")
    if any({k: p.get(k) for k in keep} != {k: parts[0].get(k) for k in keep} for p in parts):
        raise ValueError("the records were not produced with the same settings")
    out = {k: parts[0][k] for k in keep}
    out["rows"] = [row for part in parts for row in part["rows"]]
    out["processes"] = [part.get("environment") for part in parts]
    return out


def command_smoke(args) -> dict:
    ns = argparse.Namespace(**{**vars(args), "smoke": True, "out": None, "variants": None})
    return {"collisions": len(command_collisions(ns)["orders"]), "accounting": len(command_accounting(ns)["rows"]),
            "latency": len(command_latency(ns)["rows"]), "local": len(command_local(ns)["rows"]),
            "revision": len(command_revision(ns)["rows"]), "study": len(command_study(ns)["rows"])}


def _finite(value):
    if isinstance(value, float):          # six significant digits (never decimal places: they erase small values)
        return float(f"{value:.6g}") if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _write(path: str, record: dict) -> None:
    environment = {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
                   "cuda_available": torch.cuda.is_available(),
                   "cpu_capability": torch.backends.cpu.get_cpu_capability()}
    text = json.dumps(_finite(dict(record, environment=environment)), indent=1, sort_keys=True, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(text + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["study", "local", "revision", "collisions", "latency", "accounting",
                                            "combine", "smoke"])
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--seed-list", default="0,1,2")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--val-batches", type=int, default=48)
    parser.add_argument("--decode-steps", type=int, default=200)
    parser.add_argument("--prefill-runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    result = {"study": command_study, "local": command_local, "revision": command_revision,
              "collisions": command_collisions, "latency": command_latency, "accounting": command_accounting,
              "combine": command_combine, "smoke": command_smoke}[args.command](args)
    if args.out and args.command != "study":
        _write(args.out, result)
    elif not args.out:
        print(json.dumps(_finite(result), indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
