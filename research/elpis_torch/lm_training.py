"""R6: offline autoregressive training of ERSA-family models on ordinary token streams.

One path for every variant: ``ErsaLM`` or ``EvidenceLM`` (R5 retrieval on or off), dense or sparse (routed
FFN + CGC experts) channels. Causal next-token prediction; nothing here reads task labels, gold sources or
oracle addresses, and generation sees only its prompt.

Data. ``iter_documents`` streams documents from files (one file, one document; read lazily). ``document_batches``
makes rows that never cross a document: ``BOS doc EOS`` cut into windows of ``seq + 1`` tokens, the last window
padded with ``PAD``; targets at padding are ignored (``IGNORE``). A training row therefore starts from a fresh
recurrent state, and no context leaks between documents.

Training. A row of ``seq`` inputs is fed in chunks of ``chunk`` through the model's explicit state. Loss: the
summed cross-entropy over valid targets divided by their count (per batch), plus the routers' configured auxiliary
losses averaged over chunks. ``tbptt=False``: one graph through every chunk (full backpropagation through the
ERSA state across the row). ``tbptt=True``: backward after each chunk and the state DETACHED (truncated
backpropagation; gradients do not cross chunk boundaries). Gradient clipping, the warmup-cosine schedule of
``TrainConfig``, deterministic seeding, and refusal of any non-finite loss (training stops, status ``DIVERGED``).

What receives gradients with R5 on and no labels: the core, the routers and experts, and the evidence path only
through the blocks it actually selected (the readout and the selection prior); the coarse index and the hard
top-K selection get no direct signal from next-token loss. Label-free learning of addressing is NOT claimed.

Checkpoints are EXPLICIT OFFLINE ARTIFACTS (``save_lm_checkpoint``): architecture, expert topology, tokenizer
identity, parameters and their digest, step and seed, optionally the optimizer state. No live inference state
(ERSA cells, rings, caches) is ever serialized.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .ersa_model import ErsaConfig, ErsaLM
from .model import parameter_digest, sample_next
from .ple import PLEConfig
from .sparse_channel import SparseConfig, aux_loss
from .tokenizer import ByteTokenizer
from .training import TrainConfig

IGNORE = -100
SCHEMA = "elpis-torch-r7-lm/1"                     # R7: adds the PLE configuration and its addressing digest
READABLE_SCHEMAS = ("elpis-torch-r6-lm/1", SCHEMA)  # an R6 checkpoint is a model without PLE


@dataclass(frozen=True)
class LMConfig:
    seq: int = 256                  # tokens per training row (inputs)
    chunk: int = 64                 # tokens per forward call
    tbptt: bool = False             # True: detach the recurrent state between chunks
    eval_batches: int = 4

    def __post_init__(self):
        for name in ("seq", "chunk", "eval_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def tokenizer_identity(tokenizer) -> dict:
    """What a checkpoint records about its tokenizer (and checks on load)."""
    info = {"name": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
            "control": {k: getattr(tokenizer, k) for k in ("BOS", "EOS", "PAD") if hasattr(tokenizer, k)},
            "byte_fallback": bool(getattr(tokenizer, "byte_fallback", False))}
    info["digest"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return info


def check_vocabulary(model, tokenizer) -> None:
    """The model's embedding/output size must equal the tokenizer's vocabulary; control ids must be inside it."""
    size = model.config.vocab_size
    if tokenizer.vocab_size != size or model.embed.weight.shape[0] != size:
        raise ValueError(f"tokenizer vocabulary {tokenizer.vocab_size} != model vocabulary {size}")
    for name in ("BOS", "EOS", "PAD"):
        if hasattr(tokenizer, name) and not 0 <= getattr(tokenizer, name) < size:
            raise ValueError(f"control id {name} lies outside the vocabulary")


# -- data -------------------------------------------------------------------------------------------------------

def iter_documents(paths, *, suffixes=(".py", ".md", ".txt")):
    """Documents (bytes) from files and directories, in sorted order, read one at a time."""
    for root in paths:
        root = Path(root)
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes) if root.is_dir() else [root]
        for path in files:
            yield path.read_bytes()


def document_rows(documents, tokenizer, seq: int):
    """Rows of ``seq + 1`` token ids that never cross a document, and their byte counts."""
    pad = getattr(tokenizer, "PAD")
    for document in documents:
        ids = tokenizer.encode_bytes(document, bos=True, eos=True) if isinstance(tokenizer, ByteTokenizer) else \
            tokenizer.encode_bytes(document)
        for start in range(0, max(1, len(ids) - 1), seq):
            window = ids[start:start + seq + 1]
            if len(window) < 2:
                continue
            yield window + [pad] * (seq + 1 - len(window)), len(window) - 1


def document_batches(rows, batch: int, *, seed: int, ignore_pad: int):
    """Shuffled (seeded) batches: ``(inputs [B, seq], targets [B, seq])`` with padded targets ignored."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    for at in range(0, len(rows) - batch + 1, batch):
        part = torch.tensor([r for r, _ in rows[at:at + batch]], dtype=torch.long)
        inputs, targets = part[:, :-1], part[:, 1:].clone()
        valid = torch.zeros_like(targets, dtype=torch.bool)
        for i, (_, length) in enumerate(rows[at:at + batch]):
            valid[i, :length] = True
        targets[~valid] = IGNORE
        yield inputs, targets


# -- training ---------------------------------------------------------------------------------------------------

def _values(gradient: torch.Tensor) -> torch.Tensor:
    """The stored values of a gradient (a lookup table's gradient is sparse: only the rows it addressed)."""
    return gradient.coalesce().values() if gradient.is_sparse else gradient


def _detach(state) -> None:
    state.cells = [c.detach() for c in state.cells]
    state.tails = [None if t is None else t.detach() for t in state.tails]


def batch_loss(model, inputs, targets, lm: LMConfig, *, backward: bool):
    """Mean next-token loss over valid targets (+ routers' auxiliary losses). With ``lm.tbptt`` and
    ``backward`` the gradient is accumulated chunk by chunk and the state is detached between chunks."""
    batch, length = inputs.shape
    count = int((targets != IGNORE).sum())
    if count == 0:
        raise ValueError("a batch without a single valid target")
    state = model.new_state(batch)
    precision = torch.promote_types(model.embed.weight.dtype, torch.float32)      # never below the model's
    total = torch.zeros((), dtype=precision)
    cross = 0.0
    aux_terms, graph = [], None
    for at in range(0, length, lm.chunk):
        logits = model(inputs[:, at:at + lm.chunk], state)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).to(precision),
                             targets[:, at:at + lm.chunk].reshape(-1), ignore_index=IGNORE, reduction="sum") / count
        extra = aux_loss(model)
        cross += float(ce.detach())
        part = ce if extra is None else ce + extra.to(precision) / max(1, -(-length // lm.chunk))
        if extra is not None:
            aux_terms.append(float(extra.detach()))
        if backward and lm.tbptt:
            part.backward()
            _detach(state)
        elif backward:
            graph = part if graph is None else graph + part
        total = total + part.detach()
    if graph is not None:
        graph.backward()
    return total, {"targets": count, "cross_entropy": cross,
                   "aux": sum(aux_terms) / max(1, len(aux_terms)) if aux_terms else 0.0}


@torch.no_grad()
def evaluate_lm(model, batches, lm: LMConfig, *, bytes_per_target: float = 1.0) -> dict:
    """Held-out mean cross-entropy per target token (nats) and bits per byte (byte tokens: 1 byte each)."""
    was = model.training
    model.eval()
    total, targets = 0.0, 0
    for inputs, target in batches:
        _, info = batch_loss(model, inputs, target, lm, backward=False)
        total += info["cross_entropy"] * info["targets"]
        targets += info["targets"]
    model.train(was)
    nats = total / max(1, targets)
    return {"nats_per_token": nats, "bits_per_byte": nats / math.log(2) / bytes_per_target, "targets": targets}


def train_lm(model, train_batches, config: TrainConfig, lm: LMConfig, *, val_batches=None, log=None) -> dict:
    """Train in place for ``config.steps`` batches. Returns the history and the final status."""
    torch.manual_seed(config.seed)
    named = dict(model.named_parameters())
    trainable = [p for p in named.values() if p.requires_grad]
    # R7: lookup tables (sparse gradients) take lazy Adam, no weight decay, at ple.lr_multiplier x the learning rate:
    # a row that a step does not address is left exactly as it was by that step
    tables = [p for name, p in named.items() if p.requires_grad and name.endswith("ple.table.weight")]
    dense = [p for p in trainable if all(p is not t for t in tables)]
    optimizer = torch.optim.AdamW([{"params": [p for p in dense if p.ndim >= 2], "weight_decay": config.weight_decay},
                                   {"params": [p for p in dense if p.ndim < 2], "weight_decay": 0.0}],
                                  lr=config.lr, betas=(0.9, 0.95))
    lazy = torch.optim.SparseAdam(tables, lr=config.lr, betas=(0.9, 0.95)) if tables else None
    multiplier = model.ersa.ple.lr_multiplier if tables else 1.0
    history, status, step = [], "COMPLETED", 0
    iterator = iter(train_batches)
    while step < config.steps:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            break
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = config.lr_at(step)
        if lazy is not None:
            for group in lazy.param_groups:
                group["lr"] = config.lr_at(step) * multiplier
            lazy.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        loss, info = batch_loss(model, inputs, targets, lm, backward=True)
        if not torch.isfinite(loss) or any(p.grad is not None and not bool(torch.isfinite(_values(p.grad)).all())
                                           for p in trainable):
            status = "DIVERGED"
            optimizer.zero_grad(set_to_none=True)
            if lazy is not None:
                lazy.zero_grad(set_to_none=True)
            break
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        if lazy is not None and any(t.grad is not None for t in tables):
            lazy.step()
        step += 1
        if (config.eval_every and step % config.eval_every == 0) or step == config.steps:
            record = {"step": step, "loss": float(loss), "aux": info["aux"]}
            if val_batches is not None:
                record["val"] = evaluate_lm(model, val_batches, lm)
            history.append(record)
            if log is not None:
                log(record)
    return {"status": status, "steps": step, "history": history, "optimizer": optimizer, "table_optimizer": lazy}


@torch.no_grad()
def generate(model, prompt: torch.Tensor, count: int, *, chunk: int = 64, temperature: float = 0.0,
             seed: int = 0, prefill_calls: int = 1) -> torch.Tensor:
    """Free-running continuation: the prompt is fed in ``prefill_calls`` separate calls (each chunked), then one
    token at a time through the model's state. Only the prompt and the model's own samples are ever inputs."""
    was = model.training
    model.eval()
    state = model.new_state(prompt.shape[0])
    generator = torch.Generator().manual_seed(seed)
    logits = None
    pieces = torch.tensor_split(prompt, prefill_calls, dim=1)
    for piece in pieces:
        for at in range(0, piece.shape[1], chunk):
            logits = model(piece[:, at:at + chunk], state)
    out = []
    for _ in range(count):
        token = sample_next(logits[:, -1], temperature=temperature, generator=generator)
        out.append(token)
        logits = model(token[:, None], state)
    model.train(was)
    return torch.stack(out, dim=1)


# -- checkpoints ------------------------------------------------------------------------------------------------

def architecture(model) -> dict:
    ersa = {k: v for k, v in dataclasses.asdict(model.ersa).items() if k not in ("model", "sparse", "ple")}
    out = {"family": "evidence" if hasattr(model, "evidence") else "ersa", "model_config": model.config.to_dict(),
           "ersa_config": ersa, "sparse_config": None if model.ersa.sparse is None else model.ersa.sparse.to_dict(),
           "ple_config": None if model.ersa.ple is None else model.ersa.ple.to_dict(),
           "ple_addressing_digest": None if model.ple is None else model.ple.addresser.digest()}
    if hasattr(model, "evidence"):
        out["evidence_config"] = dataclasses.asdict(model.evidence)
    return out


def build_from(arch: dict):
    m = ModelConfig.from_dict(arch["model_config"])
    sparse = None if arch["sparse_config"] is None else SparseConfig(**{**arch["sparse_config"],
                                                                        "layers": tuple(arch["sparse_config"]["layers"])})
    ersa_fields = dict(arch["ersa_config"])
    if ersa_fields.get("attn_layers") is not None:
        ersa_fields["attn_layers"] = tuple(ersa_fields["attn_layers"])
    ple = None if arch.get("ple_config") is None else PLEConfig.from_dict(arch["ple_config"])
    ersa = ErsaConfig(m, sparse=sparse, ple=ple, **ersa_fields)
    if arch["family"] == "evidence":
        from .evidence_model import EvidenceConfig, EvidenceLM
        return EvidenceLM(ersa, EvidenceConfig(**arch["evidence_config"]))
    return ErsaLM(ersa)


def save_lm_checkpoint(model, path, *, tokenizer, step: int, seed: int, optimizer=None, extra: dict | None = None) -> str:
    """EXPLICIT OFFLINE ARTIFACT: architecture, tokenizer identity, parameters, digest, step, seed (and optionally
    the optimizer state for an explicit offline resumption). Returns the parameter digest."""
    digest = parameter_digest(model)
    record = {"schema": SCHEMA, "architecture": architecture(model), "tokenizer": tokenizer_identity(tokenizer),
              "parameter_digest": digest, "step": step, "seed": seed, **(extra or {})}
    payload = {"record": record, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return digest


def load_lm_checkpoint(path, *, tokenizer=None):
    """A frozen model from a checkpoint: architecture rebuilt, weights loaded strictly, digest and (optionally)
    tokenizer identity verified."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    record = saved["record"]
    if record.get("schema") not in READABLE_SCHEMAS:
        raise ValueError(f"not an {' or '.join(READABLE_SCHEMAS)} checkpoint")
    if tokenizer is not None and tokenizer_identity(tokenizer)["digest"] != record["tokenizer"]["digest"]:
        raise ValueError("the tokenizer does not match the checkpoint's tokenizer identity")
    model = build_from(record["architecture"])
    recorded = record["architecture"].get("ple_addressing_digest")
    if (None if model.ple is None else model.ple.addresser.digest()) != recorded:
        raise ValueError("the checkpoint's PLE addressing does not match this code's addressing for its configuration")
    model.load_state_dict(saved["state_dict"], strict=True)
    if parameter_digest(model) != record["parameter_digest"]:
        raise ValueError("the stored weights do not match the recorded parameter digest")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, record
