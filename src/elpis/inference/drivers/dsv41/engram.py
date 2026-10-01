"""DeepSeek normalized-token address intake and gated associative write.

The historical AddressScheme arithmetic is reused unchanged. Unlike CompactTarget,
rows are concatenated, projected to per-copy keys/shared value and gated.
"""
import numpy as np
from ...associative import DSV41Parameters, _prime
from ...contracts import Code, RowIdentity, require
from ...text import V41Tokenizer
from .numerics import F32, linear, sigmoid


def build_parameters(tokenizer, config):
    """Boundary-only donor normalization, prime layout and per-layer RNG."""
    require(type(tokenizer) is V41Tokenizer and tokenizer.identity == config.tokenizer,
            Code.IDENTITY, "Engram tokenizer")
    require(bool(config.engram_layers), detail="DSV41 tower requires an Engram layer")
    from tokenizers import Regex, normalizers
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    backend = tokenizer._encoder
    keys, mapping = {}, []
    for token in range(tokenizer.vocab_size):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else normalizer.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        mapping.append(keys[key])
    vocab = len(keys)
    multipliers, primes, offsets, rows, seen = [], [], [], [], set()
    for layer in config.engram_layers:
        rng = np.random.default_rng(10007 * layer)
        bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
        multipliers.append(tuple(int(x) for x in rng.integers(0, bound, size=config.engram_order,
                                                             dtype=np.int64) * 2 + 1))
        sizes = []
        for _ in range(config.engram_order - 1):
            current = config.engram_bucket - 1
            for _ in range(config.engram_heads):
                current += 1
                while current in seen or not _prime(current):
                    current += 1
                seen.add(current)
                sizes.append(current)
        starts, total = [], 0
        for size in sizes:
            starts.append(total)
            total += size
        primes.append(tuple(sizes))
        offsets.append(tuple(starts))
        rows.append(total)
    pad = tokenizer.control_tokens["<｜▁pad▁｜>"]
    return DSV41Parameters(tokenizer.identity, tuple(mapping), vocab, mapping[pad], config.engram_layers,
                           config.engram_order, config.engram_heads, config.engram_dim,
                           tuple(multipliers), tuple(primes), tuple(offsets), tuple(rows))


def gated_write(stream, rows, weights, c):
    kv = linear(rows.reshape(-1), weights["wkv"])
    key, value = kv[:c.hc_mult*c.dimension].reshape(c.hc_mult, c.dimension), kv[c.hc_mult*c.dimension:]
    rstd = (F32(1) / np.sqrt(np.mean(stream * stream, axis=-1, dtype=F32) + F32(c.norm_eps)) *
            (F32(1) / np.sqrt(np.mean(key * key, axis=-1, dtype=F32) + F32(c.norm_eps))))
    dot = (np.sum(stream * (weights["q_weight"] * weights["k_weight"]) * key, axis=-1, dtype=F32)
           * rstd * F32(c.dimension ** -.5))
    gate = sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), F32(1e-6))), dot))
    return stream + gate[:, None] * value


class EngramLayer:
    def __init__(self, config, rows, weights):
        self.config, self.rows, self.weights = config, rows, weights
        require(rows.workers == 1, Code.UNSUPPORTED, "tower uses caller execution thread, no row thread pool")
        self.bank_id = rows.table.bank.digest

    def apply(self, stream, row_ids):
        values = self.rows.lookup(tuple(RowIdentity(self.bank_id, r) for r in row_ids))
        return gated_write(stream, values, self.weights, self.config)
