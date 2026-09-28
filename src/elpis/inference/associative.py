"""Value-state n-gram schemes; synthetic qualification, not trained-table compatibility.

Derived mechanics: DeepSeek V4.1 inference/engram.py (MIT,
UPSTREAM_REVISION_UNPINNED, local SHA256
11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897),
and vLLM ngram_embedding.py (Apache-2.0, revision
82daf9f5756e1868be0aa751afaec4726beca12a). Exact donor paths and licenses:
LICENSES/PROVENANCE.md. No donor runtime imports.
"""
from dataclasses import dataclass
import numpy as np
from .contracts import Code, Bank, ContractError, digest_value, identity, integer, require

DSV41_ENGRAM = 'DSV41_ENGRAM/v1'
QWEN38_PLE = 'QWEN38_PLE/v1'


def load_parameters(artifact, *, expected_digest):
    """Parse an already decoded JSON artifact; callers pin its semantic identity."""
    require(type(artifact) is dict, detail='parameter artifact')
    def freeze(x):
        return tuple(freeze(v) for v in x) if type(x) is list else x
    values = {k: freeze(v) for k, v in artifact.items()}
    cls = {DSV41_ENGRAM: DSV41Parameters, QWEN38_PLE: QwenPLEParameters}.get(values.get('schema'))
    require(cls is not None, Code.IDENTITY, 'unknown scheme')
    try:
        parameters = cls(**values)
    except TypeError as exc:
        raise ContractError(Code.INVALID, 'parameter fields') from exc
    require(parameters.digest == expected_digest, Code.IDENTITY, 'parameter digest')
    return parameters


def _prime(n):
    # Deterministic Miller-Rabin over the unsigned 64-bit domain.
    integer(n, 2)
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _tuple(x, length):
    require(type(x) is tuple and len(x) == length, detail='frozen array dimensions')


def _geometry(order, heads, multipliers, sizes, offsets, rows, vocab):
    integer(order, 2, 32)
    integer(heads, 1, 1024)
    _tuple(multipliers, order)
    _tuple(sizes, (order - 1) * heads)
    _tuple(offsets, len(sizes))
    for m in multipliers:
        integer(m, 1, ((1 << 63) - 1) // vocab)
        require(m % 2 == 1, detail='odd multiplier')
    offset = 0
    for size, start in zip(sizes, offsets):
        require(_prime(size), detail='prime modulus')
        integer(start)
        require(start == offset, detail='cumulative offset')
        offset += size
    integer(rows, offset)


@dataclass(frozen=True)
class DSV41Parameters:
    tokenizer: str
    token_map: tuple[int, ...]
    compressed_vocab: int
    pad: int
    layers: tuple[int, ...]
    order: int
    heads: int
    head_dimension: int
    multipliers: tuple[tuple[int, ...], ...]
    primes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, ...], ...]
    table_rows: tuple[int, ...]
    dead: int = -1
    schema: str = DSV41_ENGRAM

    def __post_init__(self):
        require(self.schema == DSV41_ENGRAM and self.dead == -1, detail='scheme/dead sentinel')
        require(type(self.tokenizer) is str and bool(self.tokenizer), detail='tokenizer')
        integer(self.compressed_vocab, 1, (1 << 31) - 1)
        integer(self.pad, 0, self.compressed_vocab - 1)
        require(type(self.token_map) is tuple and len(self.token_map) > 0)
        for t in self.token_map:
            integer(t, 0, self.compressed_vocab - 1)
        require(len(set(self.token_map)) == self.compressed_vocab, detail='compressed vocabulary')
        require(type(self.layers) is tuple and len(self.layers) > 0)
        for layer in self.layers:
            integer(layer)
        require(len(set(self.layers)) == len(self.layers), detail='duplicate layer')
        integer(self.head_dimension, 1)
        for values in (self.multipliers, self.primes, self.offsets, self.table_rows):
            _tuple(values, len(self.layers))
        seen = set()
        for m, p, o, r in zip(self.multipliers, self.primes, self.offsets, self.table_rows):
            _geometry(self.order, self.heads, m, p, o, r, self.compressed_vocab)
            require(r == sum(p), detail='table row count')
            require(len(set(p)) == len(p) and not seen.intersection(p), detail='global prime uniqueness')
            seen.update(p)

    @property
    def digest(self):
        return identity('dsv41.parameters', self)


@dataclass(frozen=True)
class QwenPLEParameters:
    tokenizer: str
    vocab: int
    eos: int
    order: int
    heads: int
    embedding_dimension: int
    multipliers: tuple[int, ...]
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    padded_rows: int
    layer: int = 0
    schema: str = QWEN38_PLE

    def __post_init__(self):
        require(self.schema == QWEN38_PLE, detail='scheme')
        require(type(self.tokenizer) is str and bool(self.tokenizer))
        integer(self.vocab, 1)
        integer(self.eos, 0, self.vocab - 1)
        integer(self.layer)
        _geometry(self.order, self.heads, self.multipliers, self.sizes,
                  self.offsets, self.padded_rows, self.vocab)
        require(len(set(self.sizes)) == len(self.sizes), detail='duplicate modulus')
        integer(self.embedding_dimension, 1)
        require(self.embedding_dimension % ((self.order - 1) * self.heads) == 0,
                detail='embedding dimensions')

    @property
    def digest(self):
        return identity('qwen38.parameters', self)


@dataclass(frozen=True)
class History:
    scheme: str
    parameters: str
    tokenizer: str
    position: int
    tail: tuple[int, ...]  # newest first

    @property
    def digest(self):
        return identity('ngram.history', self)


@dataclass(frozen=True)
class HashResult:
    scheme: str
    parameters: str
    input_history: str
    rows: tuple  # [token][layer][column]
    active: tuple[bool, ...]
    history: History

    @property
    def digest(self):
        return identity('ngram.result', self)


class AddressScheme:
    def __init__(self, parameters, *, expected_digest, tokenizer, scheme):
        require(type(parameters) in (DSV41Parameters, QwenPLEParameters), detail='parameter type')
        digest_value(expected_digest)
        require(parameters.digest == expected_digest and tokenizer == parameters.tokenizer and
                scheme == parameters.schema, Code.IDENTITY, 'address scheme/artifact/tokenizer')
        self.parameters = parameters

    def initial(self):
        p = self.parameters
        pad = p.dead if type(p) is DSV41Parameters else p.eos
        return History(p.schema, p.digest, p.tokenizer, 0, (pad,) * (p.order - 1))

    def hash(self, history, tokens, mask=None, *, expected_history):
        p = self.parameters
        ds = type(p) is DSV41Parameters
        require(type(history) is History, detail='history type')
        require(history.digest == expected_history, Code.STALE, 'history predecessor')
        require((history.scheme, history.parameters, history.tokenizer) ==
                (p.schema, p.digest, p.tokenizer), Code.IDENTITY, 'history binding')
        integer(history.position)
        _tuple(history.tail, p.order - 1)
        vocab = p.compressed_vocab if ds else p.vocab
        for t in history.tail:
            integer(t, -1 if ds else 0, vocab - 1)
        require(type(tokens) is tuple, detail='tokens must be a value')
        for t in tokens:
            integer(t, 0, (len(p.token_map) if ds else p.vocab) - 1)
        mask = (True,) * len(tokens) if mask is None else mask
        _tuple(mask, len(tokens))
        require(all(type(x) is bool for x in mask), detail='mask')
        require(ds or all(mask), Code.UNSUPPORTED, 'PLE has no dead-token scheme')
        tail = history.tail
        output = []
        for token, active in zip(tokens, mask):
            current = (p.token_map[token] if active else p.dead) if ds else token
            blocked = False
            shifted = []
            for lookback, value in enumerate((current,) + tail):
                if ds:
                    blocked = blocked or value == p.dead
                    shifted.append(p.pad if blocked else value)
                else:
                    # EOS ends preceding history; EOS at the current position
                    # does not erase its own prefix (vLLM raw-token semantics).
                    shifted.append(p.eos if blocked else value)
                    if lookback > 0:
                        blocked = blocked or value == p.eos
            layers = []
            geometries = zip(p.multipliers, p.primes, p.offsets) if ds else [(p.multipliers, p.sizes, p.offsets)]
            for mult, sizes, offsets in geometries:
                products = np.asarray(shifted, dtype=np.int64) * np.asarray(mult, dtype=np.int64)
                mixed, row = products[0], []
                for i in range(1, p.order):
                    mixed = np.bitwise_xor(mixed, products[i])
                    for head in range(p.heads):
                        col = (i - 1) * p.heads + head
                        row.append(int(np.remainder(mixed, np.int64(sizes[col]))) + offsets[col])
                layers.append(tuple(row))
            output.append(tuple(layers))
            tail = (current,) + tail[:-1]
        end = History(p.schema, p.digest, p.tokenizer, history.position + len(tokens), tail)
        return HashResult(p.schema, p.digest, history.digest, tuple(output), mask, end)

    def validate_bank(self, bank: Bank):
        p = self.parameters
        require((bank.scheme, bank.parameters, bank.tokenizer) ==
                (p.schema, p.digest, p.tokenizer), Code.IDENTITY, 'bank scheme')
        if type(p) is DSV41Parameters:
            require(bank.layer in p.layers, Code.IDENTITY, 'bank layer')
            rows, dim = p.table_rows[p.layers.index(bank.layer)], p.head_dimension
        else:
            require(bank.layer == p.layer, Code.IDENTITY, 'bank layer')
            rows, dim = p.padded_rows, p.embedding_dimension // ((p.order - 1) * p.heads)
        require((bank.rows, bank.dimension) == (rows, dim), Code.IDENTITY, 'bank geometry')
