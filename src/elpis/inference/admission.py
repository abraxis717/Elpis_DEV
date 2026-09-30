"""Context admission: the only way external context reaches a principal sequence.

Elpis keeps context outside the model, in HACF. Before a sequence begins, the
runtime resolves digest-addressed HACF objects into verified bytes, and this
module turns them into one immutable :class:`ContextAdmission`. It binds:

* the context snapshot and target model/tokenizer the sequence runs under;
* the provenance: the corpus manifest digest and the digests of the address
  proposals the objects came from;
* the admitted objects, in deterministic order, each with its HACF object
  digest, the digest and size of the admitted bytes and their model-visible
  token rendering;
* the explicit budget (objects, bytes, tokens) and how many proposed objects
  were left out to honour it.

It carries tokens, digests and counts only. There are no key/value tensors,
hidden vectors or latents in it: context enters the model as ordinary input
tokens, never as vectors added to activations.

Rendering is the model's tokenizer applied to the admitted bytes. Synthetic
nibbles remain a fixture-only domain. Production text uses an explicitly
admitted V4.1 tokenizer handle; it is never loaded during a sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

from elpis.substrate.digests import raw_sha256

from .contracts import Code, digest_value, identity, integer, require
from .structural import AddressProposal
from .text import DSV41_RENDERER, V41Tokenizer

__all__ = (
    "AdmittedObject", "ContextAdmission", "ContextBudget", "SYNTHETIC_NIBBLE16", "admit_context",
    "render_synthetic_nibble16",
)

SYNTHETIC_NIBBLE16 = "elpis.inference.render.synthetic-nibble16.r0"
_SYNTHETIC_TOKENIZER_PREFIX = "synthetic-raw16-"


def render_synthetic_nibble16(data: bytes) -> tuple[int, ...]:
    """SYNTHETIC renderer: each byte becomes its high and low nibble (tokens 0..15)."""
    require(type(data) is bytes, detail="renderer input")
    return tuple(token for byte in data for token in (byte >> 4, byte & 15))


_RENDERERS = {SYNTHETIC_NIBBLE16: (lambda tokenizer: tokenizer.startswith(_SYNTHETIC_TOKENIZER_PREFIX),
                                   render_synthetic_nibble16, 16)}


@dataclass(frozen=True)
class ContextBudget:
    max_objects: int
    max_bytes: int
    max_tokens: int

    def __post_init__(self):
        for value in (self.max_objects, self.max_bytes, self.max_tokens):
            integer(value, 1)


@dataclass(frozen=True)
class AdmittedObject:
    object: str               # HACF object (chunk) digest
    content: str              # plain SHA-256 of the admitted bytes (HACF's norm_digest for chunk text)
    size: int                 # admitted byte count
    tokens: tuple[int, ...]   # model-visible rendering of those bytes

    def __post_init__(self):
        digest_value(self.object)
        digest_value(self.content)
        integer(self.size)
        require(type(self.tokens) is tuple and all(type(t) is int and t >= 0 for t in self.tokens),
                detail="admitted tokens")


@dataclass(frozen=True)
class ContextAdmission:
    """Immutable, bounded context for exactly one sequence. Not authority; no tensors."""
    context_snapshot: str
    model: str
    tokenizer: str
    renderer: str
    corpus: str
    proposals: tuple[str, ...]
    objects: tuple[AdmittedObject, ...]
    omitted: int
    budget: ContextBudget

    def __post_init__(self):
        for value in (self.context_snapshot, self.model, self.corpus):
            digest_value(value)
        require(type(self.tokenizer) is str and bool(self.tokenizer), detail="admission tokenizer")
        require(self.renderer in (*_RENDERERS, DSV41_RENDERER), Code.UNSUPPORTED, "admission renderer")
        if self.renderer == DSV41_RENDERER:
            digest_value(self.tokenizer)
        else:
            accepts, _, vocab = _RENDERERS[self.renderer]
            require(accepts(self.tokenizer), Code.IDENTITY, "renderer/tokenizer")
        require(type(self.proposals) is tuple and all(type(p) is str for p in self.proposals),
                detail="admission proposals")
        for proposal in self.proposals:
            digest_value(proposal)
        require(type(self.objects) is tuple and all(type(o) is AdmittedObject for o in self.objects),
                detail="admitted objects")
        if self.renderer != DSV41_RENDERER:
            require(all(t < vocab for o in self.objects for t in o.tokens), detail="renderer vocabulary")
        require(len({o.object for o in self.objects}) == len(self.objects), detail="duplicate admitted object")
        integer(self.omitted)
        require(type(self.budget) is ContextBudget, detail="admission budget")
        require(len(self.objects) <= self.budget.max_objects and
                sum(o.size for o in self.objects) <= self.budget.max_bytes and
                sum(len(o.tokens) for o in self.objects) <= self.budget.max_tokens,
                Code.LIMIT, "admission exceeds its budget")

    @property
    def tokens(self) -> tuple[int, ...]:
        """The admitted context as one model input, objects in admission order."""
        return tuple(token for o in self.objects for token in o.tokens)

    @property
    def digest(self):
        return identity("context-admission", self)


def admit_context(*, model, tokenizer, context_snapshot, corpus, proposals, resolved, omitted, budget,
                  renderer=SYNTHETIC_NIBBLE16, text_tokenizer=None) -> ContextAdmission:
    """Build the admission from proposals and resolved object bytes.

    ``resolved`` is a tuple of ``(object digest, bytes)`` in the order the
    runtime resolved them (it has already verified each object's bytes).
    Every admitted object must be one the proposals named, and every
    proposal must be bound to this snapshot and corpus. Objects are admitted
    whole, in order, while they fit the budget; the rest count as omitted.
    """
    if renderer == DSV41_RENDERER:
        require(type(text_tokenizer) is V41Tokenizer and text_tokenizer.identity == tokenizer,
                Code.IDENTITY, "admitted text tokenizer")
        render = text_tokenizer.encode_content
    else:
        require(text_tokenizer is None, Code.IDENTITY, "synthetic renderer cannot use production tokenizer")
        require(renderer in _RENDERERS, Code.UNSUPPORTED, "admission renderer")
        accepts, render, _vocab = _RENDERERS[renderer]
        require(type(tokenizer) is str and accepts(tokenizer), Code.UNSUPPORTED, "renderer/tokenizer")
    require(type(budget) is ContextBudget, detail="admission budget")
    require(type(proposals) is tuple and all(type(p) is AddressProposal for p in proposals),
            detail="admission proposals")
    named = set()
    for proposal in proposals:
        require(proposal.context_snapshot == context_snapshot, Code.STALE, "proposal snapshot")
        require(proposal.corpus == corpus, Code.STALE, "proposal corpus")
        named.update(proposal.objects)
    require(type(resolved) is tuple, detail="resolved objects")
    integer(omitted)
    objects, size, tokens = [], 0, 0
    for index, entry in enumerate(resolved):
        require(type(entry) is tuple and len(entry) == 2 and type(entry[1]) is bytes, detail="resolved object")
        digest, data = entry
        require(digest in named, Code.IDENTITY, "resolved object was not proposed")
        if len(objects) == budget.max_objects or size + len(data) > budget.max_bytes:
            omitted += len(resolved) - index
            break
        rendered = render(data)
        if (len(objects) == budget.max_objects or size + len(data) > budget.max_bytes or
                tokens + len(rendered) > budget.max_tokens):
            omitted += len(resolved) - index
            break
        objects.append(AdmittedObject(digest, raw_sha256(data).hexdigest(), len(data), rendered))
        size += len(data)
        tokens += len(rendered)
    return ContextAdmission(context_snapshot, model, tokenizer, renderer, corpus,
                            tuple(p.digest for p in proposals), tuple(objects), omitted, budget)
