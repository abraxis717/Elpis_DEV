"""Codec rendering of structural memory: the real V4.1 tokenizer over real HACF objects.

The DSV4 codec renders resolved HACF objects into the token communication
space. No DSV model runs here and nothing is generated: the runtime composes
no model execution (docs/ELPIS_MISSION.md).
"""
import pytest

from elpis.inference.admission import ContextBudget, admit_context
from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.text import DSV41_RENDERER

from ..inference.test_text import official, tokenizer  # noqa: F401  (fixtures)
from .conftest import POSITIVE

CONTEXT = initial_snapshot().digest


def empty_admission(tokenizer, model="a" * 64):
    return admit_context(model=model, tokenizer=tokenizer.identity, context_snapshot=CONTEXT,
                         corpus="c" * 64, proposals=(), resolved=(), omitted=0,
                         budget=ContextBudget(4, 4096, 2048), renderer=DSV41_RENDERER,
                         text_tokenizer=tokenizer)


def test_real_hacf_content_is_rendered_by_the_admitted_tokenizer(runtime, tokenizer, ingress, corpus):
    prepared = runtime.admit_context(ingress=ingress, task=POSITIVE, corpus_root=corpus[1],
                                     corpus_manifest=corpus[0].corpus_manifest_json,
                                     context_snapshot=CONTEXT, model="a" * 64, tokenizer=tokenizer.identity,
                                     budget=ContextBudget(4, 4096, 2048), max_document_bytes=1 << 20,
                                     text_tokenizer=tokenizer)
    admission = prepared.admission
    assert admission.renderer == DSV41_RENDERER and admission.objects
    assert [r.record.kind for r in runtime.history.records()] == ["ingress.proposal", "context.admission"]


def test_context_budget_and_tokenizer_binding(tokenizer):
    with pytest.raises(ContractError):
        empty_admission(tokenizer, model="not-a-digest")
    admission = empty_admission(tokenizer)
    assert admission.tokenizer == tokenizer.identity
    with pytest.raises(ContractError, match="admitted text tokenizer"):
        admit_context(model="a" * 64, tokenizer="b" * 64, context_snapshot=CONTEXT,
                      corpus="c" * 64, proposals=(), resolved=(), omitted=0,
                      budget=ContextBudget(1, 2, 3), renderer=DSV41_RENDERER, text_tokenizer=tokenizer)
