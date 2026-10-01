"""Text composition at the sequence boundary; generated text is an inert output.

Timing separates encoder, prefill/admission, model next, detokenizer and emitter.
No generated content is ingested into HACF or given mutation authority.
"""
from dataclasses import dataclass
from time import perf_counter_ns

from elpis.inference.admission import ContextAdmission
from elpis.inference.contracts import Code, require
from elpis.identity import content_digest
from elpis.inference.principal import PrincipalEngine, PrincipalRequest, PrincipalResult
from elpis.inference.text import ChatMessage, DSV41_RENDERER, V41Tokenizer
from .history import ReceiptRecord


@dataclass(frozen=True)
class TextResult:
    principal: PrincipalResult
    request: PrincipalRequest
    tokenizer: str
    text: str
    timings_ns: tuple[tuple[str, int], ...]


def run_text(runtime, engine, state, text, admission, *, tokenizer, request_id,
             max_new_tokens, expected_state, emit=None, thinking=False, effort=None,
             resident_experts=None):
    require(type(engine) is PrincipalEngine and type(tokenizer) is V41Tokenizer,
            detail="text engine/tokenizer")
    require(type(admission) is ContextAdmission and admission.renderer == DSV41_RENDERER,
            Code.IDENTITY, "production context renderer")
    require(engine.target.config.tokenizer == admission.tokenizer == tokenizer.identity,
            Code.IDENTITY, "model/admission/tokenizer mismatch")
    require(engine.target.config.vocab == tokenizer.vocab_size, Code.IDENTITY, "model vocabulary")
    require(emit is None or callable(emit), detail="text emitter")
    start = perf_counter_ns()
    messages = (ChatMessage("user", text),) if type(text) is str else text
    prompt = tokenizer.encode_chat(messages, thinking=thinking, effort=effort)
    request = PrincipalRequest(request_id, prompt, max_new_tokens, tokenizer.stop_tokens)
    decoder = tokenizer.decoder(preserve_special=True)
    times = {"encode": perf_counter_ns() - start, "prefill": 0, "model_next": 0,
             "decode": 0, "emit": 0, "finalize": 0}
    parts = []

    def output(piece):
        if piece:
            parts.append(piece)
            if emit is not None:
                start = perf_counter_ns()
                emit(piece)
                times["emit"] += perf_counter_ns() - start

    start = perf_counter_ns()
    sequence = engine.begin(state, request, admission, expected_state=expected_state,
                            resident_experts=resident_experts)
    times["prefill"] = perf_counter_ns() - start
    try:
        while not sequence.done:
            start = perf_counter_ns()
            token = sequence.next()
            times["model_next"] += perf_counter_ns() - start
            if token is None:
                break
            start = perf_counter_ns()
            piece = decoder.push(token)
            times["decode"] += perf_counter_ns() - start
            output(piece)
        start = perf_counter_ns()
        tail = decoder.finish()
        times["decode"] += perf_counter_ns() - start
        output(tail)
    except BaseException:
        # Output may have escaped but no successful receipt is recorded. The
        # caller retains its old committed state; partial text is only a proposal.
        sequence.stop("OUTPUT_ABORTED")
        engine.finalize(sequence)
        raise
    start = perf_counter_ns()
    result = engine.finalize(sequence)
    times["finalize"] = perf_counter_ns() - start
    text = "".join(parts)
    record = None
    if result.commit is not None:
        commit = result.commit
        runtime.history.record(ReceiptRecord.of(
            "inference", "principal.commit", commit.digest,
            admission=commit.admission, request=commit.request, state=commit.state,
            outputs=str(len(commit.outputs)), stop=commit.stop_reason))
        output_id = content_digest("elpis.runtime.text-output.v1", dict(commit=commit.digest,
                             tokenizer=tokenizer.identity, text=text, authority="NONE"))
        record = runtime.history.record(ReceiptRecord.of(
            "inference", "text.output", output_id, commit=commit.digest,
            tokenizer=tokenizer.identity, authority="NONE"))
    return TextResult(result, request, tokenizer.identity, text, tuple(times.items())), record
