"""Tokenizer-only measurements; no neural throughput or model-quality claim.

python -m tests.inference.text_boundary_benchmark /absolute/tokenizer.json
"""
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

from elpis.inference.text import ChatMessage, RECIPE_SHA256, V41Tokenizer
from elpis.substrate.boundary import RootCapability


def run(path):
    path = Path(path).absolute()
    loads = []
    for _ in range(5):
        start = perf_counter_ns()
        with RootCapability(path.parent) as root:
            tokenizer = V41Tokenizer.load(root, path.name, expected_sha256=RECIPE_SHA256)
        loads.append(perf_counter_ns() - start)
    text = "Hello, Elpis. Příliš žluťoučký kůň. 你好，世界！ 👩🏽‍💻\n" * 20
    messages = (ChatMessage("system", "Be brief."), ChatMessage("user", text))
    encode_times, decode_times, token_times = [], [], []
    ids = tokenizer.encode(text)
    for _ in range(100):
        start = perf_counter_ns()
        prompt = tokenizer.encode_chat(messages)
        encode_times.append(perf_counter_ns() - start)
        decoder = tokenizer.decoder()
        start = perf_counter_ns()
        restored = "".join(decoder.push(token) for token in ids) + decoder.finish()
        decode_times.append(perf_counter_ns() - start)
        assert restored == text
    decoder = tokenizer.decoder()
    for token in ids:
        start = perf_counter_ns()
        decoder.push(token)
        token_times.append(perf_counter_ns() - start)
    decoder.finish()
    return dict(
        schema="elpis.text-boundary.measurement.v1", tokenizer=tokenizer.identity,
        sha256=tokenizer.sha256, vocabulary=tokenizer.vocab_size,
        workload=dict(bytes=len(text.encode()), prompt_tokens=len(prompt), decoded_tokens=len(ids), repeats=100),
        median_load_ms=median(loads) / 1e6, median_chat_encode_ms=median(encode_times) / 1e6,
        median_decode_ms=median(decode_times) / 1e6,
        median_decode_ns_per_token=median(decode_times) / len(ids),
        p95_individual_decode_ns=sorted(token_times)[int(len(token_times) * .95)],
        byte_material_python_bytes=sys.getsizeof(tokenizer._material) + sum(map(sys.getsizeof, tokenizer._material)),
        utf8_carry_bound_bytes=3,
        hello_prompt_ids=tokenizer.encode_chat((ChatMessage("user", "Hello, Elpis."),)),
        model_tokens_per_second=None, model_prefill_ms=None, model_working_set_bytes=None,
        fms_hits=None, fms_misses=None, expert_activity=None,
        reason="No compatible production model instantiated; tokenizer timing only.")


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=2))
