"""The H-ECS Rust harnesses under the sanitizers (research/hecs_r{0,1,2}/CMakeLists.txt).

The crates run an uninstrumented Rust test process with the sanitizer runtime preloaded. Under ASan two CI runs
(204: hecs_r1, 215: hecs_r0) crashed inside the ASan runtime's thread bookkeeping when libtest created replacement
test threads while others were alive. The repair serializes the harness under ASan only. This gate keeps it:

* ASan: the cargo test command passes ``-- --test-threads=1`` (no overlapping test-thread lifetimes);
* TSan: the harness stays parallel (concurrent K1 use is still exercised where races are detected);
* every configuration still runs the crate's whole test suite through the sanitizer runner (nothing excluded).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CRATES = ("hecs_r0", "hecs_r1", "hecs_r2")


@pytest.mark.parametrize("crate", CRATES)
def test_asan_serializes_the_harness_and_tsan_keeps_it_parallel(crate):
    text = (REPO / "research" / crate / "CMakeLists.txt").read_text(encoding="utf-8")
    block = re.search(r"set\(_hecs_harness_args\)\nif\((\w+)\)\n  set\(_hecs_harness_args (.+?)\)\nendif\(\)", text)
    assert block, f"{crate}: the sanitizer harness arguments are missing"
    assert block.group(1) == "ELPIS_ENABLE_ASAN", f"{crate}: serialization must apply to ASan only"
    assert block.group(2) == "-- --test-threads=1"
    command = re.search(r"add_test\(NAME research\.%s_mechanics(.+?)\)\nset_tests_properties" % crate, text, re.S)
    assert command and "${_hecs_harness_args}" in command.group(1) and "${_hecs_runner}" in command.group(1)
    # Nothing is filtered or skipped: the harness arguments carry no test-name filter or --skip.
    assert "--skip" not in text and "--exact" not in text
