# Elpis Torch R5 — baseline note (audit before any R5 change)

Recorded on 2026-11-17, before the first R5 edit. Everything below was read from the repository or measured in
this container; nothing is quoted from memory.

## Repository identity

| Item | Value |
|---|---|
| Checkout | `/home/user/Elpis_DEV` (managed cloud container). `/mnt/primesauce` does not exist here; the owner said to ignore it. |
| Remote | `origin https://github.com/abraxis717/Elpis_DEV` (fetch and push) |
| Branch at audit | `claude/r4-context-runtime` at `5a5953327b39dac6500eb1904f8bc8749a5da9ba`, clean (`git status --short` empty) |
| R5 work branch | created from `origin/claude/funny-turing-obbdy5` at `80198afd145ce5210630baff11d923a798ff95ce` |
| `origin/main` | `5f5f4b8bc2cd14756da0294474d795910d4fe439` (merge of PR #72) |

Merge evidence for R4 / R4.1:

* `80198af` is "Merge pull request #75 from abraxis717/claude/r4-context-runtime". Its tree is identical to the
  R4.1 head `5a59533` (`git diff 5a59533 80198af` is empty).
* **PR #75 was merged into the R3 branch `claude/funny-turing-obbdy5`, not into `main`.** `80198af` is not an
  ancestor of `origin/main` (15 commits, R3 + R4 + R4.1, are on the R3 branch only). R3's own PR to `main`
  (abraxis717/Elpis_DEV#74) was still open at the audit.
* Owner direction on 2026-11-17: the next PR must include everything (R3, R4, R4.1, R5) and target `main`.

## Current implementation (read from the code)

| Module | What it is (R3 / R4 / R4.1) |
|---|---|
| `ersa.py`, `ersa_model.py` | The core: ERSA (gated delta rule; `serial_scan` reference, exact block-parallel `chunked_scan`) + CGC channel. `ErsaLM.forward(tokens, state, *, return_pre, logit_positions, **extra)` with the hooks `_chunk_context`, `_extra_mix`, `_commit_context`. |
| `history.py` | `TokenHistory` (exact int32 FIFO ring, absolute positions, `EvictedError` on evicted or future positions) and `BlockIndex` (ring of detached learned block keys, absolute block ids, `count` occupancy). Neither carries a session or epoch identity. |
| `history_model.py` | `ErsaHistoryLM`: R3 learned addressing. Block keys (2 x 32 per 32-token block) from a bidirectional block encoder; score = cosine x exp(scale) - softplus(recency) log1p(distance); top-2; reread of the selected blocks from exact token ids; readout attention with predecessor keys, recency rank, log-distance slope, selection prior and a null logit; modes `off` / `learned` / `oracle` / `oracle_override`; `addressing_loss` (training only). The read is computed at every position. |
| `context_runtime.py` | R4/R4.1 runtime: deterministic `exact_topk` (ties to the newer block), `tiled_select` over occupied ring views, bounded `BlockCache`, `ContextSession` (bounded streaming, current-position logits, reset, budget refusal, `RevisionWatch`), key replay. |
| `longrange_tasks.py`, `program_tasks.py` | R3 synthetic tasks: 2-token facts with rebinding, aliases, links, direct / associative / 2-hop / 3-hop probes (vocabulary 593); token-level program language with an exact checker (vocabulary 124). |
| `experiments/run_r3.py`, `run_r4.py` | R3 training and evaluation (ladder 1K-1M, density, distractors, program generation); R4 benchmarks, parity, Spark-style comparator, GPU harness (NOT_RUN without CUDA). |

Not present before R5: an epoch or session identity for ring addresses, a coarse-to-fine retrieval cascade, a
reranker, hard-negative task generators, scoped program symbols, or any retrieval metric beyond recall@K.

## Test command and result at the audit head

```bash
python -B -m pytest tests/research/elpis_torch
```

384 collected: 379 passed, 5 skipped (CUDA tests and an optional dependency; skips are not PASS), 1 warning,
3 min 13 s on this container (Python 3.13.18, torch 2.15.0+cpu, 4 cores, no CUDA). `pyproject.toml` already
adds `-q`, so a second `-q` hides the summary line; counts above are from the progress output.

## R3 baseline numbers verified against the stored records

Recomputed from `research/elpis_torch/results/r3/facts_history_s{0,1,2}.json` (learned mode; direct probes;
recall@2 over every probe kind whose source block is eligible):

| tokens | direct accuracy (s0 / s1 / s2) | recall@2 | accuracy given source block selected | direct probes per seed |
|---:|---|---|---|---:|
| 1,024 | 0.97 / 0.96 / 0.96 | 0.967 / 0.951 / 0.965 | 0.99 / 1.00 / 1.00 | 483 |
| 4,096 | 0.84 / 0.74 / 0.79 | 0.720 / 0.602 / 0.677 | 0.96 / 0.98 / 0.94 | 257 |
| 16,384 | 0.83 / 0.71 / 0.77 | 0.753 / 0.647 / 0.703 | 0.99 / 0.99 / 0.96 | 128 |
| 65,536 | 0.95 / 0.79 / 0.82 | 0.780 / 0.667 / 0.760 | 1.00 / 0.98 / 0.88 | 57 |
| 262,144 | 0.81 / 0.65 / 0.76 | 0.775 / 0.627 / 0.690 | 0.98 / 0.86 / 0.94 | 68 |
| 1,048,576 | 0.76 / 0.31 / 0.41 | 0.513 / 0.313 / 0.475 | 0.86 / 0.67 / 0.63 | 29 |

These equal `EVIDENCE_R3.md` section 2 after rounding. No numerical discrepancy was found. Two reading notes:
the column "oracle address" of that table is the `oracle_override` mode in the records, and the 1M row is one
stream with 29 direct probes per seed, so its uncertainty is wide.
