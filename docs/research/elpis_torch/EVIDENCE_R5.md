# Elpis Torch R5 — learned evidence retrieval over a 1,048,576-token exact history: archived evidence

Research only; CPU (4 cores, one thread per run), tiny models (266,563 parameters, 149,099 in the evidence path),
synthetic tasks. Design: `DESIGN_R5.md`. Audit: `BASELINE_R5.md`. Records: `research/elpis_torch/results/r5/`.
Closed out in R6 P0 (reconciliation of the post-merge commit `fe82cc8`).

**Status of every number below: HISTORICAL.** The R5 checkpoints were written to the R5 session's ephemeral
container and are not repository artifacts; no result here was reproduced in R6. Each record carries the
`parameter_digest` of the weights that produced it (seed 0 / 1 / 2 of the fine-tuned run:
`512e9419…`, `54b12673…`, `074f0daa…`). A result JSON is not a substitute for those weights.

## 1. What was built

Exact FIFO ring of 2^20 tokens with session / epoch / slot / generation identity; a coarse learned index (4
descriptors x 64 per 32-token block, a vectorized scan of all live blocks, deterministic top-64); a mandatory exact
reread of the candidates' original tokens; a fine reranker (token-level match, coarse score, learned
supersession feature) injecting the top 4 and exposing the top 8; a bounded working context (4 blocks + the
query's own block) read in the last block of the unchanged ERSA/CGC core. Training: Phase A (2,000 steps, 16 x
1,024 tokens, every parameter) then Phase B (curriculum 4K -> 16K -> 64K -> 262K -> 1M, core frozen, 296 steps)
with `coarse_source_loss`, `fine_evidence_loss` and the answer loss; training-only gold labels.

Two runs exist. **First run**: summed supersession (`supersession="sum"`), complete ladder, three seeds.
**Fine-tuned run**: the first run's checkpoints fine-tuned through the same curriculum with the bounded form
(`supersession="max"`); its ladder is INCOMPLETE (section 3).

## 2. Results

### First full run (summed supersession), seeds 0 / 1 / 2

| set | tokens | probes | accuracy | coarse recall@64 | fine recall@4 | fine recall@8 | conditional accuracy |
|---|---:|---:|---|---|---|---|---|
| records | 1,024 | 279 | 0.961 / 0.961 / 0.953 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.950 / 0.950 / 0.946 |
| records | 65,536 | 289 | 0.952 / 0.962 / 0.945 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.996 | 1.000 / 1.000 / 0.996 | 0.949 / 0.960 / 0.945 |
| records | 262,144 | 285 | 0.958 / 0.930 / 0.909 | 1.000 / 1.000 / 1.000 | 0.996 / 0.996 / 0.978 | 1.000 / 0.996 / 0.993 | 0.960 / 0.930 / 0.925 |
| records | 1,048,576 | 281 | 0.961 / 0.922 / 0.897 | 1.000 / 1.000 / 1.000 | 0.996 / 0.971 / 0.919 | 1.000 / 0.985 / 0.959 | 0.963 / 0.943 / 0.968 |
| revision | 1,024 | 721 | 0.964 / 0.961 / 0.920 | 1.000 / 1.000 / 1.000 | 0.998 / 0.998 / 0.996 | 1.000 / 1.000 / 1.000 | 0.950 / 0.948 / 0.911 |
| revision | 65,536 | 896 | 0.963 / 0.963 / 0.953 | 0.996 / 0.998 / 0.999 | 0.996 / 0.998 / 0.998 | 0.996 / 0.998 / 0.999 | 0.964 / 0.963 / 0.952 |
| revision | 262,144 | 863 | 0.964 / 0.956 / 0.954 | 0.999 / 0.999 / 1.000 | 0.999 / 0.995 / 0.998 | 0.999 / 0.998 / 0.999 | 0.965 / 0.960 / 0.955 |
| revision | 1,048,576 | 881 | 0.968 / 0.951 / 0.930 | 0.999 / 0.998 / 1.000 | 0.999 / 0.992 / 0.972 | 0.999 / 0.994 / 0.980 | 0.968 / 0.957 / 0.951 |
| scoped | 1,024 | 266 | 0.962 / 0.951 / 0.962 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.955 / 0.942 / 0.955 |
| scoped | 65,536 | 302 | 0.874 / 0.818 / 0.818 | 1.000 / 1.000 / 1.000 | 1.000 / 0.983 / 0.973 | 1.000 / 1.000 / 1.000 | 0.871 / 0.828 / 0.836 |
| scoped | 262,144 | 288 | 0.795 / 0.764 / 0.729 | 1.000 / 1.000 / 1.000 | 0.974 / 0.949 / 0.934 | 0.996 / 0.989 / 0.978 | 0.804 / 0.788 / 0.765 |
| scoped | 1,048,576 | 280 | 0.693 / 0.639 / 0.646 | 1.000 / 1.000 / 1.000 | 0.884 / 0.840 / 0.807 | 0.949 / 0.927 / 0.873 | 0.757 / 0.753 / 0.793 |
| lexical | 1,024 | 332 | 0.973 / 0.964 / 0.943 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.966 / 0.955 / 0.937 |
| lexical | 65,536 | 329 | 0.964 / 0.948 / 0.906 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.987 | 1.000 / 1.000 / 0.997 | 0.962 / 0.946 / 0.913 |
| lexical | 262,144 | 331 | 0.988 / 0.967 / 0.915 | 1.000 / 1.000 / 0.997 | 1.000 / 0.988 / 0.978 | 1.000 / 0.997 / 0.984 | 0.988 / 0.978 / 0.933 |
| lexical | 1,048,576 | 342 | 0.950 / 0.906 / 0.877 | 1.000 / 0.997 / 0.997 | 0.997 / 0.970 / 0.916 | 0.997 / 0.979 / 0.934 | 0.952 / 0.929 / 0.945 |
| low_overlap | 1,024 | 192 | 0.990 / 0.990 / 0.990 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.988 / 0.988 / 0.988 |
| low_overlap | 65,536 | 192 | 0.943 / 0.938 / 0.901 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.938 / 0.932 / 0.893 |
| low_overlap | 262,144 | 192 | 0.969 / 0.953 / 0.932 | 1.000 / 1.000 / 1.000 | 0.995 / 0.989 / 0.961 | 0.995 / 0.995 / 0.983 | 0.972 / 0.961 / 0.954 |
| low_overlap | 1,048,576 | 192 | 0.922 / 0.880 / 0.818 | 1.000 / 1.000 / 1.000 | 0.989 / 0.962 / 0.853 | 0.995 / 0.989 / 0.908 | 0.929 / 0.910 / 0.936 |
| r3facts | 1,024 | 1088 | 0.999 / 0.999 / 0.999 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.999 / 0.999 / 0.999 |
| r3facts | 65,536 | 150 | 0.993 / 1.000 / 0.993 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.993 / 1.000 / 0.993 |
| r3facts | 262,144 | 142 | 1.000 / 0.986 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 0.986 / 1.000 |
| r3facts | 1,048,576 | 156 | 0.923 / 0.885 / 0.904 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.923 / 0.885 / 0.904 |

### Fine-tuned run (bounded supersession), seeds 0 / 1 / 2 (— = not evaluated)

| set | tokens | probes | accuracy | coarse recall@64 | fine recall@4 | fine recall@8 | conditional accuracy |
|---|---:|---:|---|---|---|---|---|
| records | 1,024 | 279 | 0.957 / 0.961 / 0.964 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.946 / 0.950 / 0.955 |
| records | 65,536 | 289 | 0.965 / 0.965 / 0.969 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.964 / 0.964 / 0.967 |
| records | 262,144 | 285 | 0.961 / 0.961 / 0.961 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.960 / 0.960 / 0.960 |
| records | 1,048,576 | 281 | 0.979 / 0.979 / 0.982 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.978 / 0.978 / 0.982 |
| revision | 1,024 | 721 | 0.946 / 0.952 / 0.921 | 1.000 / 1.000 / 1.000 | 0.979 / 0.979 / 0.974 | 1.000 / 1.000 / 1.000 | 0.954 / 0.956 / 0.923 |
| revision | 65,536 | 896 | 0.895 / 0.891 / 0.885 | 0.998 / 0.998 / 0.999 | 0.920 / 0.916 / 0.907 | 0.998 / 0.996 / 0.995 | 0.964 / 0.962 / 0.963 |
| revision | 262,144 | 863 | 0.863 / 0.858 / 0.867 | 0.999 / 0.999 / 1.000 | 0.894 / 0.889 / 0.885 | 0.994 / 0.997 / 0.999 | 0.961 / 0.960 / 0.972 |
| revision | 1,048,576 | 881 | — / 0.848 / 0.849 | — / 0.998 / 1.000 | — / 0.871 / 0.868 | — / 0.995 / 0.998 | — / 0.964 / 0.969 |
| scoped | 1,024 | 266 | 0.962 / 0.955 / 0.962 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.955 / 0.946 / 0.955 |
| scoped | 65,536 | 302 | 0.967 / 0.964 / 0.944 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.966 / 0.963 / 0.942 |
| scoped | 262,144 | 288 | 0.924 / 0.927 / 0.899 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.919 / 0.923 / 0.894 |
| scoped | 1,048,576 | 280 | — / 0.939 / 0.879 | — / 1.000 / 1.000 | — / 1.000 / 0.993 | — / 1.000 / 0.996 | — / 0.938 / 0.883 |
| lexical | 1,024 | 332 | 0.970 / 0.970 / 0.964 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.963 / 0.963 / 0.959 |
| lexical | 65,536 | 329 | 0.976 / 0.970 / 0.964 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.974 / 0.968 / 0.962 |
| lexical | 262,144 | 331 | 0.988 / 0.985 / 0.976 | 1.000 / 1.000 / 0.997 | 1.000 / 1.000 / 0.994 | 1.000 / 1.000 / 0.994 | 0.988 / 0.984 / 0.981 |
| lexical | 1,048,576 | 342 | — / 0.962 / — | — / 0.997 / — | — / 0.997 / — | — / 0.997 / — | — / 0.964 / — |
| low_overlap | 1,024 | 192 | 0.990 / 0.990 / 0.990 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.988 / 0.988 / 0.988 |
| low_overlap | 65,536 | 192 | 0.953 / 0.948 / 0.958 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.949 / 0.944 / 0.955 |
| low_overlap | 262,144 | 192 | 0.990 / 0.990 / 0.984 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.989 / 0.989 / 0.983 |
| r3facts | 1,024 | 1088 | 0.998 / 0.996 / 0.995 | 1.000 / 1.000 / 1.000 | 0.999 / 0.999 / 0.999 | 1.000 / 1.000 / 1.000 | 0.999 / 0.997 / 0.997 |
| r3facts | 65,536 | 150 | 0.993 / 0.993 / 0.993 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.993 / 0.993 / 0.993 |
| r3facts | 262,144 | 142 | 1.000 / 0.986 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 0.986 / 1.000 |

### Revision and scope at 1,048,576 tokens

| run | set | latest-binding accuracy (revised probes) | stale top-1 rate | wrong-scope top-1 rate |
|---|---|---|---|---|
| first | records | 0.959 / 0.918 / 0.912 | 0.020 / 0.007 / 0.007 | — / — / — |
| first | revision | 0.965 / 0.953 / 0.932 | 0.007 / 0.007 / 0.008 | — / — / — |
| first | scoped | 0.685 / 0.637 / 0.657 | 0.000 / 0.000 / 0.007 | 0.011 / 0.000 / 0.011 |
| first | lexical | 0.939 / 0.896 / 0.844 | 0.004 / 0.022 / 0.017 | — / — / — |
| fine-tuned | records | 0.973 / 0.973 / 0.980 | 0.483 / 0.463 / 0.442 | — / — / — |
| fine-tuned | revision | — / 0.826 / 0.827 | — / 0.670 / 0.630 | — / — / — |
| fine-tuned | scoped | — / 0.945 / 0.918 | — / 0.418 / 0.438 | — / 0.000 / 0.000 |
| fine-tuned | lexical | — / 0.952 / — | — / 0.448 / — | — / — / — |

### Training ablations (seed 0, reduced budget, summed supersession): coarse recall@64 / fine recall@8 / accuracy

| variant | records 64K | records 262K | records 1M | scoped 64K | scoped 262K | scoped 1M |
|---|---|---|---|---|---|---|
| 1 key per block | 0.986 / 0.986 / 0.903 | 0.949 / 0.942 / 0.842 | 0.930 / 0.919 / 0.829 | 0.997 / 0.790 / 0.434 | 0.985 / 0.641 / 0.340 | 0.785 / 0.444 / 0.246 |
| 2 keys | 0.996 / 0.993 / 0.924 | 0.996 / 0.996 / 0.902 | 0.993 / 0.978 / 0.911 | 0.997 / 0.997 / 0.841 | 0.996 / 0.996 / 0.799 | 0.993 / 0.964 / 0.693 |
| 4 keys (reference) | 1.000 / 0.993 / 0.910 | 1.000 / 0.985 / 0.877 | 1.000 / 0.971 / 0.879 | 1.000 / 1.000 / 0.748 | 1.000 / 0.963 / 0.656 | 1.000 / 0.862 / 0.536 |
| 8 keys | 1.000 / 0.993 / 0.896 | 1.000 / 0.982 / 0.835 | NOT_RUN | 1.000 / 0.946 / 0.550 | 1.000 / 0.865 / 0.458 | NOT_RUN |

### R3 architecture trained on the same R5 mixture (seed 0): accuracy (all probes)

| set | 1K | 4K | 16K | 64K | 262K | 1M |
|---|---|---|---|---|---|---|
| records | 0.774 | 0.831 | 0.861 | 0.799 | 0.737 | 0.553 |
| revision | 0.574 | 0.639 | 0.676 | 0.730 | 0.753 | 0.676 |
| scoped | 0.451 | 0.481 | 0.444 | 0.401 | 0.340 | 0.237 |

Probe counts are per seed. "Historical" probes need retrieval (the evidence is in an earlier block); conditional
accuracy is the accuracy of historical probes whose gold block was injected. R3's own benchmark streams at
1,024 tokens: R5 answers 1.000 / 1.000 / 1.000 of direct questions (fine-tuned 1.000 / 1.000 / 0.998); R3 itself
answered 0.97 / 0.96 / 0.96. At 1M (first run) 0.889 on every seed; R3: 0.76 / 0.31 / 0.41.

### Diagnosis of the supersession feature (seed 2, first run, 1M; CONSOLE-ONLY, no JSON record)

| set | measured | as trained (sum) | supersession switched off at evaluation |
|---|---|---|---|
| no hard negatives | accuracy / fine recall@4 | 0.810 / 0.853 | 0.957 / 1.000 |
| records | accuracy / fine recall@4 | 0.893 / 0.919 | 0.982 / 1.000 |
| scoped | accuracy / fine recall@4 | 0.640 / 0.807 | 0.920 / 0.993 |
| revision-heavy | latest-binding accuracy / stale top-1 | 0.932 / 0.008 | 0.819 / 0.720 |

The summed feature penalizes old evidence for every newer weak match (losses grew with distance); removing it
fixes that but breaks revision-heavy correctness. The bounded form keeps most of both (fine-tuned run), but the
revision-heavy set still loses its gold block at ranks 5-8 under top-4 injection: fine recall@4 0.87-0.92 against
fine recall@8 0.994-0.999 at 64K-1M, conditional accuracy 0.96-0.97 when the gold is injected. Weakening
supersession helps some tasks and severely damages revision-heavy correctness; both directions are preserved
here as found.

## 3. What is incomplete or was not run

* Fine-tuned 1M ladder: seed 0 has records only; seed 1 lacks low_overlap and r3facts; seed 2 lacks lexical,
  low_overlap and r3facts.
* Evaluation-time controls (oracle source, coarse only, no reranker, coarse top-16/32/128, shuffled and
  untrained index): NOT_RUN on either run.
* Block-size ablations (16 / 64): NOT_RUN. Eight keys per block at 1M: NOT_RUN.
* Top-8 versus top-4 injection: a validation-only comparison was started and stopped; NOT_RUN. R6 adds the
  explicit, tested evaluation override needed for it (`evidence_model.evaluation_state_dict`).
* Multi-hop retrieval rounds: not attempted (single-hop did not qualify). CUDA: NOT_RUN (no GPU).

## 4. Gates at 1,048,576 tokens (R5 directive; historical)

| Gate | Threshold | Status |
|---|---|---|
| coarse recall@64 | >= 0.99 | PASS on every evaluated cell (>= 0.997) |
| fine recall@4 or @8 | >= 0.97 | PASS through @8 on every evaluated fine-tuned cell (>= 0.995); @4 fails on revision-heavy (0.87) |
| conditional answer accuracy | >= 0.95 | FAIL: scoped 0.88-0.94 |
| end-to-end accuracy | >= 0.95 | FAIL: revision-heavy 0.85, scoped 0.88-0.94 (records 0.98, lexical 0.96) |
| latest-binding accuracy | >= 0.95 | FAIL: revision-heavy 0.83 (records 0.97-0.98) |
| scoped symbol accuracy | >= 0.95 | FAIL: 0.879 / 0.939 |
| stale reads | 0 | PASS: 0 in every record (e.g. 30,213 and 66,768 rereads at 1M) |
| oracle use at inference | 0 | PASS (labels never change learned outputs: test) |
| unbounded state growth | 0 | PASS (constant state bytes at every length: records and tests) |
| runtime storage writes | 0 | PASS (audit-hook test with a negative control) |
| 1K regression | none | PASS (R3 benchmark streams 1.000 vs R3 0.96-0.97) |

**Verdict (historical): PARTIALLY QUALIFIED.** Demonstrated at 1M on these synthetic tasks: coarse addressing,
exact rereading with zero stale reads, bounded state, and end-to-end accuracy of 0.96-0.98 where revisions are
moderate. Unqualified: revision-heavy evidence selection, scoped-symbol use, the complete matrix. R5 is NOT a
reliable million-token language context; it is a synthetic retrieval substrate with measured limits.

## 5. Cost (fine-tuned run, seed 1, 1M, two streams, one CPU thread; PERFORMANCE_ONLY)

| set | tokens/s (ingest + reads) | coarse scan ms/query | exact reread ms/query | rerank ms/query | readout ms/query |
|---|---:|---:|---:|---:|---:|
| records | 17,997 | 3.4 | 142.4 | 4.3 | 1.2 |
| revision | 11,406 | 1.6 | 119.1 | 2.9 | 1.0 |
| scoped | 15,892 | 4.4 | 156.8 | 5.7 | 1.6 |
| lexical | 13,464 | 4.2 | 181.1 | 5.8 | 1.5 |

Per query, the scan is 32,768 x 4 x 64 = 8.4M multiply-adds; re-encoding the 64 candidates dominates (no cache).
Persistent state per stream: ring 4 MiB + index 32 MiB (float32) or 16 MiB (bfloat16) + ids 0.25 MiB = 38.0 /
21.2 MB (36.25 / 20.25 B per token); measured 75,779,072 B for two streams (float32), equal to the formula.

## 6. Reproducing (recipe; new checkpoints carry `elpis-torch-r5-checkpoint/2` metadata)

```bash
python -B -m research.elpis_torch.experiments.run_r5 train --seed S --steps 2000 --threads 1 --out r5_sS.pt
python -B -m research.elpis_torch.experiments.run_r5 train --seed S --init r5_sS.pt --supersession max --threads 1 --out r5max_sS.pt
python -B -m research.elpis_torch.experiments.run_r5 eval --ckpt r5max_sS.pt --lengths 1048576 --threads 1 --out final_sS_1048576.json
```

Seeds S = 0, 1, 2; evaluation streams come from fixed test seeds (`run_r5.eval_set`); training data from
`TRAIN_SEED + seed * 10^7 + step`. Schema /2 records the model, ERSA and evidence configurations with their
digests, the vocabulary layout digest and the SHA-256 of the task generators' source; `run_r5.load` verifies the
parameter digest and refuses undeclared topology changes (`tests/research/elpis_torch/test_r6.py`). A small
reproducibility smoke (two identical two-step runs, identical digests, checkpoint round trip) is a test. The R5
checkpoints themselves used schema /1 and are not available.
