# ECS_C R19 — prospective fixed-root Boolean composition DEV protocol

## Authority and motivation

R18 passed mechanical DEV verification but its retrospective audit (`R18_RETROSPECTIVE_SHORTCUT_AUDIT.json`, SHA-256 `9fb6a313cfa093feead709ba128a54b84b57c7c6b724a98041948ed3f007686d`) showed that a root-operator-only rule reproduced all 448 outcomes while the learned primitive gates were correct on only 6/10 cases. R18 source, model, results, audit and all earlier research remain read-only. No post-hoc relabeling, scientific rescue, or promotion of R18. This is a **fresh DEV synthetic experiment**, not an independent scientific qualification.

The checkout is pinned at `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`. The runner verifies frozen R18 inputs plus `git diff --quiet HEAD -- src native docs tests`. Research outputs stay in the existing `research/ecsc_hca_r0/external/locomo_r0/` directory. No new worktree, model service, external LLM, CUDA, QUAL data, canonical code changes, or Git mutation.

## Pre-registered falsifier and data split

Train on 640 authored formal Boolean statements: **256** fixed-root paired examples (128 paired counterfactuals), **192** primitive truth-table observations and **192** additional composite expressions. Training includes both true and false labels across leaf/operator combinations and fixed-root structural counterfactuals. Independent evaluation names and items do not appear in training; exact text overlap across splits is forbidden.

Evaluation: **544** examples, with gold labels emitted to a separate immutable `R19_EVAL_GOLD.json` not opened by the predictor:

| Group | Examples | Test |
|---|---:|---|
| fixed_root_pairs | 256 (128 pairs) | Identical ordered non-bracket tokens, identical outer AND/OR and NOT parity, opposite oracle labels from internal grouping |
| deep_fixed_root_pairs | 128 (64 pairs) | Same falsifier with deeper nested neutral wrappers |
| primitive_generalization | 96 | Independently named terminals and primitive NOT, AND and OR truth-table probes |
| unknown_connective_refusal | 32 | Unknown operator (`xor`) must refuse |
| malformed_parenthesis_refusal | 32 | Incorrect grouping must refuse |

The root-only rule is guaranteed to score **at most one correct member per opposing-label pair**. A grouping-blind deterministic classifier also cannot correctly answer both members because every pair has exactly the same ordered non-bracket tokens. The primary diagnostic is **both-correct pairs**, not just aggregate accuracy. Refusals enter the denominators; no dropped cases or score rescues. The simple root-only rule, supplied-parser learned reducer, and raw-token n-gram logistic classifier are evaluated on the same frozen public rows.

## Learners, limits and interpretation

1. **Supplied-parser reference:** the *unchanged* R18 parenthesis parser and 12-parameter probabilistic Boolean reducer, retrained on R19's balanced examples with exactly the frozen R18 training hyperparameters (110 epochs, batch 16, rate 0.065). Source parser is **authored, not learned**.
2. **Parser-free raw-token baseline:** hash-bucket logistic classifier over 1–3-gram tokens, retaining parentheses as tokens, 1,024 dimensions, 16 epochs, learning rate 0.065; no call to the grammar parser or Boolean oracle during inference. This is a **deliberately simple** raw-token comparator, *not* a state-of-the-art learned parser or Elpis's final model.
3. **Root-only shortcut control:** after the same *supplied* parser validates the input, ignore all leaves/subtrees, output AND=false or OR=true with any outer NOT parity. Cannot solve both sides of a discriminative pair.

The formal Boolean labels are authored by the same generator that creates the expressions. None of these controlled-grammar results establishes open-domain semantics, evidence-based proposition admission, learned natural-language parsing, general intelligence, or ECS_G qualification. The structural reference and the raw baseline must be interpreted separately.

## Sequential authority and output

`prepare` generates and validates exact pinned SHA-256 of `R19_TRAIN.json`, `R19_EVAL_PUBLIC.json` and `R19_EVAL_GOLD.json`. `infer` reads only TRAIN and EVAL_PUBLIC, trains both models, and publishes frozen model/predictions. `score` replays training and all predictions **before** it opens EVAL_GOLD, then writes the report exactly once. An existing file may be accepted only when bytes match identically; conflicts or source drift are HOLD, not repaired. A single-writer advisory lock excludes concurrent R19 launches. Tests and runner preserve live stdout/stderr; no `set -e`, cleanup killing, timeouts or service calls.

**Required reporting:** group-wise structured/raw/root exact matches and refusals, and pair-level both/one/neither for fixed-root and deep fixed-root. No threshold tuning, changing epochs, feature sets or candidate selection after evaluation. Whether success or failure, preserve the frozen result and investigate shortcomings in a *separate subsequent* study.
