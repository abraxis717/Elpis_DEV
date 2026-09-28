# Third-party mechanism provenance (inference)

The inference subsystem (`src/elpis/inference`) contains independent
implementations of mechanisms whose design follows the upstream sources below.
No upstream code is imported at runtime. No upstream model weights, trained
tables, tokenizer maps or evaluation data are included or downloaded.

| Mechanism | Upstream reference | Identity | Role | License | License text |
|---|---|---|---|---|---|
| DSV4.1 engram addressing (`associative.DSV41Parameters`) | DeepSeek V4.1 `inference/engram.py` | revision unpinned; SHA-256 11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897 | primary architecture | MIT | `DeepSeek-V41-MIT.txt` |
| Local/global context, sparse index, MoE experts (`global_context`, `experts`, `drivers/dsv4/target`) | DeepSeek V4.1 `inference/model.py` | revision unpinned | primary architecture | MIT | `DeepSeek-V41-MIT.txt` |
| V4-style hash and row engine (`rows`) | ds4.c `ds4_engram.c`, `ds4_engram.h` | 0aaea5a238fb41a35106a551e73c8409dfb751ac | implementation reference / independent oracle | MIT | `DS4-MIT.txt` |
| Qwen PLE n-gram addressing (`associative.QwenPLEParameters`) | vLLM `ngram_embedding.py`, `tests/test_ple.py` | 82daf9f5756e1868be0aa751afaec4726beca12a | implementation reference / independent oracle | Apache-2.0 | `Apache-2.0.txt` |
| Markov drafter and greedy verifier (`speculative`) | DeepSpec `deepspec/modeling/dspark/markov_head.py`, `eval/dspark/draft_ops.py`, `eval/dspark/evaluator.py` | 005e03b81cec38b7da6399833d609ee89a2587f2 | primary architecture | MIT | `DeepSpec-MIT.txt` |
| Context lifetimes, compaction, forks (`context`) | ZCode `context/context-types.ts`, `compaction/compact-contract.ts`, `state/session-fork.ts` | 872ad960de7ec172591f7e1952f7849229f94521 | implementation reference | Apache-2.0 | `Apache-2.0.txt` |

Comparative-only references in the beta (DeepSeek engram demo, NeMo PLE and
QSA references) informed no code here.

Golden values: the beta evaluated Qwen golden rows with the isolated vLLM
reference function and DeepSeek golden rows with the unmodified compiled DS4
hash implementation. The frozen address artifacts used by the tests are
synthetic maps and constants. They do not establish compatibility with any
production trained table; the production V4.1 tokenizer map is unavailable.

No NOTICE file was present in the Apache-2.0 upstream subsets that were used;
the repository `NOTICE` records attribution and adaptation scope without
inventing upstream notices.
