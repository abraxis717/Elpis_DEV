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
production trained table; its trained compressed address-token map is unavailable.

No NOTICE file was present in the Apache-2.0 upstream subsets that were used;
the repository `NOTICE` records attribution and adaptation scope without
inventing upstream notices.

# DeepSeek recipe V4.1 text boundary

`elpis.inference.text` implements the text-only system/user/assistant prompt
semantics of `deepseek-ai/deepseek-recipe`, revision
`8cadfede7063c896b944e7bae05daa3549ae97ea` (tree
`3ed5db20847e09a9f3d157660e27c8d2c3141fc3`). Source files:
`deepseek-recipe-encoding/src/v4/{mod,dsv41}.rs`. MIT notice is preserved in
`DeepSeek-recipe-MIT.txt`. No donor runtime or framework is vendored/imported.

The caller-supplied tokenizer is the recipe's `static/tokenizers/v41/tokenizer.json`,
SHA-256 `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`.
It is not shipped as a weight/model artifact here. The recipe records comparison
against `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` revision
`6821d6ad3681a4b137b066b76094fa82ebd0a380`, with image token 129264 reassigned
from `<｜deepseek_image｜>` (`special=false`) to `<｜image｜>` (`special=true`).
The original import revision was not recorded by the donor; the comparison
revision is not represented as that import revision. Its MIT notice is preserved
in `DeepSeek-tokenizer-MIT.txt`. See `docs/inference/DSV41_TEXT_BOUNDARY.md`.
The tokenizer license notice's CRLF line endings were normalized to LF; its
license text is unchanged.
