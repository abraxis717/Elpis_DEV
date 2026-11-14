# Provenance of the CGC mathematics in `research/elpis_torch`

## Upstream source

| | |
|---|---|
| Repository | https://github.com/abraxis717/CGC |
| Commit | `0895c2bc732a377b5a5f0d063ae13cc72684ec9a` (2026-11-09, `main`) |
| License | Apache-2.0 (the upstream `LICENSE` is byte-identical to this repository's `LICENSE`, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`) |
| File | `src/cgc/ecsg_e0.py` |
| File SHA-256 | `92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084` (equals the digest the upstream `SOURCE_SHA256SUMS.txt` and its R97W experiment pin as `SHA_E0`) |

The same mathematics is stated in this repository by `native/ECS_G/README.md` (the qualified native kernel)
and implemented by `native/ECS_G/src/ecsg_math.c`. Neither is modified by this work.

## What was copied

`upstream/cgc_ecsg_e0.py` is the upstream file, byte for byte. It is never edited. Its digest is checked by
`tests/research/elpis_torch/test_e0_reference.py`, which loads it from the file and uses it only as an oracle.
Nothing else was copied: not the `CGC120` wrapper, not the R97W experiment, not the native C sources, no
weights, no evidence.

## What was written here, and how it relates to the upstream

| File | Relation to the upstream mathematics | Changes |
|---|---|---|
| `e0.py` | A separate float64 implementation of the same E0 cell (`phi`, `phi'`, `f_W`, the mean-squared-error gradient, one explicit gradient step). Written after reading the upstream file and the ECS_G README; it follows their formulas and the upstream operation order. | Equal to the upstream bitwise on every tested case. Adds the unpacked moment form `(mu, M, T3)` (the upstream packs the upper triangle). Raises `E0Error` with descriptive messages instead of the upstream's code strings. No K1, no FMS, no state. |
| `cgc.py` `CGCCore` | `C` E0 cells sharing one input, evaluated as one matrix product. With `preact_bound=None` each output channel equals `e0.forward` of that channel's `[d, N]` weight. | New tensor layout `[d, C, N]` (the upstream `CGC120` holds four separate `[5, 6]` cells). `phi` in Horner form (agrees with the reference to round-off, not bitwise). Runs in float32 or float64 and on inputs of any leading shape. |
| `cgc.py` `preact_bound` | **Extension, not E0.** `u -> b tanh(u/b)` before `phi`. Deviates from E0 by at most `\|u\|^3 / (3 b^2)` in the pre-activation and converges to E0 as `b -> inf`. | New. Exists to cap the cubic's growth. |
| `cgc.py` `CGCTransform` | **Architecture around E0.** Optional linear bottleneck `P_in` and linear readout `P_out`. Without the bound, cell `c` is exactly E0 on the full hidden input with weight `P_in^T W_c` (a rank constraint on `W`). | New. E0 has neither a readout nor a factorized weight. |
| `model.py`, `session.py` | Use `CGCTransform` inside a transformer block and adapt the core weights `W` per session. | New. The session's update is plain gradient descent on a language-model loss; it is not the G1 recurrence, not K1 and not the R97W Reptile procedure. |

## What is not claimed

* None of the tensorized or bounded operators is a qualified ECS_G kernel. The qualified kernel is
  `native/ECS_G`, in its frozen regime `d = 6`, `N in {36, 48, 72}`, float64.
* The R97W result (a 120-parameter CGC against a parameter-matched GELU network on synthetic meta-learning
  tasks) says nothing about language models, and nothing here reproduces or extends it.
* Agreement with the upstream file is established on the tested shapes and inputs
  (`tests/research/elpis_torch/test_e0_reference.py`), not for all inputs.
