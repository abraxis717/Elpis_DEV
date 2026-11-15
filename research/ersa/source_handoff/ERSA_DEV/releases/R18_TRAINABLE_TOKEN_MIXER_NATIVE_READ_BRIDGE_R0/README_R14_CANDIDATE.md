# ERSA R14 — single-call C-native bounded recurrent selector

**Target disposition:** `PASS_FROZEN_R13_SINGLE_CALL_C_RECURRENCE_DIFFERENTIAL_ONLY`.

R14 preserves all **94** frozen R13 files byte for byte and adds one bounded native
select→continue→select→halt loop. It uses the **real R11 C-owned SSC exact-state
bank** as memory authority, but a **host-authorized, generation-pinned snapshot**
is still materialized before the loop begins. The entire four-hop recurrent decision
(including 8-feature extraction, scoring, canonical top-2 ranking, confidence,
pointer veto and learned or deterministic halt) occurs inside **one C call**.
It is not a direct pointer-to-live-C-bank kernel, not a GPU kernel, and not a
neural transformer replacement.

## Implementation

- New native entry: `native/ersa_recur_r14.c:ersa14_recur(...)`.
- New host adapter: `integration/ersa_recur_r14.py`.
- R11 owner RLock is held across exact-state materialization and native execution.
  Crucially, SSC *reads advance their own generation*, so the captured generation
  is pinned **after** materialization; premature pre-read generation pinning
  would incorrectly reject every request.
- Authority is a host-provided validated 0/1 admission mask; the host owns its
  provenance. The fixture uses fixed addresses `1000+i` and up to 8192 slots.
- Max hops = 4, feature width = 8, vector width = 4, head parameters = 8+6.
- No runtime learning, weight persistence, source-tree mutation, GPU use or disk IO.
- Invalid masks (including 257 -> uint8 1), values, non-finite weights and
  invalid budgets fail closed. Negative C return does not modify output receipts.
- A new **native cyclic-stop veto** refuses paths that revisit the same slot
  before claiming a final answer. This strengthens R13's weaker fixed-2 rule,
  which occasionally accepted a duplicate path as a terminal answer.

## Tests, baselines and science

- All 777 frozen ERSA R13 tests remain collected, unchanged and required.
- 198 new pytest parametrizations exercise exact host/native decision parity,
  cyclic-stop exceptions, stale generation, authorization, malformed state,
  C output atomicity, no-terminal abstention and the 8192 record ceiling.
- Science pairs 64 held-out episodes in each of four families against all four
  original R13 policies (1024 total decisions). Its digest excludes machine
  dependent timings; timings are included separately in the printed report.
- The same strong key-plus-goal deterministic controller is retained. Learned
  **superiority is not established**; in the positive family the deterministic
  rule wins on the current held-out evaluation.
- The R13 predecessor’s supervised pointer-chain task has strong template
  structure. **Open-world semantic retrieval, GDN parity, real autoregressive
  generation, end-to-end native SSC admission and neural utility are HOLD.**

## Host execution, no persistent project writes

Extract the ZIP to `/dev/shm` and execute `python3 -B tools/qualify_ersa_r14.py`.
All build artifacts, Python caches, and training intermediates must stay in
volatile memory. The qualifier requires exact R2 and SSC predecessor pins at
known original paths, or explicit `/dev/shm` overrides. It repeats file hashes
after tests. `=== SEND BACK ===` prints decisive metrics and blockers.

**No OS-level write-prevention attestation is claimed.** State remains bounded
but the full recurrent GDN alternative and end-to-end semantic learning remain
unqualified. Do not promote to canonical Elpis authority based on this packet.
