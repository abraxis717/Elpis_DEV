# CGC-LAO post-freeze host admission correction

**Status:** research-only host-interface hardening. This is additive to the
frozen SLB1/SLB2 v1–v4 evidence and does not change the qualified algorithm,
its experimental source or its holdout results.

## Why a strict host is necessary

The historical `v2/src/cgc_lao_v2/host.py` accepted an admitted duplicate
provider with `max_readings=1` and allowed a Contract I episode to advance on
one reading at a step that requires two. It also did not reject NaN metadata
before comparing declared correlation bounds. These are host-admission defects,
not failures of the frozen v4 evaluation, whose synthetic harness supplied
its complete `DUP2_4` schedule.

Do **not** change that historically hashed source or any frozen v4 specification
or evidence. For any new integration, use `host_closure.strict_host.StrictResearchHost`
and `strict_admit` instead. The old host is retained *only* for replay of the
published historical experiment, and is not an approved new integration seam.

## Strict contract

- Contract I requires an audited provider with **two-reading capability** and
  an actual **two-read delivery at each of steps 0–3**; exactly one reading at
  steps 4–7. A missing or extra reading fails closed, zeroizes the learner,
  and refuses further episode actions until a fresh reset.
- Contract II routes only the first reading to the conservative frozen fallback.
  A provider declaring only one reading cannot be promoted to Contract I.
- Correlation declarations and audited bounds must be finite and within [0,1].
  The host's correlation threshold may not exceed the published SLB2 v4
  **rho_admit=0.25**. A malformed, unverifiable or inconsistent declaration
  never authorizes Contract I.
- Strict host works with native C11 and the Python scalar test backend; the
  latter is now explicitly zeroized on failure and close as well.
- The registry's mechanism digest must be verified by the host. It does **not**
  certify a real measurement provider. No Level C production admission follows.

## Verify in the Elpis_DEV repository

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  research/cgc_lao/solver_lab/host_closure/test_strict_host.py -v
```

The SCIENTIFIC pytest lane also imports this suite through
`tests/research/cgc_lao_solver_lab/test_host_closure.py`.

This correction is a **post-freeze integration-boundary supplement**, not a
new scientific v5, and does not open or regenerate any holdout partition.
