# ECS Binary Observation R0

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `UNQUALIFIED` · `NO_WORLD_REPRESENTATION_CLAIM` · `NO_BFM` · `NO_H-ECS`

**Disposition: `INCONCLUSIVE`** (`evidence/disposition`; `docs/research/ECS_BINARY_OBSERVER_R0_RESULTS.md`).

* Mechanics pass.
* GRAY4 and SPROJ are useful on QUAL.
* The p99 latency gate is unresolvable on the host it ran on, although the deterministic observer cost is at most
  0.52% of the median canonical QUERY.
* A Bernoulli-transition follow-up is not justified.

## The question

Can a deterministic encoder `B = Encode(observed_computation, frozen_law)` produce a compact, fixed-width binary
record of actual native ECS computations that satisfies all of the following?

* It is reproducible.
* It never moves ECS.
* It preserves demonstrably useful information.
* It costs a strictly bounded, measured amount.

The ADR (`ADR.md`, written before any code) fixes the only admissible observation point: the caller-owned result of
one accepted canonical QUERY (`elpis_ecsg_k1_query_identity`). That result is the query rows X, the answers
`f_W(X)` and the 32-byte retained-state digest.

## Pieces

| Piece | Role |
|---|---|
| `native/ebo_r0.{h,c}` → `libelpis_ebo_r0` | The observer: law admission, `ebo_encode` (64-byte EBO1 record), `ebo_record_check`, and a caller-allocated SPSC ring with `DROP_NEWEST`. No allocation, I/O or ECS symbol; the observation path calls nothing. |
| `native/test_ebo_r0.c` (ctest `research.test_ebo_r0`) | Native mechanics against the real K1 library: refusals without mutation, ties, the bounded window, non-interference, zero allocations, the threaded handoff. |
| `native/ebo_r0_bench.c` | Direct-native performance driver: interleaved `query_identity` alone vs `query_identity` + encode + push. |
| `law.py` | The pure-Python reference law, bitwise equal to the native observer. Also holds the science-only RAW16/RAW4 comparators. |
| `observer.py` | ctypes binding of the observer. It imports no `elpis` module. |
| `protocol.py` | The preregistration: worlds, registries, calibration rules, gates, performance rule, disposition. |
| `lab.py`, `run.py` | Laboratory and runner, with write-once evidence under `evidence/`. |

## Record EBO1

64 bytes:

* header;
* law tag;
* 8-byte state-identity prefix;
* 8-byte input identity;
* a 256-bit payload over the first `min(rows, 64)` answers.

There are two candidate laws, both calibrated on DESIGN only:

* `GRAY4`: 16 fixed levels, Gray-coded, 4 bits per row.
* `SPROJ`: 256 fixed sparse ±1 projections of the normalized answers, thresholded.

## Run order

```
python -m research.ecs_binary_observer_r0.run design --build <build>
python -m research.ecs_binary_observer_r0.run dev    --build <build>
python -m research.ecs_binary_observer_r0.run perf   --build <build> --stage dev --raw-dir <scratch>
(commit) python -m research.ecs_binary_observer_r0.run freeze --build <build>   (commit)
python -m research.ecs_binary_observer_r0.run qual   --build <build>
python -m research.ecs_binary_observer_r0.run perf   --build <build> --stage qual --raw-dir <scratch>
python -m research.ecs_binary_observer_r0.run disposition --build <build>
```

Every phase refuses to overwrite its evidence. QUAL and perf-QUAL refuse unless the freeze still holds. Raw timing
telemetry stays out of Git; its SHA-256 is recorded.

## Non-claims

* The record is a lossy code of one forward map on caller-chosen rows. It is not a world representation, a state
  representation, or a model of ECS.
* It is not wired into `src/` or `native/runtime`.
* It implies nothing for Bernoulli Flow Models or H-ECS.
