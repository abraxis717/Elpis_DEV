# ECS Binary Observation R0: what DEV changed, and what it did not

Written after DEV and before the freeze.

* The gate thresholds, the performance rule, the sample sizes, the registries and the two laws (`evidence/design/*.bin`)
  are unchanged since the preregistration commit `11a6441`.
* DESIGN ran at `11a6441` and DEV at `a9be051`, both on the same laboratory source (source digest `2ada336a...`,
  observer library `d3c684e5...`).

## DEV results (evidence/dev, evidence/perf_dev)

* **Mechanics.** All checks passed:
  * 1248 records, with native and reference bitwise equal and repeatable;
  * OFF/ON identical in all 12 worlds;
  * no other failures.
* **Science.** Both encoders passed S1, S2a, S2b, S3 and S4 on the 12 DEV worlds:
  * GRAY4 S3 R^2 0.859, S4 0.880;
  * SPROJ S3 0.643, S4 0.629;
  * RAW16 S3 0.789;
  * every control was at or below 0.
* **Performance rehearsal.** INCONCLUSIVE for both encoders at every row count.
  * The tail of the canonical QUERY on this shared cloud host is too noisy to resolve the preregistered 1.5 us
    p99 threshold. The A/A (OFF vs OFF) 95% intervals span roughly ±5 to ±10 us.
  * Resolving ±0.5 us would need roughly a hundred times more samples, and probably a quiet host.
  * **The rule and the sample sizes were not changed.** The QUAL performance run is expected to be INCONCLUSIVE on
    this host for the same reason.

## Changes after DEV (all mechanical; no law, gate or threshold moved)

1. **Encoder implementation (`native/ebo_r0.c`), outputs bitwise unchanged.** The DEV rehearsal measured the
   isolated SPROJ cost at 1.26 us, which is 0.99% of the QUERY median at 64 rows. Three changes:
   * the input-identity hash keeps its four lanes in registers;
   * GRAY4 finds the level by a branchless search over the strictly increasing thresholds instead of 15 compares;
   * SPROJ reads each signed term from a `[u, -u]` table using an index derived at law admission (`ebo_law.term`).

   The same operations are performed in the same order:
   * The current library reproduces the DESIGN record digest and the DEV record digest bitwise.
   * All parity tests pass.
   * Isolated cost after the change (scratch measurement, not evidence):
     * GRAY4: 0.02 to 0.55 us;
     * SPROJ: 0.47 to 0.93 us.
   * The QUAL performance run measures this implementation. The DEV rehearsal measured the earlier one.
2. **Native SPSC test (`native/test_ebo_r0.c`).** The test-side retry loop for the terminator record counted its own
   failed pushes as drops, so the loss accounting could overshoot. This was a defect in the test, not the observer.
   The drop count is now taken before the terminator.
3. **Library audit (`run.py: audit_library`).** The audit now follows local (static) calls transitively. After the
   change, `ebo_encode` calls the non-inlined static `encode_sproj`. An external call other than the
   stack-protector failure path still fails the audit.
4. **Repository boundary registrations (outside this lineage):**
   * `tests/boundary/test_mission.py` pins the research native targets and now includes `elpis_ebo_r0`.
   * `tests/boundary/test_one_ecs.py` lists this lineage's files that name the K1 ABI (`elpis_ecsg_k1_*`), with a
     reason.
