# H-ECS R3 successor C2 — Q2 QUAL pre-registration (unchanged candidate, fresh registry)

`RESEARCH_ONLY` · `QUAL2_RELEASED_SCIENCE_UNSTARTED` · `CANDIDATE_CHANGED=NO` · `NO_R4` · `NO_INTEGRATION`

This record was committed before any Q2 seed ran.

**Chronology:**
1. C2 frozen (`../freeze/`, `8a98d9c`).
2. Fresh DEV 8/8 PASS (`../dev/`, `b64e32f`).
3. Q1 QUAL released; seeds 0–2 committed.
4. Seed 3 killed by a container restart.
5. Q1 closed as mechanically incomplete, registry spent, no replay (`../qual_q1_incomplete/`, `24fed33`).
6. C2 continues unchanged under Q2 (this record).
7. Q2 executes once.

**Contents:**
- `PROTOCOL_R3_C2_Q2.md` — the Q2 law. It inherits `PROTOCOL_R3_C2.md` by digest and changes only the execution identity and the registry.
- `C2_Q2_REGISTRY.json` — 12 fresh, never-executed seeds, law `HECS|R3-SUCCESSOR|C2|Q2|QUAL|<i>`, no rejections. It is disjoint from every H-ECS registry in Git and from 4096 indices of every known domain (61,457 excluded values), including the spent Q1 registry. No world outcomes were inspected and there was no task screening.
- `QUAL2_RELEASE.txt` — the write-once Q2 release (`777747bc…`). It binds:
  - the Q2 engine, observer and supervisor;
  - every source;
  - the hecs_r2 library tree and native K1;
  - the C2 protocol and registries;
  - the DEV pass disposition;
  - the Q1 incident and Q1 release;
  - the flags `CANDIDATE_CHANGED=NO`, `RETUNED_AFTER_Q1=NO`, `OLD_QUAL_RESULTS_USED_FOR_SELECTION=NO`, `Q1_REGISTRY_REUSED=NO` and `Q2_REGISTRY_FRESH=YES`.
- `source/` — the Q2 engine, observer and supervisor sources.
  - The engine differs from the frozen C2 engine in 10 diff lines. These cover the QUAL registry constant, the QUAL seal path and the DESIGN-mode refusal of the spent Q1 seeds.
  - The observer differs in 8 lines, adding the `C2_Q2_QUAL` registry.
- `IDENTITY.txt` — computational identity on a DESIGN world. The Q2 engine binary's five-file output is byte-identical to the frozen C2 engine source's output for the same world. The Q2 observer stays byte-identical to R3-BP on the legacy cohort.

Raw telemetry is not in Git.
