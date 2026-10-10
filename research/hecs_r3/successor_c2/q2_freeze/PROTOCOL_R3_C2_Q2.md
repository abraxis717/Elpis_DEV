# H-ECS R3 successor C2 — Q2 qualification protocol (fresh QUAL of the unchanged candidate)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `CANDIDATE_CHANGED=NO` · `NO_R4` · `NO_INTEGRATION`

## Chronology and authority

1. C2 was frozen and pre-registered (`PROTOCOL_R3_C2.md`, sha256 `09f7278c…`, commit `8a98d9c`).
2. Fresh DEV passed 8/8 (`DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE`, disposition `2ac195de…`, commit `b64e32f`).
3. The Q1 QUAL release was issued (`d29b0622…`). Q1 seeds 0–2 were committed.
4. Q1 seed 3 was started, then killed by an external container restart.

**Q1 disposition (operator decision; incident record `Q1_INCIDENT.txt`, `05dda853…`):**
- `CURRENT_QUAL_EXECUTION=INCOMPLETE_INFRASTRUCTURE_INTERRUPTION`
- `CURRENT_QUAL_REGISTRY=SPENT` (all 12 seeds)
- `C2_SCIENTIFIC_QUAL_VERDICT=NOT_ADJUDICATED`

This is not a scientific failure, and nothing was replayed.

The Q1 tree is preserved byte-for-byte, with manifest digest `PRESERVED_TREE_MANIFEST_SHA256` in the incident record. The Q1 results on seeds 0–2 are historical evidence only. They have no role in any decision. The spent Q1 seeds are never DESIGN data and never QUAL data again.

## Question

Does the **unchanged** C2 candidate pass the unchanged frozen validity hierarchy at every seed, binding and bound horizon of a fresh, never-executed, 12-seed held-out QUAL registry? It is executed once.

## The scientific object is unchanged

Everything in `PROTOCOL_R3_C2.md` is unchanged and binding:
- the candidate law (HD/L2 ROLLOUT: 4096 updates, terminal weight 1.0, six C0 windows, λ = 0.37, unchanged rate law and clip);
- every other binding and arm, and native K1 (`8d8b7ffc…`);
- the evidence law (R3-BP arithmetic and cohort law, gates TASK → S1 → M, thresholds 0.95 / 0.90 / 0.01, every seed);
- the execution law (one shot, in order, no retry, no replay, no replacement, no rescue, no retuning);
- the pre-registered, candidate-independent TASK risk.

## The only changes: execution identity and registry

**Q2 QUAL registry (fresh, never executed).** `seed_i = u64 big-endian of SHA-256(ASCII("HECS|R3-SUCCESSOR|C2|Q2|QUAL|" + decimal(i)))[0..8]`, i = 0..11, with no rejections:

`614105602224108045`, `13431973024848494695`, `8015593928169052856`, `4368982056838705008`, `12192592809587942193`, `97214863508322549`, `6955956690753739111`, `18105131785740719551`, `17684696095471157875`, `11665294448906309688`, `16885114571500087964`, `12577501025278960659`

It is disjoint (`C2_Q2_REGISTRY.json`) from every H-ECS registry in Git and from the first 4096 indices of every known domain. That includes the spent Q1 registry, C2 DEV, the C2 DESIGN seeds, the R3-AM domains and the hecs-r1/r2/r3 domains. No world outcome was inspected, and there is no task screening or redraw.

**Artifacts.** Each differs from the frozen C2 artifact only where the registry and execution identity require.
- **Engine.** `r3_bn_c2_dev_engine.rs` `69ee89e58817ca836cf2143e07acdacded4a08dc723083759394333299209ea8` differs from the frozen C2 engine source `5345d0ac…` only in three places:
  - the QUAL registry constant;
  - the QUAL seal path (`release/QUAL2_AUTHORITY.txt`);
  - a DESIGN-mode refusal of the spent Q1 seeds.

  It is built with the C2 build command at `build/hecs_r3_successor_c2_q2_engine_r0`, giving binary `38b08ff26c092c88527008ae5ec93578c11a1e022db402bb177a4e26fcc5b0ee`. The terminal module, training core, emitter, Cargo files and hecs_r2 library tree are byte-identical to the C2 freeze. Computational identity: on a DESIGN world, the Q2 binary's five-file output is byte-identical to the frozen C2 engine source's output for the same world (Q2 freeze record).
- **Observer.** `r3_c2_observer.rs` `07b9edcb74f75c0085f3ccd4014896b5adf56810ed2f0f84410e88b06422cb79`, binary `93c4a5e8461743509c65ef5f8a1f718e79ff6710275203444a6cfd8e321de372`. It differs from the C2 observer only by the added `C2_Q2_QUAL` cohort registry. Per-seed and cohort arithmetic are unchanged, and its legacy outputs are byte-identical to R3-BP.
- **Supervisor.** `r3_c2_q2_supervisor.rs`, the frozen C2 supervisor re-pinned to the Q2 artifacts with a single QUAL2 phase. Its release requires and binds:
  - the committed DEV pass disposition;
  - the Q1 incident record and the Q1 release;
  - the C2 protocol and registries;
  - an unclosed Q1.

  The release states `CANDIDATE_CHANGED=NO`, `RETUNED_AFTER_Q1=NO`, `OLD_QUAL_RESULTS_USED_FOR_SELECTION=NO`, `Q1_REGISTRY_REUSED=NO` and `Q2_REGISTRY_FRESH=YES`.

## Outcome law

- **Every gate passes on all 12 seeds:** `WORLD_MODEL_VALID`, recorded as C2_WORLD_MODEL_VALID; the constituent blocker is closed.
- **A genuine scientific failure:** `*_INVALID_ON_QUAL`, recorded as C2_QUAL_FAIL. The Q2 registry is then spent and never rerun.
- **An infrastructure interruption:** recorded as mechanics, never as a scientific verdict.
