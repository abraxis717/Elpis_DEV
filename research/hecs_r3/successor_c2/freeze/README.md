# H-ECS R3 successor C2 — frozen candidate, pre-registered before DEV (2026-10-09)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `DEV_RELEASED_SCIENCE_UNSTARTED` · `NO_QUAL_RELEASE` · `NO_R4` · `NO_INTEGRATION`

This directory records the frozen candidate C2 and its DEV release. It was committed before any C2 DEV seed was executed.

**Candidate.** C2 is the frozen R3 successor engine with one change, at the HD/L2 ROLLOUT arm only. That arm is trained from the native G1 W for 4096 updates. Each update uses the C0 window picks and the unchanged R3 rollout objective (λ = 0.37), plus a true-target terminal anchor `1.0 · 0.5‖ẑ₄ − z₄‖²/(LATENT·var)` with an exact reverse-mode gradient through all four transitions. The rate law and clip are unchanged. Everything else is the C0/C1 code path. C2 is not compute-matched to C0 or C1, and no superiority claim is made.

**Why.** The successor 8/8 closure (`research/hecs_r3/closure/successor_dev_8_of_8/`) failed only HD/L2 h=4. DESIGN on the 8 burned worlds plus 80 DESIGN-only worlds showed three things:
- The failure is a lower-tail sub-population: C0 fails HD/L2 in about 25–31% of fresh worlds.
- The 256-update fine-tune is under-converged.
- Terminal anchoring plus a convergent budget removes the tail: 40/40 DESIGN-train worlds (minimum h4 capture 0.972) and 48/48 held-out DESIGN worlds (minimum 0.976).

A full-hierarchy pass with this engine on 32 DESIGN worlds passed every gate in 31. The remaining world was TASK-invalid through the candidate-independent MATCHED-world T3 criterion, a risk pre-registered in the protocol. All details and per-variant aggregates are in `DESIGN_RECORD.json`.

**Contents.**
- `PROTOCOL_R3_C2.md` — the frozen law: candidate, registries, evidence, execution, build.
- `C2_REGISTRIES.json` — the fresh 8-seed DEV registry and the reserved 12-seed held-out QUAL registry, with the derivation law and exclusions.
- `DESIGN_RECORD.json` — DESIGN evidence and the selection rule.
- `DEV_RELEASE.txt` — the write-once DEV release. It binds the engine, observer, K1, supervisor, protocol, sources, library tree, registries and DESIGN record.
- `source/` — the C2 engine and support module, the C2 observer and the C2 supervisor. The unchanged support modules and the hecs_r2 library are at the closure and source-closure commits.

Raw telemetry is not in Git.
