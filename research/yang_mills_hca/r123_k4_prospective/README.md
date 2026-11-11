# R123: Preregistered K4 spectral-correction causal shadow

**Research only. R93 remains `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.**

Exact base: `501be02d33201ce9c95e211d6ba9a1b531656f43` on the published R122 Claude branch.

This prospective experiment fixes, before data consumption, the K4 self-retention correction in post-sigmoid gate-value space:

- NATIVE (0.0) baseline;
- 0.25 fraction primary, selected as the only nontrivial discovery candidate with zero observed induced readout reversals across all four consumed amplitude scales;
- 0.50 fraction secondary, has no rescue authority over a primary failure.

The target is the frozen R30 teacher's `k4` small-amplitude limit, used as a disclosed **offline diagnostic oracle**. No learning, autonomously inferred teacher calibration, checkpoint mutation or production integration is claimed.

The runner performs a known-data preflight, generates and publishes a new fixed source/protocol commit, verifies its remote hash, **then and only then** executes fresh seeds and graph classes once. It publishes either a scientific PASS or scientifically negative result in a separate evidence commit. Git working state remains under `/dev/shm` and is preserved for troubleshooting; GitHub is the only remote publication target.

The complete source, explicit selected graphs, numerical gates, hashes and all decision thresholds live in `make_protocol.py`, `PROTOCOL_R123.json`, and `qualify.py`. The runner refuses if the preregistration is not remotely visible, prior evidence exists, inherited source pins differ, or the graph/seed audit fails. Full results are placed in the research `evidence/` directory only after measurement terminates.

The primary scientific gate requires at ordinary scale, independently for both checkpoint cohorts and k=0/2/6: at least 20% mean H20 full-fixture-state NSE improvement; 3/4 graph classes improved; no graph mean worse by more than 0.10 absolute NSE; H8 pooled relative deterioration no more than 5%; and teacher-forced H20 pooled relative deterioration no more than 10%. It additionally requires zero induced readout reversals at scales `2^-20`, `2^-2` and `1` across all twenty steps and all six checkpoints. Above-clamp `2^2` is separately reported, not a rescue gate.

Readout NSE, paired-state NSE, induced extinguishments, graph-block results, exact guarded/reference parity, and R93 disposition are separately reported. The four new incidence classes exclude the complete historical inventory and R122; the experiment has limited independent graph-level statistical power.
