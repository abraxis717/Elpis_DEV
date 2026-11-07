"""CGC-LAO solver lab: bounded causal learning from selected, possibly corrupted feedback (RESEARCH_ONLY).

Self-contained research package (numpy only). Not packaged, never imported by ``src/elpis``, no runtime
authority. Modules: ``rng`` counter streams; ``env`` SLB1 simulator and potential outcomes; ``selector``
acquisition rules (PROBE_EARLY port); ``cgc`` float32 port of the checked-in CGC reference; ``filters``
the bounded reliability-mixture filter; ``agents`` baselines and candidates; ``harness`` causal rollouts;
``stats`` paired router-seed intervals; ``evaluate`` runs and evidence records; ``train_cgc`` and
``calibrate`` TRAIN-only fitting; ``candidates`` research configurations.
"""
