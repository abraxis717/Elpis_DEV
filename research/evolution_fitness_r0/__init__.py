"""Evolution Fitness R0 infrastructure: an independent fitness environment. RESEARCH_ONLY. UNQUALIFIED. NO_CLAIM.

A tiny deterministic environment (state, action, transition, feedback, terminal evaluation, replay identity) with
frozen world partitions, negative baselines and an evaluator that recomputes fitness itself from replayed
trajectories, so that an evolution policy (``elpis.evolution.policy``) can pin an evaluator that measures something
outside the candidate. Nothing here is packaged or imported by ``src/`` (README.md).
"""
