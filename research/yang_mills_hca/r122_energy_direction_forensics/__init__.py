"""R122: energy-normalized fidelity, directional safety and recurrent error localization. RESEARCH_ONLY.

A failure-localization experiment on the frozen 88-parameter HCA (six R77/R78 checkpoints)
over the R118 sparse AWR substrate with the R119 EXACT_GUARDED kernel. Nothing is trained,
no controller is fitted, and no result here clears ``ENERGY_NORMALIZED_DENOMINATOR_HOLD``
(R93). See README.md and PROTOCOL_R122.json.

Importing this package imports nothing heavy; torch is needed only where the frozen oracle
(the R30 teacher) is used.
"""
