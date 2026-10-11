"""HACF -> ECS Scientific Bridge R0 infrastructure. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. UNQUALIFIED.

Turns validated HACF structural retrieval evidence into an ECS observation through an *identified* observation map,
and asks a K1 state a read-only QUERY with it. It never learns, never writes back to HACF and never turns
retrieved evidence into authority. Nothing here is packaged or imported by ``src/``; it imports the canonical
structure and K1 bindings in one direction only (README.md).
"""
