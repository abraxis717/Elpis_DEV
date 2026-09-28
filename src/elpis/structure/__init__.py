"""Elpis structure: structural memory and representation.

Structure is the system's memory of *what things are and how they relate*:

* HACF (native): content-addressed cascade/graph identity, chunked
  content-addressed corpus, FMS-resident exact vector index, context graph,
  deterministic hybrid retrieval producing canonical RetrievalBundles.
* The semantic core (native): typed hypergraph with segments, snapshots and
  query overlays; context-deficit control; evidence typing and admission;
  bounded semantic views; semantic topology IR; Grid81 structural packets and
  read-only structural observations.
* ``elpis.structure.retrieval``: the Python-facing retrieval stage (query
  derivation, bounded hybrid retrieval through an explicitly loaded bridge,
  bundle validation, budgets, evidence envelopes).
* ``elpis.structure.grid81``: the bounded Grid81 representation (D4 group
  semantics, typed and structural-group projections, canonical substrate).

Structure never grants semantic truth, execution or mutation authority by
itself. Canonical Grid81 state changes only through the pipeline's explicit
canonical writer path.
"""

__all__ = ("retrieval",)
