"""Grid81: the bounded 9x9 structural representation and its canonical state.

* ``semantics`` — D4 dihedral actions over the 81-cell grid, pair orbits and
  passive structural contracts;
* ``typed`` — typed projection of transition rows with D4 orbit identity;
* ``groups`` — structural-group evidence, proposals, orderings and conflicts;
* ``canonical`` — the read-only, fail-closed reader of published canonical
  Grid81 state and its runtime reduction.

Everything here is read-only with respect to canonical state. Canonical Grid81
state changes only through ``elpis.pipeline.canonical``.
"""
