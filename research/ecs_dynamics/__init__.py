"""ECS neural-dynamics research laboratory.

RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. NO_PRODUCTION_NEURAL_CLAIM.

This package is a synthetic laboratory for questions about coarse observables,
target-relative sufficiency, delay memory and finite-time dynamical
diagnostics. It is not packaged with ``elpis-dev``, it is not imported by any
production module, and nothing in it can mutate ECS state. The tanh family is a
probe, not the production architecture; the cubic family is an exact algebraic
control. No result here says anything about a trained or production model.

Submodules are imported explicitly; importing this package loads nothing else.
"""

RESEARCH_ONLY = True
NO_RUNTIME_AUTHORITY = True
NO_PRODUCTION_NEURAL_CLAIM = True

# The four papers named in the brief were not reachable from this environment
# (egress policy). The requester then supplied the PDFs; the laboratory's
# paper-derived parts rest on those files, identified by SHA-256 below. The
# PDFs themselves are not committed.
PAPER_ACCESS = {
    "2609.19288": "AVAILABLE_USER_SUPPLIED_PDF",
    "2609.29834": "AVAILABLE_USER_SUPPLIED_PDF",
    "2609.19424": "AVAILABLE_USER_SUPPLIED_PDF",
    "2609.07341": "AVAILABLE_USER_SUPPLIED_PDF",
}
PAPER_SHA256 = {
    "2609.19288": "595988809f49614c81ae065860ec3fc64a8ae02a027b36b09b1016c901daff10",
    "2609.29834": "34edc64b4811662a26323b5578ebb4f61d8ba31fbd32830a659bfd6bd985f2e4",
    "2609.19424": "422ea8452ad9b8f049c93a41b170207da58aaee7ca1f74da07c9b0b6b7682a5e",
    "2609.07341": "0c3afa02e8cd3d4dc7abf1513de63da4abd481d2fd2fcde1ef0c662e518051b7",
}
# Paper-derived statements are listed, with section and equation citations,
# in docs/research/ECS_DYNAMICS_RESULTS.md and in each result's ``sources``.
PAPER_DERIVED_CLAIMS = "CITED_PER_RESULT"
