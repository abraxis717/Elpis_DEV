"""Elpis Torch R0: a trainable CGC-native decoder-only transformer. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY.

An architectural research prototype (PyTorch, optional dependency), isolated from the canonical runtime:
nothing in ``src/elpis`` or ``native/`` imports it, and it claims no language intelligence
(docs/research/elpis_torch/DESIGN.md, docs/research/elpis_torch/EVIDENCE_R0.md).

Importing this package does not import PyTorch; its submodules do:

    e0          float64 reference of the CGC / ECS_G E0 cubic cell (validated against the pinned upstream)
    cgc         batched E0 operators for hidden states ``[batch, sequence, hidden]``
    config      model configuration and validation
    tokenizer   reversible byte-level tokenizer
    model       the decoder-only transformer (FFN, CGC and FFN+CGC channel transformations)
    training    offline teacher-forced training (explicit artifacts only)
    session     bounded, session-local, volatile CGC fast weights
    data        generated token tasks with disjoint training and held-out splits
    accounting  parameter, theoretical compute and memory accounting
"""

PHASE = "ELPIS_TORCH_GENERATIVE_SUBSTRATE_R0"
CLASSIFICATION = "RESEARCH_ONLY NO_RUNTIME_AUTHORITY NO_LANGUAGE_CLAIM"
