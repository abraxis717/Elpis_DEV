"""Structural Group Projection Compiler.

Compiles sealed typed projections into structural-group artifacts:
  StructuralGroupEvidenceV1
  StructuralGroupProposalV1
  ProposalOrderingV1
  StructuralConflictEvidenceV1

This package does NOT implement:
  adjudication, selection, routing, model loading, adapter loading,
  dispatch, activation, capability issuance, capability consumption,
  ECS mutation, recursive execution.
"""
