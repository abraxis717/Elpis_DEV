"""Independent evolution policy authority: evolution may propose; it may not authorize or evaluate itself.

Before this module every authority-bearing value of an evolution attempt came from its caller: the gate's allowed
component scopes, resource budget and evaluation contract were constructor arguments, an assertion chose its own
edit budget, and :class:`~elpis.evolution.promotion.EvaluationEvidence` was a record anyone could fill in (its
gates, deltas and scope verdict were self-reported). A candidate could therefore authorize and evaluate itself.

An :class:`EvolutionPolicy` is a catalog document (``elpis.evolution-policy.v1``) whose SHA-256 must equal an
**independent pin** from trusted configuration (``RuntimeConfig.evolution_policy_sha256`` for the managed runtime;
the same pattern as the codec and native authorities). It fixes, outside any candidate:

* **objective**: what is optimized (the promotion law's held-out delta, maximized);
* **evaluator**: the one evaluator implementation (``module:qualname`` and the SHA-256 of its defining source
  file, *measured* by the authority, never supplied by a candidate);
* **candidate scopes** a candidate may edit, and **protected scopes** (the evaluator, the policy, the fitness
  environment, ...) it may never touch; the two are disjoint;
* **budget**: the largest edit budget an assertion may claim, the largest resource cost, the most candidates one
  selection may weigh;
* **evaluation contract**: the four frozen partition manifests (EVOLVE, CALIBRATION, HELD_OUT, OOD; pairwise
  distinct), the noise envelope and the OOD regression floor; it may only narrow the fixed promotion law
  (:mod:`elpis.evolution.promotion`);
* **side effects**: ``CANDIDATE_WORKSPACE_ONLY`` (an evaluation must leave both workspaces byte-identical; the
  child is materialized only through an approved grant);
* **confinement**: ``TRUSTED_OPERATOR_CALLBACKS``. Elpis does not sandbox the evaluator or an attempt's ``advance``
  callback: both are trusted operator code. The evaluator is pinned by digest; ``Runtime.evolve`` is an operator
  operation (``elpis.runtime.persistence``). Nothing here confines a malicious in-process caller;
* **promotion**: operator approval is required (``requires_operator_approval`` must be true): no candidate is
  materialized autonomously.

An :class:`EvolutionPolicyAuthority` binds a policy to its evaluator callable (measured) and is the only issuer of:

* the gates the managed runtime accepts (:meth:`EvolutionPolicyAuthority.gate`; ``Runtime.evolve`` refuses any
  other gate, and re-verifies an issued one against the policy);
* evaluations (:meth:`EvolutionPolicyAuthority.evaluate`): the authority verifies both workspaces against their
  manifests, computes the changed components and the source-scope verdict itself, refuses a candidate that
  carries or edits the evaluator, runs the pinned evaluator, verifies the evaluation had no side effect, and builds
  the evidence with the policy's partitions and noise envelope. Self-reported evidence is never selected
  (``EVALUATION_NOT_INDEPENDENT``);
* selections (:meth:`EvolutionPolicyAuthority.select`) and, against an explicit operator approval digest, promotion
  grants (:meth:`EvolutionPolicyAuthority.approve`), without which :meth:`EvolutionPolicyAuthority.materialize`
  writes nothing.

Non-claims: the pins authenticate exact bytes, not an author; the evaluator digest covers its defining module's
source file only (not its imports or the interpreter); issuance registries are process-local and are no protection
against in-process code that constructs its own policy and pin. No evaluator, fitness environment or promotion is
qualified by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
from threading import Lock
from types import MappingProxyType
from typing import Callable

from .digests import domain_digest, require_digest
from .path_gate import EvolutionPathGate
from .promotion import (
    NOISE_ENVELOPE,
    OOD_REGRESSION_FLOOR,
    REQUIRED_PARTITIONS,
    RESOURCE_COST_LIMIT,
    CandidateRecord,
    EvaluationEvidence,
    HarnessManifest,
    SelectionReceipt,
    atomic_materialize,
    digest_tree,
    select,
)

__all__ = ("EVOLUTION_POLICY_SCHEMA", "EvaluatorReport", "EvolutionPolicy", "EvolutionPolicyAuthority",
           "EvolutionPolicyError", "PromotionGrant", "implementation_identity", "policy_of")

EVOLUTION_POLICY_SCHEMA = "elpis.evolution-policy.v1"
SIDE_EFFECTS = "CANDIDATE_WORKSPACE_ONLY"
CONFINEMENT = "TRUSTED_OPERATOR_CALLBACKS"
OBJECTIVE_METRIC, OBJECTIVE_DIRECTION = "held_out_delta", "MAXIMIZE"
MAX_CANDIDATES = 64


class EvolutionPolicyError(ValueError):
    """A refusal of the evolution policy authority, with a stable code."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _refuse(code: str, detail: str = ""):
    raise EvolutionPolicyError(code, detail)


def _hex(value, what):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _refuse("EVOLUTION_POLICY", what + ": 64 lowercase hex characters")
    return value


def _int(value, what, low, high):
    if type(value) is not int or not low <= value <= high:
        _refuse("EVOLUTION_POLICY", f"{what}: an int in [{low}, {high}]")
    return value


def _text(value, what):
    if type(value) is not str or not value.strip():
        _refuse("EVOLUTION_POLICY", what + ": a non-empty string")
    return value


def _fields(data, names, what):
    if type(data) is not dict or set(data) != set(names):
        _refuse("EVOLUTION_POLICY", what + " fields: " + ", ".join(sorted(names)))
    return data


def _scope(value, what):
    path = PurePosixPath(_text(value, what))
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        _refuse("EVOLUTION_POLICY", what + ": a normalized relative path")
    return value


def _under(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + "/")


def implementation_identity(fn) -> tuple[str, str]:
    """``(module:qualname, SHA-256 of the defining source file)`` of a function or class, measured here."""
    qualname = getattr(fn, "__qualname__", None)
    if not callable(fn) or type(qualname) is not str or "<" in qualname:
        _refuse("EVALUATOR_IDENTITY", "a named, importable function or class is required")
    try:
        path = inspect.getsourcefile(fn)
    except TypeError:
        path = None
    if not path:
        _refuse("EVALUATOR_IDENTITY", "the implementation has no source file to measure")
    with open(path, "rb") as handle:
        source = handle.read()
    return f"{fn.__module__}:{qualname}", hashlib.sha256(source).hexdigest()


def _source_path(fn) -> Path:
    return Path(inspect.getsourcefile(fn)).resolve()


@dataclass(frozen=True)
class EvaluatorReport:
    """What the pinned evaluator measures. The authority adds everything else (scope, partitions, envelope)."""

    correctness_pass: bool
    leakage_pass: bool
    resource_cost: int
    held_out_delta: int
    ood_delta: int

    def __post_init__(self):
        if type(self.correctness_pass) is not bool or type(self.leakage_pass) is not bool:
            raise TypeError("evaluator gates must be bool")
        for name in ("resource_cost", "held_out_delta", "ood_delta"):
            if type(getattr(self, name)) is not int:
                raise TypeError(name + " must be an int")
        if self.resource_cost < 0:
            raise ValueError("resource_cost must be non-negative")


class EvolutionPolicy:
    """An independently pinned evolution policy (``elpis.evolution-policy.v1``); immutable."""

    __slots__ = ("_document", "digest", "source", "provenance", "objective_id", "evaluator_id", "evaluator",
                 "evaluator_sha256", "candidate_scopes", "protected_scopes", "max_edit_budget", "max_resource_cost",
                 "max_candidates", "partitions", "noise_envelope", "ood_regression_floor", "resource_budget_digest",
                 "evaluation_contract_digest")

    def __init__(self, document: bytes, *, expected_sha256: str):
        _hex(expected_sha256, "policy pin")
        if type(document) is not bytes or hashlib.sha256(document).hexdigest() != expected_sha256:
            _refuse("EVOLUTION_POLICY_UNPINNED", "the policy document does not match its independent pin")

        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    _refuse("EVOLUTION_POLICY", "duplicate key")
                out[key] = value
            return out

        try:
            data = json.loads(document, object_pairs_hook=unique)
        except (json.JSONDecodeError, UnicodeError):
            _refuse("EVOLUTION_POLICY", "malformed policy document")
        _fields(data, ("schema", "source", "provenance", "objective", "evaluator", "candidate_scopes",
                       "protected_scopes", "budget", "evaluation_contract", "side_effects", "confinement",
                       "promotion"), "policy")
        if data["schema"] != EVOLUTION_POLICY_SCHEMA:
            _refuse("EVOLUTION_POLICY", "policy schema")
        if data["provenance"] not in ("deployment", "test-fixture"):
            _refuse("EVOLUTION_POLICY", "provenance: deployment or test-fixture")
        objective = _fields(data["objective"], ("objective_id", "metric", "direction"), "objective")
        if (objective["metric"], objective["direction"]) != (OBJECTIVE_METRIC, OBJECTIVE_DIRECTION):
            _refuse("EVOLUTION_POLICY", "objective: the promotion law's held_out_delta, MAXIMIZE")
        evaluator = _fields(data["evaluator"], ("evaluator_id", "implementation", "implementation_sha256"),
                            "evaluator")
        if _text(evaluator["implementation"], "evaluator implementation").count(":") != 1:
            _refuse("EVOLUTION_POLICY", "evaluator implementation: module:qualname")
        candidate_scopes, protected_scopes = data["candidate_scopes"], data["protected_scopes"]
        for scopes, what in ((candidate_scopes, "candidate_scopes"), (protected_scopes, "protected_scopes")):
            if type(scopes) is not list or not scopes or len(set(scopes)) != len(scopes):
                _refuse("EVOLUTION_POLICY", what + ": a non-empty list of distinct paths")
            for scope in scopes:
                _scope(scope, what)
        for c in candidate_scopes:
            for p in protected_scopes:
                if _under(c, p) or _under(p, c):
                    _refuse("EVOLUTION_POLICY", f"candidate scope {c!r} overlaps protected scope {p!r}")
        budget = _fields(data["budget"], ("max_edit_budget", "max_resource_cost", "max_candidates"), "budget")
        _int(budget["max_edit_budget"], "max_edit_budget", 0, 1 << 20)
        _int(budget["max_resource_cost"], "max_resource_cost", 0, RESOURCE_COST_LIMIT)
        _int(budget["max_candidates"], "max_candidates", 1, MAX_CANDIDATES)
        contract = _fields(data["evaluation_contract"], ("partitions", "noise_envelope", "ood_regression_floor"),
                           "evaluation_contract")
        partitions = _fields(contract["partitions"], REQUIRED_PARTITIONS, "partitions")
        for name, value in partitions.items():
            _hex(value, "partition " + name)
        if len(set(partitions.values())) != len(partitions):
            _refuse("EVOLUTION_POLICY", "the four partitions must be distinct (no partition is another)")
        if contract["noise_envelope"] != NOISE_ENVELOPE or type(contract["noise_envelope"]) is not int:
            _refuse("EVOLUTION_POLICY", "noise_envelope: the promotion law's own")
        _int(contract["ood_regression_floor"], "ood_regression_floor", OOD_REGRESSION_FLOOR, 1 << 20)
        if data["side_effects"] != SIDE_EFFECTS:
            _refuse("EVOLUTION_POLICY", "side_effects: " + SIDE_EFFECTS)
        if data["confinement"] != CONFINEMENT:
            _refuse("EVOLUTION_POLICY", "confinement: " + CONFINEMENT)
        promotion = _fields(data["promotion"], ("requires_operator_approval",), "promotion")
        if promotion["requires_operator_approval"] is not True:
            _refuse("EVOLUTION_POLICY", "promotion requires operator approval")
        values = {
            "_document": document, "digest": expected_sha256, "source": _text(data["source"], "source"),
            "provenance": data["provenance"], "objective_id": _text(objective["objective_id"], "objective_id"),
            "evaluator_id": _text(evaluator["evaluator_id"], "evaluator_id"),
            "evaluator": evaluator["implementation"],
            "evaluator_sha256": _hex(evaluator["implementation_sha256"], "evaluator implementation_sha256"),
            "candidate_scopes": tuple(candidate_scopes), "protected_scopes": tuple(protected_scopes),
            "max_edit_budget": budget["max_edit_budget"], "max_resource_cost": budget["max_resource_cost"],
            "max_candidates": budget["max_candidates"], "partitions": MappingProxyType(dict(partitions)),
            "noise_envelope": contract["noise_envelope"], "ood_regression_floor": contract["ood_regression_floor"],
        }
        values["resource_budget_digest"] = domain_digest("elpis.evolution-policy.resource-budget.v1", {
            "policy": expected_sha256, "max_edit_budget": budget["max_edit_budget"],
            "max_resource_cost": budget["max_resource_cost"], "max_candidates": budget["max_candidates"]})
        values["evaluation_contract_digest"] = domain_digest("elpis.evolution-policy.evaluation-contract.v1", {
            "policy": expected_sha256, "objective": [objective["objective_id"], OBJECTIVE_METRIC, OBJECTIVE_DIRECTION],
            "evaluator": [evaluator["evaluator_id"], evaluator["implementation"],
                          evaluator["implementation_sha256"]],
            "partitions": sorted([k, v] for k, v in partitions.items()),
            "noise_envelope": contract["noise_envelope"], "ood_regression_floor": contract["ood_regression_floor"]})
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("EvolutionPolicy is immutable")

    def intact(self) -> bool:
        return hashlib.sha256(self._document).hexdigest() == self.digest


@dataclass(frozen=True)
class PromotionGrant:
    """An operator-approved promotion of one selection (issued by :meth:`EvolutionPolicyAuthority.approve`)."""

    policy_digest: str
    selection_receipt_digest: str
    selected_harness_manifest_digest: str
    operator_approval_digest: str


_LOCK = Lock()
_GATES: dict[int, tuple[EvolutionPathGate, "EvolutionPolicyAuthority"]] = {}


def policy_of(gate) -> "EvolutionPolicyAuthority | None":
    """The authority that issued ``gate``, or None for a gate built any other way."""
    found = _GATES.get(id(gate))
    return found[1] if found is not None and found[0] is gate else None


class EvolutionPolicyAuthority:
    """One pinned policy bound to its measured evaluator: the issuer of gates, evaluations, selections, grants."""

    def __init__(self, policy: EvolutionPolicy, evaluator: Callable[..., EvaluatorReport]):
        if type(policy) is not EvolutionPolicy or not policy.intact():
            _refuse("EVOLUTION_POLICY_UNPINNED", "an intact, pinned EvolutionPolicy is required")
        if implementation_identity(evaluator) != (policy.evaluator, policy.evaluator_sha256):
            _refuse("EVALUATOR_UNPINNED", "the evaluator is not the policy's pinned implementation")
        self.policy = policy
        self._evaluator = evaluator
        self._evaluations: dict[str, tuple[str, int]] = {}   # evidence digest -> (candidate manifest, edit count)
        self._selections: dict[str, SelectionReceipt] = {}
        self._grants: dict[str, PromotionGrant] = {}

    def require(self) -> None:
        """Re-verify the policy document and the evaluator's measured identity (every use)."""
        if not self.policy.intact():
            _refuse("EVOLUTION_POLICY_UNPINNED", "the policy document changed")
        if implementation_identity(self._evaluator) != (self.policy.evaluator, self.policy.evaluator_sha256):
            _refuse("EVALUATOR_UNPINNED", "the evaluator changed since admission")

    # -- attempts --------------------------------------------------------------------------------------------------
    def gate(self) -> EvolutionPathGate:
        """A path gate configured from the policy alone (scopes, budgets, contract)."""
        self.require()
        gate = EvolutionPathGate(allowed_component_scopes=self.policy.candidate_scopes,
                                 resource_budget_digest=self.policy.resource_budget_digest,
                                 evaluation_contract_digest=self.policy.evaluation_contract_digest,
                                 max_edit_budget=self.policy.max_edit_budget)
        with _LOCK:
            _GATES[id(gate)] = (gate, self)
        return gate

    def verify_gate(self, gate) -> None:
        """Refuse unless ``gate`` was issued by this authority and still carries exactly the policy's values."""
        self.require()
        if policy_of(gate) is not self:
            _refuse("EVOLUTION_POLICY_UNAUTHORIZED", "the gate was not issued by this policy authority")
        if (gate.allowed_component_scopes, gate.resource_budget_digest, gate.evaluation_contract_digest,
                gate.max_edit_budget) != (frozenset(self.policy.candidate_scopes), self.policy.resource_budget_digest,
                                          self.policy.evaluation_contract_digest, self.policy.max_edit_budget):
            _refuse("EVOLUTION_POLICY_UNAUTHORIZED", "the gate's configuration is not the policy's")

    # -- evaluation ------------------------------------------------------------------------------------------------
    def _changed(self, parent: HarnessManifest, candidate: HarnessManifest) -> tuple[str, ...]:
        before, after = dict(parent.component_content_digests), dict(candidate.component_content_digests)
        return tuple(sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p)))

    def evaluate(self, *, parent: HarnessManifest, candidate: HarnessManifest, parent_root, candidate_root
                 ) -> CandidateRecord:
        """Evaluate ``candidate`` against ``parent`` with the pinned evaluator. The candidate supplies its workspace
        and manifest; every verdict and binding in the evidence is computed here."""
        self.require()
        if type(parent) is not HarnessManifest or type(candidate) is not HarnessManifest:
            _refuse("EVALUATION_INVALID", "harness manifests are required")
        if candidate.parent_harness_manifest_digest != parent.digest:
            _refuse("STALE_PARENT_BINDING", "the candidate is not a child of this parent")
        parent_root, candidate_root = Path(parent_root).resolve(), Path(candidate_root).resolve()
        for root, manifest, what in ((parent_root, parent, "parent"), (candidate_root, candidate, "candidate")):
            if digest_tree(root) != manifest.component_content_digests:
                _refuse("WORKSPACE_MANIFEST_MISMATCH", f"the {what} workspace is not its manifest")
        source = _source_path(self._evaluator)
        changed = self._changed(parent, candidate)
        content = dict(candidate.component_content_digests)
        if any(source.is_relative_to(root) for root in (parent_root, candidate_root)) or any(
                content.get(p) == self.policy.evaluator_sha256 for p in changed):
            # The evaluator would be loaded from, or carried by, the workspace under evaluation.
            _refuse("EVALUATOR_IN_CANDIDATE", "a candidate never carries or edits its evaluator")
        scope_pass = all(any(_under(p, s) for s in self.policy.candidate_scopes) for p in changed) and not any(
            _under(p, s) for p in changed for s in self.policy.protected_scopes)
        report = self._evaluator(parent_root=parent_root, candidate_root=candidate_root,
                                 partitions=self.policy.partitions)
        if type(report) is not EvaluatorReport:
            _refuse("EVALUATION_INVALID", "the evaluator must return an EvaluatorReport")
        for root, manifest in ((parent_root, parent), (candidate_root, candidate)):
            if digest_tree(root) != manifest.component_content_digests:
                _refuse("EVALUATION_SIDE_EFFECT", "the evaluation changed a workspace")
        evidence = EvaluationEvidence(
            candidate_harness_manifest_digest=candidate.digest,
            parent_harness_manifest_digest=parent.digest,
            evaluation_contract_digest=self.policy.evaluation_contract_digest,
            path_transition_receipt_digest=candidate.path_transition_receipt_digest,
            partition_manifest_digests=tuple(sorted(self.policy.partitions.items())),
            correctness_pass=report.correctness_pass,
            leakage_pass=report.leakage_pass,
            resource_pass=report.resource_cost <= self.policy.max_resource_cost,
            source_scope_pass=scope_pass,
            resource_cost=report.resource_cost,
            held_out_delta=report.held_out_delta,
            ood_delta=report.ood_delta,
            noise_envelope=self.policy.noise_envelope,
        )
        self._evaluations[evidence.digest] = (candidate.digest, len(changed))
        return CandidateRecord(candidate, evidence, len(changed))

    # -- selection and promotion ---------------------------------------------------------------------------------
    def select(self, parent: HarnessManifest, candidates) -> SelectionReceipt:
        """The promotion law over evaluations this authority issued; anything else is rejected, never weighed."""
        self.require()
        candidates = tuple(candidates)
        if len(candidates) > self.policy.max_candidates:
            _refuse("EVOLUTION_BUDGET", "more candidates than the policy admits to one selection")
        admitted, refused = [], []
        for record in candidates:
            if type(record) is not CandidateRecord:
                _refuse("EVALUATION_INVALID", "candidate records are required")
            issued = self._evaluations.get(record.evidence.digest)
            if issued != (record.manifest.digest, record.edit_count):
                refused.append((record.manifest.candidate_manifest_digest, "EVALUATION_NOT_INDEPENDENT"))
            elif record.evidence.ood_delta < self.policy.ood_regression_floor:
                refused.append((record.manifest.candidate_manifest_digest, "OOD_REGRESSION_BEYOND_POLICY"))
            else:
                admitted.append(record)
        law = select(parent, admitted, self.policy.evaluation_contract_digest, dict(self.policy.partitions))
        receipt = SelectionReceipt(
            parent_harness_manifest_digest=law.parent_harness_manifest_digest,
            selected_harness_manifest_digest=law.selected_harness_manifest_digest,
            selected_candidate_manifest_digest_or_null=law.selected_candidate_manifest_digest_or_null,
            selected_evaluation_evidence_digest_or_null=law.selected_evaluation_evidence_digest_or_null,
            disposition=law.disposition,
            eligible_candidate_manifest_digests=law.eligible_candidate_manifest_digests,
            rejected=tuple(sorted(law.rejected + tuple(refused))),
        )
        self._selections[receipt.digest] = receipt
        return receipt

    def approve(self, receipt: SelectionReceipt, *, operator_approval_digest: str) -> PromotionGrant:
        """Operator approval of one selection this authority issued (a SELECT_CHALLENGER) as a promotion grant."""
        self.require()
        require_digest(operator_approval_digest)
        if type(receipt) is not SelectionReceipt or self._selections.get(receipt.digest) != receipt:
            _refuse("PROMOTION_UNAUTHORIZED", "the selection was not issued by this policy authority")
        if receipt.disposition != "SELECT_CHALLENGER":
            _refuse("PROMOTION_UNAUTHORIZED", "only a selected challenger can be promoted")
        grant = PromotionGrant(self.policy.digest, receipt.digest, receipt.selected_harness_manifest_digest,
                               operator_approval_digest)
        self._grants[receipt.digest] = grant
        return grant

    def materialize(self, grant: PromotionGrant, *, parent_dir, dest_dir, edits: list[dict],
                    expected_manifest: HarnessManifest) -> dict[str, str]:
        """Materialize exactly the granted child (``atomic_materialize``); anything else writes nothing."""
        self.require()
        if type(grant) is not PromotionGrant or self._grants.get(grant.selection_receipt_digest) != grant:
            _refuse("PROMOTION_UNAUTHORIZED", "no operator-approved grant of this policy authority")
        if type(expected_manifest) is not HarnessManifest or \
                expected_manifest.digest != grant.selected_harness_manifest_digest:
            _refuse("PROMOTION_UNAUTHORIZED", "the manifest is not the granted selection")
        return atomic_materialize(parent_dir, dest_dir, edits, expected_manifest)
