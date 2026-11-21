"""R13 P3: the session-owned bridge between ``EvidenceLM`` and the Geodome cognitive substrate. RESEARCH_ONLY.

The seam (the final block, unchanged; nothing of the backbone is modified or recomputed upstream):

    pre        = the final block's pre-channel stream (``EvidenceLM.forward(..., return_pre=True)``; it already holds
                 the R5 evidence read, if any)
    query      = P_query(pre at the final position)
    latent     = reason(query, admitted evidence, execution plan)          (Geodome working state, bounded rounds)
    refined    = pre + g * P_out(latent)                                    (final position only)
    logits     = head(final_norm(refined + channel(channel_norm(refined))))

``channel`` is the final block's ACTUAL channel, sparse routing included, evaluated on the same T rows of the chunk
as the forward evaluated it (the full pre-channel rows are captured from that very forward), so with ``g = 0`` the
refined logits equal the backbone's bit for bit (the routed channel's batch variance is not provoked).

``CognitiveSession`` (batch 1, final-position query, frozen backbone):

    observe(tokens)          consume one chunk ONCE through EvidenceLM (state committed by the model itself);
                             keep its final-position logits (the safe bypass) and pre-channel rows
    enrol(vertices)          make Geodome vertices resident in the bounded working state (all or none)
    admit_evidence(addrs)    validate exact historical ``BlockAddress`` identities against the ring (session, epoch,
                             block, slot generation, canonical form, liveness at the query) and causality (strictly
                             before the query's own block), then reread and encode the ORIGINAL tokens through the
                             existing exact-reread and causal-encoder path
    reason(plan)             bounded internal computation; no LM token step; publishes all enrolled rows or nothing
    readout()                the latent and the refined final-position logits
    prediction()             the refined logits if a readout exists for the current observation, else the
                             backbone's (a refused cognitive operation never rewinds or repeats the observation)
    reset(), close()         invalidate every session-owned state and handle

Invariants. Nothing but ``observe`` advances the external clock: ``admit_evidence``, ``reason`` and ``readout`` change
no ERSA cell or tail, window cache, PLE history, ring token, index key or id, fine key, or position (they may move
the ring's accounted reread counters). Every operation is anchored to the session identity, the backbone revision
(``RevisionWatch``) and the current query position. No graph crosses an external step (``observe`` detaches the
working state, which keeps values only, so a training graph spans one ``reason`` call and its ``readout``);
inference runs under ``no_grad``. Nothing is written to storage; nothing is serialized.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from ..context_runtime import BudgetError, ModelRevisionError, RevisionWatch, SessionClosedError
from ..evidence_ring import BlockAddress, StaleAddressError
from ..lastblock import capture_pre
from .activation import Execution, execute
from .kernels import CognitiveKernels
from .plan import ExecutionPlan, PlanError, compile_plan
from .state import TensorWorkingState
from .topology import substrate as frozen_substrate

_SESSIONS = itertools.count(1)


class CognitiveError(RuntimeError):
    """A refused cognitive operation. ``code`` names the rule; nothing was published."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class Observation:
    serial: int                         # observations of this session, epoch-scoped
    position: int                       # absolute position of the final (query) token
    logits: torch.Tensor                # [1, vocab]: the backbone's prediction there (the bypass)
    pre: torch.Tensor                   # [1, d]: the final block's pre-channel stream there (return_pre)
    rows: torch.Tensor                  # [1, T, d]: every row the final channel consumed in that forward


@dataclass(frozen=True)
class AdmittedEvidence:
    serial: int                         # the observation it was admitted for
    addresses: tuple                    # BlockAddress, admission order
    encoded: torch.Tensor               # [n, s + 1, encoder width]: the exact reread, re-encoded


@dataclass(frozen=True)
class Reasoning:
    serial: int
    latent: torch.Tensor                # [D]
    execution: Execution


@dataclass(frozen=True)
class Readout:
    latent: torch.Tensor                # [D]
    logits: torch.Tensor                # [1, vocab]: refined final-position logits


class CognitiveSession:
    """Batch-1 cognitive session over a FROZEN ``EvidenceLM`` (see the module docstring)."""

    def __init__(self, model, kernels: CognitiveKernels, *, origin: int = 0, evidence_options: dict | None = None,
                 budget_bytes: int | None = None, residency: str | None = None):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("a cognitive session needs a frozen backbone (eval mode, no parameter requires grad)")
        if not hasattr(model, "evidence") or not hasattr(model, "lookup"):
            raise TypeError("the cognitive session binds the R5 EvidenceLM")
        dtype = model.embed.weight.dtype
        if kernels.gate.dtype != dtype or kernels.query_in.in_features != model.config.d_model:
            raise ValueError("the kernels must match the backbone's width and dtype (the seam is exact)")
        if kernels.evidence_query.shape[1] != model.encoder.width:
            raise ValueError("the evidence ingress must match the backbone's block-encoder width")
        self.model, self.kernels, self.config = model, kernels, kernels.config
        self.substrate = frozen_substrate()
        self.origin = origin
        self.evidence_options = dict(evidence_options or {})
        if set(self.evidence_options) - {"mode", "rerank", "coarse_top"}:
            raise ValueError("evidence_options: mode, rerank or coarse_top only")
        self._residency = residency or self.config.residency
        self.session_id = next(_SESSIONS)
        self.epoch = 0
        self._closed = False
        self.state = self.working = None
        self._allocate()
        if budget_bytes is not None and self.projected_bytes()["total"] > budget_bytes:
            self.close()
            raise BudgetError("projected cognitive-session memory exceeds the budget")

    # -- lifetime ---------------------------------------------------------------------------------------------
    def _allocate(self) -> None:
        self.state = self.model.new_state(1, origin=self.origin)
        self.working = TensorWorkingState(self.substrate, capacity=self.config.capacity, width=self.config.width,
                                          dtype=self.kernels.gate.dtype, residency=self._residency)
        self._revision = RevisionWatch(self.model)
        self._refused = False
        self._observation = self._evidence = self._reasoning = self._readout = None
        self._serial = 0

    def reset(self) -> None:
        """A clean session in the same identity: backbone state, ring (new epoch), working state (new Geodome
        epoch), observation, evidence and results all dropped; every earlier handle and address refused."""
        self._live()
        self.state.reset()
        self.working.reset()
        self.epoch += 1
        self._revision = RevisionWatch(self.model)
        self._refused = False
        self._observation = self._evidence = self._reasoning = self._readout = None

    def close(self) -> None:
        if self.working is not None:
            self.working.close()
        self.state = self.working = None
        self._observation = self._evidence = self._reasoning = self._readout = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        raise TypeError("a cognitive session is volatile and is never serialized")

    def __reduce_ex__(self, protocol):
        raise TypeError("a cognitive session is volatile and is never serialized")

    def __deepcopy__(self, memo):
        raise TypeError("a cognitive session is volatile and is never copied")

    def _live(self) -> None:
        if self._closed or self.state is None:
            raise SessionClosedError("the cognitive session is closed")

    def _check(self) -> None:
        self._live()
        if self._refused or not self._revision.unchanged(self.model):
            self._refused = True
            raise ModelRevisionError("the backbone changed under this cognitive session; reset() to continue")

    def _current(self) -> Observation:
        if self._observation is None:
            raise CognitiveError("NO_OBSERVATION", "observe a chunk first: reasoning is anchored to a query")
        return self._observation

    # -- the external clock -------------------------------------------------------------------------------------
    @torch.no_grad()
    def observe(self, tokens: torch.Tensor) -> torch.Tensor:
        """Consume ``tokens [1, T]`` once through the backbone; returns its final-position logits ``[1, vocab]``.
        A failure inside the backbone leaves its state unchanged (the model commits only after success)."""
        self._check()
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be an int64 tensor [1, T] (batch one)")
        length = tokens.shape[1]
        if not 1 <= length <= self.model.ersa.chunk:
            raise ValueError(f"1 <= T <= {self.model.ersa.chunk} tokens per observation")
        last = torch.tensor([length - 1], device=tokens.device)
        with capture_pre(self.model) as captured:
            out = self.model(tokens, self.state, return_pre=True, logit_positions=last, **self.evidence_options)
        logits, pre = out[0], out[1]
        rows = captured.pre
        if rows is None or rows.shape[1] != length or not torch.equal(rows[:, -1:], pre):
            raise CognitiveError("SEAM", "the captured pre-channel rows disagree with return_pre")
        self._serial += 1
        self._observation = Observation(self._serial, self.state.position - 1, logits[:, 0], pre[:, 0], rows)
        self._evidence = self._reasoning = self._readout = None
        self.working.detach()                      # no training graph crosses an external step
        return logits[:, 0]

    # -- working state --------------------------------------------------------------------------------------------
    def enrol(self, vertices) -> tuple:
        """Make Geodome ``vertices`` resident (zero rows); all or none. Returns their handles."""
        self._check()
        return self.working.acquire(vertices)

    # -- evidence ------------------------------------------------------------------------------------------------
    def admit_evidence(self, addresses) -> int:
        """Admit exact historical blocks for the current query; all or none. Returns the count admitted."""
        self._check()
        observation = self._current()
        addresses = tuple(addresses)
        if not 1 <= len(addresses) <= self.config.max_evidence:
            raise CognitiveError("EVIDENCE_COUNT", f"1 to {self.config.max_evidence} addresses")
        ring, block = self.state.ring, self.model.evidence.block
        query_block = observation.position // block
        seen = set()
        for address in addresses:
            if type(address) is not BlockAddress:
                raise CognitiveError("EVIDENCE_TYPE", "an EvidenceRing BlockAddress is required")
            if address.block in seen:
                raise CognitiveError("EVIDENCE_DUPLICATE")
            seen.add(address.block)
            if address.block >= query_block:
                raise CognitiveError("EVIDENCE_CAUSALITY", "only blocks strictly before the query's own block")
            if ring.address(address.block) != address:          # canonical form; StaleAddressError if not live
                raise StaleAddressError("not the ring's canonical address of that block")
            ring.check(address, at=observation.position)
        blocks = torch.tensor([a.block for a in addresses], dtype=torch.long, device=ring.device)
        with torch.no_grad():
            encoded = self.model._reread_encoded(self.state, blocks, self.model.embed.weight.device)[0]
        if not bool(torch.isfinite(encoded).all()):
            raise CognitiveError("EVIDENCE_NONFINITE")
        self._evidence = AdmittedEvidence(observation.serial, addresses, encoded)
        self._reasoning = self._readout = None
        return len(addresses)

    # -- internal computation -------------------------------------------------------------------------------------
    def reason(self, plan: ExecutionPlan) -> Execution:
        """Run ``plan`` on the working state; publish every enrolled row, or nothing (``CognitiveError``).

        One call is one bounded differentiable horizon: the rows start from the working state's VALUES (it holds no
        graph), so under ``enable_grad`` the graph spans this call's ingress, rounds and readout only (plus the
        ``readout`` injection), never an earlier call or an earlier observation."""
        self._check()
        observation = self._current()
        evidence = self._evidence if self._evidence is not None and self._evidence.serial == observation.serial \
            else None
        available = 0 if evidence is None else len(evidence.addresses)
        c = self.config
        compiled = compile_plan(plan, self.working, max_rounds=c.max_rounds, max_arcs=c.max_arcs,
                                max_visits=c.max_visits, max_evidence=c.max_evidence, evidence_available=available)
        result, latent = self.compute(compiled, self.working.gather(compiled.slots), observation, evidence)
        if not bool(torch.isfinite(result.rows).all()) or not bool(torch.isfinite(latent).all()):
            raise CognitiveError("NONFINITE", "the computation produced a non-finite value; nothing was published")
        if self.working.epoch != compiled.epoch:
            raise CognitiveError("STALE_PLAN", "the working state was reset during the computation")
        self.working.publish(compiled.handles, compiled.slots, result.rows)
        self._reasoning = Reasoning(observation.serial, latent, result)
        self._readout = None
        return result

    def readout(self) -> Readout:
        """The latent of the current observation's last reasoning and the refined final-position logits."""
        self._check()
        observation = self._current()
        reasoning = self._reasoning
        if reasoning is None or reasoning.serial != observation.serial:
            raise CognitiveError("NO_REASONING", "reason() for the current observation first")
        logits = self.refine(reasoning.latent, observation)
        if not bool(torch.isfinite(logits).all()):
            raise CognitiveError("NONFINITE", "the refined logits are not finite")
        self._readout = Readout(reasoning.latent, logits)
        return self._readout

    # -- the pure computation (no validation, no publication; ``reason`` and ``readout`` wrap these) ---------------
    def compute(self, compiled, rows: torch.Tensor, observation: Observation, evidence) -> tuple:
        """Ingress, rounds and readout on ``rows [N, D]`` (in ``compiled.slots`` order): ``(Execution, latent [D])``.
        Writes nothing; every step is a tensor operation (no host extraction on this path)."""
        kernels = self.kernels
        if compiled.query.numel():
            query = kernels.query(observation.pre)                        # [1, D]
            rows = rows.index_copy(0, compiled.query, query.expand(compiled.query.numel(), -1))
        if compiled.evidence.numel():
            items = kernels.evidence(evidence.encoded[:compiled.evidence.numel()])
            rows = rows.index_copy(0, compiled.evidence, items)
        result = execute(rows, compiled, kernels)
        return result, kernels.readout(result.rows.index_select(0, compiled.readout))

    def refine(self, latent: torch.Tensor, observation: Observation) -> torch.Tensor:
        """The refined final-position logits ``[1, vocab]``: ``pre + g P_out(latent)`` at the final position, then
        the backbone's own tail over the observation's T rows. Writes nothing."""
        delta = self.kernels.injection(latent)                              # [d]
        rows = observation.rows
        return self.final_logits(torch.cat((rows[:, :-1], rows[:, -1:] + delta), dim=1))

    def final_logits(self, rows: torch.Tensor) -> torch.Tensor:
        """The backbone's own tail on pre-channel ``rows [1, T, d]``: the final block's actual channel (sparse
        routing included) over every row, then the final norm and head at the last position."""
        model = self.model
        block = model.blocks[-1]
        x = rows + block.channel(block.channel_norm(rows))
        x = model.final_norm(x[:, -1:])
        out = torch.nn.functional.linear(x, model.embed.weight) if model.head is None else model.head(x)
        return out[:, 0]

    def prediction(self) -> tuple:
        """``(logits [1, vocab], source)``: the refined logits of the current observation's last COMMITTED readout
        (``"cognitive"``), else the backbone's cached logits (``"backbone"``, the safe bypass). A refused operation
        publishes nothing, so it leaves this answer as it was; it never rewinds or repeats the observation."""
        self._check()
        observation = self._current()
        if self._readout is not None and self._reasoning is not None and self._reasoning.serial == observation.serial:
            return self._readout.logits, "cognitive"
        return observation.logits, "backbone"

    # -- accounting -------------------------------------------------------------------------------------------------
    def projected_bytes(self) -> dict:
        """EXACT: the backbone state (its interface's fixed bytes), the working state, the kernel parameters; the
        frozen index is shared by the process (counted once, not per session)."""
        backbone = self.model.context_interface().fixed_bytes(1)["total"]
        working = self.working.nbytes()["total"]
        kernels = sum(p.numel() * p.element_size() for p in self.kernels.parameters())
        return {"backbone_state": backbone, "working_state": working, "kernel_parameters": kernels,
                "total": backbone + working + kernels,
                "shared_index_payload": self.substrate.index.payload_bytes()}


__all__ = ("CognitiveSession", "CognitiveError", "ExecutionPlan", "PlanError", "Observation", "Readout")
