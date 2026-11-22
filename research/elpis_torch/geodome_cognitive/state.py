"""R13 P2: the bounded, volatile working state of a cognitive session. RESEARCH_ONLY.

Identity and lifecycle are Geodome's OWN ``NodeState`` (width 1, unused rows): node ownership (the instance's
owner token), allocation generation per slot, the epoch that ``reset`` advances, and the bounded capacity. A handle
is a Geodome ``StateHandle``; it is valid only in the state that issued it, in its epoch and slot generation
(``NodeState.read`` refuses anything else with ``ValueError``). Nothing of that logic is reimplemented here.

The latent rows ``[rows, width]`` are a PyTorch tensor indexed by the same slots:
* sparse residency: ``rows = capacity`` (K, the bounded resident-node count, at most 4096 = Geodome's enrolment cap);
* dense residency: ``rows = 144002`` (slot = vertex id), for the residency parity gate only.

Writes are transactional: ``publish`` validates every handle, the shape, dtype, device and finiteness of the whole
batch BEFORE replacing the latent tensor (out of place). The state holds VALUES only (published rows are stored
detached): a training graph lives in the result of one bounded computation, never here, so no graph can span two
computations or an external step (``detach`` is kept as the explicit step boundary). ``reset`` invalidates every
handle (Geodome epoch) and zeroes the rows. Nothing is ever written to storage; the state cannot be pickled or
copied.
"""
from __future__ import annotations

import torch

MAX_SPARSE_CAPACITY = 4096          # Geodome's MAX_ENROLLED: the most vertices one scheduler may enrol
MAX_WIDTH = 1024


class WorkingStateError(RuntimeError):
    """A refused working-state operation (closed state, invalid tensor, capacity). Nothing was changed."""


class TensorWorkingState:
    def __init__(self, substrate, *, capacity: int, width: int, dtype=torch.float32, residency: str = "sparse",
                 device="cpu"):
        if residency not in ("sparse", "dense"):
            raise ValueError("residency must be 'sparse' or 'dense'")
        if type(width) is not int or not 1 <= width <= MAX_WIDTH:
            raise ValueError(f"width must be an integer in [1, {MAX_WIDTH}]")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("the latent state is float32 (trainable) or float64 (reference)")
        if residency == "sparse" and (type(capacity) is not int or not 1 <= capacity <= MAX_SPARSE_CAPACITY):
            raise ValueError(f"sparse capacity must be an integer in [1, {MAX_SPARSE_CAPACITY}]")
        self.substrate = substrate
        self.nodes = substrate.geodome.new_state(substrate.index, width=1, precision="f32", mode=residency,
                                                 capacity=None if residency == "dense" else capacity)
        self.residency, self.width, self.dtype = residency, width, dtype
        self.capacity = self.nodes.capacity
        self.latent = torch.zeros(self.capacity, width, dtype=dtype, device=device)
        self.version = 0
        self._closed = False

    # -- identity -------------------------------------------------------------------------------------------
    @property
    def epoch(self) -> int:
        self._live()
        return self.nodes.epoch

    @property
    def resident(self) -> int:
        self._live()
        return self.nodes.resident

    def handle(self, vertex: int):
        """The current handle of a resident vertex (``KeyError('NONRESIDENT')`` otherwise)."""
        self._live()
        return self.nodes.handle(vertex)

    def acquire(self, vertices) -> tuple:
        """Make ``vertices`` resident (zero rows): all or none. Refuses a resident vertex or an overflow first."""
        self._live()
        vertices = tuple(vertices)
        if any(type(v) is not int or not 0 <= v < self.substrate.vertices for v in vertices):
            raise ValueError("VERTEX_RANGE")
        if len(set(vertices)) != len(vertices):
            raise ValueError("DUPLICATE_VERTEX")
        for v in vertices:
            try:
                self.nodes.handle(v)
            except KeyError:
                continue
            raise ValueError("ALREADY_RESIDENT")
        if self.nodes.resident + len(vertices) > self.capacity:
            raise OverflowError("STATE_CAPACITY")
        taken = []
        try:
            for v in vertices:
                taken.append(self.nodes.acquire_vertex(v))
        except Exception:
            for h in taken:
                self.nodes.release(h)
            raise
        self.latent = self.latent.index_fill(0, self._slots(taken), 0.0)
        self.version += 1
        return tuple(taken)

    def validate(self, handles) -> None:
        """Every handle is this state's, of the current epoch and slot generation (Geodome's own check)."""
        self._live()
        for h in handles:
            self.nodes.read(h)

    def slots(self, handles) -> torch.Tensor:
        self.validate(handles)
        return self._slots(handles)

    def _slots(self, handles) -> torch.Tensor:
        return torch.tensor([h.slot for h in handles], dtype=torch.long, device=self.latent.device)

    # -- rows -----------------------------------------------------------------------------------------------
    def gather(self, slots: torch.Tensor) -> torch.Tensor:
        self._live()
        return self.latent.index_select(0, slots)

    def publish(self, handles, slots: torch.Tensor, rows: torch.Tensor) -> None:
        """Replace the rows of ``handles`` atomically: everything is validated before anything changes. The state
        keeps VALUES only (``rows`` are stored detached): a training graph lives in the caller's result of one
        bounded computation and never in the working state, so no graph can span two computations."""
        self.validate(handles)
        if not isinstance(rows, torch.Tensor) or rows.shape != (len(handles), self.width):
            raise WorkingStateError(f"rows must be [{len(handles)}, {self.width}]")
        if rows.dtype != self.dtype or rows.device != self.latent.device:
            raise WorkingStateError("rows have the wrong dtype or device")
        if not bool(torch.isfinite(rows).all()):
            raise WorkingStateError("NONFINITE_ROWS: nothing was published")
        if not torch.equal(slots, self._slots(handles)):
            raise WorkingStateError("slots do not belong to the handles")
        self.latent = self.latent.index_copy(0, slots, rows.detach())
        self.version += 1

    def release(self, handles) -> None:
        """Make the vertices of ``handles`` non-resident and zero their rows: all or none. The whole batch is checked
        before anything changes: it must be non-empty, every handle valid (this state's, of the current epoch and
        slot generation) and every vertex named once. A repeated handle would otherwise be released once and then
        refused as stale, leaving a partial update (R13 owner review, hardening 1)."""
        self._live()
        handles = tuple(handles)
        if not handles:
            raise ValueError("EMPTY_BATCH")
        self.validate(handles)
        if len({h.vertex for h in handles}) != len(handles):
            raise ValueError("DUPLICATE_HANDLE")
        slots = self._slots(handles)
        for h in handles:
            self.nodes.release(h)
        self.latent = self.latent.index_fill(0, slots, 0.0)
        self.version += 1

    def detach(self) -> None:
        """Cut any training graph the rows carry (the external-step boundary). Values are kept."""
        self._live()
        self.latent = self.latent.detach()

    def reset(self) -> None:
        """Every handle becomes stale (Geodome epoch), every row zero, no graph kept."""
        self._live()
        self.nodes.reset()
        self.latent = torch.zeros_like(self.latent.detach())
        self.version += 1

    def close(self) -> None:
        self.nodes, self.latent, self._closed = None, None, True

    def nbytes(self) -> dict:
        self._live()
        payload = self.nodes.payload()
        out = {"latent_rows": self.latent.numel() * self.latent.element_size(),
               "geodome_identity_metadata": payload["logical_metadata_bytes"],
               "geodome_unused_rows": payload["reserved_state_bytes"]}
        out["total"] = sum(out.values())
        return out

    def _live(self) -> None:
        if self._closed:
            raise WorkingStateError("the working state is closed")

    def __getstate__(self):
        raise TypeError("a working state is volatile and is never serialized")

    def __reduce_ex__(self, protocol):
        raise TypeError("a working state is volatile and is never serialized")

    def __deepcopy__(self, memo):
        raise TypeError("a working state is volatile and is never copied")
