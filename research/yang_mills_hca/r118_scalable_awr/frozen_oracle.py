"""R118-A frozen oracle: the authentic frozen HCA, loaded in-process from verified bytes.

The historical modules import one another by bare name (``import r64``,
``import portable_mechanics as pm`` ...). They are served by a meta-path finder
that answers *only* for those names and executes the exact bytes that were
authenticated against the manifest (no re-read between check and execution,
no bytecode written anywhere). Each module's ``__file__`` points into a
volatile staged copy so that protocol JSON reads resolve there and any
historical write path could only ever touch scratch. No ``run()``/``save``
path is called by this module.

Global side effects of importing the frozen modules are historical and
documented: ``r64``, ``r77`` and ``r78`` call ``torch.set_num_threads(1)`` at
import. ``FrozenOracle.torch_threads_before/after`` record it.

The oracle is a process singleton (``get_oracle()``): the staged copy lives
until interpreter exit and is then removed.
"""
from __future__ import annotations

import atexit
import importlib
import importlib.abc
import importlib.util
import pathlib
import sys
import threading

from . import provenance as P

FROZEN_MODULES = (
    "r30_source",
    "r62_source",
    "r63_source",
    "portable_mechanics",
    "make_proto",
    "r64",
    "r77",
    "r78",
)


class _VerifiedSourceFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve exactly the frozen module names from authenticated in-memory bytes."""

    def __init__(self, directory: pathlib.Path, sources: dict[str, bytes]):
        self._dir = directory
        self._sources = dict(sources)
        self.executed: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self._sources:
            return importlib.util.spec_from_loader(fullname, self, origin=str(self._dir / (fullname + ".py")))
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        path = self._dir / (name + ".py")
        module.__file__ = str(path)
        code = compile(self._sources[name], str(path), "exec", dont_inherit=True)
        self.executed.append(name)
        exec(code, module.__dict__)


class FrozenOracle:
    def __init__(self):
        self.manifest_report = P.verify_manifest()
        clash = [n for n in FROZEN_MODULES if n in sys.modules]
        if clash:
            raise P.ProvenanceError("FROZEN_MODULE_NAME_ALREADY_LOADED:" + ",".join(clash))
        self._scratch_cm = P.volatile_scratch("ym_r118_oracle_")
        self.staged_dir = self._scratch_cm.__enter__()
        atexit.register(self.close)
        staged = P.stage_snapshot(self.staged_dir)
        sources = {}
        for name in FROZEN_MODULES:
            data = (self.staged_dir / (name + ".py")).read_bytes()
            if P.sha256_bytes(data) != staged[name + ".py"]:
                raise P.ProvenanceError("STAGED_SOURCE_DRIFT:" + name)
            sources[name] = data
        self.source_sha256 = {n + ".py": P.sha256_bytes(b) for n, b in sources.items()}
        import torch

        self.torch = torch
        self.torch_threads_before = torch.get_num_threads()
        finder = _VerifiedSourceFinder(self.staged_dir, sources)
        sys.meta_path.insert(0, finder)
        try:
            self.r78 = importlib.import_module("r78")
        finally:
            sys.meta_path.remove(finder)
        self.torch_threads_after = torch.get_num_threads()
        self.executed_modules = tuple(finder.executed)
        if sorted(self.executed_modules) != sorted(FROZEN_MODULES):
            raise P.ProvenanceError("FROZEN_MODULE_SET:" + ",".join(self.executed_modules))
        self.r77 = sys.modules["r77"]
        self.r64 = sys.modules["r64"]
        self.r63 = sys.modules["r63_source"]
        self.r62 = sys.modules["r62_source"]
        self.r30 = sys.modules["r30_source"]
        self.pm = sys.modules["portable_mechanics"]
        for name, mod in (("r64", self.r64), ("r77", self.r77), ("r78", self.r78), ("portable_mechanics", self.pm)):
            if pathlib.Path(mod.__file__).parent != self.staged_dir:
                raise P.ProvenanceError("MODULE_ROOT_NOT_STAGED:" + name)
        self._models = None
        self._r77_report = None
        self.r61 = self.pm.read_json("FROZEN_PROTOCOL_R61.json")

    def close(self):
        cm = getattr(self, "_scratch_cm", None)
        if cm is not None:
            self._scratch_cm = None
            cm.__exit__(None, None, None)

    # -- checkpoints -------------------------------------------------------
    def r77_report(self) -> dict:
        if self._r77_report is None:
            # Historical gate: source pins, parent archive pin, graph/arm/seed drift checks.
            self._r77_report = self.r78.source_gate()
        return self._r77_report

    def models(self) -> dict:
        """The six frozen HCAs via the historical R78 reconstruction path, re-verified."""
        if self._models is None:
            models = self.r78.models_from_r77(self.r77_report())
            if sorted(models) != list(P.CHECKPOINTS):
                raise P.ProvenanceError("RECONSTRUCTED_SET")
            for key, m in models.items():
                n = sum(p.numel() for p in m.parameters())
                if n != P.PARAMETER_COUNT:
                    raise P.ProvenanceError("PARAMETER_COUNT:" + key)
                if any(p.requires_grad for p in m.parameters()):
                    raise P.ProvenanceError("TRAINABLE_AT_INFERENCE:" + key)
                pin = self.r64.parameter_pin(m)
                independent = P.checkpoint_hash(m.gates.weight.tolist(), m.gates.bias.tolist())
                if not (pin == independent == P.CHECKPOINT_SHA256[key]):
                    raise P.ProvenanceError("CHECKPOINT_HASH:" + key)
            self._models = models
        return self._models

    def parameter_pins(self) -> dict:
        return {k: self.r64.parameter_pin(m) for k, m in self.models().items()}

    def weights(self, checkpoint: str):
        """Copies of (W[8,10], b[8]) as float64 numpy arrays for derived backends."""
        m = self.models()[checkpoint]
        return m.gates.weight.detach().numpy().copy(), m.gates.bias.detach().numpy().copy()

    # -- exact historical operator -----------------------------------------
    def step(self, state, adj, model):
        """One historical HCA graph step (``r64.rollout`` uses the same adj for both channels)."""
        return self.r62.graph_step_split(state, adj, adj, model, self.torch)

    def graph_adjs(self, graph):
        """Historical (BASE_SEPARABLE, PARTICIPANT_NONFLAT) link dicts for a 4-relation fixture."""
        return self.pm.graph_adjs(graph, self.r61)


_LOCK = threading.Lock()
_ORACLE: FrozenOracle | None = None


def get_oracle() -> FrozenOracle:
    global _ORACLE
    with _LOCK:
        if _ORACLE is None:
            _ORACLE = FrozenOracle()
        return _ORACLE
