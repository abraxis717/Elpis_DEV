"""R13 P1: the Geodome Reference API v1.0.0, admitted as a FROZEN dependency. RESEARCH_ONLY.

Source authority. HexaFlake (https://github.com/abraxis717/HexaFlake, Apache-2.0) at commit
``4e5fe56670c61a0d8906770b03033658c3087c3a``, directory ``Geodome/``. The seven modules the Geodome core imports
(``Hexaflag.py``, ``GEO144K/g1c_compact.py``, ``GEO144K/g2b_bridge.py``, ``geodome.py``, ``geodome_index.py``,
``geodome_state.py``, ``geodome_scheduler.py``) are vendored BYTE FOR BYTE in ``research/elpis_torch/upstream/
geodome_v1/``, with the upstream ``LICENSE``, ``NOTICE`` and ``PUBLIC.sha256``. Nothing there is ever edited.

Import mechanism (``load_geodome``):
1. every vendored file's SHA-256 is checked against the pin below AND against its line in the vendored upstream
   manifest ``PUBLIC.sha256`` (itself pinned by digest);
2. the Geodome module names must not already be loaded from anywhere else (no shadowing);
3. the vendored directory is put on ``sys.path`` only for the import, then removed;
4. every Geodome module must resolve inside the vendored directory (as Geodome's own C9 consumer requires);
5. the API constants (version, source pin, mapping and edge digests) must equal the pins.

``substrate()`` builds the compact frequency-120 identity index ONCE per process (``geodome.build_index``, which
itself rejects any identity drift) and caches it. It is never rebuilt per token or per reasoning round.

Geodome v1.0.0 is qualified as a deterministic computational substrate, not as a learned cognitive architecture.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

GEODOME_REPOSITORY = "https://github.com/abraxis717/HexaFlake"
GEODOME_COMMIT = "4e5fe56670c61a0d8906770b03033658c3087c3a"
GEODOME_API_VERSION = "1.0.0"
GEODOME_SOURCE_PIN = "993429cc94e9e3dcfd42df40502fa5b57175d883"      # the frozen upstream commit geodome.py names
GEODOME_LICENSE = "Apache-2.0"
VENDORED = Path(__file__).resolve().parents[1] / "upstream" / "geodome_v1"
MANIFEST = "PUBLIC.sha256"
MANIFEST_SHA256 = "e44d288fec19a09c17be0ce503f011d400b499e4d393f1275d305e2b23d31cb8"

# vendored path -> (path in the upstream manifest, SHA-256)
FILES = {
    "Hexaflag.py": ("Hexaflag.py", "190ddd3128d646135dcdc3553968edf798a30d43835e5d51e281d068249b2648"),
    "GEO144K/g1c_compact.py": ("GEO144K/g1c_compact.py",
                               "4961c9f346dc4061aa2a7db61ea7a59fe700636e86f4490ff33fce2e162a2087"),
    "GEO144K/g2b_bridge.py": ("GEO144K/g2b_bridge.py",
                              "18d226749b3c74b8b4d62f961c027414c3d1783b7c2101a8010c1b239a870d39"),
    "geodome.py": ("Geodome/geodome.py", "a9e6a6fe0de6ee29500ff90a2c4b767ab6c63a9fac98255e5031d98a0adc3eca"),
    "geodome_index.py": ("Geodome/geodome_index.py",
                         "9b8a3061da0d0a310c5b683988c9063e19ecbc2cbd1434000c2a5ccd3eedd26d"),
    "geodome_state.py": ("Geodome/geodome_state.py",
                         "83f3b2d1414f62d4afb0301e7cdd80464c1e8703acc6b6968304a7b61509c7be"),
    "geodome_scheduler.py": ("Geodome/geodome_scheduler.py",
                             "556566221ece714fdca027471c503e81cc68335a41a6f6195d533a3b29d54d9c"),
    "LICENSE": ("LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "NOTICE": ("NOTICE", "7349216806d083f687ccb94fdba9738f20663e2fb5248084fc3e488178235ae3"),
}
MODULES = ("geodome", "geodome_index", "geodome_state", "geodome_scheduler", "Hexaflag", "GEO144K",
           "GEO144K.g1c_compact", "GEO144K.g2b_bridge")

# The identity invariants of the qualified frequency-120 substrate.
IDENTITY = {"vertices": 144002, "edges": 432000, "arcs": 864000, "capacity": 823543, "depth": 7,
            "mapping_sha256": "abe96202b554608bd8f5e702b70459daaaf4ab33eb7dfd058723fd4d1724f51b",
            "edge_sha256": "712a8fc3d94685aeb614456ba8c88f6df7e6c6ea8a4c19e86d4d3865ad014231",
            "index_payload_bytes": 11502238}


class GeodomeSourceError(RuntimeError):
    """The vendored Geodome sources differ from the pin, or would not be imported from the vendored tree."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root: Path = VENDORED) -> dict:
    """Every vendored file equals its pin and its line in the vendored upstream manifest. Returns the digests."""
    manifest_path = root / MANIFEST
    if not manifest_path.is_file() or _sha(manifest_path) != MANIFEST_SHA256:
        raise GeodomeSourceError("the vendored upstream manifest PUBLIC.sha256 is missing or differs from its pin")
    manifest = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    out = {MANIFEST: MANIFEST_SHA256}
    for local, (upstream, pinned) in FILES.items():
        path = root / local
        if not path.is_file() or path.is_symlink():
            raise GeodomeSourceError(f"vendored file missing: {local}")
        digest = _sha(path)
        if digest != pinned or manifest.get(upstream) != pinned:
            raise GeodomeSourceError(f"vendored file differs from the pin: {local}")
        out[local] = digest
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    unexpected = [name for name in extra if name not in FILES and name != MANIFEST]
    if unexpected:
        raise GeodomeSourceError(f"unpinned files in the vendored tree: {unexpected}")
    return out


_LOADED: dict = {}


def load_geodome():
    """The ``geodome`` facade module, imported once from the verified vendored tree (see the module docstring)."""
    if "module" in _LOADED:
        return _LOADED["module"]
    verify_sources()
    root = str(VENDORED)
    for name in MODULES:
        found = sys.modules.get(name)
        if found is not None and not _inside(found):
            raise GeodomeSourceError(f"a module named {name!r} is already loaded from elsewhere: refused")
    sys.path.insert(0, root)
    bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # never write caches into the vendored tree
    try:
        import geodome                                            # noqa: PLC0415 (isolated, verified import)
    finally:
        sys.path.remove(root)
        sys.dont_write_bytecode = bytecode
    for name in MODULES:
        if name in sys.modules and not _inside(sys.modules[name]):
            raise GeodomeSourceError(f"{name} did not resolve inside the vendored tree")
    if (geodome.API_VERSION, geodome.SOURCE_PIN, geodome.MAPPING_SHA256, geodome.EDGE_SHA256) != (
            GEODOME_API_VERSION, GEODOME_SOURCE_PIN, IDENTITY["mapping_sha256"], IDENTITY["edge_sha256"]):
        raise GeodomeSourceError("the Geodome API constants differ from the pin")
    _LOADED["module"] = geodome
    return geodome


def _inside(module) -> bool:
    files = [getattr(module, "__file__", None)] + list(getattr(module, "__path__", []) or [])
    files = [f for f in files if f]
    return bool(files) and all(Path(f).resolve().is_relative_to(VENDORED) for f in files)


def source_record() -> dict:
    """The dependency identity for records: repository, commit, license, pins and the verified digests."""
    return {"repository": GEODOME_REPOSITORY, "commit": GEODOME_COMMIT, "api_version": GEODOME_API_VERSION,
            "source_pin": GEODOME_SOURCE_PIN, "license": GEODOME_LICENSE,
            "vendored": "research/elpis_torch/upstream/geodome_v1", "files": verify_sources(),
            "identity": dict(IDENTITY)}
