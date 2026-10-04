"""Mission gate: architectural checks over a repository tree (docs/ELPIS_MISSION.md).

Every check takes a repository root and returns a list of violations, so the
same gate runs against the working tree and against forensic trees (the
sidecar commits it must reject). Canonical code is ``src/`` and ``native/``;
``research/`` and ``tests/`` are noncanonical and never scanned here.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

# elpis.inference modules that form the DSV4 communication codec. Everything
# else in elpis.inference is DSV model-execution machinery and noncanonical.
CODEC_MODULES = (
    "elpis.inference.contracts",
    "elpis.inference.text",
    "elpis.inference.admission",
    "elpis.inference.structural",
)

# Names that encode ECS-as-DSV-conditioning (f4e1f05, 75313fb). They describe
# the forbidden graph; no canonical code may define or use them.
SIDECAR_IDENTIFIERS = re.compile(
    r"\b(TurnConditioning|TurnObservation|ConditioningProjection|FEATURE_CONDITIONING|"
    r"WorldModelLoop|DriveMap|conditioning_projection|conditioning_vector|observe_distribution|"
    r"accepts_conditioning|CONFIG_V2)\b")

# DSV-specific tensor roles the generic substrate core must never name.
DSV_ROLES = re.compile(r"(?i)(dsv4|deepseek|engram|\bmoe\b|\bexperts?\b|\bw[123]\b)")

# The qualified ECS_G kernel. Changing these sources requires an explicit,
# reviewed update of the pins together with the kernel qualification.
ECSG_KERNEL_PINS = {
    "native/ECS_G/src/ecsg_math.c": "fbfc48e5e36498df3dc8d33fa44620a61d697fa7bd00b8c7303929c4dc0cf235",
    "native/ECS_G/src/ecsg_state.c": "c9887848fa0ddcfff71952a2010dd9968d980973cbe56cc0fa056c81f7b0b821",
    "native/ECS_G/include/elpis/ecsg_math.h": "94f3c4e7a9bf26b08b267679c9b707e8376cc4e2867be760ac825760ce95f80e",
    "native/ECS_G/include/elpis/ecsg_state.h": "34b173c4873fdac398876cf2eab426d88aca0884a3fa33640a570ae5c03a93c5",
    # Runtime R1 executor: bitwise-qualified against the reference above
    # (tests/ECS_G/test_executor_differential.py) and bound to its measured
    # evidence (research/ecs_runtime_r1/evidence).
    "native/ECS_G/src/ecsg_executor.c": "1d94739c52543da2080ccac2aeb640ef84b650161fa740c01f706b3f73b0fa8f",
    "native/ECS_G/include/elpis/ecsg_executor.h": "67502a5e4cf3a14800760e661dac59dd17bbd736fcbb146e60faada0ee7ebde2",
}

_CODE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp", ".map")


def _canonical_files(root: Path):
    for top in ("src", "native"):
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".egg-info"):
                if any(part.endswith(".egg-info") for part in path.parts):
                    continue
                yield path


def _module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root / "src").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imports_of(root: Path, path: Path) -> set[str]:
    """Absolute names imported by ``path`` (relative imports resolved, nested imports included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _module_name(root, path).split(".")
    package = current if path.name == "__init__.py" else current[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module.split(".") if node.level == 0 else (
                package[: len(package) - (node.level - 1)] + (node.module.split(".") if node.module else []))
            module = ".".join(base)
            names.add(module)
            names.update(module + "." + alias.name for alias in node.names)
    return names


def _python(root: Path, package: str):
    base = root / "src" / "elpis" / package
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts) if base.is_dir() else []


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _strip_comments(text: str, suffix: str) -> str:
    if suffix == ".py":
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


# ----------------------------------------------------------------------------- checks

def sidecar_conditioning(root: Path) -> list[str]:
    """(1) No ECS-as-DSV-conditioning names or modules in canonical code."""
    out = []
    for path in _canonical_files(root):
        if path.suffix not in _CODE_SUFFIXES:
            continue
        match = SIDECAR_IDENTIFIERS.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            out.append(f"{path.relative_to(root)}: {match.group(0)}")
    for name in ("src/elpis/runtime/world_model.py", "src/elpis/inference/conditioning.py"):
        if (root / name).exists():
            out.append(f"{name}: sidecar module present")
    return out


def runtime_model_execution(root: Path) -> list[str]:
    """(2) The canonical runtime reaches elpis.inference only through codec modules."""
    out = []
    for path in _python(root, "runtime"):
        for name in sorted(imports_of(root, path)):
            if not _under(name, "elpis.inference") or name == "elpis.inference":
                continue
            if not any(_under(name, codec) for codec in CODEC_MODULES):
                out.append(f"{path.relative_to(root)}: imports {name}")
    return out


def canonical_tower(root: Path) -> list[str]:
    """(3) No DSV4.1 tower module in the canonical tree."""
    return [str(p.relative_to(root)) for p in _canonical_files(root)
            if re.search(r"(?i)dsv41", str(p.relative_to(root)))]


def ecs_independence(root: Path) -> list[str]:
    """(4) ECS mathematics is independent; only the residency adapter may use generic FMS."""
    out = []
    for path in _python(root, "ECS_G"):
        for name in imports_of(root, path):
            if _under(name, "elpis") and not _under(name, "elpis.ECS_G"):
                out.append(f"{path.relative_to(root)}: imports {name}")
            if _under(name, "numpy") or _under(name, "research"):
                out.append(f"{path.relative_to(root)}: imports {name}")
    base = root / "native" / "ECS_G"
    for path in sorted(base.rglob("*")) if base.is_dir() else []:
        if path.suffix in (".c", ".h"):
            for inc in re.findall(r'#\s*include\s*"([^"]+)"', path.read_text(encoding="utf-8")):
                relative = str(path.relative_to(base))
                allowed = {
                    "include/elpis/ecsg_fms.h": {"elpis/fms.h"},
                    "src/ecsg_fms.c": {"elpis/sha256.h"},
                    "tests/test_ecsg_fms.c": {"elpis/fms_pal_posix.h"},
                    "tests/test_ecsg_fms_alloc.c": {"elpis/fms_pal_posix.h"},
                    "tests/test_ecsg_fms_performance.c": {"elpis/fms_pal_posix.h"},
                }
                if not inc.startswith("elpis/ecsg_") and inc not in allowed.get(relative, set()):
                    out.append(f"{path.relative_to(root)}: includes {inc}")
    return out


def codec_independence(root: Path) -> list[str]:
    """(5) The codec/inference package depends on no ECS, runtime or research module and holds no ECS state."""
    out = []
    for path in _python(root, "inference"):
        for name in imports_of(root, path):
            for lane in ("elpis.ECS_G", "elpis.ECS_C", "elpis.runtime", "research"):
                if _under(name, lane):
                    out.append(f"{path.relative_to(root)}: imports {name}")
        if re.search(r"\bWorldState\b|\becsg_", path.read_text(encoding="utf-8")):
            out.append(f"{path.relative_to(root)}: references ECS_G state")
    return out


def ecs_c_separation(root: Path) -> list[str]:
    """(6) Continuity history (ECS_C) and active geometric state (ECS_G) do not depend on each other."""
    out = []
    for package, other in (("ECS_C", "elpis.ECS_G"), ("ECS_G", "elpis.ECS_C")):
        for path in _python(root, package):
            out += [f"{path.relative_to(root)}: imports {n}" for n in imports_of(root, path) if _under(n, other)]
    return out


def fms_genericity(root: Path) -> list[str]:
    """(7) The generic substrate/FMS core names no DSV-specific tensor role (comments excluded)."""
    out = []
    for base in (root / "src" / "elpis" / "substrate", root / "native" / "substrate"):
        for path in sorted(base.rglob("*")) if base.is_dir() else []:
            if path.suffix in _CODE_SUFFIXES and "__pycache__" not in path.parts:
                code = _strip_comments(path.read_text(encoding="utf-8", errors="replace"), path.suffix)
                match = DSV_ROLES.search(code)
                if match:
                    out.append(f"{path.relative_to(root)}: {match.group(0)}")
                semantic = re.search(r"(?i)\b(ecsg_\w*|elpis_ecsg_\w*|g1|s3|retention)\b", code)
                if semantic:
                    out.append(f"{path.relative_to(root)}: ECS semantics {semantic.group(0)}")
    return out


def ecsg_kernel_unchanged(root: Path) -> list[str]:
    """(10) The qualified ECS_G kernel sources match their pinned digests."""
    out = []
    for name, digest in ECSG_KERNEL_PINS.items():
        path = root / name
        if not path.is_file():
            out.append(f"{name}: missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            out.append(f"{name}: changed")
    return out


STATIC_CHECKS = {
    "sidecar_conditioning": sidecar_conditioning,
    "runtime_model_execution": runtime_model_execution,
    "canonical_tower": canonical_tower,
    "ecs_independence": ecs_independence,
    "codec_independence": codec_independence,
    "ecs_c_separation": ecs_c_separation,
    "fms_genericity": fms_genericity,
    "ecsg_kernel_unchanged": ecsg_kernel_unchanged,
}


def violations(root: Path) -> dict[str, list[str]]:
    return {name: found for name, check in STATIC_CHECKS.items() if (found := check(root))}
