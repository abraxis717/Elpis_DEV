"""R118-A provenance: authenticate the frozen YM/AWR host snapshot.

Standard library only. RESEARCH_ONLY. Nothing here executes historical code.

Authority chain (all pinned below, all read-only):

* ``frozen_host/INTAKE_MANIFEST.json`` is pinned by its own SHA-256, as published
  at commit 11cbed44cbef92054da9d944ce3472fc3294a60f.
* Every preserved file is checked against the manifest (SHA-256 and byte size),
  the on-disk file set must equal the manifest's file set, and the historical
  ``SOURCE_SHA256_ALL.txt`` ledger must agree with the manifest.
* The six R77 checkpoints are authenticated from the sealed R77 result archive:
  ``sha256(struct.pack('<88d', weights_row_major + bias))`` must equal the
  archived ``checkpoint_sha256`` and the pins below.

Execution of historical code happens only from a *volatile staged copy* of the
verified bytes (``stage_snapshot``), mirroring the historical ``run.sh``
protocol (extraction to ``/dev/shm``). This matters: ``r77.run()`` and
``r78.run()`` write their result archive into their own directory, which would
overwrite the frozen evidence if executed in place.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import platform
import shutil
import struct
import sys
import tempfile
import zipfile

COMPONENT = pathlib.Path(__file__).resolve().parent
YM_ROOT = COMPONENT.parent
FROZEN_HOST = YM_ROOT / "frozen_host"
MANIFEST_PATH = FROZEN_HOST / "INTAKE_MANIFEST.json"
SNAPSHOT_DIR = FROZEN_HOST / "source" / "frozen_snapshot"
SNAPSHOT_PREFIX = "source/frozen_snapshot/"

PUBLICATION_COMMIT = "11cbed44cbef92054da9d944ce3472fc3294a60f"
MANIFEST_SHA256 = "b0415964e83c2dba01d2a1b6bd9f2f55dad53f147a3747119433a3e07524d855"
MANIFEST_FILE_COUNT = 39
R77_ARCHIVE = "YM_AWR_R77_RESULT.zip"
R78_ARCHIVE = "YM_AWR_R78_RESULT.zip"
R77_ARCHIVE_SHA256 = "93fd66f3ef5589ff7af40f0fcdba6c790661429a0bf2a7b857df5f0caf599f57"
R78_ARCHIVE_SHA256 = "7646e113fa8dcef48ff2b0ae724504ddc745d68fe7a8fd6699cb73ef662ab40e"
R77_RESULT_JSON_SHA256 = "ad995bd790fba42afaf31d288c3923d2c0c6f8528085140ab33130acacc4c1ed"

# Frozen checkpoint identities (R77 training records; re-derived by R78).
CHECKPOINT_SHA256 = {
    "54007/STATE8": "3212d1d8a512d010dafc464e7b8fb90adac55f631bbf7d61a9b7baef317a5c0a",
    "54007/STATE8_SIGNED8": "25e555914e000f2c521a71b5d062c716969428a288681fc42226013b32d87d70",
    "54007/STATE8_TF1": "1c764544ac67a16234dfa62383b678b05745af87ef89b5f570ad48a7458cdc7e",
    "54031/STATE8": "4c37c40b9d7656021db39d38ed7caca43877940c395a686fbba0c6c91a636a03",
    "54031/STATE8_SIGNED8": "711d47c25c96fd20229b65f7e54508e34a701fa2e0500b21bb59a65a4367995f",
    "54031/STATE8_TF1": "abb9906d0259f4854f1f44942f546aa27e250be070aa42574084772591e640f3",
}
CHECKPOINTS = tuple(sorted(CHECKPOINT_SHA256))
PARAMETER_COUNT = 88
RESULT_JSON_LIMIT = 8_000_000


class ProvenanceError(RuntimeError):
    """A provenance contract was violated. Never caught to manufacture success."""


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise ProvenanceError(why)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_constant(token: str):
    raise ProvenanceError("NONFINITE_JSON_CONSTANT:" + token)


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_default(obj):
    """numpy scalars -> Python scalars for evidence files; anything else is an error."""
    item = getattr(obj, "item", None)
    if item is not None and getattr(obj, "shape", None) == ():
        return item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def load_manifest() -> dict:
    raw = MANIFEST_PATH.read_bytes()
    _need(sha256_bytes(raw) == MANIFEST_SHA256, "MANIFEST_PIN")
    manifest = json.loads(raw, parse_constant=_reject_constant)
    _need(manifest.get("schema") == "elpis.ym_hca.host_source_intake.v1", "MANIFEST_SCHEMA")
    _need(len(manifest["files"]) == MANIFEST_FILE_COUNT, "MANIFEST_COUNT")
    return manifest


def verify_manifest() -> dict:
    """Authenticate every preserved file; return a compact report."""
    manifest = load_manifest()
    files = manifest["files"]
    for rel, rec in sorted(files.items()):
        path = FROZEN_HOST / rel
        _need(path.is_file() and not path.is_symlink(), "MISSING_OR_LINK:" + rel)
        _need(path.stat().st_size == rec["bytes"], "SIZE:" + rel)
        _need(sha256_file(path) == rec["sha256"], "SHA256:" + rel)
    on_disk = {
        str(p.relative_to(FROZEN_HOST)).replace(os.sep, "/")
        for p in (FROZEN_HOST / "source").rglob("*")
        if p.is_file() or p.is_symlink()
    }
    extra = sorted(on_disk - set(files))
    _need(not extra, "UNMANIFESTED_FILES:" + ",".join(extra))
    # The historical ledger must agree with the manifest entry for every file it names.
    ledger = {}
    for line in (SNAPSHOT_DIR / "SOURCE_SHA256_ALL.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        _need(name not in ledger and len(digest) == 64, "LEDGER_SCHEMA")
        ledger[name] = digest
    for name, digest in ledger.items():
        rec = files.get(SNAPSHOT_PREFIX + name)
        _need(rec is not None and rec["sha256"] == digest, "LEDGER_DISAGREES:" + name)
    _need(files[SNAPSHOT_PREFIX + R77_ARCHIVE]["sha256"] == R77_ARCHIVE_SHA256, "R77_ARCHIVE_PIN")
    _need(files[SNAPSHOT_PREFIX + R78_ARCHIVE]["sha256"] == R78_ARCHIVE_SHA256, "R78_ARCHIVE_PIN")
    return {
        "manifest_sha256": MANIFEST_SHA256,
        "files_verified": len(files),
        "ledger_entries_verified": len(ledger),
        "unmanifested_files": 0,
    }


def snapshot_digest() -> str:
    """One digest over (name, sha256) of every manifest entry, for result headers."""
    manifest = load_manifest()
    lines = "".join(f"{rec['sha256']}  {rel}\n" for rel, rec in sorted(manifest["files"].items()))
    return sha256_bytes(lines.encode())


def read_sealed_result(name: str, limit: int = RESULT_JSON_LIMIT) -> tuple[dict, str]:
    """Bounded, CRC-checked, canonical, NaN-rejecting read of a sealed result archive."""
    path = SNAPSHOT_DIR / name
    expected = {R77_ARCHIVE: R77_ARCHIVE_SHA256, R78_ARCHIVE: R78_ARCHIVE_SHA256}[name]
    data = path.read_bytes()
    _need(sha256_bytes(data) == expected, "ARCHIVE_PIN:" + name)
    import io

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        _need([i.filename for i in infos] == ["result.json"], "ARCHIVE_MEMBERS:" + name)
        _need(0 < infos[0].file_size <= limit, "ARCHIVE_MEMBER_SIZE:" + name)
        _need(z.testzip() is None, "ARCHIVE_CRC:" + name)
        raw = z.read("result.json")
    _need(len(raw) == infos[0].file_size, "ARCHIVE_LENGTH:" + name)
    obj = json.loads(raw, parse_constant=_reject_constant)
    _need(canonical_json(obj) == raw, "ARCHIVE_NOT_CANONICAL:" + name)
    return obj, sha256_bytes(raw)


def checkpoint_hash(weights: list, bias: list) -> str:
    """The R77/R78 parameter pin: row-major (8x10) weights then 8 biases, little-endian float64."""
    values = [float(v) for row in weights for v in row] + [float(v) for v in bias]
    _need(len(values) == PARAMETER_COUNT, "PARAMETER_COUNT")
    return sha256_bytes(struct.pack("<88d", *values))


def checkpoint_records() -> dict:
    """Authenticated weight records of the six frozen checkpoints (stdlib only)."""
    r77, json_sha = read_sealed_result(R77_ARCHIVE)
    _need(json_sha == R77_RESULT_JSON_SHA256, "R77_RESULT_JSON_PIN")
    training = r77["training"]
    _need(sorted(training) == list(CHECKPOINTS), "CHECKPOINT_SET")
    out = {}
    for key in CHECKPOINTS:
        rec = training[key]
        _need(len(rec["weights"]) == 8 and all(len(row) == 10 for row in rec["weights"]), "WEIGHT_SHAPE:" + key)
        _need(len(rec["bias"]) == 8, "BIAS_SHAPE:" + key)
        digest = checkpoint_hash(rec["weights"], rec["bias"])
        _need(digest == rec["checkpoint_sha256"], "CHECKPOINT_RECORD_HASH:" + key)
        _need(digest == CHECKPOINT_SHA256[key], "CHECKPOINT_PIN:" + key)
        out[key] = {"weights": rec["weights"], "bias": rec["bias"], "sha256": digest}
    return out


@contextlib.contextmanager
def volatile_scratch(prefix: str = "ym_r118_"):
    """A private scratch directory on a RAM-backed filesystem, removed on exit."""
    base = "/dev/shm" if os.path.isdir("/dev/shm") else None
    _need(base is not None, "NO_VOLATILE_MEMORY_FILESYSTEM")
    path = tempfile.mkdtemp(prefix=prefix, dir=base)
    try:
        yield pathlib.Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=False)


def stage_snapshot(dest: pathlib.Path) -> dict:
    """Copy every verified snapshot file into ``dest``; return {name: sha256}.

    Bytes are hashed from the same buffer that is written, then the staged copy
    is re-read and re-hashed, so the staged tree is exactly the manifest tree.
    """
    manifest = load_manifest()
    staged = {}
    for rel, rec in sorted(manifest["files"].items()):
        if not rel.startswith(SNAPSHOT_PREFIX):
            continue
        name = rel[len(SNAPSHOT_PREFIX):]
        _need("/" not in name and not name.startswith("."), "UNSAFE_NAME:" + name)
        data = (SNAPSHOT_DIR / name).read_bytes()
        _need(len(data) == rec["bytes"] and sha256_bytes(data) == rec["sha256"], "STAGE_SOURCE:" + name)
        target = dest / name
        with open(target, "xb") as f:
            f.write(data)
        _need(sha256_file(target) == rec["sha256"], "STAGE_COPY:" + name)
        staged[name] = rec["sha256"]
    return staged


def environment_fingerprint() -> dict:
    """Interpreter, library and hardware identity recorded with every result."""
    info = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "cpu_model": None,
        "cpu_flags_avx512f": None,
    }
    with contextlib.suppress(OSError):
        text = pathlib.Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name") and info["cpu_model"] is None:
                info["cpu_model"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and info["cpu_flags_avx512f"] is None:
                info["cpu_flags_avx512f"] = " avx512f " in (" " + line.split(":", 1)[1] + " ")
    mods = sys.modules
    if "numpy" in mods:
        info["numpy"] = mods["numpy"].__version__
    if "torch" in mods:
        torch = mods["torch"]
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    return info
