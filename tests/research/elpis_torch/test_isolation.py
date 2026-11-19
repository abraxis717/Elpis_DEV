"""The prototype stays where it belongs: research only, PyTorch optional, canonical authority untouched.

These run with or without PyTorch (they import nothing from the prototype that needs it).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib

from .conftest import REPO

LAB = REPO / "research" / "elpis_torch"
CANONICAL = (REPO / "src", REPO / "native")
SOURCE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp", ".rs", ".toml", ".cmake", ".txt", ".json")
PROPOSAL = REPO / "DEV" / "proposals" / "CGC_NATIVE_GENERATIVE_SUBSTRATE_R0.md"


def test_canonical_code_does_not_know_the_prototype_exists():
    offenders = []
    for root in CANONICAL:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                if "elpis_torch" in text or re.search(r"(?m)^\s*(import|from)\s+torch\b", text):
                    offenders.append(str(path.relative_to(REPO)))
    assert not offenders


def test_pytorch_is_not_a_dependency_of_the_distribution():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project["dependencies"]) + [d for extra in project["optional-dependencies"].values() for d in extra]
    assert not [d for d in declared if d.lower().startswith("torch")]
    assert project["dependencies"] == []
    policy = json.loads((REPO / "ELPIS_SYSTEM.json").read_text(encoding="utf-8"))["boundary_policy"]
    assert "torch" in policy["forbidden_python_imports"] and "research" in policy["forbidden_python_imports"]


def test_importing_the_package_does_not_import_pytorch():
    probe = "import sys, research.elpis_torch as lab; print(lab.PHASE, 'torch' in sys.modules)"
    run = subprocess.run([sys.executable, "-B", "-c", probe], capture_output=True, text=True, cwd=REPO,
                         env={"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["ELPIS_TORCH_GENERATIVE_SUBSTRATE_R0", "False"]


def test_no_weights_or_build_products_are_committed_with_the_prototype():
    forbidden = tuple(json.loads((REPO / "ELPIS_SYSTEM.json").read_text())["boundary_policy"]["forbidden_file_suffixes"])
    roots = (LAB, REPO / "tests" / "research" / "elpis_torch", REPO / "docs" / "research" / "elpis_torch")
    files = [p for root in roots for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files and not [str(p) for p in files if p.name.endswith(forbidden)]
    assert max(p.stat().st_size for p in files) < 400_000         # proportional: no evidence archive


def test_the_only_storage_writers_are_the_explicit_offline_operations():
    """Sources of the package that can open a file for writing: checkpoint savers and the experiments' --out."""
    writers = {}
    for path in sorted(LAB.rglob("*.py")):
        if "upstream" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        hits = re.findall(r"torch\.save\(|\bopen\([^)]*[\"'][wax+b]+[\"']|\.write_text\(|\.write_bytes\(|pickle\.dump", text)
        if hits:
            writers[str(path.relative_to(LAB))] = len(hits)
    assert writers == {"training.py": 1, "experiments/run_r0.py": 1, "experiments/run_r1.py": 2,
                       "experiments/run_r2.py": 1, "experiments/run_r3.py": 2, "experiments/run_r4.py": 1,
                       "experiments/run_r5.py": 3, "experiments/run_r6.py": 1, "experiments/run_r7.py": 2,
                       "experiments/run_r8.py": 1, "experiments/run_r9.py": 1, "lm_training.py": 1}
    for module in ("session.py", "scroll_session.py", "scroll.py", "lastblock.py", "ersa.py", "ersa_model.py",
                   "history.py", "history_model.py", "longrange_tasks.py", "program_tasks.py", "scaling.py",
                   "context_runtime.py", "spark_reference.py", "evidence_ring.py", "evidence_model.py",
                   "evidence_tasks.py", "sparse_channel.py", "ple.py"):
        source = (LAB / module).read_text(encoding="utf-8")
        assert "save_checkpoint" not in source and "torch.save" not in source and "open(" not in source, module


def test_the_ple_store_only_ever_reads():
    """R7: the inference store opens its file read only and reads rows with positioned reads; nothing in it can
    write, map, truncate or replace a file (the offline writer lives in experiments/run_r7.py)."""
    source = (LAB / "ple_store.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    opens = re.findall(r"\bopen\([^)]*\)", code)
    assert opens == ["open(self.path, os.O_RDONLY)"], opens
    assert not re.search(r"os\.(write|pwrite|ftruncate|truncate|rename|replace|remove|unlink)|\.write\(|mmap|"
                         r"O_WRONLY|O_RDWR|O_CREAT|torch\.save", code)


def test_the_deferred_systems_are_not_implemented():
    """No supervisor, expert selector, halting agent, energy function or persistent learner hides in the package.

    ERSA left this list in R2, by the owner's directive: it is now the token mixer (ersa.py, ersa_model.py).
    Token-routed channel experts left it in R6, by the owner's directive (ELPIS_TORCH_R6 P2): a learned per-token
    router over FFN + CGC experts inside the channel sublayer (sparse_channel.py). An expert SELECTOR in the
    agentic sense (RALPH as selector, an action policy, a halting agent, a controller) stays absent.
    Parametric lookup embeddings were admitted for evaluation in R7, by the owner's directive (ELPIS_TORCH_R7):
    hashed n-gram rows fused by a learned gate at one block (ple.py, read-only stores in ple_store.py). The names
    below stay forbidden: no Engram reproduction and no general memory or retrieval framework.
    """
    names = {p.stem.lower() for p in LAB.rglob("*.py")}
    assert not names & {"ralph", "moe", "engram", "mhc", "hca", "awr", "gdn", "halt", "energy", "router",
                        "controller", "supervisor", "memory", "retrieval", "policy", "agent", "planner"}
    code = "\n".join(p.read_text(encoding="utf-8") for p in LAB.rglob("*.py") if "upstream" not in p.parts)
    assert not re.search(r"(?im)^\s*(class|def)\s+\w*(ralph|halt|energy|metacog|supervis|policy|planner)\w*", code)
    routed = [name for name in re.findall(r"(?im)^\s*(?:class|def)\s+(\w*(?:expert|router)\w*)", code)]
    assert set(routed) <= {"ChannelExpert", "TokenRouter"}, routed          # only the R6 channel mechanism


def test_the_mission_does_not_name_the_prototype_and_the_amendment_is_only_a_proposal():
    """The prototype claims no place in the mission; what it would take is a proposal the owner may decline."""
    assert "elpis_torch" not in (REPO / "docs" / "ELPIS_MISSION.md").read_text(encoding="utf-8")
    text = " ".join(PROPOSAL.read_text(encoding="utf-8").split())
    for needle in ("PROPOSAL", "NOT ADOPTED", "not repository authority", "docs/ELPIS_MISSION.md",
                   "prohibited topology", "research/elpis_torch"):
        assert needle in text, needle


def test_the_historical_ersa_archive_is_byte_for_byte_what_its_manifest_says():
    """R2 preserves the archive unchanged: every file it lists is present with the recorded size and digest."""
    import hashlib
    root = REPO / "research" / "ersa" / "source_handoff"
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == len(manifest["files"]) == 131
    for entry in manifest["files"]:
        data = (root / "ERSA_DEV" / entry["path"]).read_bytes()
        assert len(data) == entry["size"] and hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    tracked = sorted(p for p in (root / "ERSA_DEV").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert len(tracked) == 131                              # nothing added beside the archived sources


def test_the_elpis_native_ersa_shares_no_code_path_with_the_archive_or_its_donor():
    """Independence: the new layer imports nothing from the archive, from FLA or from the studied projects."""
    for path in sorted(LAB.rglob("*.py")):
        if "upstream" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0].strip()
            if code.startswith(("import ", "from ")):
                assert not re.search(r"research\.ersa|\bfla\b|naive_recurrent_gsa|chipformer|qgpu|qwentpu|einops", code), (
                    path.name, line)
    text = (LAB / "PROVENANCE.md").read_text(encoding="utf-8")
    for needle in ("7b423a174fa58e6f0d2f3156ed496dcb67f53361", "0cab57038a8b218dabe481f994174c3437405e3d",
                   "e2695f6a0a818ebc68e2e14b6b990eddf04c51b3", "856720f3e5d27c0fc1c19f370a39487853e722ae", "MIT",
                   "Songlin Yang"):
        assert needle in text, needle
    licence = REPO / "research/ersa/source_handoff/ERSA_DEV/worktrees/R35_DONOR_BACKED_SLOT_WRITE_READ_R0/third_party/fla_gsa/LICENSE.txt"
    assert "MIT License" in licence.read_text(encoding="utf-8")       # the donor's notice stays with its file
