"""Storage preflight is not production-model admission."""
import hashlib
import json
import struct

import pytest

from elpis.inference.contracts import ContractError
from elpis.inference.production import ParameterArtifact, bind_production_target, inspect_parameters
from elpis.substrate.boundary import RootCapability


def artifact(tmp_path, header=None, data=None):
    header = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}} if header is None else header
    data = bytes(8) if data is None else data
    encoded = json.dumps(header).encode()
    raw = struct.pack("<Q", len(encoded)) + encoded + data
    (tmp_path / "test.safetensors").write_bytes(raw)
    return ParameterArtifact("test.safetensors", len(raw), hashlib.sha256(raw).hexdigest())


def test_pinned_inventory_is_never_executable(tmp_path):
    spec = artifact(tmp_path)
    with RootCapability(tmp_path) as root:
        inventory = inspect_parameters(root, spec, max_bytes=1024)
    assert inventory.tensors[0].shape == (2,)
    with pytest.raises(ContractError, match="PRODUCTION_DRIVER_UNIMPLEMENTED"):
        bind_production_target(inventory=inventory, tokenizer=None, architecture="DeepSeek-V4.1")


@pytest.mark.parametrize("header,data", [
    ({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, bytes(8)),
    ({"w": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}, bytes(12)),
    ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
      "b": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, bytes(8)),
    ({"w": {"dtype": "GUESS_FP4", "shape": [2], "data_offsets": [0, 8]}}, bytes(8)),
])
def test_incompatible_storage_refused(tmp_path, header, data):
    spec = artifact(tmp_path, header, data)
    with RootCapability(tmp_path) as root, pytest.raises(ContractError):
        inspect_parameters(root, spec, max_bytes=1024)


def test_digest_and_budget_refused(tmp_path):
    spec = artifact(tmp_path)
    with RootCapability(tmp_path) as root:
        with pytest.raises(ContractError, match="budget"):
            inspect_parameters(root, spec, max_bytes=1)
        path = tmp_path / spec.path
        data = path.read_bytes()
        path.write_bytes(data[:-1] + b"1")
        with pytest.raises(ContractError, match="SHA-256"):
            inspect_parameters(root, spec, max_bytes=1024)
