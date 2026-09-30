"""Read-only parameter preflight. Inspection does not create model authority.

No production DSV4.1 arithmetic driver has been qualified. In particular,
CompactTarget cannot become one by replacing its tensors/tokenizer strings.
This module inventories pinned safetensors without importing an execution
framework, and refuses binding until an architecture-specific driver exists.
"""
from dataclasses import dataclass
import json
import math
import os
import struct

from elpis.substrate.boundary import RootCapability, read_exact, stamp
from elpis.substrate.digests import raw_sha256
from .contracts import Code, ContractError, digest_value, integer, require


@dataclass(frozen=True)
class ParameterArtifact:
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        require(type(self.path) is str and bool(self.path), detail="artifact path")
        integer(self.size, 1)
        digest_value(self.sha256)


@dataclass(frozen=True)
class TensorInventory:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass(frozen=True)
class ParameterInventory:
    artifact: ParameterArtifact
    tensors: tuple[TensorInventory, ...]


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, detail="duplicate safetensors key")
        result[key] = value
    return result


def inspect_parameters(root: RootCapability, artifact: ParameterArtifact, *, max_bytes: int,
                       max_header_bytes: int = 16 << 20) -> ParameterInventory:
    """Stream-verify caller-pinned bytes, validate storage spans, return metadata.

    No parameters are converted, mapped writable, relocated or installed. The
    returned inventory is a preflight observation, never an executable target.
    """
    require(type(artifact) is ParameterArtifact, detail="parameter artifact")
    integer(max_bytes, 1)
    integer(max_header_bytes, 2)
    require(artifact.size <= max_bytes, Code.LIMIT, "parameter artifact budget")
    fd = root.open_file(artifact.path)
    try:
        before = os.fstat(fd)
        require(before.st_size == artifact.size and artifact.size >= 10, Code.IDENTITY, "parameter size")
        header_size = struct.unpack("<Q", read_exact(fd, 8, 0))[0]
        require(2 <= header_size <= min(max_header_bytes, artifact.size - 8), Code.LIMIT, "parameter header")
        header = read_exact(fd, header_size, 8)
        digest = raw_sha256()
        for offset in range(0, artifact.size, 1 << 20):
            digest.update(read_exact(fd, min(1 << 20, artifact.size - offset), offset))
        require(digest.hexdigest() == artifact.sha256, Code.IDENTITY, "parameter SHA-256")
        require(stamp(before) == stamp(os.fstat(fd)), Code.IDENTITY, "parameters changed during inspection")
    finally:
        os.close(fd)
    try:
        document = json.loads(header, object_pairs_hook=_object)
    except (ValueError, UnicodeError) as exc:
        raise ContractError(Code.ENCODING, "parameter JSON") from exc
    require(type(document) is dict, detail="parameter header object")
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
             "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
    tensors = []
    for name, entry in document.items():
        if name == "__metadata__":
            require(type(entry) is dict and all(type(v) is str for v in entry.values()), detail="metadata")
            continue
        require(type(entry) is dict and set(entry) == {"dtype", "shape", "data_offsets"}, detail="tensor header")
        dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
        require(type(dtype) is str and dtype in sizes, Code.UNSUPPORTED, "storage dtype")
        require(type(shape) is list and len(shape) <= 16, detail="tensor rank")
        for dim in shape:
            integer(dim, 0, 1 << 40)
        require(type(offsets) is list and len(offsets) == 2, detail="tensor offsets")
        start, end = offsets
        integer(start, 0, artifact.size - header_size - 8)
        integer(end, start, artifact.size - header_size - 8)
        require(end - start == math.prod(shape) * sizes[dtype], Code.ENCODING, "tensor shape/storage size")
        tensors.append(TensorInventory(name, dtype, tuple(shape), start + 8 + header_size, end - start))
    cursor = 8 + header_size
    for tensor in sorted(tensors, key=lambda t: (t.offset, t.size)):
        require(tensor.offset == cursor, Code.ENCODING, "tensor overlap/hole")
        cursor += tensor.size
    require(cursor == artifact.size, Code.ENCODING, "unaccounted parameter bytes")
    return ParameterInventory(artifact, tuple(sorted(tensors, key=lambda t: t.name)))


def bind_production_target(*, inventory, tokenizer, architecture):
    """Fail closed: a storage inventory is not architecture qualification.

    This explicit refusal must be replaced by a qualified driver with exact
    tensor, layer, routing, addressing and numerical contracts. It must never
    delegate to the synthetic CompactTarget. See the qualification report.
    """
    require(type(inventory) is ParameterInventory, detail="parameter inventory")
    raise ContractError(Code.UNSUPPORTED,
                        "PRODUCTION_DRIVER_UNIMPLEMENTED: " + str(architecture))
