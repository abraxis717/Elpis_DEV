"""Raw-byte digests, distinct from canonical structured identity.

``raw_sha256`` is ordinary SHA-256 over deployment bytes (catalog pins,
native library pins). ``raw_digest`` is the exact canonical-identity digest of
a raw byte string, computed in bounded chunks. Its domain,
``elpis.inference.raw-bytes.r0``, is persisted identity: page maps and asset
manifests pinned by deployments depend on it, so it keeps its historical
spelling even though the mechanism now lives in the substrate.
"""
import binascii
import hashlib

from elpis.identity import content_digest

from .contracts import require

_RAW_DIGEST_CHUNK = 64 * 1024
_RAW_DIGEST_PREFIX = b'elpis.inference.raw-bytes.r0\x00{"__bytes__":"'
_RAW_DIGEST_SUFFIX = b'"}'


def raw_sha256(data=b""):
    require(
        isinstance(data, (bytes, bytearray, memoryview)),
        detail="raw SHA-256 bytes",
    )
    return hashlib.sha256(bytes(data))


def raw_digest(data):
    """Exact canonical raw-bytes identity with bounded contiguous intermediates."""
    require(
        isinstance(data, (bytes, bytearray, memoryview)),
        detail='raw byte content',
    )
    view = memoryview(data)
    if view.c_contiguous:
        octets = view.cast('B')
    else:
        octets = memoryview(bytes(data))
    digest = raw_sha256()
    digest.update(_RAW_DIGEST_PREFIX)
    for start in range(0, len(octets), _RAW_DIGEST_CHUNK):
        digest.update(binascii.hexlify(octets[start:start+_RAW_DIGEST_CHUNK]))
    digest.update(_RAW_DIGEST_SUFFIX)
    return digest.hexdigest()


def identity(kind, value):
    """Canonical identity of substrate records.

    The ``elpis.inference.<kind>.r0`` domain family is persisted identity
    (asset manifests and content maps are pinned by deployment catalogs), so
    it is retained verbatim.
    """
    return content_digest('elpis.inference.' + kind + '.r0', value)
