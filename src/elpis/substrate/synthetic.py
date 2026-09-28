"""Explicitly self-authorized synthetic qualification ONLY.

This class is not a production provenance boundary. It deliberately authorizes
candidate bytes for generated fixtures, while exercising the same secure open,
full admission verification, native sealing and page-read implementation.
"""
from dataclasses import asdict
import json
import os

from .authority import AssetIdentity, NativeIdentity, PinnedAuthority
from .boundary import RootCapability, read_exact
from .contracts import Code, require
from .digests import raw_sha256
from .file_assets import FMSFileAssets, _inspect_fd


class SyntheticFileAssets(FMSFileAssets):
    provenance = 'synthetic-test'

    def __init__(self, *, root, library, **kwargs):
        with RootCapability(root) as boundary:
            fd = boundary.open_file(library)
            try:
                size = os.fstat(fd).st_size
                digest = raw_sha256()
                for offset in range(0, size, 1024 * 1024):
                    digest.update(read_exact(fd, min(1024 * 1024, size-offset), offset))
            finally:
                os.close(fd)
        native = NativeIdentity('synthetic-provider', size, digest.hexdigest())
        document = json.dumps(dict(schema='elpis.inference-authority.v1',
                                   source='explicit synthetic fixture self-authorization',
                                   provenance='synthetic-test', assets=[],
                                   libraries=[asdict(native)])).encode()
        authority = PinnedAuthority(document, expected_sha256=raw_sha256(document).hexdigest())
        super().__init__(root=root, library=library, authority=authority,
                         library_id=native.library_id, **kwargs)

    @staticmethod
    def _check_authority(authority):
        require(type(authority) is PinnedAuthority and authority.provenance == 'synthetic-test',
                Code.IDENTITY, 'synthetic fixture authority required')

    def _authorized_identity(self, asset_id, path, manifest):
        fd = self._boundary.open_file(path)
        try:
            _, digest, _ = _inspect_fd(fd, manifest.page_size)
        finally:
            os.close(fd)
        return AssetIdentity(asset_id or 'synthetic:' + manifest.digest,
                             manifest.size, manifest.page_size, digest, manifest.digest)
