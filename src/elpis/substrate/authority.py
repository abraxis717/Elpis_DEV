"""Deployment-pinned catalog. Inspection never creates production authority.

The expected catalog SHA-256 must arrive through trusted deployment configuration,
not alongside untrusted candidates. This authenticates exact catalog bytes; it is
not a signature service or protection against a malicious in-process caller.

The catalog schema string ``elpis.inference-authority.v1`` is persisted in
deployment catalogs and is therefore kept verbatim.
"""
from dataclasses import dataclass
import json
from types import MappingProxyType

from .contracts import Code, ContractError, digest_value, integer, require
from .digests import raw_sha256


@dataclass(frozen=True)
class AssetIdentity:
    asset_id: str
    size: int
    page_size: int
    sha256: str
    manifest_digest: str

    def __post_init__(self):
        require(type(self.asset_id) is str and bool(self.asset_id), detail='asset identifier')
        integer(self.size, 1)
        integer(self.page_size, 1, 16 * 1024 * 1024)
        digest_value(self.sha256)
        digest_value(self.manifest_digest)


@dataclass(frozen=True)
class NativeIdentity:
    library_id: str
    size: int
    sha256: str

    def __post_init__(self):
        require(type(self.library_id) is str and bool(self.library_id), detail='library identifier')
        integer(self.size, 1)
        digest_value(self.sha256)


@dataclass(frozen=True, init=False)
class PinnedAuthority:
    source: str
    provenance: str
    digest: str
    assets: object
    libraries: object
    schema: str

    def __init__(self, document: bytes, *, expected_sha256: str):
        digest_value(expected_sha256)
        require(type(document) is bytes, detail='catalog bytes')
        require(raw_sha256(document).hexdigest() == expected_sha256,
                Code.IDENTITY, 'independent catalog pin')

        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, Code.IDENTITY, 'duplicate catalog key')
                result[key] = value
            return result

        try:
            data = json.loads(document, object_pairs_hook=unique)
            require(type(data) is dict and set(data) == {
                'schema', 'source', 'provenance', 'assets', 'libraries'}, detail='catalog fields')
            require(data['schema'] == 'elpis.inference-authority.v1', detail='catalog schema')
            require(type(data['source']) is str and bool(data['source']), detail='authority source')
            require(data['provenance'] in ('deployment', 'synthetic-test'), detail='authority provenance')
            require(type(data['assets']) is list and type(data['libraries']) is list)
            assets = tuple(AssetIdentity(**entry) for entry in data['assets'])
            libraries = tuple(NativeIdentity(**entry) for entry in data['libraries'])
            require(len({a.asset_id for a in assets}) == len(assets), detail='duplicate asset identifier')
            require(len({a.library_id for a in libraries}) == len(libraries), detail='duplicate library identifier')
        except (TypeError, KeyError, json.JSONDecodeError, UnicodeError) as exc:
            raise ContractError(Code.IDENTITY, 'malformed authority catalog') from exc
        for key, value in dict(source=data['source'], provenance=data['provenance'],
                               digest=expected_sha256, schema=data['schema'],
                               assets=MappingProxyType({a.asset_id: a for a in assets}),
                               libraries=MappingProxyType({a.library_id: a for a in libraries})).items():
            object.__setattr__(self, key, value)
