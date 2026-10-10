"""Test-only bridge catalog. Never use this as deployment pin provisioning.

In tests, the fixture author is the trust root and deliberately pins exactly
one copied test artifact. Production must receive an independently configured
catalog and its expected SHA-256 through trusted deployment configuration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from elpis.substrate.authority import PinnedAuthority


def pin_bridge(source: Path, workspace: Path, library_id: str, *, sha_override=None):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / Path(source).name
    shutil.copyfile(source, target)
    data = target.read_bytes()
    spec = {
        'schema': 'elpis.inference-authority.v1',
        'source': 'structural bridge test fixture, NOT deployment',
        'provenance': 'deployment',
        'assets': [],
        'libraries': [{'library_id': library_id, 'size': len(data),
                       'sha256': sha_override if sha_override is not None
                                 else hashlib.sha256(data).hexdigest()}],
    }
    catalog = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
    pin = hashlib.sha256(catalog).hexdigest()
    return target, workspace, PinnedAuthority(catalog, expected_sha256=pin)
