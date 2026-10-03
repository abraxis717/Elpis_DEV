"""R10A sealed DSV4.1 native backend admission qualification."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

from elpis.inference.contracts import ContractError
from research.dsv41_tower.native_backend import (
    ABI_VERSION,
    CAPABILITY_NAMES,
    DSV41NativeBackend,
    REQUIRED_CAPABILITIES,
)
from elpis.substrate.authority import PinnedAuthority


def authority_for(lib, *, native_sha=None, library_id="dsv41-r10a-test"):
    lib = Path(lib).resolve()
    digest = native_sha
    if digest is None:
        digest = sha256(lib.read_bytes()).hexdigest()
    document = json.dumps(
        {
            "schema": "elpis.inference-authority.v1",
            "source": "DSV41_NATIVE_BACKEND_R10A_TEST",
            "provenance": "synthetic-test",
            "assets": [],
            "libraries": [
                {
                    "library_id": library_id,
                    "size": lib.stat().st_size,
                    "sha256": digest,
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return PinnedAuthority(
        document,
        expected_sha256=sha256(document).hexdigest(),
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_backend_admission.py "
            "/path/to/libelpis_dsv41_native.so"
        )

    lib = Path(sys.argv[1]).resolve()
    library_id = "dsv41-r10a-test"
    authority = authority_for(lib, library_id=library_id)

    backend = DSV41NativeBackend(
        lib.parent,
        lib.name,
        authority=authority,
        library_id=library_id,
    )

    assert backend.abi_version == ABI_VERSION == 1
    assert backend.capabilities & REQUIRED_CAPABILITIES == REQUIRED_CAPABILITIES
    assert backend.capability_names[:23] == CAPABILITY_NAMES
    assert len(CAPABILITY_NAMES) == 23

    # Exercise the three orchestration scratch-query families through the
    # sealed-loaded object, not through a direct ctypes.CDLL pathname.
    assert backend.local_attention_scratch_floats(
        24, 12, 2, 32, 2, 4, 5
    ) > 0
    assert backend.compressed_attention_scratch_floats(
        12, 2, 32, 2, 32, 2, 4, 5, 8, 3
    ) > 0
    assert backend.layer_frame_scratch_floats(
        4, 24, 128, 8, True
    ) > 0

    # Independent native identity is authoritative. A catalog whose own bytes
    # are correctly pinned but whose native SHA is wrong must fail admission.
    bad = authority_for(
        lib,
        native_sha="0" * 64,
        library_id="dsv41-r10a-bad",
    )
    try:
        DSV41NativeBackend(
            lib.parent,
            lib.name,
            authority=bad,
            library_id="dsv41-r10a-bad",
        )
    except ContractError:
        pass
    else:
        raise AssertionError("wrong native SHA unexpectedly admitted")

    # Likewise, an unlisted identifier cannot select a candidate by pathname.
    try:
        DSV41NativeBackend(
            lib.parent,
            lib.name,
            authority=authority,
            library_id="not-authorized",
        )
    except ContractError:
        pass
    else:
        raise AssertionError("unlisted native identifier unexpectedly admitted")

    print("PASS_DSV41_NATIVE_BACKEND_ADMISSION_R10A")
    print(f"native_abi={backend.abi_version}")
    print(f"native_capabilities=0x{backend.capabilities:x}")
    print(f"required_capabilities=0x{REQUIRED_CAPABILITIES:x}")
    print(f"capability_count={len(backend.capability_names)}")
    print(f"native_sha256={backend.identity.sha256}")


if __name__ == "__main__":
    main()
