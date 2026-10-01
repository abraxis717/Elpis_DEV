"""R10B full seven-layer Python-vs-native DSV4.1 tower A/B."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from elpis.inference.drivers.dsv41.engram import build_parameters
from elpis.inference.drivers.dsv41.fixtures import fixture_config, make_fixture
from elpis.inference.drivers.dsv41.native_backend import DSV41NativeBackend
from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.synthetic import SyntheticFileAssets


RTOL = 1e-4
ATOL = 1e-5


def _authority(lib):
    lib = Path(lib).resolve()
    library_id = "dsv41-r10b-ab"
    document = json.dumps(
        {
            "schema": "elpis.inference-authority.v1",
            "source": "DSV41_NATIVE_DRIVER_R10B_AB",
            "provenance": "synthetic-test",
            "assets": [],
            "libraries": [{
                "library_id": library_id,
                "size": lib.stat().st_size,
                "sha256": sha256(lib.read_bytes()).hexdigest(),
            }],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        PinnedAuthority(
            document,
            expected_sha256=sha256(document).hexdigest(),
        ),
        library_id,
    )


def test_full_seven_layer_python_native_ab(
    native_workspace, fms_file_library, v41
):
    lib = Path(os.environ["ELPIS_DSV41_NATIVE_LIBRARY"]).resolve()
    authority, library_id = _authority(lib)
    backend = DSV41NativeBackend(
        lib.parent,
        lib.name,
        authority=authority,
        library_id=library_id,
    )

    config = fixture_config(v41, seed=1041, max_tokens=32)
    parameters = build_parameters(v41, config)

    py_provider = SyntheticFileAssets(
        root=native_workspace,
        library=fms_file_library,
        warm_bytes=1 << 20,
        staging_bytes=1 << 16,
    )
    native_provider = SyntheticFileAssets(
        root=native_workspace,
        library=fms_file_library,
        warm_bytes=1 << 20,
        staging_bytes=1 << 16,
    )

    py_target = native_target = None
    py_state = native_state = None
    try:
        py_target, py_meta = make_fixture(
            py_provider,
            native_workspace / "r10b-python",
            v41,
            seed=1041,
            config=config,
            parameters=parameters,
        )
        native_target, native_meta = make_fixture(
            native_provider,
            native_workspace / "r10b-native",
            v41,
            seed=1041,
            config=config,
            parameters=parameters,
            native_backend=backend,
        )

        assert py_meta == native_meta
        assert py_target.model_identity == native_target.model_identity
        assert py_target.config.digest == native_target.config.digest
        assert py_target.store.manifest.digest == native_target.store.manifest.digest
        assert py_target.state_bytes == native_target.state_bytes

        py_state = py_target.window_initial()
        native_state = native_target.window_initial()
        py_experts = py_target.admit_stream()
        native_experts = native_target.admit_stream()

        tokens = (
            v41.encode(
                "Hello, Elpis. Multi layer attention and memory exercise."
            ) * 2
        )[:24]
        assert len(tokens) >= 12

        worst_layer = 0.0
        worst_logits = 0.0

        for position, token in enumerate(tokens):
            py_target.window_step(py_state, token, experts=py_experts)
            native_target.window_step(
                native_state, token, experts=native_experts
            )

            layer_error = float(np.max(
                np.abs(
                    native_state.layer_streams - py_state.layer_streams
                ),
                initial=0.0,
            ))
            logits_error = float(np.max(
                np.abs(native_state.logits - py_state.logits),
                initial=0.0,
            ))
            worst_layer = max(worst_layer, layer_error)
            worst_logits = max(worst_logits, logits_error)

            np.testing.assert_allclose(
                native_state.layer_streams,
                py_state.layer_streams,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"position={position} layer_streams",
            )
            np.testing.assert_allclose(
                native_state.logits,
                py_state.logits,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"position={position} logits",
            )

            assert native_state.selected == py_state.selected
            assert tuple(a.count for a in native_state.attention) == tuple(
                a.count for a in py_state.attention
            )
            assert native_state.history == py_state.history
            assert native_state.step == py_state.step
            assert int(np.argmax(native_state.logits)) == int(
                np.argmax(py_state.logits)
            )

            assert native_target.store.staged_bytes == 0
            assert py_target.store.staged_bytes == 0
            assert native_target.store.high_water <= native_target.store.staging_budget
            assert py_target.store.high_water <= py_target.store.staging_budget

        assert native_state.attention[1].count > 0
        assert native_state.attention[4].count > 0
        assert native_state.selected[2] == native_state.selected[3]
        assert native_state.selected[5] == native_state.selected[6]

        print("PASS_DSV41_NATIVE_DRIVER_FULL_TOWER_AB_R10B")
        print(f"tokens={len(tokens)}")
        print(f"layer_stream_max_abs={worst_layer:.9g}")
        print(f"logits_max_abs={worst_logits:.9g}")
        print(
            "native_expert_staging_high_water="
            f"{native_target.store.high_water}"
        )
        print(
            "python_expert_staging_high_water="
            f"{py_target.store.high_water}"
        )
    finally:
        if py_target is not None and py_state is not None and not py_state.closed:
            py_target.release_window(py_state)
        if (
            native_target is not None and native_state is not None
            and not native_state.closed
        ):
            native_target.release_window(native_state)
        py_provider.close()
        native_provider.close()

    assert py_state.closed and py_state.nbytes == 0
    assert native_state.closed and native_state.nbytes == 0
