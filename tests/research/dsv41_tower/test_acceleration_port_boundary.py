"""Architecture guard: Elpis exposes a vendor-neutral acceleration port.

Concrete device kernels are post-compiled/provider-side artifacts. The base
DSV4.1 repository must not acquire a vendor SDK as a build prerequisite.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_base_dsv41_build_is_vendor_sdk_free():
    build_files = (
        ROOT / "CMakeLists.txt",
        ROOT / "research/dsv41_tower/native/CMakeLists.txt",
        ROOT / "pyproject.toml",
    )
    forbidden = (
        "cudatoolkit",
        "enable_language(cuda",
        "languages c cxx cuda",
        "cuda_runtime.h",
        "hip_runtime",
        "hip/hip",
        "metal/metal.h",
        "vulkan/vulkan.h",
        "oneapi",
    )
    for path in build_files:
        text = path.read_text().lower()
        for marker in forbidden:
            assert marker not in text, (path, marker)


def test_inference_tree_has_no_vendor_kernel_sources():
    root = ROOT / "research/dsv41_tower/native"
    forbidden_suffixes = {".cu", ".cuh", ".hip", ".metal"}
    offenders = sorted(
        p.relative_to(ROOT).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in forbidden_suffixes
    )
    assert offenders == []


def test_substrate_exposes_backend_only_stream_without_vendor_runtime():
    header = (ROOT / "native/substrate/include/elpis/execution.h").read_text()
    doc = (ROOT / "native/substrate/EXECUTION.md").read_text()
    assert "ELPIS_EXEC_BACKEND_ONLY" in header
    assert "elpis_exec_backend" in header
    assert "post-compiled" in doc
    assert "base build contains no real device backend" in doc.lower()


if __name__ == "__main__":
    test_base_dsv41_build_is_vendor_sdk_free()
    test_inference_tree_has_no_vendor_kernel_sources()
    test_substrate_exposes_backend_only_stream_without_vendor_runtime()
    print("PASS_DSV41_ACCELERATION_PORT_BOUNDARY")
