"""Truthful classification for formulaic/configured staging metrics."""
from __future__ import annotations

import numpy as np

from elpis.inference.drivers.dsv4.fixtures import make_fixture


def test_file_provider_staging_high_water_is_analytical_bound(provider):
    f, _, manifest, asset = provider
    with f.acquire(asset, 0, 16):
        pass
    stats = f.stats()
    assert stats["staging_high_water_kind"] == "analytical_bound"
    assert stats["staging_high_water"] == 4 * manifest.page_size
    assert stats["staging_high_water"] <= f.staging_budget


def test_row_engine_staging_upper_bound_is_analytical_bound(provider,tmp_path):
    f, *_ = provider
    target, _, _ = make_fixture(f, tmp_path / "rows_target")
    bank = target.rows.table.bank
    from elpis.inference.contracts import RowIdentity
    requests = tuple(RowIdentity(bank.digest, i % bank.rows) for i in range(4))
    target.rows.lookup(requests)
    metrics = target.rows.last_metrics
    stride = bank.dimension * 4
    assert metrics["row_staging_upper_bound_kind"] == "analytical_bound"
    assert metrics["row_staging_upper_bound"] == target.rows.workers * (
        stride + bank.dimension * 32
    )


def test_expert_bank_staging_high_water_is_configured_reservation(provider,tmp_path):
    f, *_ = provider
    target, _, _ = make_fixture(f, tmp_path / "expert_target")
    experts = target.experts
    x = np.zeros(target.config.dimension, dtype="<f4")
    experts.execute(x, model=experts.model, layer=0, route=(0,), weights=(1.0,))
    metrics = experts.last_metrics
    manifest = experts.manifest(0, 0)
    reservation = sum(t.size for t in manifest.tensors) * 3
    assert metrics["staging_high_water_kind"] == "configured_reservation"
    assert metrics["staging_high_water"] == reservation
    assert metrics["staging_high_water"] <= experts.staging_budget
