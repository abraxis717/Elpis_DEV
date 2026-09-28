"""No model weights, no build products, no network in tests."""
from __future__ import annotations

import socket

import pytest

from ._system import REPO, SYSTEM, repository_files
from ..conftest import NetworkAccessForbidden

MAX_FILE_BYTES = 2 * 1024 * 1024


def test_no_weights_or_binary_build_products_committed():
    suffixes = tuple(SYSTEM["boundary_policy"]["forbidden_file_suffixes"])
    offenders = [str(p.relative_to(REPO)) for p in repository_files() if p.name.endswith(suffixes)]
    assert not offenders


def test_no_oversized_files():
    offenders = [
        (str(p.relative_to(REPO)), p.stat().st_size)
        for p in repository_files()
        if p.is_file() and p.stat().st_size > MAX_FILE_BYTES
    ]
    assert not offenders


def test_network_guard_is_active():
    with pytest.raises(NetworkAccessForbidden):
        socket.create_connection(("example.invalid", 443), timeout=0.1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkAccessForbidden):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()
    with pytest.raises(NetworkAccessForbidden):
        socket.getaddrinfo("example.invalid", 443)
