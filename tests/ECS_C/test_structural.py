"""Byte-level contract tests; fixtures are synthetic software data only."""
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import itertools
from pathlib import Path
import os
import struct
import subprocess
import sys

import pytest

from elpis.ECS_C.structural import (
    AuthorityBundle, EXPECTED, Candidate, EditAddress, FrozenTheta,
    MutationRequest, Opcode, R0Error, Status, equivalent, mutate, permute,
    materialize_authority, verify_authority,
)


from elpis.ECS_C.structural_authority import (
    HEADER_ANCHOR, MANIFEST_ANCHOR, authority_entry_anchor, read_authority_anchor,
)

import elpis

IMPORT_ROOT = Path(elpis.__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent


def theta_bytes(n=36):
    # Unique columns; signed zero, subnormal and extreme finite values survive.
    values = [struct.pack("<d", float(row * n + i + 1))
              for row in range(6) for i in range(n)]
    values[n] = bytes.fromhex("0000000000000080")
    values[2 * n] = bytes.fromhex("0100000000000000")
    values[3 * n] = bytes.fromhex("ffffffffffffef7f")
    return b"".join(values)


@pytest.fixture
def candidate():
    return Candidate.bind(FrozenTheta(36, theta_bytes()))


@pytest.fixture
def authority_copy():
    anchors = (HEADER_ANCHOR, MANIFEST_ANCHOR,
               *(authority_entry_anchor(name) for name in EXPECTED))
    return {anchor: read_authority_anchor(anchor) for anchor in anchors}


def edit(candidate, opcode, slot=0):
    return mutate(candidate, MutationRequest(opcode, candidate.digest,
                  None if opcode is Opcode.ABSTAIN else candidate.address(slot)))


def test_manifest_pass(authority_copy):
    assert verify_authority() == verify_authority(authority_copy.__getitem__)


def test_unknown_anchor_fails_closed():
    for name in ("unknown", "../README.md", "", None):
        with pytest.raises(KeyError):
            read_authority_anchor(name)
        with pytest.raises(KeyError):
            authority_entry_anchor(name)


def test_header_tamper_and_manifest_byte_pin(authority_copy):
    original = authority_copy[HEADER_ANCHOR]
    authority_copy[HEADER_ANCHOR] += b"\n"
    with pytest.raises(R0Error, match="AUTHORITY_HEADER_DIGEST"):
        verify_authority(authority_copy.__getitem__)
    authority_copy[HEADER_ANCHOR] = original
    authority_copy[MANIFEST_ANCHOR] = authority_copy[MANIFEST_ANCHOR].replace(b"\n", b"\r\n")
    with pytest.raises(R0Error, match="AUTHORITY_MANIFEST_DIGEST"):
        verify_authority(authority_copy.__getitem__)


@pytest.mark.parametrize("fault", ["tamper", "missing", "manifest_missing", "duplicate",
    "bad_digest", "malformed", "traversal", "extra", "omitted", "non_ascii", "self_consistent"])
def test_manifest_fails_closed(authority_copy, fault):
    lines = authority_copy[MANIFEST_ANCHOR].decode().splitlines()
    name = "MUTATION_GRAMMAR.json"
    anchor = authority_entry_anchor(name)
    if fault in ("tamper", "self_consistent"):
        authority_copy[anchor] = b"{}"
        if fault == "self_consistent":
            lines = [hashlib.sha256(b"{}").hexdigest() + "  " + name
                     if line.endswith(name) else line for line in lines]
            authority_copy[MANIFEST_ANCHOR] = ("\n".join(lines) + "\n").encode()
    elif fault == "missing":
        del authority_copy[anchor]
    elif fault == "manifest_missing":
        del authority_copy[MANIFEST_ANCHOR]
    elif fault == "non_ascii":
        authority_copy[MANIFEST_ANCHOR] = b"\xff"
    else:
        if fault == "duplicate": lines.append(lines[0])
        if fault == "bad_digest": lines[0] = "g" + lines[0][1:]
        if fault == "malformed": lines[0] = lines[0].replace("  ", " ")
        if fault == "traversal": lines[0] = "0" * 64 + "  ../README.md"
        if fault == "extra": lines.append("0" * 64 + "  EXTRA.json")
        if fault == "omitted": lines.pop()
        authority_copy[MANIFEST_ANCHOR] = ("\n".join(lines) + "\n").encode()
    with pytest.raises(R0Error): materialize_authority(authority_copy.__getitem__)
    with pytest.raises(R0Error): FrozenTheta(36, theta_bytes(), authority_copy.__getitem__)


def test_verified_snapshot_survives_reader_change(authority_copy):
    theta = FrozenTheta(36, theta_bytes(), authority_copy.__getitem__)
    bound = Candidate.bind(theta)
    original_digest = bound.digest
    authority_copy[authority_entry_anchor("README.md")] = b"tamper"
    with pytest.raises(R0Error): materialize_authority(authority_copy.__getitem__)
    assert Candidate.bind(theta).digest == original_digest
    assert bound.materialize() == theta.data
    changed = edit(bound, Opcode.DISABLE_COLUMN).candidate
    moved, _ = permute(changed, tuple(reversed(range(36))))
    assert moved.theta.authority is changed.theta.authority is theta.authority
    assert equivalent(changed, moved)
    assert theta.authority.content_identity == verify_authority()
    with pytest.raises(FrozenInstanceError): theta.authority.header = b"tamper"


@pytest.mark.parametrize("fault", ["mutable_header", "mutable_manifest", "mutable_entries",
    "mutable_entry", "mutable_bytes", "duplicate", "extra", "missing", "tamper"])
def test_bundle_construction_is_verified_and_immutable(fault):
    bundle = materialize_authority()
    fields = dict(header=bundle.header, manifest=bundle.manifest, entries=bundle.entries)
    if fault == "mutable_header": fields["header"] = bytearray(bundle.header)
    if fault == "mutable_manifest": fields["manifest"] = bytearray(bundle.manifest)
    if fault == "mutable_entries": fields["entries"] = list(bundle.entries)
    if fault == "mutable_entry": fields["entries"] = (list(bundle.entries[0]),) + bundle.entries[1:]
    if fault == "mutable_bytes": fields["entries"] = ((bundle.entries[0][0], bytearray(bundle.entries[0][1])),) + bundle.entries[1:]
    if fault == "duplicate": fields["entries"] += bundle.entries[:1]
    if fault == "extra": fields["entries"] += (("EXTRA.json", b"{}"),)
    if fault == "missing": fields["entries"] = bundle.entries[1:]
    if fault == "tamper": fields["header"] += b"\n"
    with pytest.raises(R0Error): AuthorityBundle(**fields)


def test_directory_is_only_reader_transport(tmp_path):
    bundle = materialize_authority()
    anchors = (HEADER_ANCHOR, MANIFEST_ANCHOR,
               *(authority_entry_anchor(name) for name in EXPECTED))
    identities = []
    for location in (tmp_path / "first", tmp_path / "relocated"):
        location.mkdir()
        # Only this outer fixture adapter knows the physical layout.
        files = {anchor: location / str(i) for i, anchor in enumerate(anchors)}
        for anchor, path in files.items(): path.write_bytes(read_authority_anchor(anchor))
        def reader(anchor):
            return files[anchor].read_bytes()
        theta = FrozenTheta(36, theta_bytes(), reader)
        identities.append((theta.authority.content_identity, Candidate.bind(theta).digest))
        assert theta.authority == bundle
    assert identities[0] == identities[1]


@pytest.mark.parametrize("n", [36, 48, 72])
def test_active_disabled_restore_width_determinism(n):
    theta = FrozenTheta(n, theta_bytes(n))
    bound = Candidate.bind(theta)
    assert bound.materialize() == theta.data
    disabled = edit(bound, Opcode.DISABLE_COLUMN).candidate
    expected = bytearray(theta.data)
    for row in range(6): expected[row * n * 8:row * n * 8 + 8] = b"\0" * 8
    assert disabled.materialize() == bytes(expected) == disabled.materialize()
    restored = edit(disabled, Opcode.RESTORE_COLUMN).candidate
    assert restored.materialize() == theta.data
    assert restored.digest == bound.digest
    assert restored.width == disabled.width == n
    assert restored.theta is disabled.theta is bound.theta


def test_initial_zero_columns_nonwritable():
    raw = bytearray(theta_bytes())
    for row in range(6):
        raw[row * 36 * 8:row * 36 * 8 + 8] = struct.pack("<d", -0.0)
    bound = Candidate.bind(FrozenTheta(36, raw))
    assert bound.mask[0] is Status.FROZEN_ZERO
    assert bound.materialize() == raw
    for op in (Opcode.DISABLE_COLUMN, Opcode.RESTORE_COLUMN):
        with pytest.raises(R0Error, match="MUTATION_PRECONDITION"): edit(bound, op)
    with pytest.raises(TypeError):
        replace(bound, mask=(Status.ACTIVE,) + bound.mask[1:])


@pytest.mark.parametrize("bits", [0x7ff0000000000000, 0xfff0000000000000,
    0x7ff8000000000123, 0x7ff0000000000001, 0xfff8000000000042])
def test_reject_nonfinite_without_normalizing(bits):
    raw = struct.pack("<Q", bits) + theta_bytes()[8:]
    with pytest.raises(R0Error, match="THETA_NONFINITE"): FrozenTheta(36, raw)


@pytest.mark.parametrize("n", [True, 0, 1, 35, 37, 81, 36.0])
def test_width_scope(n):
    with pytest.raises(R0Error, match="WIDTH_SCOPE"): FrozenTheta(n, theta_bytes())


def test_theta_shape_and_encoding():
    with pytest.raises(R0Error, match="THETA_SHAPE"): FrozenTheta(36, theta_bytes()[:-8])
    with pytest.raises(R0Error, match="THETA_ENCODING"):
        FrozenTheta(36, theta_bytes(), encoding="binary64-be-c-order")


@pytest.mark.parametrize("mask", [(), [Status.ACTIVE] * 35, [Status.ACTIVE] * 37,
    [True] * 36, [1] * 36, ["ACTIVE"] * 36, [Status.FROZEN_ZERO] * 36])
def test_public_candidate_construction_is_closed(candidate, mask):
    with pytest.raises(TypeError):
        Candidate(candidate.theta, mask)
    with pytest.raises(TypeError):
        replace(candidate, mask=mask)


@pytest.mark.parametrize("slot", [-1, 36, 100, True, 1.0, "0", None])
def test_invalid_slot(candidate, slot):
    with pytest.raises(R0Error, match="SLOT_BOUNDS"): candidate.address(slot)
    request = MutationRequest(Opcode.DISABLE_COLUMN, candidate.digest,
                             replace(candidate.address(0), slot_index=slot))
    with pytest.raises(R0Error, match="SLOT_BOUNDS"): mutate(candidate, request)


@pytest.mark.parametrize("opcode", ["OPTIMIZE", "DISABLE", "", 0, None])
def test_invalid_opcode(candidate, opcode):
    with pytest.raises(R0Error, match="OPCODE"):
        MutationRequest.from_packet(dict(op=opcode, address=None,
                                        expected_candidate_digest=candidate.digest))


@pytest.mark.parametrize("field", ["values", "theta", "N", "width", "primitive", "d",
    "adjacency", "graph", "modules", "S3", "optimizer", "learning_rate", "operations"])
def test_injection_closed_surface(candidate, field):
    packet = dict(op="DISABLE_COLUMN", address=asdict(candidate.address(0)),
                  expected_candidate_digest=candidate.digest)
    with pytest.raises(R0Error, match="PACKET_FIELDS"):
        MutationRequest.from_packet({**packet, field: [0.5]})
    packet["address"][field] = [0.5]
    # d and N are legitimate address fields, but incorrect values reject later.
    with pytest.raises(R0Error): mutate(candidate, MutationRequest.from_packet(packet))


def test_identity_and_stale_bindings(candidate):
    same = Candidate.bind(FrozenTheta(36, theta_bytes()))
    assert candidate.digest == same.digest
    other = Candidate.bind(FrozenTheta(36, struct.pack("<d", 9.0) + theta_bytes()[8:]))
    assert candidate.theta.digest != other.theta.digest
    req = MutationRequest(Opcode.DISABLE_COLUMN, candidate.digest, candidate.address(0))
    with pytest.raises(R0Error, match="STALE_CANDIDATE"): mutate(other, req)
    with pytest.raises(R0Error, match="STALE_ADDRESS"):
        mutate(other, replace(req, expected_candidate_digest=other.digest))
    disabled = mutate(candidate, req).candidate
    with pytest.raises(R0Error, match="STALE_CANDIDATE"): mutate(disabled, req)


@pytest.mark.parametrize("field,value", [
    ("ontology_id", "other"), ("ontology_version", "R1"),
    ("primitive_contract_digest", "0" * 64), ("d", 5), ("d", 6.0),
    ("N", 48), ("N", 36.0), ("operational_gauge_id", "other"),
    ("ordered_frozen_sidecar_digest", "0" * 64),
    ("expected_pre_status", "DISABLED"), ("pre_state_equivalence_digest", "0" * 64)])
def test_every_address_binding(candidate, field, value):
    req = MutationRequest(Opcode.DISABLE_COLUMN, candidate.digest,
                          replace(candidate.address(0), **{field: value}))
    with pytest.raises(R0Error): mutate(candidate, req)


def test_idempotence_is_only_abstain(candidate):
    abstain = edit(candidate, Opcode.ABSTAIN)
    assert not abstain.changed and abstain.candidate is candidate
    assert edit(abstain.candidate, Opcode.ABSTAIN) == abstain
    with pytest.raises(R0Error, match="MUTATION_PRECONDITION"):
        edit(candidate, Opcode.RESTORE_COLUMN)
    disabled = edit(candidate, Opcode.DISABLE_COLUMN).candidate
    with pytest.raises(R0Error, match="MUTATION_PRECONDITION"):
        edit(disabled, Opcode.DISABLE_COLUMN)


def test_immutable_public_interfaces():
    data = bytearray(theta_bytes())
    theta = FrozenTheta(36, memoryview(data))
    candidate = Candidate.bind(theta)
    original = candidate.materialize()
    data[:] = b"\0" * len(data)
    with pytest.raises(TypeError):
        Candidate(theta, [Status.ACTIVE] * 36)
    for obj, name, value in ((theta, "data", b""), (theta, "width", 72),
                             (candidate, "mask", ()), (candidate, "theta", None)):
        with pytest.raises((FrozenInstanceError, AttributeError)): setattr(obj, name, value)
    with pytest.raises(TypeError): theta.data[0] = 0
    with pytest.raises(TypeError): candidate.mask[0] = Status.DISABLED
    materialized = bytearray(candidate.materialize())
    materialized[:] = b"\0" * len(materialized)
    assert theta.data == candidate.materialize() == original


def test_permutation_preserves_bytes_status_and_reverse_addresses(candidate):
    candidate = edit(candidate, Opcode.DISABLE_COLUMN, 2).candidate
    moved, addresses = permute(candidate, tuple(reversed(range(36))))
    assert equivalent(candidate, moved)
    assert candidate.equivalence_digest == moved.equivalence_digest
    assert candidate.digest != moved.digest
    for old, new in enumerate(addresses):
        assert candidate.theta.column(old) == moved.theta.column(new)
        assert candidate.mask[old] is moved.mask[new]
    with pytest.raises(R0Error, match="STALE_ADDRESS"):
        mutate(moved, MutationRequest(Opcode.DISABLE_COLUMN, moved.digest, candidate.address(0)))


@pytest.mark.parametrize("order", [tuple([0] * 36), tuple(range(35)),
    tuple(range(35)) + (True,), tuple(range(35)) + (36,)])
def test_duplicate_missing_malformed_addresses(candidate, order):
    with pytest.raises(R0Error, match="PERMUTATION"): permute(candidate, order)


def test_duplicate_column_multiplicity_and_signed_zero():
    raw = b"".join(struct.pack("<d", float(row + 1)) * 36 for row in range(6))
    candidate = Candidate.bind(FrozenTheta(36, raw))
    left = edit(candidate, Opcode.DISABLE_COLUMN, 0).candidate
    right = edit(candidate, Opcode.DISABLE_COLUMN, 1).candidate
    assert equivalent(left, right)
    assert not equivalent(left, candidate)
    changed = Candidate.bind(FrozenTheta(36, b"\0" * 8 + raw[8:]))
    negative = Candidate.bind(FrozenTheta(36, struct.pack("<d", -0.0) + raw[8:]))
    assert not equivalent(changed, negative)


def test_non_equivalent_masks_never_alias_in_bounded_exhaustive_space(candidate):
    canonical, identities = set(), set()
    for bits in itertools.product((Status.ACTIVE, Status.DISABLED), repeat=6):
        state = candidate
        for slot, status in enumerate(bits):
            if status is Status.DISABLED:
                state = edit(state, Opcode.DISABLE_COLUMN, slot).candidate
        canonical.add(state.canonical_state)
        identities.add(state.digest)
    assert len(canonical) == len(identities) == 64


def test_no_observation_or_optimization_identity_surface(candidate):
    assert set(Candidate.__dataclass_fields__) == {"theta", "mask"}
    assert set(MutationRequest.__dataclass_fields__) == {"op", "address", "expected_candidate_digest"}
    assert {op.value for op in Opcode} == {"ABSTAIN", "DISABLE_COLUMN", "RESTORE_COLUMN"}
    observation = {"S3": [0] * 83}
    digest = candidate.digest
    observation["S3"][0] = 10
    assert candidate.digest == digest
    disabled = edit(candidate, Opcode.DISABLE_COLUMN, 4).candidate
    assert disabled.theta is candidate.theta
    assert all(disabled.mask[i] is candidate.mask[i] for i in range(36) if i != 4)


def test_packet_positive_and_abstain_address_negative(candidate):
    packet = dict(op="DISABLE_COLUMN", address=asdict(candidate.address(0)),
                  expected_candidate_digest=candidate.digest)
    assert mutate(candidate, MutationRequest.from_packet(packet)).changed
    with pytest.raises(R0Error, match="EDIT_ADDRESS"):
        MutationRequest.from_packet({**packet, "op": "ABSTAIN"})


def test_fresh_process_digests_and_materialization(candidate):
    code = (
        "import hashlib; from test_structural import theta_bytes; "
        "from elpis.ECS_C.structural import Candidate,FrozenTheta; "
        "c=Candidate.bind(FrozenTheta(36,theta_bytes())); "
        "print(c.digest,c.equivalence_digest,hashlib.sha256(c.materialize()).hexdigest())"
    )
    expected = " ".join((candidate.digest, candidate.equivalence_digest,
                         hashlib.sha256(candidate.materialize()).hexdigest()))
    for seed in ("1", "77"):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=seed,
                   PYTHONPATH=os.pathsep.join((str(IMPORT_ROOT), str(TESTS_ROOT))))
        assert subprocess.check_output([sys.executable, "-B", "-c", code],
                                       cwd=TESTS_ROOT, env=env, text=True).strip() == expected


@pytest.mark.parametrize("field", ["ontology_id", "ontology_version",
    "primitive_contract_digest", "ordered_frozen_sidecar_digest",
    "operational_gauge_id", "expected_pre_status", "pre_state_equivalence_digest"])
def test_address_string_fields_are_strictly_typed(candidate, field):
    class EqualToAnything:
        def __eq__(self, other): return True
    address = replace(candidate.address(0), **{field: EqualToAnything()})
    with pytest.raises(R0Error, match="ADDRESS_FIELD_TYPE"):
        mutate(candidate, MutationRequest(Opcode.DISABLE_COLUMN, candidate.digest, address))


# Explicit bounded decoder census at the largest supported width.
#
# Address domain = {None} U {0, ..., 71}
# Opcode domain  = {ABSTAIN, DISABLE_COLUMN, RESTORE_COLUMN}
#
# Therefore the complete structural decode surface is 3 * 73 = 219 rows.
_DECODE_TOTALITY_ROWS_72 = tuple(
    (opcode, slot)
    for opcode in tuple(Opcode)
    for slot in (None, *range(72))
)


@pytest.fixture(scope="module")
def decode_candidate72():
    return Candidate.bind(
        FrozenTheta(72, theta_bytes(72))
    )


@pytest.mark.parametrize(
    ("opcode", "slot"),
    _DECODE_TOTALITY_ROWS_72,
    ids=[
        f"{opcode.value}-{'none' if slot is None else slot}"
        for opcode, slot in _DECODE_TOTALITY_ROWS_72
    ],
)
def test_mutation_packet_decode_totality_72(
    decode_candidate72, opcode, slot
):
    """Every bounded opcode/address pair has one deterministic decode result."""
    candidate = decode_candidate72

    packet = {
        "op": opcode.value,
        "expected_candidate_digest": candidate.digest,
        "address": (
            None
            if slot is None
            else asdict(candidate.address(slot))
        ),
    }

    should_decode = (
        (opcode is Opcode.ABSTAIN and slot is None)
        or
        (opcode is not Opcode.ABSTAIN and slot is not None)
    )

    if not should_decode:
        with pytest.raises(R0Error, match="EDIT_ADDRESS"):
            MutationRequest.from_packet(packet)
        return

    request = MutationRequest.from_packet(packet)

    assert request.op is opcode
    assert request.expected_candidate_digest == candidate.digest

    if slot is None:
        assert request.address is None
    else:
        assert request.address == candidate.address(slot)
