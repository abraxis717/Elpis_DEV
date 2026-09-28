from __future__ import annotations

import importlib

from elpis.structure.grid81.semantics import D4, canonical_bytes, transform_index

def _semantics_permutations() -> list[tuple[int, ...]]:
    return [
        tuple(transform_index(index, element) for index in range(81))
        for element in D4
    ]


def test_consumer_packages_import_alongside_semantics():
    imported = {
        "typed_projection": importlib.import_module("elpis.structure.grid81.typed"),
        "structural_group_projection": importlib.import_module("elpis.structure.grid81.groups"),
        "deterministic_adjudicator": importlib.import_module(
            "elpis.pipeline.adjudication"
        ),
    }
    assert all(module is not None for module in imported.values())


def test_typed_and_group_d4_actions_are_set_equivalent_to_semantics():
    typed_d4 = importlib.import_module("elpis.structure.grid81.typed.d4")
    group_orbit = importlib.import_module("elpis.structure.grid81.groups.orbit")

    semantics_perms = _semantics_permutations()
    typed_perms = [tuple(perm) for perm in typed_d4.D4_TRANSFORMS]
    group_perms = [tuple(perm) for perm in group_orbit._load_d4_transforms()]

    assert len(set(semantics_perms)) == 8
    assert set(typed_perms) == set(semantics_perms)
    assert set(group_perms) == set(semantics_perms)


def test_consumer_reflection_slot_mapping_is_explicit_not_conflated():
    typed_d4 = importlib.import_module("elpis.structure.grid81.typed.d4")
    group_orbit = importlib.import_module("elpis.structure.grid81.groups.orbit")

    semantics_perms = _semantics_permutations()
    typed_perms = [tuple(perm) for perm in typed_d4.D4_TRANSFORMS]
    group_perms = [tuple(perm) for perm in group_orbit._load_d4_transforms()]

    # The consumers call slot 4 "fh" and slot 5 "fv" using direction-of-flip
    # terminology. The semantics package names reflections by their mirror axis.
    # The geometric actions are equal after this observed 4 <-> 5 mapping.
    assert typed_perms[:4] == semantics_perms[:4]
    assert group_perms[:4] == semantics_perms[:4]
    assert typed_perms[4] == semantics_perms[5]
    assert typed_perms[5] == semantics_perms[4]
    assert group_perms[4] == semantics_perms[5]
    assert group_perms[5] == semantics_perms[4]
    assert typed_perms[6:] == semantics_perms[6:]
    assert group_perms[6:] == semantics_perms[6:]


def test_serialized_semantics_pair_crosses_consumer_canonical_boundaries(
    valid_pair_payload,
):
    typed_canonical = importlib.import_module("elpis.structure.grid81.typed.canonical")
    group_canonical = importlib.import_module("elpis.structure.grid81.groups.canonical")
    adjudication_canonical = importlib.import_module(
        "elpis.pipeline.adjudication.canonical"
    )

    payload = valid_pair_payload.to_dict()
    semantics_value = canonical_bytes(payload)
    typed_value = typed_canonical.canonicalize(payload)
    group_value = group_canonical.canonical_json_bytes(payload)
    adjudication_value = adjudication_canonical.canonical_json(payload).encode("utf-8")

    assert typed_value == semantics_value
    assert group_value == semantics_value
    assert adjudication_value == semantics_value
