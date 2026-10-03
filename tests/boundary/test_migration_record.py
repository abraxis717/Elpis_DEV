"""migration/BETA_MIGRATION.json is complete, consistent and points at real files."""
from __future__ import annotations

from ._system import MIGRATION, REPO, SYSTEM

LANDED = {"MIGRATED", "EXTRACTED_PRIMITIVE", "COLLAPSED_INTO_SUBSYSTEM", "TEST_FIXTURE_ONLY"}
FIELDS = {
    "donor_repository", "donor_commit", "donor_path", "destination", "disposition",
    "migration_type", "semantic_code_changed", "reason",
}


def test_record_header_matches_system_authority():
    assert MIGRATION["schema"] == "elpis.beta-migration.v1"
    donor = MIGRATION["donor"]
    assert donor["repository"] == SYSTEM["donor"]["repository"]
    assert donor["migration_basis_commit"] == SYSTEM["donor"]["migration_basis_commit"]
    assert donor["migration_basis_tree"] == SYSTEM["donor"]["migration_basis_tree"]
    assert donor["donor_mutated"] is False


def test_every_record_is_well_formed():
    allowed = set(MIGRATION["dispositions"])
    for record in MIGRATION["records"]:
        assert set(record) == FIELDS, record
        assert record["donor_commit"] == SYSTEM["donor"]["migration_basis_commit"]
        assert record["disposition"] in allowed, record
        assert record["reason"].strip(), record
        if record["disposition"] in LANDED:
            assert record["destination"], record
        else:
            assert record["destination"] is None, record
            assert record["migration_type"] == "NOT_MIGRATED", record


def _current_destination(destination: str) -> str:
    """Resolve an historical migration destination through later declared relocations."""
    relocations = SYSTEM.get("post_migration_relocations", [])
    matches = sorted(
        (
            (entry["from"], entry["to"])
            for entry in relocations
            if isinstance(entry, dict)
            and type(entry.get("from")) is str
            and type(entry.get("to")) is str
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for old, new in matches:
        if destination == old:
            return new
        if destination.startswith(old + "/"):
            return new + destination[len(old):]
    return destination


def test_landed_destinations_exist():
    missing = []
    for record in MIGRATION["records"]:
        if record["disposition"] in LANDED:
            for dest in record["destination"].split(" + "):
                historical = dest.strip()
                current = _current_destination(historical)
                if not (REPO / current).exists():
                    missing.append({"historical": historical, "current": current})
    assert not missing, missing


def test_post_migration_relocations_are_real():
    for entry in SYSTEM.get("post_migration_relocations", []):
        assert set(entry) == {"from", "to", "reason"}
        assert entry["from"] != entry["to"]
        assert entry["reason"].strip()
        assert (REPO / entry["to"]).exists(), entry
