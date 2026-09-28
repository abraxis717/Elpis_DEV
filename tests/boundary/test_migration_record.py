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


def test_landed_destinations_exist():
    missing = []
    for record in MIGRATION["records"]:
        if record["disposition"] in LANDED:
            for dest in record["destination"].split(" + "):
                if not (REPO / dest.strip()).exists():
                    missing.append(dest)
    assert not missing, missing
