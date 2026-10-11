"""A codec may describe itself; it may not authorize itself (elpis.runtime.codec_authority).

Every attack below is refused before the codec is even asked to encode, so before any ECS call: the complete K1
retained state and (managed) every continuity slot byte are unchanged. The admitted maps are TEST_ONLY TRAINING=NONE
SEMANTICS=NONE interface fixtures; nothing here is a codec or a claim about meaning.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from elpis.runtime import Runtime
from elpis.runtime.codec_authority import (CODEC_AUTHORITY_SCHEMA, QUALIFIED_CODEC_RECORDS, AdmittedCodec,
                                           CodecAuthority, admit_codec)
from elpis.runtime.cognition import LearnRequest, QueryRequest, run_learn, run_query, run_turn
from elpis.runtime.composition import CompositionError

from ._turn_fixtures import (FIXTURE, LEARN, TEST_CODEC_AUTHORITY, ByteTokens, FixtureMap, admitted, catalog,
                             codec_id, fixture_authority, fixture_entry)
from .test_codec_ecs_turn import _config, k1, world  # noqa: F401 (module fixture)

QUERY_ONLY = ("QUERY_ENCODE", "QUERY_DECODE")
LEARN_ONLY = ("LEARN_ENCODE",)


class ForgedQualified(FixtureMap):
    """A fixture that claims a qualification it does not have."""
    classification = "QUALIFIED ECS<->DSV semantic codec"


def _ops(codec):
    """Every cognitive operation with ``codec``: (name, unmanaged call)."""
    return (("query", lambda s: run_query(s, QueryRequest("q", ByteTokens(), codec))),
            ("learn", lambda s: run_learn(s, LearnRequest("l", ByteTokens(), codec, LEARN))),
            ("legacy", lambda s: run_turn(s, "t", tokenizer=ByteTokens(), codec=codec, authority=LEARN)))


def _refused_everywhere(k1, codec, code, fixture, ops=("query", "learn", "legacy")):
    with world(k1) as state:
        before = state.snapshot()
        for name, call in _ops(codec):
            if name not in ops:
                continue
            with pytest.raises(CompositionError) as info:
                call(state)
            assert info.value.code == code, (name, info.value)
            assert state.snapshot() == before and state.epoch == 0
        assert fixture.calls == []   # refused before the codec was asked to encode


# --- the authority is independent of the candidate ---------------------------------------------------------------

def test_a_raw_codec_map_never_runs_whatever_it_declares(k1):
    for fixture in (FixtureMap(), ForgedQualified()):
        _refused_everywhere(k1, fixture, "CODEC_UNADMITTED", fixture)


def test_an_admitted_codec_cannot_be_forged_outside_admission():
    with pytest.raises(CompositionError) as info:
        AdmittedCodec(object(), FixtureMap(), TEST_CODEC_AUTHORITY, None, ())
    assert info.value.code == "CODEC_UNADMITTED"
    codec = admitted(FixtureMap())
    with pytest.raises(AttributeError):
        codec.entry = None
    with pytest.raises(AttributeError):
        TEST_CODEC_AUTHORITY.digest = "0" * 64


def test_forged_classification_is_refused(k1):
    forged = ForgedQualified()
    # The catalog admits the class only under the classification the authority fixes; the claim is not it.
    authority = catalog([fixture_entry(forged, identifier="forged")])
    with pytest.raises(CompositionError) as info:
        admit_codec(forged, authority, "forged")
    assert info.value.code == "CODEC_CLASSIFICATION"
    # A classification changed after admission is refused at the operation, before encode.
    fixture = FixtureMap()
    codec = admitted(fixture)
    fixture.classification = "QUALIFIED"
    _refused_everywhere(k1, codec, "CODEC_CLASSIFICATION", fixture)
    # A catalog cannot launder the claim either.
    for entry in (dict(classification="QUALIFIED TEST_ONLY TRAINING=NONE SEMANTICS=NONE"),
                  dict(classification="TEST_ONLY TRAINING=NONE (no semantics marker)")):
        with pytest.raises(CompositionError) as info:
            catalog([fixture_entry(FixtureMap(), **entry)])
        assert info.value.code == "CODEC_AUTHORITY"


def test_no_catalog_can_admit_a_qualified_codec_without_a_pinned_record():
    assert dict(QUALIFIED_CODEC_RECORDS) == {}   # no ECS<->DSV semantic codec is qualified
    entry = fixture_entry(FixtureMap(), classification="QUALIFIED")
    entry.update(qualification="QUALIFIED", qualification_record="research/some_codec/qual.json")
    for provenance in ("deployment", "test-fixture"):
        with pytest.raises(CompositionError) as info:
            catalog([entry], provenance=provenance)
        assert info.value.code == "CODEC_UNQUALIFIED"
    research = fixture_entry(FixtureMap(), classification="RESEARCH_ONLY candidate")
    research["qualification"] = "RESEARCH_ONLY"
    with pytest.raises(CompositionError) as info:
        catalog([research], provenance="test-fixture")    # a test catalog admits TEST_ONLY only
    assert info.value.code == "CODEC_AUTHORITY"
    assert catalog([research], provenance="deployment").codecs[research["codec_id"]].qualification.value == \
        "RESEARCH_ONLY"


def test_an_unknown_codec_identity_is_refused():
    fixture = FixtureMap()
    with pytest.raises(CompositionError) as info:
        admit_codec(fixture, TEST_CODEC_AUTHORITY, "test-fixture:nobody")
    assert info.value.code == "CODEC_UNKNOWN" and fixture.calls == []


def test_the_right_identity_with_the_wrong_bytes_is_refused(k1):
    admitted_fixture = FixtureMap()
    other = FixtureMap(steps=99)        # different parameter bytes presented under the admitted identity
    with pytest.raises(CompositionError) as info:
        admit_codec(other, TEST_CODEC_AUTHORITY, codec_id(admitted_fixture))
    assert info.value.code == "CODEC_IDENTITY"
    # Wrong implementation source bytes pinned by the authority.
    entry = fixture_entry(admitted_fixture)
    entry["implementation_sha256"] = hashlib.sha256(b"another implementation").hexdigest()
    with pytest.raises(CompositionError) as info:
        admit_codec(admitted_fixture, catalog([entry]), entry["codec_id"])
    assert info.value.code == "CODEC_IDENTITY"
    # Another class presented under the fixture's entry.
    forged = ForgedQualified()
    with pytest.raises(CompositionError) as info:
        admit_codec(forged, fixture_authority(admitted_fixture), codec_id(admitted_fixture))
    assert info.value.code == "CODEC_IDENTITY"
    # After admission: mutated parameters, an instance override, a class-level patch.
    fixture = FixtureMap()
    codec = admitted(fixture)
    fixture.steps = 1000
    _refused_everywhere(k1, codec, "CODEC_IDENTITY", fixture)
    fixture = FixtureMap()
    codec = admitted(fixture)
    fixture.encode = lambda tokens: None
    fixture.encode_query = lambda tokens: None
    _refused_everywhere(k1, codec, "CODEC_IDENTITY", fixture)

    class Patched(FixtureMap):
        pass
    patched = Patched()
    codec = admitted(patched)
    Patched.decode_query = lambda self, readout: (1,)
    _refused_everywhere(k1, codec, "CODEC_IDENTITY", patched)


def test_the_right_bytes_under_the_wrong_authority_are_refused(k1, tmp_path):
    fixture = FixtureMap()
    with pytest.raises(CompositionError) as info:   # a document that is not its pin
        CodecAuthority(TEST_CODEC_AUTHORITY._document + b" ", expected_sha256=TEST_CODEC_AUTHORITY.digest)
    assert info.value.code == "CODEC_AUTHORITY"
    own = admitted(fixture, fixture_authority(fixture))   # correct bytes, self-made authority
    with world(k1) as state, Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(state)
        before, durable = state.snapshot(), runtime.continuity.snapshot()
        slots = tuple((runtime.continuity.directory / n).read_bytes() for n in ("continuity.a", "continuity.b"))
        for call in (lambda: runtime.run_query(state, QueryRequest("q", ByteTokens(), own)),
                     lambda: runtime.run_learn(state, LearnRequest("l", ByteTokens(), own, LEARN)),
                     lambda: runtime.run_turn(state, "t", tokenizer=ByteTokens(), codec=own, authority=LEARN)):
            with pytest.raises(CompositionError) as info:
                call()
            assert info.value.code == "CODEC_AUTHORITY"
        assert state.snapshot() == before and runtime.continuity.snapshot() == durable
        assert slots == tuple((runtime.continuity.directory / n).read_bytes() for n in ("continuity.a", "continuity.b"))
        assert fixture.calls == [] and runtime.fault is None
        # The same codec admitted by the deployment-pinned authority runs.
        assert runtime.run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap()))).text == "ok"


def test_a_runtime_without_a_deployment_pinned_codec_authority_admits_no_codec(k1, tmp_path):
    from dataclasses import replace
    config = replace(_config(tmp_path / "c"), codec_authority_sha256=None)
    fixture = FixtureMap()
    with world(k1) as state, Runtime(config) as runtime:
        runtime.anchor_cognition(state)
        with pytest.raises(CompositionError) as info:
            runtime.run_query(state, QueryRequest("q", ByteTokens(), admitted(fixture)))
        assert info.value.code == "CODEC_AUTHORITY" and fixture.calls == []
        with pytest.raises(CompositionError) as info:   # fail closed still comes first without any codec
            runtime.run_query(state, QueryRequest("q", ByteTokens()))
        assert info.value.code == "ECS_CODEC_UNQUALIFIED"


def test_a_query_only_codec_never_learns_and_a_learn_only_codec_never_answers(k1):
    fixture = FixtureMap()
    query_only = admitted(fixture, fixture_authority(fixture, capabilities=QUERY_ONLY))
    _refused_everywhere(k1, query_only, "CODEC_CAPABILITY", fixture, ops=("learn", "legacy"))
    with world(k1) as state:
        assert run_query(state, QueryRequest("q", ByteTokens(), query_only)).text == "ok"
    fixture = FixtureMap()
    learn_only = admitted(fixture, fixture_authority(fixture, capabilities=LEARN_ONLY))
    _refused_everywhere(k1, learn_only, "CODEC_CAPABILITY", fixture, ops=("query", "legacy"))   # legacy decodes
    with world(k1) as state:
        assert run_learn(state, LearnRequest("l", ByteTokens(), learn_only, LEARN)).epoch_after == 6


def test_a_revoked_codec_is_refused():
    fixture = FixtureMap()
    authority = fixture_authority(fixture, status="REVOKED")
    with pytest.raises(CompositionError) as info:
        admit_codec(fixture, authority, codec_id(fixture))
    assert info.value.code == "CODEC_REVOKED"


def test_results_report_the_admitted_classification(k1):
    with world(k1) as state:
        result = run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap())))
        assert result.codec == FIXTURE and "TEST_ONLY" in result.codec and "SEMANTICS=NONE" in result.codec


def test_the_catalog_schema_is_strict():
    good = json.loads(TEST_CODEC_AUTHORITY._document)
    for mutate in (lambda d: d.update(schema="elpis.ecs-codec-authority.v0"),
                   lambda d: d.update(provenance="anyone"),
                   lambda d: d.update(extra=1),
                   lambda d: d["codecs"][0].update(capabilities=["QUERY_ENCODE", "QUERY_ENCODE"]),
                   lambda d: d["codecs"][0].update(capabilities=["TELEPATHY"]),
                   lambda d: d["codecs"][0].update(status="PENDING"),
                   lambda d: d["codecs"].append(dict(d["codecs"][0])),
                   lambda d: d["codecs"][0].update(parameters_sha256="F" * 64)):
        data = json.loads(json.dumps(good))
        mutate(data)
        document = json.dumps(data).encode()
        with pytest.raises(CompositionError) as info:
            CodecAuthority(document, expected_sha256=hashlib.sha256(document).hexdigest())
        assert info.value.code == "CODEC_AUTHORITY"
    assert good["schema"] == CODEC_AUTHORITY_SCHEMA
