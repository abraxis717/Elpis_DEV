"""Frozen synthetic DS4/vLLM oracle cases; provenance in InferenceInfrastructure."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from elpis.inference.associative import AddressScheme, load_parameters
from elpis.inference.contracts import Bank, ContractError

FIX = Path(__file__).resolve().parents[1] / 'fixtures' / 'inference'


def fixture(name):
    f = json.loads((FIX / (name + '.json')).read_text())
    p = load_parameters(f['parameters'], expected_digest=f['parameter_digest'])
    return f, p, AddressScheme(p, expected_digest=p.digest, tokenizer=p.tokenizer, scheme=p.schema)


@pytest.mark.parametrize('name', ['dsv41', 'qwen38'])
def test_frozen_oracle_and_every_split(name):
    f, p, s = fixture(name)
    tokens, mask = tuple(f['tokens']), tuple(f['mask'])
    h = s.initial()
    result = s.hash(h, tokens, mask, expected_history=h.digest)
    assert json.loads(json.dumps(result.rows)) == f['golden_rows']
    for split in range(len(tokens) + 1):
        a = s.hash(h, tokens[:split], mask[:split], expected_history=h.digest)
        b = s.hash(a.history, tokens[split:], mask[split:], expected_history=a.history.digest)
        assert a.rows + b.rows == result.rows
        assert b.history == result.history
    state, rows = h, ()
    for token, active in zip(tokens, mask):
        r = s.hash(state, (token,), (active,), expected_history=state.digest)
        state, rows = r.history, rows + r.rows
    assert rows == result.rows and state == result.history


@pytest.mark.parametrize('name', ['dsv41', 'qwen38'])
def test_identity_and_token_defects(name):
    _, p, s = fixture(name)
    for key, bad in [('expected_digest', '0'*64), ('tokenizer', 'wrong'), ('scheme', 'DS_ENGRAM_DEMO/v1')]:
        args = dict(expected_digest=p.digest, tokenizer=p.tokenizer, scheme=p.schema)
        args[key] = bad
        with pytest.raises(ContractError, match='IDENTITY'):
            AddressScheme(p, **args)
    h = s.initial()
    for token in (-1, 10**30, True, 1.0):
        with pytest.raises(ContractError):
            s.hash(h, (token,), expected_history=h.digest)
    with pytest.raises(ContractError, match='STALE'):
        s.hash(h, (1,), expected_history='0'*64)
    with pytest.raises(ContractError, match='IDENTITY'):
        bad = replace(h, parameters='0'*64)
        s.hash(bad, (1,), expected_history=bad.digest)


def test_planted_parameter_defects():
    f, p, _ = fixture('dsv41')
    for change in [dict(pad=8), dict(dead=-2), dict(token_map=(9,)*16),
                   dict(compressed_vocab=9), dict(layers=(1,1)), dict(head_dimension=0),
                   dict(multipliers=((2,5,7,11),p.multipliers[1])),
                   dict(primes=(p.primes[0],p.primes[0])),
                   dict(offsets=(tuple(reversed(p.offsets[0])),p.offsets[1])),
                   dict(table_rows=(1,2))]:
        with pytest.raises(ContractError):
            replace(p, **change)
    # A valid-but-different odd multiplier is rejected by the frozen artifact pin.
    changed = replace(p, multipliers=((1,5,7,11),p.multipliers[1]))
    with pytest.raises(ContractError, match='IDENTITY'):
        AddressScheme(changed, expected_digest=p.digest, tokenizer=p.tokenizer, scheme=p.schema)
    artifact = dict(f['parameters'])
    del artifact['multipliers']
    with pytest.raises(ContractError):
        load_parameters(artifact, expected_digest=p.digest)


def test_dead_barrier_and_eos_reset_are_not_zero_substitution():
    _, p, s = fixture('dsv41')
    h = s.initial()
    r = s.hash(h, (7,3,4), (True,False,True), expected_history=h.digest)
    fresh = s.hash(h, (4,), expected_history=h.digest)
    assert r.rows[-1] == fresh.rows[-1]
    assert not r.active[1]
    # A mutation dropping dead-token blocking must alter the frozen row sequence.
    bad = s.hash(h, (7,3,4), expected_history=h.digest)
    assert bad.rows != r.rows
    _, p, s = fixture('qwen38')
    h = s.initial()
    r = s.hash(h, (8,p.eos,4), expected_history=h.digest)
    assert r.rows[-1] == s.hash(h,(4,),expected_history=h.digest).rows[-1]
    with pytest.raises(ContractError, match='UNSUPPORTED'):
        s.hash(h,(4,),(False,),expected_history=h.digest)


def test_bank_identity_geometry():
    _, p, s = fixture('dsv41')
    bank = Bank('bank','model',p.tokenizer,p.schema,p.digest,p.layers[0],p.table_rows[0],p.head_dimension,'1'*64,'2'*64)
    s.validate_bank(bank)
    for bad in [replace(bank,parameters='3'*64),replace(bank,tokenizer='wrong'),
                replace(bank,layer=99),replace(bank,rows=1),replace(bank,dimension=1)]:
        with pytest.raises(ContractError):
            s.validate_bank(bad)
