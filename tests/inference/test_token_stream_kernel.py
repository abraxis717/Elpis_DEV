# DSV4 stream kernel: numeric parity without canonical identity work.
import numpy as np

import elpis.identity as canonical
import elpis.substrate.digests as digests
import elpis.inference.associative as associative
import elpis.inference.global_context as global_context
import elpis.inference.rows as rows
import elpis.inference.experts as experts
import elpis.inference.target as target_module
import elpis.inference.drivers.dsv4.target as dsv4


def _numeric_equal(stream, legacy):
    assert stream.tokens == legacy.tokens
    assert stream.history == legacy.history
    assert stream.local_keys == legacy.local_keys
    assert stream.local_values == legacy.local_values
    assert stream.pending == legacy.pending
    assert len(stream.global_pool) == len(legacy.global_pool)
    for a,b in zip(stream.global_pool, legacy.global_pool):
        assert (a.object_id,a.end_position,a.key,a.value) == (b.object_id,b.end_position,b.key,b.value)
    assert np.array_equal(np.asarray(stream.hidden,dtype="<f4"),np.asarray(legacy.hidden,dtype="<f4"))
    assert np.array_equal(np.asarray(stream.logits,dtype="<f4"),np.asarray(legacy.logits,dtype="<f4"))


def test_stream_step_numeric_parity_and_no_canonical_identity(target, monkeypatch):
    model=target[0]
    context="1"*64

    legacy=model.initial(context)
    expected=[]
    for token in (1,2,3,4,5,6):
        legacy,_=model.step(legacy,token,expected_state=legacy.digest)
        expected.append(legacy)

    stream=model.initial_stream(context)

    def forbidden(*args,**kwargs):
        raise AssertionError("canonical/content identity executed in stream_step")

    monkeypatch.setattr(canonical,"content_digest",forbidden)
    monkeypatch.setattr(digests,"content_digest",forbidden)
    for module in (associative,global_context,rows,experts,target_module,dsv4):
        if hasattr(module,"identity"):
            monkeypatch.setattr(module,"identity",forbidden)

    assert not hasattr(stream,"digest")
    for token,legacy_state in zip((1,2,3,4,5,6),expected):
        stream=model.stream_step(stream,token)
        assert not hasattr(stream,"digest")
        _numeric_equal(stream,legacy_state)
