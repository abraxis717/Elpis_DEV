from __future__ import annotations
from dataclasses import replace
from elpis.inference.context import initial_snapshot
from elpis.inference.transaction import InferenceRequest, InferenceEngine

def test_fresh_runtime_accepts_matching_numerical_profile(target):
    rt=InferenceEngine(target[0]); state=rt.initial(initial_snapshot())
    committed=rt.execute(
        state,InferenceRequest("p",state.context.digest,"PREFILL",(1,2,3)),
        expected_state=state.digest
    ).state
    restarted=InferenceEngine(target[0])
    result=restarted.execute(
        committed,InferenceRequest("g",committed.context.digest,"GREEDY",count=1),
        expected_state=committed.digest
    )
    assert result.receipt.terminal=="COMMITTED"

def test_fresh_runtime_rejects_mismatched_numerical_profile_before_replay(target):
    rt=InferenceEngine(target[0]); state=rt.initial(initial_snapshot())
    committed=rt.execute(
        state,InferenceRequest("p",state.context.digest,"PREFILL",(1,2,3)),
        expected_state=state.digest
    ).state
    forged=replace(committed,neural=replace(committed.neural,numerical_profile="0"*64))
    restarted=InferenceEngine(target[0])
    result=restarted.execute(
        forged,InferenceRequest("g",forged.context.digest,"GREEDY",count=1),
        expected_state=forged.digest
    )
    assert result.state==forged
    assert result.receipt.terminal=="FAILED"
    assert result.receipt.failure=="UNSUPPORTED"
    assert forged.digest not in restarted._validated_states


def test_numerical_profile_v2_is_yaml_stdout_and_processor_independent(monkeypatch,capsys):
    import platform
    import sys
    import types
    import warnings
    from elpis.inference.target import numerical_profile

    def forbidden_processor():
        raise AssertionError("platform.processor must not participate in numerical identity")

    monkeypatch.setattr(platform,"processor",forbidden_processor)
    monkeypatch.setitem(sys.modules,"yaml",None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first=numerical_profile()
    first_io=capsys.readouterr()
    assert first_io.out=="" and first_io.err==""
    assert not [w for w in caught if issubclass(w.category,UserWarning)]

    stub=types.ModuleType("yaml")
    stub.dump=lambda *args,**kwargs: "environment-dependent-format"
    monkeypatch.setitem(sys.modules,"yaml",stub)
    second=numerical_profile()
    second_io=capsys.readouterr()

    assert second==first
    assert second_io.out=="" and second_io.err==""
