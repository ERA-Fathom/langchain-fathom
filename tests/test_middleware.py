"""Middleware behavior with a stand-in AgentMiddleware base and a stubbed read."""
import langchain_fathom.middleware as mw
from langchain_fathom import FathomMiddleware, FathomCoherenceError


class AI:
    def __init__(self, tool_calls): self.tool_calls = tool_calls


def _verdict(coherent, findings):
    from fathom_read.ops import Verdict
    return Verdict.from_dict({"coherent": coherent, "findings": findings, "ops_read": 3, "ops_rejected": 0, "live_facts": 1})


def test_raise_on_finding(monkeypatch):
    monkeypatch.setattr(mw, "read", lambda *a, **k: _verdict(False, [
        {"kind": "superseded_value", "step": 2, "key": "user.city", "detail": "step 2 answers 'Denver'...", "cites": 1}]))
    m = FathomMiddleware(on_finding="raise")
    state = {"messages": [AI([{"name": "set_value", "args": {"key": "user.city", "value": "Denver"}, "id": "a"}])]}
    try:
        m.after_agent(state)
        assert False, "expected FathomCoherenceError"
    except FathomCoherenceError as e:
        assert "superseded_value" in str(e)


def test_store_on_finding(monkeypatch):
    monkeypatch.setattr(mw, "read", lambda *a, **k: _verdict(True, []))
    m = FathomMiddleware(on_finding="store")
    state = {"messages": [AI([{"name": "set_value", "args": {"key": "a", "value": "1"}, "id": "x"}])]}
    out = m.after_agent(state)
    assert out["fathom"]["coherent"] is True


def test_no_tool_calls_returns_none(monkeypatch):
    monkeypatch.setattr(mw, "read", lambda *a, **k: _verdict(True, []))
    m = FathomMiddleware()
    assert m.after_agent({"messages": []}) is None
