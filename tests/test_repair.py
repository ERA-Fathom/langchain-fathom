"""The repair middleware, against a stubbed service and a stub model call."""
import pytest

import langchain_fathom.repair as R
from langchain_fathom import FathomRepairMiddleware, FathomKeyError, FathomRepairError, REPAIR_STATE_KEY

from fathom_read.client import ReadError
from fathom_read.ops import RegroundVerdict


class AI:
    """Enough of an AIMessage for the middleware: tool calls and a pydantic-style copy."""

    def __init__(self, tool_calls):
        self.tool_calls = list(tool_calls)

    def model_copy(self, update):
        return AI(update["tool_calls"])


class Response:
    def __init__(self, result):
        self.result = list(result)


class Request:
    def __init__(self, messages):
        self.messages = list(messages)
        self.overridden_with = None

    def override(self, **kw):
        out = Request(kw.get("messages", self.messages))
        out.overridden_with = kw
        return out


def call(name, key, value, cid):
    return {"name": name, "args": {"key": key, "value": value}, "id": cid, "type": "tool_call"}


def verdict(decision, kept=(), dropped=(), facts=(), proposals=0):
    return RegroundVerdict.from_dict({
        "decision": decision, "keep": [k.get("value") for k in kept], "kept": list(kept),
        "dropped": list(dropped), "facts": list(facts), "proposals": proposals,
        "read": {"coherent": True, "findings": 0, "ops_read": 0, "live_facts": 0}})


@pytest.fixture
def mw(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "k-test")
    return FathomRepairMiddleware(record=False)


# -- the key ---------------------------------------------------------------------------

def test_key_is_required_at_construction(monkeypatch):
    monkeypatch.delenv("FATHOM_API_KEY", raising=False)
    with pytest.raises(FathomKeyError) as e:
        FathomRepairMiddleware()
    assert "fathom key" in str(e.value)


def test_key_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "from-env")
    assert FathomRepairMiddleware(key="explicit").key == "explicit"


def test_bad_arguments_are_refused(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "k")
    with pytest.raises(ValueError):
        FathomRepairMiddleware(on_error="explode")
    with pytest.raises(ValueError):
        FathomRepairMiddleware(max_reasks=-1)


# -- the three decisions ---------------------------------------------------------------

def test_proceed_leaves_the_response_alone(mw, monkeypatch):
    monkeypatch.setattr(R, "reground", lambda *a, **k: verdict("proceed", proposals=1))
    ai = AI([call("set_value", "user.city", "Denver", "c1")])
    out = mw.wrap_model_call(Request([]), lambda req: Response([ai]))
    assert out.result[0] is ai


def test_filter_drops_the_contradicting_call_and_keeps_the_alternative(mw, monkeypatch):
    monkeypatch.setattr(R, "reground", lambda *a, **k: verdict(
        "filter",
        kept=[{"index": 1, "op": "set", "kind": "fact", "key": "user.zip", "value": "80202"}],
        dropped=[{"index": 0, "kind": "fact", "key": "user.city", "value": "Denver",
                  "findings": ["duplicate_commit"]}],
        proposals=2))
    calls = [call("set_value", "user.city", "Denver", "c1"),
             call("set_value", "user.zip", "80202", "c2")]
    out = mw.wrap_model_call(Request([]), lambda req: Response([AI(calls)]))
    kept = out.result[0].tool_calls
    assert [c["id"] for c in kept] == ["c2"]


def test_reground_reasks_once_then_proceeds(mw, monkeypatch):
    monkeypatch.setattr(R, "reground", lambda *a, **k: verdict(
        "reground",
        dropped=[{"index": 0, "kind": "fact", "key": "q", "value": "vector db",
                  "findings": ["duplicate_commit"]}],
        facts=[{"kind": "query", "key": "q", "holds": ["vector db"], "because": "duplicate_commit"}],
        proposals=1))
    seen = []

    def handler(req):
        seen.append(req)
        return Response([AI([call("set_value", "q", "vector db", "c1")])])

    mw.wrap_model_call(Request([]), handler)
    assert len(seen) == 2, "one original call and one re-ask"
    note = seen[1].overridden_with["messages"][-1].content
    assert "COMMITTED STATE CHECK" in note
    assert "vector db" in note


def test_max_reasks_zero_never_reasks(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "k")
    m = FathomRepairMiddleware(max_reasks=0, record=False)
    monkeypatch.setattr(R, "reground", lambda *a, **k: verdict("reground", proposals=1))
    seen = []

    def handler(req):
        seen.append(req)
        return Response([AI([call("set_value", "q", "x", "c1")])])

    m.wrap_model_call(Request([]), handler)
    assert len(seen) == 1


# -- what never reaches the service ------------------------------------------------------

def test_a_turn_without_tool_calls_costs_nothing(mw, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the repair should not have been called")

    monkeypatch.setattr(R, "reground", boom)
    out = mw.wrap_model_call(Request([]), lambda req: Response([AI([])]))
    assert out.result[0].tool_calls == []


def test_unmapped_tools_never_become_proposals(mw, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a tool the map does not name is not a proposal")

    monkeypatch.setattr(R, "reground", boom)
    ai = AI([{"name": "totally_unknown_tool", "args": {"x": 1}, "id": "c1", "type": "tool_call"}])
    out = mw.wrap_model_call(Request([]), lambda req: Response([ai]))
    assert out.result[0] is ai


def test_unmapped_tools_survive_a_filter(mw, monkeypatch):
    """The verdict's indices point at proposals, so an unmapped call must not be dropped by index."""
    monkeypatch.setattr(R, "reground", lambda *a, **k: verdict(
        "filter",
        kept=[{"index": 1, "op": "set", "kind": "fact", "key": "user.zip", "value": "80202"}],
        dropped=[{"index": 0, "kind": "fact", "key": "user.city", "findings": ["duplicate_commit"]}],
        proposals=2))
    calls = [{"name": "search_web", "args": {"q": "hi"}, "id": "u1", "type": "tool_call"},
             call("set_value", "user.city", "Denver", "c1"),
             call("set_value", "user.zip", "80202", "c2")]
    out = mw.wrap_model_call(Request([]), lambda req: Response([AI(calls)]))
    assert [c["id"] for c in out.result[0].tool_calls] == ["u1", "c2"]


# -- failure ---------------------------------------------------------------------------

def test_unreachable_repair_lets_the_agent_proceed(mw, monkeypatch):
    def down(*a, **k):
        raise ReadError("could not reach the repair")

    monkeypatch.setattr(R, "reground", down)
    ai = AI([call("set_value", "user.city", "Denver", "c1")])
    out = mw.wrap_model_call(Request([]), lambda req: Response([ai]))
    assert out.result[0] is ai


def test_on_error_raise_stops_the_run(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "k")
    m = FathomRepairMiddleware(on_error="raise", record=False)

    def down(*a, **k):
        raise ReadError("the daily limit for this key is reached")

    monkeypatch.setattr(R, "reground", down)
    with pytest.raises(FathomRepairError):
        m.wrap_model_call(Request([]), lambda req: Response([AI([call("set_value", "a", "1", "c1")])]))


# -- the log ----------------------------------------------------------------------------

def test_the_decision_is_recorded(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "k")
    m = FathomRepairMiddleware()
    monkeypatch.setattr(R, "reground", lambda *a, **k: verdict("proceed", proposals=1))
    out = m.wrap_model_call(Request([]), lambda req: Response([AI([call("set_value", "a", "1", "c1")])]))
    command = getattr(out, "command", None)
    if command is None:  # pragma: no cover - older langchain cannot carry the update
        pytest.skip("this langchain cannot return a state update from wrap_model_call")
    entry = command.update[REPAIR_STATE_KEY][0]
    assert entry["decision"] == "proceed"
    assert entry["reasks"] == 0
