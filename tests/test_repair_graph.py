"""The repair inside a real agent.

A stub stands in for the service, and a scripted fake model stands in for the provider, so this
runs with no network and no key. What is real is the graph: the middleware has to drop a tool call
before that tool executes, and the run's repair log has to survive the graph and reach the caller.
"""
import pytest

pytest.importorskip("langchain.agents", reason="needs langchain>=1.0 installed")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain.agents import create_agent

import langchain_fathom.repair as R
from langchain_fathom import FathomRepairMiddleware, REPAIR_STATE_KEY
from fathom_read.ops import RegroundVerdict

WROTE = []


@tool
def set_value(key: str, value: str) -> str:
    """Write a value to the record."""
    WROTE.append((key, value))
    return "ok"


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _model():
    return ScriptedModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "set_value", "args": {"key": "user.city", "value": "Denver"},
             "id": "dup", "type": "tool_call"},
            {"name": "set_value", "args": {"key": "user.zip", "value": "80202"},
             "id": "new", "type": "tool_call"},
        ]),
        AIMessage(content="done"),
    ]))


def _filter_verdict(*a, **k):
    return RegroundVerdict.from_dict({
        "decision": "filter",
        "keep": ["80202"],
        "kept": [{"index": 1, "op": "set", "kind": "fact", "key": "user.zip", "value": "80202"}],
        "dropped": [{"index": 0, "op": "set", "kind": "fact", "key": "user.city",
                     "value": "Denver", "findings": ["duplicate_commit"]}],
        "facts": [],
        "proposals": 2,
        "read": {"coherent": False, "findings": 1, "ops_read": 0, "live_facts": 0},
    })


def test_the_repair_drops_a_call_before_the_tool_runs(monkeypatch):
    WROTE.clear()
    monkeypatch.setenv("FATHOM_API_KEY", "k-test")
    monkeypatch.setattr(R, "reground", _filter_verdict)

    agent = create_agent(model=_model(), tools=[set_value],
                         middleware=[FathomRepairMiddleware()])
    result = agent.invoke({"messages": [{"role": "user", "content": "write the record"}]})

    assert WROTE == [("user.zip", "80202")], "the contradicting write reached the tool"
    assert REPAIR_STATE_KEY in result, "the graph dropped the repair log"
    log = result[REPAIR_STATE_KEY]
    assert len(log) == 1
    assert log[0]["decision"] == "filter"
    assert log[0]["dropped"][0]["findings"] == ["duplicate_commit"]
    assert log[0]["reasks"] == 0


def test_control_without_the_repair_both_writes_land(monkeypatch):
    """Control: the same agent and the same script, with no repair in the way."""
    WROTE.clear()
    agent = create_agent(model=_model(), tools=[set_value])
    agent.invoke({"messages": [{"role": "user", "content": "write the record"}]})
    assert WROTE == [("user.city", "Denver"), ("user.zip", "80202")]


def test_the_read_and_the_repair_ride_together(monkeypatch):
    """Both middlewares on one agent, each keeping its own declared key."""
    WROTE.clear()
    monkeypatch.setenv("FATHOM_API_KEY", "k-test")
    monkeypatch.setattr(R, "reground", _filter_verdict)

    import langchain_fathom.middleware as M
    from langchain_fathom import FathomMiddleware, STATE_KEY
    from fathom_read.ops import Verdict
    monkeypatch.setattr(M, "read", lambda *a, **k: Verdict.from_dict(
        {"coherent": True, "findings": [], "ops_read": 1, "ops_rejected": 0, "live_facts": 1}))

    agent = create_agent(model=_model(), tools=[set_value],
                         middleware=[FathomRepairMiddleware(), FathomMiddleware(on_finding="store")])
    result = agent.invoke({"messages": [{"role": "user", "content": "write the record"}]})

    assert result[REPAIR_STATE_KEY][0]["decision"] == "filter"
    assert result[STATE_KEY]["coherent"] is True
