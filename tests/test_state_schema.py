"""The verdict has to survive a real graph.

on_finding="store" returns {"fathom": verdict} from after_agent, and LangGraph keeps that key only
when a schema declares it. This test runs a real agent with a fake model and no network, and fails
if the verdict does not reach the caller. The control below builds the same agent with the key
undeclared, which is the 0.1.0 shape the deepagents orchestrator dropped.
"""
import pytest

pytest.importorskip("langchain.agents", reason="needs langchain>=1.0 installed")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

import langchain_fathom.middleware as mw
from langchain_fathom import FathomMiddleware, STATE_KEY


@tool
def set_value(key: str, value: str) -> str:
    """Write a value to the record."""
    return "ok"


class ScriptedModel(GenericFakeChatModel):
    """A fake model that accepts bound tools and replays a fixed script."""

    def bind_tools(self, tools, **kwargs):
        return self


def _model():
    return ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "set_value", "args": {"key": "user.city", "value": "Denver"},
                         "id": "call_1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )


def _stub_read(monkeypatch):
    from fathom_read.ops import Verdict

    verdict = Verdict.from_dict(
        {"coherent": True, "findings": [], "ops_read": 1, "ops_rejected": 0, "live_facts": 1}
    )
    monkeypatch.setattr(mw, "read", lambda *a, **k: verdict)


def test_verdict_survives_the_graph(monkeypatch):
    _stub_read(monkeypatch)
    agent = create_agent(model=_model(), tools=[set_value],
                         middleware=[FathomMiddleware(on_finding="store")])
    result = agent.invoke({"messages": [{"role": "user", "content": "set the city"}]})
    assert STATE_KEY in result, "the graph dropped the verdict; the state key is not declared"
    assert result[STATE_KEY]["coherent"] is True
    assert result[STATE_KEY]["ops_read"] == 1


def test_undeclared_key_is_dropped(monkeypatch):
    """Control: the same verdict, written by a middleware that declares no schema, does not arrive."""
    _stub_read(monkeypatch)

    class Undeclared(FathomMiddleware):
        state_schema = AgentMiddleware.state_schema

    agent = create_agent(model=_model(), tools=[set_value],
                         middleware=[Undeclared(on_finding="store")])
    result = agent.invoke({"messages": [{"role": "user", "content": "set the city"}]})
    assert STATE_KEY not in result
