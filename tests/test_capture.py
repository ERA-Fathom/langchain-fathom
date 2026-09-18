"""Tests for the single-attachment capture. Synthetic callback events, no agent and no model."""

from langchain_fathom.capture import FathomCapture


def build():
    """An orchestrator delegating two sub-agents. Sub-agent two repeats sub-agent one's query."""
    cap = FathomCapture()
    cap.note_chain("orch", None)
    cap.start_tool("task", "t1", "orch", inputs={"description": "research a"})
    cap.end_tool("t1", "delegated")
    cap.note_chain("sub1", "t1")
    cap.start_tool("tavily_search", "s1", "sub1", inputs={"query": "agent memory"})
    cap.end_tool("s1", "## Cognee\n**URL:** https://example.com/cognee")
    cap.start_tool("task", "t2", "orch", inputs={"description": "research b"})
    cap.end_tool("t2", "delegated")
    cap.note_chain("sub2", "t2")
    cap.start_tool("tavily_search", "s2", "sub2", inputs={"query": "agent memory"})
    cap.end_tool("s2", "none")
    cap.start_tool("tavily_search", "os", "orch", inputs={"query": "durable execution"})
    cap.end_tool("os", "none")
    return cap


def test_ancestry_attributes_subagents():
    by = {c["run_id"]: c["agent"] for c in build().finish()}
    assert by["s1"] == "research-agent (task 1)"
    assert by["s2"] == "research-agent (task 2)"
    assert by["os"] == "orchestrator"


def test_full_capture_sees_more_than_orchestrator_only():
    calls = build().finish()
    searches = [c for c in calls if c["tool"] == "tavily_search"]
    orch_only = [c for c in searches if c["agent"] == "orchestrator"]
    # the two sibling searches are invisible to the orchestrator-only view
    assert len(searches) == 3 and len(orch_only) == 1


def test_tavily_results_extracted():
    cap = FathomCapture()
    cap.start_tool("tavily_search", "s", None, inputs={"query": "x"})
    cap.end_tool("s", "## Title\n**URL:** https://example.com/p")
    assert cap.calls[0]["results"] == [["https://example.com/p", "Title"]]


class _Msg:
    def __init__(self, mtype, content="", tool_calls=None):
        self.type = mtype
        self.content = content
        self.tool_calls = tool_calls or []


def test_merge_and_final_message():
    cap = FathomCapture()
    cap.note_chain("orch", None)
    cap.start_tool("tavily_search", "os", "orch", inputs={"query": "x"})
    cap.end_tool("os", "none")
    messages = [
        _Msg("ai", "", [{"name": "write_todos", "args": {"todos": ["a"]}, "id": "t"}]),
        _Msg("ai", "done"),
    ]
    trace = cap.trace(messages)
    assert "write_todos" in [c["tool"] for c in trace["calls"]]
    assert trace["final_message"] == "done"
