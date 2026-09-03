"""The message-to-ops mapping is pure, so it tests on plain stand-in objects, no langchain."""
from langchain_fathom.messages import ops_from_messages


class AI:
    def __init__(self, tool_calls): self.tool_calls = tool_calls


class Tool:
    def __init__(self, tool_call_id, status="success", content="ok"):
        self.tool_call_id = tool_call_id; self.status = status; self.content = content


def _call(name, args, cid): return {"name": name, "args": args, "id": cid}


def test_tool_calls_become_ops_and_errors_are_no_ops():
    messages = [
        AI([_call("set_value", {"key": "user.city", "value": "Denver"}, "a")]),
        Tool("a", status="success"),
        AI([_call("set_value", {"key": "user.city", "value": "Austin"}, "b")]),
        Tool("b", status="error", content="Error: write rejected"),
    ]
    ops = ops_from_messages(messages)
    assert [(o.op, o.key, o.ok) for o in ops] == [("set", "user.city", True), ("set", "user.city", False)]


def test_plain_text_error_result_marks_no_op():
    messages = [
        AI([_call("write_record", {"record": "r0", "content": "x"}, "a")]),
        Tool("a", content="error: disk full"),
    ]
    ops = ops_from_messages(messages)
    assert ops[0].ok is False


def test_non_state_tools_are_skipped():
    messages = [AI([_call("web_search", {"query": "postgres"}, "a")])]
    assert ops_from_messages(messages) == []
