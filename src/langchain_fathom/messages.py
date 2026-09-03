"""
Turn a LangChain message list into a committed-state op stream.

The middleware watches the messages an agent produces. Every tool the agent calls is an
action on committed state, so this module maps each tool call in the message history to an op
through the same tool mapping the fathom-read adapters use, and marks an op as a no-op when the
matching tool result reports an error. It is pure and holds no LangChain import, so it can be
tested on plain objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom_read.adapters._tools import op_from_tool, load_map
from fathom_read.ops import Op


def _tool_calls(message: Any) -> List[Dict[str, Any]]:
    """The tool calls an AI message made, as a list of {name, args, id} dicts."""
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return []
    out = []
    for c in calls:
        if isinstance(c, dict):
            out.append({"name": c.get("name", ""), "args": c.get("args", {}), "id": c.get("id")})
        else:
            out.append({"name": getattr(c, "name", ""), "args": getattr(c, "args", {}), "id": getattr(c, "id", None)})
    return out


def _is_error_result(message: Any) -> bool:
    """A tool result message that reports failure."""
    status = getattr(message, "status", None)
    if isinstance(status, str) and status.lower() == "error":
        return True
    content = getattr(message, "content", "")
    return isinstance(content, str) and content.strip().lower().startswith("error")


def _result_id(message: Any) -> Optional[str]:
    return getattr(message, "tool_call_id", None)


def ops_from_messages(messages: List[Any], mapping_path: Optional[str] = None) -> List[Op]:
    """Map an agent's message history to committed-state ops.

    Each tool call becomes an op; a tool-result message whose id matches an earlier call and
    reports an error marks that call's op as a no-op, so a failed write does not count as a commit.
    """
    mapping = load_map(mapping_path)
    # First pass: which tool_call_ids reported an error.
    errored = {mid for m in messages if (mid := _result_id(m)) is not None and _is_error_result(m)}
    ops: List[Op] = []
    step = 0
    for m in messages:
        for call in _tool_calls(m):
            ok = call["id"] not in errored
            op = op_from_tool(call["name"], call["args"], ok, step, mapping, source="tool_call")
            if op is not None:
                ops.append(op)
                step += 1
    return ops
