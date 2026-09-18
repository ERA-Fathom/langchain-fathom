"""fathom_capture.py

A single-attachment capture for LangChain and LangGraph agents. Attach one
FathomCapture at the graph root and it records every tool call the orchestrator
and its sub-agents make, attributing each to the task() delegation it ran under
by run_id ancestry. This is what closes the gap the per-agent middleware leaves,
where an orchestrator that delegates shows only its own calls unless a partner
wires a middleware into every sub-agent by hand.

Usage.
    from fathom_capture import FathomCapture
    cap = FathomCapture()
    result = agent.invoke({"messages": [HumanMessage(content=q)]},
                          config={"callbacks": [cap.handler]})
    trace = cap.trace(result.get("messages"))     # {"calls": [...], "final_message": "..."}
    verdict = fathom_read.read(fathom_read.load_ops_from(trace, "deepagents"))

The recording logic is the ToolTrace the deepagents runs already used, the one
that recovered all nine findings on the 9/16 llama run against three from the
orchestrator-only view. It is ported here as a reusable class with a lazy handler
so importing it needs no langchain, and with a pluggable result extractor so a
search tool other than tavily still yields the sources it returned.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# helpers, ported from the runner
# ---------------------------------------------------------------------------

def jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        return jsonable(x.model_dump())
    return str(x)


_TAVILY_RESULT = re.compile(r"^## (.*?)\n\*\*URL:\*\* (\S+)", re.M)


def tavily_results(text: str) -> List[list]:
    """The sources a tavily_search result carried, as [url, title] pairs."""
    return [[url, title] for title, url in _TAVILY_RESULT.findall(text or "")]


# tool name -> function(result_text) -> list of [url, title]. A partner using a
# different search tool passes {"their_search": their_extractor} to the constructor.
DEFAULT_EXTRACTORS: Dict[str, Callable[[str], List[list]]] = {"tavily_search": tavily_results}


# ---------------------------------------------------------------------------
# the capture
# ---------------------------------------------------------------------------

class FathomCapture:
    """Records every tool call in a LangChain/LangGraph run with its ancestry, and
    emits the canonical trace the deepagents adapter reads."""

    def __init__(self, result_extractors: Optional[Dict[str, Callable[[str], List[list]]]] = None):
        self.calls: List[Dict[str, Any]] = []
        self.open: Dict[str, Dict[str, Any]] = {}
        self.parent: Dict[str, Optional[str]] = {}
        self.task_runs: Dict[str, int] = {}
        self.result_extractors = dict(DEFAULT_EXTRACTORS)
        if result_extractors:
            self.result_extractors.update(result_extractors)
        self._handler = None

    # ---- core recording, callable directly so the logic tests without langchain ----

    def note_chain(self, run_id: str, parent_run_id: Optional[str]) -> None:
        self.parent[str(run_id)] = str(parent_run_id) if parent_run_id else None

    def start_tool(self, name: str, run_id: str, parent_run_id: Optional[str],
                   inputs: Any = None, input_str: str = "") -> None:
        rid = str(run_id)
        self.parent[rid] = str(parent_run_id) if parent_run_id else None
        args = inputs if isinstance(inputs, dict) else self._parse(input_str)
        self.calls.append({"n": len(self.calls), "run_id": rid,
                           "parent_run_id": str(parent_run_id) if parent_run_id else None,
                           "tool": name or "", "args": jsonable(args), "t_start": time.time()})
        self.open[rid] = self.calls[-1]
        if name == "task":
            self.task_runs[rid] = len([c for c in self.calls if c["tool"] == "task"])

    def end_tool(self, run_id: str, output: Any) -> None:
        self._close(str(run_id), output, ok=None)

    def error_tool(self, run_id: str, error: Any) -> None:
        self._close(str(run_id), f"Error: {error}", ok=False)

    @staticmethod
    def _parse(s):
        try:
            return json.loads(s)
        except (TypeError, ValueError):
            return {"input": str(s)}

    def _close(self, run_id, output, ok):
        c = self.open.pop(run_id, None)
        if c is None:
            return
        content, status = output, None
        if hasattr(output, "content"):
            content = output.content
            status = getattr(output, "status", None)
        if isinstance(content, list):
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        text = str(content or "")
        if ok is None:
            ok = not (str(status or "").lower() == "error" or text.lower().startswith("error"))
        c["ok"] = ok
        c["t_end"] = time.time()
        c["output_chars"] = len(text)
        c["output_head"] = text[:1500]
        extractor = self.result_extractors.get(c["tool"])
        if extractor:
            c["results"] = extractor(text)

    # ---- ancestry and trace ----

    def finish(self) -> List[Dict[str, Any]]:
        """Attribute each call to the task() delegation it ran under, or to the orchestrator,
        by walking the run_id parent chain until it meets a task run."""
        for c in self.calls:
            who = "orchestrator"
            p = c.get("parent_run_id")
            seen = 0
            while p and seen < 200:
                if p in self.task_runs:
                    who = f"research-agent (task {self.task_runs[p]})"
                    break
                p = self.parent.get(p)
                seen += 1
            c["agent"] = who
            c.pop("parent_run_id", None)
        return self.calls

    def trace(self, messages: Optional[list] = None) -> Dict[str, Any]:
        """The canonical trace the deepagents adapter reads, calls plus the final message.
        Pass result['messages'] so orchestrator tool calls the callback did not see are merged."""
        calls = self.finish()
        if messages:
            calls = merge_orchestrator_calls(calls, messages)
        return {"calls": calls, "final_message": final_message(messages or [])}

    # ---- the langchain handler, built lazily so import needs no langchain ----

    @property
    def handler(self):
        if self._handler is None:
            from langchain_core.callbacks import BaseCallbackHandler
            cap = self

            class _Handler(BaseCallbackHandler):
                def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
                    cap.note_chain(run_id, parent_run_id)

                def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kw):
                    cap.start_tool((serialized or {}).get("name") or "", run_id, parent_run_id,
                                   inputs=inputs, input_str=input_str)

                def on_tool_end(self, output, *, run_id, **kw):
                    cap.end_tool(run_id, output)

                def on_tool_error(self, error, *, run_id, **kw):
                    cap.error_tool(run_id, error)

            self._handler = _Handler()
        return self._handler


# ---------------------------------------------------------------------------
# orchestrator merge and final message, ported from the runner
# ---------------------------------------------------------------------------

def merge_orchestrator_calls(calls: List[dict], messages: list) -> List[dict]:
    """Add orchestrator tool calls the callback did not see, from the orchestrator's own
    message history, in message order. deepagents writes write_todos through its planning
    middleware without a tool run, so those appear only in the messages."""
    def key(name, args):
        return name, json.dumps(jsonable(args), sort_keys=True)

    seen = {key(c["tool"], c["args"]) for c in calls}
    merged = list(calls)
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name, args = tc.get("name", ""), tc.get("args", {})
            if key(name, args) in seen:
                continue
            seen.add(key(name, args))
            entry = {"n": None, "run_id": tc.get("id"), "tool": name, "args": jsonable(args),
                     "ok": True, "agent": "orchestrator", "source": "messages"}
            later = [c for c in merged if c.get("agent") == "orchestrator" and c.get("source") != "messages"
                     and _msg_index(messages, c) > _msg_index(messages, entry)]
            idx = merged.index(later[0]) if later else len(merged)
            merged.insert(idx, entry)
    for i, c in enumerate(merged):
        c["n"] = i
    return merged


def _msg_index(messages, call):
    k = (call["tool"], json.dumps(jsonable(call["args"]), sort_keys=True))
    for i, m in enumerate(messages):
        for tc in getattr(m, "tool_calls", None) or []:
            if (tc.get("name", ""), json.dumps(jsonable(tc.get("args", {})), sort_keys=True)) == k:
                return i
    return 10 ** 9


def final_message(messages: list) -> str:
    ai = [m for m in messages if _mtype(m) == "ai"]
    if not ai:
        return ""
    content = _mcontent(ai[-1])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _mtype(m):
    t = getattr(m, "type", None)
    if t:
        return t
    if isinstance(m, dict):
        return m.get("type")
    return type(m).__name__.lower().replace("message", "")


def _mcontent(m):
    if hasattr(m, "content"):
        return m.content
    if isinstance(m, dict):
        return m.get("content") or (m.get("data") or {}).get("content")
    return ""
