"""
FathomRepairMiddleware: run the repair in front of the agent's next step.

The read tells you where an agent contradicted itself. The repair acts on what the read finds,
before the contradiction ships. This middleware sits between the model and its tools. When the
model proposes its next actions, the middleware sends those actions and the run's committed state
so far to the Fathom repair, and the repair answers with one of three decisions.

    proceed   every proposed action is consistent with what the agent already committed
    filter    some are; the agent keeps its own consistent alternatives and drops the rest
    reground  none are; the committed facts go back in front of the model and it is asked again

Unlike FathomMiddleware, which only watches, this one changes what the agent does. It drops tool
calls the agent's own committed state contradicts, and it can spend one extra model call to ask
again. Turn it on deliberately.

    from langchain.agents import create_agent
    from langchain_fathom import FathomRepairMiddleware

    agent = create_agent(model="gpt-5.5", tools=[...], middleware=[FathomRepairMiddleware()])

The repair needs a free key. Run `fathom key you@example.com` from the fathom-read package and
set FATHOM_API_KEY, or pass key= here. The middleware checks for it when you construct it, so a
run fails at setup rather than partway through.

Cost, in two places. A repair call counts double against the 2,000 calls a day a free key carries.
A reground decision spends one extra model call, billed by your own provider.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fathom_read.adapters._tools import load_map, op_from_tool
from fathom_read.client import ReadError, reground

from .messages import ops_from_messages
from .state import FathomRepairState

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "langchain-fathom needs langchain>=1.0 with agent middleware: pip install 'langchain>=1.0'"
    ) from e

log = logging.getLogger("fathom")

STATE_KEY = "fathom_repair"

# Writing the run's repair log back onto agent state needs these two. Older LangChain versions
# have no way for a wrap_model_call hook to return a state update, and the repair still runs.
_Command = None
_ExtendedModelResponse = None
try:  # pragma: no cover - present from langchain 1.1
    from langchain.agents.middleware import ExtendedModelResponse as _ExtendedModelResponse
    from langgraph.types import Command as _Command
except Exception:
    pass

_SystemMessage = None
try:  # pragma: no cover
    from langchain_core.messages import SystemMessage as _SystemMessage
except Exception:
    pass


class FathomKeyError(RuntimeError):
    """Raised when FathomRepairMiddleware is built without a key for the repair."""


class FathomRepairError(RuntimeError):
    """Raised by FathomRepairMiddleware(on_error='raise') when the repair cannot be reached."""


def _last_ai_index(messages: Sequence[Any]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "tool_calls", None):
            return i
    return -1


def _tool_calls_of(message: Any) -> List[Dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    out = []
    for c in calls:
        if isinstance(c, dict):
            out.append(c)
        else:
            out.append({"name": getattr(c, "name", ""), "args": getattr(c, "args", {}),
                        "id": getattr(c, "id", None), "type": "tool_call"})
    return out


class FathomRepairMiddleware(AgentMiddleware):
    """Runs the Fathom repair on the actions the model proposes, before the tools run.

    Args:
        key: the service key. Falls back to FATHOM_API_KEY. Required, and checked here.
        endpoint: optional override for the repair endpoint.
        supersede: optional (old, new) token pairs the run is expected to migrate, such as a
            field rename, passed through to the repair.
        mapping_path: optional path to a JSON tool map for tool names outside the defaults. A
            tool the map does not name never becomes a proposal and always passes through.
        max_reasks: how many times to put the committed facts back in front of the model on a
            reground decision before proceeding with whatever it returns. Default 1, which
            matches the DBOS run where one re-ask occurred across 45 follow-up steps.
        on_error: "proceed" logs a warning and lets the agent act on its own proposal when the
            repair cannot be reached, the key is rejected, or the daily limit lands. "raise"
            raises FathomRepairError instead, which is what a measured run wants, since a
            silently skipped repair would contaminate a before-and-after.
        record: append one entry per repaired step to agent state under "fathom_repair".
        timeout: seconds to wait on the repair.
    """

    state_schema = FathomRepairState  # type: ignore[assignment]

    def __init__(
        self,
        key: Optional[str] = None,
        endpoint: Optional[str] = None,
        supersede: Optional[List[Tuple[str, str]]] = None,
        mapping_path: Optional[str] = None,
        max_reasks: int = 1,
        on_error: str = "proceed",
        record: bool = True,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        if on_error not in ("proceed", "raise"):
            raise ValueError("on_error must be 'proceed' or 'raise'")
        if max_reasks < 0:
            raise ValueError("max_reasks cannot be negative")
        self.key = key or os.environ.get("FATHOM_API_KEY")
        if not self.key:
            raise FathomKeyError(
                "the repair needs a key. Get a free one with `fathom key you@example.com`, then set "
                "FATHOM_API_KEY or pass key= to FathomRepairMiddleware. The read runs without one."
            )
        self.endpoint = endpoint or os.environ.get("FATHOM_REGROUND_ENDPOINT")
        self.supersede = supersede
        self.mapping_path = mapping_path
        self.max_reasks = max_reasks
        self.on_error = on_error
        self.record = record
        self.timeout = timeout

    # -- mapping ------------------------------------------------------------------------

    def _proposals(self, tool_calls: List[Dict[str, Any]], step: int):
        """Map the model's tool calls to proposals, keeping which call each one came from.

        A tool the map does not name returns no op, so it never becomes a proposal and the read
        holds no opinion on it. The verdict's indices point at proposals, not at tool calls,
        which is why the origin list has to be carried alongside.
        """
        mapping = load_map(self.mapping_path)
        proposals: List[Dict[str, Any]] = []
        origin: List[int] = []
        for i, call in enumerate(tool_calls):
            op = op_from_tool(call.get("name", ""), call.get("args", {}), True, step, mapping,
                              source="tool_call")
            if op is None:
                continue
            d = op.as_dict()
            proposals.append({k: d[k] for k in ("op", "kind", "key", "value", "to", "refs") if k in d})
            origin.append(i)
        return proposals, origin

    # -- the hook -----------------------------------------------------------------------

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        response = handler(request)
        entries: List[Dict[str, Any]] = []
        history = list(getattr(request, "messages", []) or [])

        for attempt in range(self.max_reasks + 1):
            ai_index = _last_ai_index(getattr(response, "result", []) or [])
            if ai_index < 0:
                return self._finish(response, entries)
            ai = response.result[ai_index]
            tool_calls = _tool_calls_of(ai)

            ops = ops_from_messages(history, self.mapping_path)
            proposals, origin = self._proposals(tool_calls, len(ops))
            if not proposals:
                return self._finish(response, entries)

            try:
                verdict = reground(ops, proposals, supersede=self.supersede, key=self.key,
                                   endpoint=self.endpoint, timeout=self.timeout)
            except ReadError as e:
                if self.on_error == "raise":
                    raise FathomRepairError(str(e)) from None
                log.warning("fathom: could not reach the repair (%s); the agent proceeds unrepaired", e)
                return self._finish(response, entries)

            entries.append({
                "decision": verdict.decision,
                "proposals": verdict.proposals,
                "dropped": [{"kind": d.get("kind"), "key": d.get("key"), "findings": d.get("findings", [])}
                            for d in verdict.dropped],
                "reasks": attempt,
                "read_findings": verdict.read.get("findings"),
            })

            if verdict.decision == "proceed":
                return self._finish(response, entries)

            if verdict.decision == "filter":
                keep_origins = {origin[k["index"]] for k in verdict.kept if k.get("index") is not None
                                and k["index"] < len(origin)}
                mapped = set(origin)
                kept_calls = [c for i, c in enumerate(tool_calls)
                              if i not in mapped or i in keep_origins]
                dropped_n = len(tool_calls) - len(kept_calls)
                log.info("fathom: the repair dropped %d of %d proposed actions the run already committed",
                         dropped_n, len(tool_calls))
                response.result[ai_index] = self._with_tool_calls(ai, kept_calls)
                return self._finish(response, entries)

            # reground. Put the committed facts back in front of the model and ask again.
            if attempt >= self.max_reasks:
                log.warning("fathom: every proposed action still contradicts the committed state after "
                            "%d re-ask(s); the agent proceeds with its own", self.max_reasks)
                return self._finish(response, entries)

            note = verdict.prompt_note([p.get("value") for p in proposals])
            request = self._with_note(request, note)
            response = handler(request)

        return self._finish(response, entries)  # pragma: no cover - loop returns above

    # -- helpers ------------------------------------------------------------------------

    @staticmethod
    def _with_tool_calls(ai: Any, tool_calls: List[Dict[str, Any]]) -> Any:
        copy = getattr(ai, "model_copy", None)
        if copy is not None:
            return copy(update={"tool_calls": tool_calls})
        ai.tool_calls = tool_calls  # pragma: no cover - non-pydantic message
        return ai

    def _with_note(self, request: Any, note: str) -> Any:
        """Append the committed facts to the turn as a system-role message."""
        messages = list(getattr(request, "messages", []) or [])
        if _SystemMessage is not None:
            messages = messages + [_SystemMessage(content=note.strip())]
        else:  # pragma: no cover - langchain_core absent
            return request
        return request.override(messages=messages)

    def _finish(self, response: Any, entries: List[Dict[str, Any]]) -> Any:
        if not entries or not self.record:
            return response
        if _ExtendedModelResponse is None or _Command is None:  # pragma: no cover - older langchain
            log.debug("fathom: this langchain cannot carry a state update out of wrap_model_call; "
                      "the repair ran and was not recorded on state")
            return response
        return _ExtendedModelResponse(
            model_response=response, command=_Command(update={STATE_KEY: entries})
        )
