"""
FathomMiddleware: read an agent's committed state after it runs and report where it broke.

The middleware observes the agent and does not change what it does. When the agent finishes,
it maps the tool calls in the message history to a committed-state op stream, sends that stream
to the Fathom read, and reports the findings. Use it to catch the step where a long agent
contradicts a decision it already made, without touching the agent's runtime path.

    from langchain.agents import create_agent
    from langchain_fathom import FathomMiddleware

    agent = create_agent(model="gpt-5.5", tools=[...], middleware=[FathomMiddleware()])
    result = agent.invoke({"messages": [...]})

By default the middleware logs any findings. Pass on_finding="raise" to fail a run that is not
coherent, which drops it into a test suite, or on_finding="store" to put the verdict on the
agent state under the "fathom" key. The read is diagnosis. The repair, which re-grounds the
agent before the contradiction ships, runs as part of the Fathom service and is not in this
package.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from fathom_read.client import read, ReadError

from .messages import ops_from_messages

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "langchain-fathom needs langchain>=1.0 with agent middleware: pip install 'langchain>=1.0'"
    ) from e

log = logging.getLogger("fathom")


class FathomCoherenceError(AssertionError):
    """Raised by FathomMiddleware(on_finding='raise') when the committed state is not coherent."""


class FathomMiddleware(AgentMiddleware):
    """Reads the agent's committed state after each run and reports contradictions.

    Args:
        on_finding: "log" reports findings through the "fathom" logger (the default); "raise"
            raises FathomCoherenceError when the run is not coherent; "store" writes the verdict
            onto the agent state under the "fathom" key.
        supersede: optional (old, new) token pairs the run is expected to migrate, such as a
            field rename, so the read also reports records left on the old value at the end.
        mapping_path: optional path to a JSON tool map for tool names outside the defaults.
        key, endpoint: optional overrides for the Fathom read; the packaged demo key is used
            otherwise, and FATHOM_API_KEY / FATHOM_ENDPOINT are honored.
    """

    def __init__(
        self,
        on_finding: str = "log",
        supersede: Optional[List[Tuple[str, str]]] = None,
        mapping_path: Optional[str] = None,
        key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        if on_finding not in ("log", "raise", "store"):
            raise ValueError("on_finding must be 'log', 'raise', or 'store'")
        self.on_finding = on_finding
        self.supersede = supersede
        self.mapping_path = mapping_path
        self.key = key or os.environ.get("FATHOM_API_KEY")
        self.endpoint = endpoint or os.environ.get("FATHOM_ENDPOINT")

    def _run_read(self, messages: List[Any]) -> Optional[Dict[str, Any]]:
        ops = ops_from_messages(messages, self.mapping_path)
        if not ops:
            return None
        try:
            verdict = read(ops, supersede=self.supersede, key=self.key, endpoint=self.endpoint)
        except ReadError as e:
            log.warning("fathom: could not reach the read (%s); skipping", e)
            return None
        return verdict.as_dict()

    def after_agent(self, state: Any, runtime: Any = None) -> Optional[Dict[str, Any]]:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        verdict = self._run_read(list(messages))
        if verdict is None:
            return None
        if verdict["coherent"]:
            log.info("fathom: committed state coherent across %d ops", verdict["ops_read"])
            return {"fathom": verdict} if self.on_finding == "store" else None
        lines = [
            f"step {f['step']} {f['kind']}: {f['detail']}" if f["step"] is not None else f"{f['kind']}: {f['detail']}"
            for f in verdict["findings"]
        ]
        message = "fathom: %d coherence finding(s)\n  %s" % (len(verdict["findings"]), "\n  ".join(lines))
        if self.on_finding == "raise":
            raise FathomCoherenceError(message)
        if self.on_finding == "store":
            return {"fathom": verdict}
        log.warning(message)
        return None
