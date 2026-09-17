"""
The agent state key the middleware writes its verdict to.

LangGraph builds an agent's state schema from the schemas the agent and its middleware declare,
and it drops any key no schema names. `FathomMiddleware(on_finding="store")` writes the verdict
under "fathom", so the key has to be declared for the verdict to survive the graph and reach the
caller. A graph that declares its own state, such as a deepagents orchestrator, dropped the
verdict before 0.1.1 for exactly that reason.

This module carries no `from __future__ import annotations`, so the annotation below evaluates to
a real type at import time rather than a string LangGraph has to resolve later.
"""
from typing import Any, Dict

from typing_extensions import Annotated, NotRequired, TypedDict

# The base state an agent already carries. LangChain moved it once, so try both homes, and fall
# back to a bare TypedDict carrying the key alone when neither import lands.
_AgentState = None
for _module, _name in (
    ("langchain.agents.middleware", "AgentState"),
    ("langchain.agents.middleware.types", "AgentState"),
):
    try:
        _AgentState = getattr(__import__(_module, fromlist=[_name]), _name)
        break
    except Exception:  # pragma: no cover - import shape varies by langchain version
        continue

# Marks the key as one the agent writes rather than one a caller passes in. Older LangChain
# versions have no such marker, and the key still survives the graph without it.
_OmitFromInput = None
try:  # pragma: no cover - present from langchain 1.1
    from langchain.agents.middleware.types import OmitFromInput as _OmitFromInput
except Exception:
    pass

Verdict = Dict[str, Any]

if _OmitFromInput is not None:
    VerdictField = Annotated[NotRequired[Verdict], _OmitFromInput]
else:  # pragma: no cover - older langchain
    VerdictField = NotRequired[Verdict]

if _AgentState is not None:

    class FathomState(_AgentState):  # type: ignore[misc,valid-type]
        """Agent state plus the key the read's verdict lands on."""

        fathom: VerdictField  # type: ignore[valid-type]

else:  # pragma: no cover - langchain absent, offline tests only

    class FathomState(TypedDict):  # type: ignore[no-redef]
        """The verdict key alone, for an environment without langchain installed."""

        fathom: VerdictField  # type: ignore[valid-type]


__all__ = ["FathomState"]
