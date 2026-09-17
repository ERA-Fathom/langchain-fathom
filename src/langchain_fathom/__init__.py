"""langchain-fathom: a coherence read for LangChain agents, and the repair, as agent middleware."""
from .middleware import FathomMiddleware, FathomCoherenceError, STATE_KEY
from .messages import ops_from_messages
from .repair import FathomRepairMiddleware, FathomKeyError, FathomRepairError
from .repair import STATE_KEY as REPAIR_STATE_KEY
from .state import FathomState, FathomRepairState

__version__ = "0.2.0"
__all__ = [
    "FathomMiddleware",
    "FathomCoherenceError",
    "FathomRepairMiddleware",
    "FathomKeyError",
    "FathomRepairError",
    "FathomState",
    "FathomRepairState",
    "STATE_KEY",
    "REPAIR_STATE_KEY",
    "ops_from_messages",
]
