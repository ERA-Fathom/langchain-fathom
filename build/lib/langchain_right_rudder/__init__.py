"""langchain-right-rudder: a coherence read for LangChain agents, and the repair, as agent middleware."""
from .middleware import RightRudderMiddleware, RightRudderCoherenceError, STATE_KEY
from .messages import ops_from_messages
from .repair import RightRudderRepairMiddleware, RightRudderKeyError, RightRudderRepairError
from .repair import STATE_KEY as REPAIR_STATE_KEY
from .state import RightRudderState, RightRudderRepairState
from .capture import RightRudderCapture

__version__ = "0.4.0"
# The names before the Right Rudder rebrand, kept so existing code keeps working.
FathomMiddleware = RightRudderMiddleware
FathomCoherenceError = RightRudderCoherenceError
FathomRepairMiddleware = RightRudderRepairMiddleware
FathomKeyError = RightRudderKeyError
FathomRepairError = RightRudderRepairError
FathomState = RightRudderState
FathomRepairState = RightRudderRepairState
FathomCapture = RightRudderCapture

__all__ = [
    "RightRudderMiddleware",
    "RightRudderCoherenceError",
    "RightRudderRepairMiddleware",
    "RightRudderKeyError",
    "RightRudderRepairError",
    "RightRudderState",
    "RightRudderRepairState",
    "RightRudderCapture",
    "STATE_KEY",
    "REPAIR_STATE_KEY",
    "ops_from_messages",
    "FathomMiddleware",
    "FathomCoherenceError",
    "FathomRepairMiddleware",
    "FathomKeyError",
    "FathomRepairError",
    "FathomState",
    "FathomRepairState",
    "FathomCapture",
]
