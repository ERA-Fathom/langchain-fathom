"""langchain-fathom: a coherence read for LangChain agents, as an agent middleware."""
from .middleware import FathomMiddleware, FathomCoherenceError, STATE_KEY
from .messages import ops_from_messages
from .state import FathomState

__version__ = "0.1.1"
__all__ = ["FathomMiddleware", "FathomCoherenceError", "FathomState", "STATE_KEY", "ops_from_messages"]
