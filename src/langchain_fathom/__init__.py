"""langchain-fathom: a coherence read for LangChain agents, as an agent middleware."""
from .middleware import FathomMiddleware, FathomCoherenceError
from .messages import ops_from_messages

__version__ = "0.1.0"
__all__ = ["FathomMiddleware", "FathomCoherenceError", "ops_from_messages"]
