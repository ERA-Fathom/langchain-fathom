"""Offline test shim: stand in for langchain.agents.middleware so the package imports without
langchain installed. Real CI installs langchain>=1.0 and skips this by importing the real class."""
import sys, types

try:
    import langchain.agents.middleware  # noqa: F401
except Exception:
    lc = sys.modules.setdefault("langchain", types.ModuleType("langchain"))
    ag = sys.modules.setdefault("langchain.agents", types.ModuleType("langchain.agents"))
    mw = types.ModuleType("langchain.agents.middleware")
    class AgentMiddleware:
        def __init__(self): pass
    mw.AgentMiddleware = AgentMiddleware
    ag.middleware = mw
    sys.modules["langchain.agents.middleware"] = mw
