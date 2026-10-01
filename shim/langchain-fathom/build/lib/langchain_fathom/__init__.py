"""langchain_fathom is now langchain_right_rudder. This module forwards every import to the new name."""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

warnings.warn(
    "langchain-fathom is now langchain-right-rudder; install langchain-right-rudder and import langchain_right_rudder instead of langchain_fathom",
    DeprecationWarning,
    stacklevel=2,
)

from langchain_right_rudder import *  # noqa: F401,F403
from langchain_right_rudder import (  # noqa: F401
    FathomMiddleware, FathomCoherenceError, FathomRepairMiddleware, FathomKeyError,
    FathomRepairError, FathomState, FathomRepairState, FathomCapture, STATE_KEY, REPAIR_STATE_KEY,
)
from langchain_right_rudder import __version__  # noqa: F401


class _Forward(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves langchain_fathom.<name> to the module langchain_right_rudder.<name>."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("langchain_fathom."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module("langchain_right_rudder." + spec.name[len("langchain_fathom."):])
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        return None


sys.meta_path.insert(0, _Forward())
