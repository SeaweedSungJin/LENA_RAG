"""Router module exporting configuration and inference helpers."""
from .config import RouterConfig
from .router import Router, RouterDecision

__all__ = ["Router", "RouterConfig", "RouterDecision"]

