"""Analysis passes over the kernel IR."""

from .base import Checker, CheckerContext
from .deadlock import DeadlockChecker
from .hazard import HazardChecker
from .memory import MemoryChecker
from .matmul import MatmulChecker
from .pipeline import PipelineChecker

__all__ = [
    "Checker",
    "CheckerContext",
    "MemoryChecker",
    "DeadlockChecker",
    "HazardChecker",
    "PipelineChecker",
    "MatmulChecker",
]
