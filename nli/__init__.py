"""Lightweight NLI-based sentence selection utilities for EchoSight (LENA)."""

from .config import NLIConfig
from .selector import NLISelector, SentenceCandidate, SectionCandidate

__all__ = [
    "NLIConfig",
    "NLISelector",
    "SectionCandidate",
    "SentenceCandidate",
]

