"""Configuration schema for the EchoSight NLI selector (LENA)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class NLIConfig:
    """Runtime knobs controlling NLI filtering and clustering."""

    model_name: str = "FacebookAI/roberta-large-mnli"
    device: str = "cuda:0"
    max_length: int = 512
    batch_size: int = 32
    question_mode: str = "statement"
    # Question-to-statement conversion
    convert_question: bool = True
    question_entail_threshold: float = 0.5
    # Pairwise graph construction
    alpha: float = 1.0
    beta: float = 1.0
    margin: float = 0.15
    tau: float = 0.25
    edge_rule: str = "avg"
    dir_margin: float = 0.0
    autocast: bool = True
    autocast_dtype: str = "fp16"
    clamp_weights: bool = True
    # Cluster trimming
    target_size: int = 3
    reduction_mode: str = "recompute"
    reduction_epsilon: float = 1e-6
    hybrid_lambda: float = 0.5
    # Sentence segmentation
    sentence_batch_size: int = 128
    keep_top_k_on_empty: int = 5

    # Optional overrides for integration scripts
    dataset_start: int = 0
    dataset_end: Optional[int] = None
    dataset_limit: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = (self.question_mode or ("statement" if self.convert_question else "question")).strip().lower()
        if mode not in {"statement", "question"}:
            mode = "statement"
        self.question_mode = mode
        self.convert_question = mode != "question"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NLIConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        kwargs = {k: v for k, v in data.items() if hasattr(cls, k)}
        extra = {k: v for k, v in data.items() if k not in kwargs}
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "question_mode": self.question_mode,
            "convert_question": self.convert_question,
            "question_entail_threshold": self.question_entail_threshold,
            "alpha": self.alpha,
            "beta": self.beta,
            "margin": self.margin,
            "tau": self.tau,
            "edge_rule": self.edge_rule,
            "dir_margin": self.dir_margin,
            "autocast": self.autocast,
            "autocast_dtype": self.autocast_dtype,
            "clamp_weights": self.clamp_weights,
            "target_size": self.target_size,
            "reduction_mode": self.reduction_mode,
            "reduction_epsilon": self.reduction_epsilon,
            "hybrid_lambda": self.hybrid_lambda,
            "sentence_batch_size": self.sentence_batch_size,
            "keep_top_k_on_empty": self.keep_top_k_on_empty,
            "dataset_start": self.dataset_start,
            "dataset_end": self.dataset_end,
            "dataset_limit": self.dataset_limit,
        }
        payload.update(self.extra)
        return payload
