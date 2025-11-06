"""Configuration helpers for the router module."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RouterConfig:
    """Runtime configuration for the router decision module.

    Parameters mirror the FrozenLlavaSAM router setup present in
    :mod:`LENA/flmm_wiki_rag`, but the implementation degrades gracefully when
    optional fields are omitted.
    """

    enabled: bool = True
    backend: str = "mmengine"
    threshold: float = 0.5
    device: str = "cuda:0"
    config_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    router_checkpoint: Optional[Path] = None
    hf_model: Optional[str] = None
    image_token: Optional[str] = None
    rag_token: Optional[str] = None
    dtype: str = "float16"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RouterConfig":
        """Load configuration values from a YAML file."""
        data = Path(path).expanduser().resolve()
        with data.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}

        known: Dict[str, Any] = {}
        unknown: Dict[str, Any] = {}
        for key, value in payload.items():
            if hasattr(cls, key):
                known[key] = value
            else:
                unknown[key] = value

        base_dir = data.parent

        def _resolve_path(raw: Optional[str]) -> Optional[Path]:
            if not raw:
                return None
            candidate = Path(str(raw)).expanduser()
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            return candidate

        cfg = cls(
            enabled=bool(known.get("enabled", True)),
            backend=str(known.get("backend", "mmengine")).lower(),
            threshold=float(known.get("threshold", 0.5)),
            device=str(known.get("device", "cuda:0")),
            config_path=_resolve_path(known.get("config_path")),
            checkpoint_path=_resolve_path(known.get("checkpoint_path")),
            router_checkpoint=_resolve_path(known.get("router_checkpoint")),
            hf_model=known.get("hf_model"),
            image_token=known.get("image_token"),
            rag_token=known.get("rag_token"),
            dtype=str(known.get("dtype", "float16")),
        )
        cfg.extra = unknown
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": self.enabled,
            "backend": self.backend,
            "threshold": self.threshold,
            "device": self.device,
            "config_path": str(self.config_path) if self.config_path else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "router_checkpoint": str(self.router_checkpoint) if self.router_checkpoint else None,
            "hf_model": self.hf_model,
            "image_token": self.image_token,
            "rag_token": self.rag_token,
            "dtype": self.dtype,
        }
        payload.update(self.extra)
        return payload
