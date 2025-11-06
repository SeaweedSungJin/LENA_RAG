"""Router inference helpers (inspired by flmm_wiki_rag)."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import RouterConfig


@dataclass
class RouterDecision:
    """Return value describing the router output."""

    prob: float
    threshold: float
    use_rag: bool
    backend: str
    extras: Dict[str, float]


class BaseRouterBackend:
    """Abstract backend interface."""

    def __init__(self, cfg: RouterConfig) -> None:
        self.cfg = cfg

    def score(self, question: str, image_path: str | Path) -> RouterDecision:  # pragma: no cover - interface
        raise NotImplementedError


class AlwaysOnRouter(BaseRouterBackend):
    """Fallback backend when no specialised router is available."""

    def score(self, question: str, image_path: str | Path) -> RouterDecision:
        prob = 1.0
        return RouterDecision(
            prob=prob,
            threshold=self.cfg.threshold,
            use_rag=True,
            backend="always_on",
            extras={"reason": 1.0},
        )


class HeuristicRouter(BaseRouterBackend):
    """Very light-weight heuristic router using question length as proxy."""

    def score(self, question: str, image_path: str | Path) -> RouterDecision:
        # Normalise question length to obtain a crude probability
        tokens = question.strip().split()
        length = len(tokens)
        # Squash with tanh for stability
        score = math.tanh(length / 12.0)
        prob = 0.5 + 0.5 * score
        use_rag = prob >= self.cfg.threshold
        return RouterDecision(
            prob=prob,
            threshold=self.cfg.threshold,
            use_rag=use_rag,
            backend="heuristic",
            extras={"question_len": float(length)},
        )


class FrozenRouterBackend(BaseRouterBackend):
    """Router backed by FrozenLlavaSAM via mmengine/xtuner."""

    def __init__(self, cfg: RouterConfig) -> None:
        super().__init__(cfg)

        # Register vendored flmm modules so mmengine config imports succeed.
        project_root = Path(__file__).resolve().parent.parent
        rag_flmm_root = project_root / "rag_flmm"
        if rag_flmm_root.exists():
            sys_path_entry = str(rag_flmm_root)
            if sys_path_entry not in sys.path:
                sys.path.insert(0, sys_path_entry)

        try:
            from mmengine.config import Config as MMConfig
            from xtuner.registry import BUILDER
            from xtuner.model.utils import guess_load_checkpoint
        except Exception as exc:  # pragma: no cover - dependency import
            raise RuntimeError("mmengine/xtuner are required for the mmengine router backend") from exc

        if cfg.config_path is None:
            raise ValueError("RouterConfig.config_path must be provided for backend 'mmengine'")

        self._MMConfig = MMConfig
        self._BUILDER = BUILDER
        self._guess_load_checkpoint = guess_load_checkpoint

        mm_cfg = MMConfig.fromfile(str(cfg.config_path))
        self._prompt_template = mm_cfg.get("prompt_template", {})
        if "INSTRUCTION" not in self._prompt_template:
            raise ValueError("prompt_template with INSTRUCTION key is required in mmengine config")

        device = torch.device(cfg.device)
        model = BUILDER.build(mm_cfg.model).to(device)
        if cfg.checkpoint_path:
            state = guess_load_checkpoint(str(cfg.checkpoint_path))
            model.load_state_dict(state, strict=False)

        router_ckpt = cfg.router_checkpoint
        if router_ckpt:
            state = guess_load_checkpoint(str(router_ckpt))
            router_state = self._collect_router_state(state)
            if router_state:
                model.load_state_dict(router_state, strict=False)

        model.eval()
        self._model = model
        self._device = device

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tok_cfg = mm_cfg.get("tokenizer", None)
            if tok_cfg is None:
                raise RuntimeError("Tokenizer config missing; cannot build router tokenizer.")
            tokenizer = BUILDER.build(tok_cfg)

            base = (
                getattr(model, "llm", None)
                or getattr(model, "llava", None)
                or getattr(model, "language_model", None)
                or getattr(model, "model", None)
            )
            if base is not None and hasattr(base, "resize_token_embeddings"):
                try:
                    base.resize_token_embeddings(len(tokenizer))
                except Exception:  # pragma: no cover - best effort
                    pass
        self._tokenizer = tokenizer

        img_proc_cfg = mm_cfg.get("image_processor", None)
        if img_proc_cfg is None:
            raise RuntimeError("image_processor missing in mmengine config")
        self._image_processor = BUILDER.build(img_proc_cfg)

        self._image_token = cfg.image_token or mm_cfg.get("image_token", "<image>")
        self._rag_token = cfg.rag_token or getattr(model, "rag_token", "[RAG]")

    def _collect_router_state(self, payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        key_frags = ("rag_router", "router_head", "router.", "mlp_router")
        router_state: Dict[str, torch.Tensor] = {}

        def _visit(obj: Dict[str, torch.Tensor]) -> None:
            for key, value in obj.items():
                if any(frag in key for frag in key_frags):
                    router_state[key] = value

        if isinstance(payload, dict):
            if "state_dict" in payload:
                _visit(payload["state_dict"])
            else:
                _visit(payload)
        return router_state

    def _build_router_sample(self, question: str, image_path: str | Path) -> Dict[str, torch.Tensor]:
        instruction = self._prompt_template["INSTRUCTION"].format(input=question)
        text_parts = [self._image_token, self._rag_token, instruction]
        text = " ".join(part for part in text_parts if part)

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        bos_id = getattr(self._tokenizer, "bos_token_id", None) or getattr(
            self._tokenizer, "cls_token_id", None
        )
        if bos_id is not None:
            token_ids = [bos_id] + token_ids

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        image = self._image_processor.preprocess(str(image_path))
        pixel_values = image["pixel_values"]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device=self._device, dtype=self._model.dtype if hasattr(self._model, "dtype") else torch.float16)
        else:
            pixel_values = torch.tensor(pixel_values, device=self._device, dtype=torch.float16)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        sample: Dict[str, torch.Tensor] = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "pixel_values": pixel_values,
        }
        return sample

    def score(self, question: str, image_path: str | Path) -> RouterDecision:
        sample = self._build_router_sample(question, image_path)
        orig_tf32_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
        orig_tf32_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
        orig_bench = getattr(torch.backends.cudnn, "benchmark", None)
        try:
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass

        with torch.no_grad():
            outputs = self._model(sample, mode="tensor")

        try:
            if orig_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul  # type: ignore[attr-defined]
            if orig_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn  # type: ignore[attr-defined]
            if orig_bench is not None:
                torch.backends.cudnn.benchmark = orig_bench  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass

        prob_tensor = outputs.get("router_prob")
        logit_tensor = outputs.get("router_logit")

        if prob_tensor is None:
            if logit_tensor is None:
                raise RuntimeError("Router model did not return router_prob or router_logit")
            prob = float(torch.sigmoid(logit_tensor).reshape(-1)[0].item())
        else:
            prob = float(prob_tensor.reshape(-1)[0].item())

        use_rag = prob >= self.cfg.threshold
        return RouterDecision(
            prob=prob,
            threshold=self.cfg.threshold,
            use_rag=use_rag,
            backend="mmengine",
            extras={},
        )


class Router:
    """Public facade choosing between available router backends."""

    def __init__(self, cfg: RouterConfig) -> None:
        backend = cfg.backend.lower()
        if backend == "mmengine":
            try:
                self._backend = FrozenRouterBackend(cfg)
                self.backend_name = "mmengine"
            except Exception as exc:
                print(f"[Router] Falling back to heuristic backend: {exc}")
                self._backend = HeuristicRouter(cfg)
                self.backend_name = "heuristic"
        elif backend == "heuristic":
            self._backend = HeuristicRouter(cfg)
            self.backend_name = "heuristic"
        elif backend in {"always_on", "none"}:
            self._backend = AlwaysOnRouter(cfg)
            self.backend_name = "always_on"
        else:
            print(f"[Router] Unknown backend '{backend}', using heuristic fallback.")
            self._backend = HeuristicRouter(cfg)
            self.backend_name = "heuristic"

    def score(self, question: str, image_path: str | Path) -> RouterDecision:
        return self._backend.score(question, image_path)
