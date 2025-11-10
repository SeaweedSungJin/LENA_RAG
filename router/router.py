"""Router inference helpers (inspired by flmm_wiki_rag)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from .config import RouterConfig
from .mmengine_loader import (
    ensure_vendor_paths,
    load_frozen_llava_components,
    resolve_router_checkpoint,
)


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
        # Break ties conservatively: prob == threshold -> NO_RAG
        use_rag = prob > self.cfg.threshold
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

        ensure_vendor_paths()

        try:
            from xtuner.model.utils import guess_load_checkpoint
        except Exception as exc:  # pragma: no cover - dependency import
            raise RuntimeError("mmengine/xtuner are required for the mmengine router backend") from exc

        if cfg.config_path is None:
            raise ValueError("RouterConfig.config_path must be provided for backend 'mmengine'")

        device = torch.device(cfg.device)

        # Allow torch.load to deserialize DeepSpeed metadata in older checkpoints.
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            import deepspeed.runtime.fp16.loss_scaler as ds_loss_scaler  # type: ignore
            import deepspeed.runtime.zero.config as ds_zero_config  # type: ignore
            import deepspeed.utils.tensor_fragment as ds_tensor_fragment  # type: ignore

            add_safe_globals(
                [
                    ds_loss_scaler.DynamicLossScaler,
                    ds_zero_config.ZeroStageEnum,
                    ds_tensor_fragment.fragment_address,
                ]
            )
        except Exception:
            pass

        orig_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load  # type: ignore[assignment]
        try:
            (
                model,
                tokenizer,
                prompt_template,
                image_processor,
                default_image_token,
                default_rag_token,
            ) = load_frozen_llava_components(
                config_path=cfg.config_path,
                device=cfg.device,
                checkpoint_path=cfg.checkpoint_path,
            )

            base_module = (
                getattr(model, "llm", None)
                or getattr(model, "llava", None)
                or getattr(model, "language_model", None)
                or getattr(model, "model", None)
            )
            try:
                first_param = next(
                    iter(
                        list(base_module.parameters())
                        if base_module is not None
                        else list(model.parameters())
                    )
                )
            except StopIteration:
                first_param = None
            self._model_dtype = (
                first_param.dtype
                if first_param is not None
                else getattr(model, "dtype", torch.float16)
            )

            self._model = model
            self._device = device
            self._debug_samples = 0

            self._tokenizer = tokenizer
            self._prompt_template = prompt_template
            if "INSTRUCTION" not in self._prompt_template:
                raise ValueError("prompt_template with INSTRUCTION key is required in mmengine config")

            self._materialize_router_head()

            router_ckpt = resolve_router_checkpoint(cfg.router_checkpoint)
            if router_ckpt:
                state = guess_load_checkpoint(str(router_ckpt))
                state = self._remap_checkpoint_keys(state)
                router_state = self._collect_router_state(state)
                mlp_keys = [k for k in router_state if "rag_router.mlp" in k]
                if router_state:
                    print(f"[Router] router parameters extracted: {len(router_state)}")
                    if not mlp_keys:
                        print(
                            "[Router] WARNING: router checkpoint is missing 'rag_router.mlp' weights. "
                            "Router head will fall back to freshly initialised parameters."
                        )
                    missing_r, unexpected_r = model.load_state_dict(router_state, strict=False)
                    if missing_r:
                        print(f"[Router] router_ckpt missing ({len(missing_r)}): {missing_r[:8]}")
                    if unexpected_r:
                        print(f"[Router] router_ckpt unexpected ({len(unexpected_r)}): {unexpected_r[:8]}")
                else:
                    print(f"[Router] router state not found in {router_ckpt}")
            elif cfg.router_checkpoint:
                print(f"[Router] router checkpoint not found: {cfg.router_checkpoint}")

            model.eval()
        finally:
            torch.load = orig_torch_load  # type: ignore[assignment]

        self._image_processor = image_processor

        self._image_token = cfg.image_token or default_image_token
        self._rag_token = cfg.rag_token or default_rag_token

    def _collect_router_state(self, payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        key_frags = ("rag_router", "router_head", "router.", "mlp_router")
        router_state: Dict[str, torch.Tensor] = {}

        def _visit(obj: Dict[str, torch.Tensor]) -> None:
            for key, value in obj.items():
                if isinstance(key, str) and any(frag in key for frag in key_frags):
                    new_key = key[len("module.") :] if key.startswith("module.") else key
                    router_state[new_key] = value
                elif isinstance(value, dict):
                    _visit(value)

        if isinstance(payload, dict):
            if "state_dict" in payload:
                _visit(payload["state_dict"])
            elif "module" in payload and isinstance(payload["module"], dict):
                _visit(payload["module"])
            else:
                _visit(payload)
        return router_state

    def _materialize_router_head(self) -> None:
        """Ensure dynamic router head modules exist before loading weights."""

        model = getattr(self, "_model", None)
        device = getattr(self, "_device", None)
        dtype = getattr(self, "_model_dtype", torch.float16)
        if model is None or device is None:
            return
        router = getattr(model, "rag_router", None)
        if router is None:
            return

        try:
            txt_dim = router.txt_proj.in_features
            vis_dim = router.vis_proj.in_features
            base_dim = getattr(model, "_router_in_dim", 3 * txt_dim)

            base_feat = torch.zeros(1, base_dim, device=device, dtype=dtype)
            txt_feat = torch.zeros(1, txt_dim, device=device, dtype=dtype)
            vis_feat = torch.zeros(1, vis_dim, device=device, dtype=dtype)

            vocab_size = None
            llm = getattr(model, "llm", None)
            if llm is not None:
                vocab_size = getattr(getattr(llm, "config", None), "vocab_size", None)
            if vocab_size is None:
                vocab_size = getattr(model, "vocab_size", None)
            if vocab_size is None and hasattr(self, "_tokenizer"):
                try:
                    vocab_size = len(self._tokenizer)
                except Exception:
                    vocab_size = None
            txt_logits = None
            if isinstance(vocab_size, int) and vocab_size > 0:
                txt_logits = torch.zeros(1, vocab_size, device=device, dtype=dtype)

            with torch.no_grad():
                router(
                    base_feat=base_feat,
                    txt_feat=txt_feat,
                    vis_feat=vis_feat,
                    txt_logits=txt_logits,
                    vis_logits=None,
                    ret_stats=None,
                )
        except Exception as exc:  # pragma: no cover - diagnostic guard
            print(f"[Router] warning: failed to materialize router head ({exc})")

    def _remap_checkpoint_keys(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(state, dict):
            return state
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("llm.base_model.model.model."):
                new_key = "llm.model." + new_key[len("llm.base_model.model.model.") :]
            elif new_key.startswith("llm.base_model.model."):
                new_key = "llm.model." + new_key[len("llm.base_model.model.") :]
            elif new_key.startswith("llm.base_model."):
                new_key = "llm." + new_key[len("llm.base_model.") :]
            remapped[new_key] = value
        return remapped

    def _build_router_sample(self, question: str, image_path: str | Path) -> Dict[str, torch.Tensor]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Router input image not found: {image_path}")

        instruction = self._prompt_template["INSTRUCTION"].format(input=question)
        text_parts: list[str] = []
        if self._image_token:
            text_parts.append(self._image_token)
        if self._rag_token:
            text_parts.append(self._rag_token)
        text_parts.append(instruction)
        text = " ".join(part for part in text_parts if part)

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        bos_id = getattr(self._tokenizer, "bos_token_id", None) or getattr(
            self._tokenizer, "cls_token_id", None
        )
        if bos_id is not None:
            token_ids = [bos_id] + token_ids

        base = (
            getattr(self._model, "llm", None)
            or getattr(self._model, "llava", None)
            or getattr(self._model, "language_model", None)
            or getattr(self._model, "model", None)
        )
        device = getattr(base, "device", None)
        if device is None:
            device = self._device
        device = torch.device(device)
        dtype = getattr(base, "dtype", None)
        if dtype is None:
            dtype = getattr(self, "_model_dtype", torch.float16)

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        with Image.open(str(image_path)) as img:
            image = img.convert("RGB")
        processed = self._image_processor.preprocess(image)
        pixel_values = processed["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = pixel_values[0]
        if isinstance(pixel_values, torch.Tensor):
            tensor = pixel_values
        else:
            tensor = torch.as_tensor(pixel_values)
        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                f"Router pixel tensor should be 3D after preprocessing, got shape {tuple(tensor.shape)}"
            )
        pixel_values = tensor.to(device=device, dtype=dtype, non_blocking=True).contiguous()

        sample: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if self._debug_samples < 3:
            self._debug_samples += 1
            try:
                print(
                    f"[Router] sample input_ids={sample['input_ids'].shape} "
                    f"pixel_values={sample['pixel_values'].shape} device={pixel_values.device}"
                )
                pv = sample["pixel_values"].float()
                print(
                    f"[Router] pixel stats min={pv.min().item():.4f} "
                    f"max={pv.max().item():.4f} mean={pv.mean().item():.4f}"
                )
            except Exception:
                pass
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

        if logit_tensor is not None:
            try:
                raw_logit = float(logit_tensor.reshape(-1)[0].item())
                print(f"[Router] raw_logit={raw_logit:.6e}")
            except Exception:
                pass

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
        decision = self._backend.score(question, image_path)
        try:
            print(
                f"[Router] backend={self.backend_name} prob={decision.prob:.4f}\n"
                f"[Router] threshold={decision.threshold:.4f} -> use_rag={decision.use_rag}"
            )
        except Exception:
            # Best-effort logging; keep inference resilient to formatting issues.
            print(
                f"[Router] backend={self.backend_name} prob={decision.prob} "
                f"threshold={decision.threshold} -> use_rag={decision.use_rag}"
            )
        return decision

    # --- VLM direct generation support ---
    def get_generation_components(self):
        """Expose VLM components for direct generation when available.

        Returns a dict with keys: model, tokenizer, image_processor,
        prompt_template, image_token.
        """
        backend = getattr(self, "_backend", None)
        if backend is None:
            return None

    def generate_vlm_answer(
        self,
        question: str,
        image_path: str | Path,
        *,
        context: Optional[str] = None,
        max_new_tokens: int = 64,
    ) -> Optional[str]:
        """Generate an answer using the VLM (image + question) when available.

        Returns a string on success, or None if the backend doesn't support
        direct generation.
        """
        comps = self.get_generation_components()
        if not comps:
            return None
        try:
            import torch
            from PIL import Image

            model = comps["model"]
            tokenizer = comps["tokenizer"]
            image_processor = comps["image_processor"]
            prompt_template = comps["prompt_template"]
            image_token = comps.get("image_token", "<image>")

            # Format prompt similar to rag_flmm integration, with context-aware instructions
            # Build prompt in the requested style. Keep image token on a separate line.
            question_text = str(question or "").strip()
            context_text = context.strip() if isinstance(context, str) else ""
            user_lines = []
            if context_text:
                user_lines.append(f"Context:\n{context_text}")
            user_lines.append(f"Question: {question_text}")
            user_lines.append("Just answer the question, no explanations.")
            user_lines.append("Short answer is:")

            parts = []
            parts.append('SYSTEM: "You always answer exactly the asked question. No extra text."')
            if image_token:
                parts.append(image_token)
            parts.append("USER:\n" + "\n".join(user_lines))
            prompt = "\n".join(parts)

            tokenized = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids))

            # Underlying HF module
            base = (
                getattr(model, "llm", None)
                or getattr(model, "language_model", None)
                or getattr(model, "model", None)
                or getattr(model, "llava", None)
                or model
            )
            try:
                device = next(base.parameters()).device  # type: ignore[attr-defined]
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = getattr(base, "dtype", torch.float16)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with Image.open(str(image_path)) as raw_img:
                img = raw_img.convert("RGB")
                processed = image_processor.preprocess(img, return_tensors="pt")

            pixel_values = processed.get("pixel_values")
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device=device, dtype=dtype)
            else:
                pixel_values = torch.tensor(pixel_values, device=device, dtype=dtype)
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)

            with torch.no_grad():
                output_ids = base.generate(  # type: ignore[attr-defined]
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                )
            generated = output_ids[:, input_ids.shape[1]:]
            answer = tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            return answer.strip()
        except Exception:
            return None
        # Only FrozenRouterBackend exposes these attributes
        for attr in ("_model", "_tokenizer", "_image_processor", "_prompt_template"):
            if not hasattr(backend, attr):
                return None
        try:
            model = backend._model
            tokenizer = backend._tokenizer
            image_processor = backend._image_processor
            prompt_template = backend._prompt_template
            image_token = getattr(backend, "_image_token", "<image>")
            return {
                "model": model,
                "tokenizer": tokenizer,
                "image_processor": image_processor,
                "prompt_template": prompt_template,
                "image_token": image_token,
            }
        except Exception:
            return None
