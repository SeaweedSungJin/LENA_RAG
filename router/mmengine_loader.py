"""Helpers to load FrozenLlavaSAM components through mmengine for routing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def ensure_vendor_paths() -> None:
    """Ensure vendored ``rag_flmm`` package is importable."""
    project_root = Path(__file__).resolve().parent.parent
    rag_flmm_root = project_root / "rag_flmm"
    if rag_flmm_root.exists():
        entry = str(rag_flmm_root)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def resolve_router_checkpoint(router_ckpt: Optional[Path]) -> Optional[Path]:
    """Resolve a router checkpoint which may be a directory or shard."""
    if router_ckpt is None:
        return None

    candidate = Path(router_ckpt)
    if not candidate.exists():
        return None

    if candidate.is_file():
        return candidate

    # Case 1: directory with 'last_checkpoint' pointer
    last_ckpt = candidate / "last_checkpoint"
    if last_ckpt.exists():
        try:
            pointer = last_ckpt.read_text(encoding="utf-8").strip()
        except Exception:
            pointer = ""
        if pointer:
            pointer_path = Path(pointer)
            if not pointer_path.is_absolute():
                pointer_path = candidate / pointer_path.name
            if pointer_path.exists():
                candidate = pointer_path

    # Case 2: DeepSpeed mp_rank shard
    shard = candidate / "mp_rank_00_model_states.pt"
    if shard.exists():
        return shard

    # Case 3: pick the first model_states file
    try:
        matches = sorted(candidate.glob("*model_states.pt"))
    except Exception:
        matches = []
    if matches:
        return matches[0]

    return candidate if candidate.exists() else None


def _remap_state_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    remapped: Dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        remapped[new_key] = value
    return remapped


def load_frozen_llava_components(
    *,
    config_path: Path,
    device: str,
    checkpoint_path: Optional[Path],
) -> Tuple[Any, Any, Dict[str, Any], Any, str, str]:
    """Load FrozenLlavaSAM model/tokenizer/image processor via mmengine."""
    import torch
    from mmengine.config import Config as MMConfig
    from xtuner.model.utils import guess_load_checkpoint
    from xtuner.registry import BUILDER

    mm_cfg = MMConfig.fromfile(str(config_path))

    model = BUILDER.build(mm_cfg.model)
    model = model.to(torch.device(device))

    if checkpoint_path is not None:
        raw_state = guess_load_checkpoint(str(checkpoint_path))
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]
        if isinstance(raw_state, dict):
            state = _remap_state_keys(raw_state)
        else:
            state = raw_state
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[FrozenLlavaSAM] base checkpoint missing ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(
                "[FrozenLlavaSAM] base checkpoint unexpected "
                f"({len(unexpected)}): {unexpected[:8]}"
            )
    model.eval()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tok_cfg = mm_cfg.get("tokenizer", None)
        if tok_cfg is None:
            raise RuntimeError("Tokenizer config missing in FrozenLlavaSAM mmengine config")
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
            except Exception:
                pass

    img_proc_cfg = mm_cfg.get("image_processor", None)
    if img_proc_cfg is None:
        raise RuntimeError("image_processor block missing in FrozenLlavaSAM config")
    image_processor = BUILDER.build(img_proc_cfg)

    prompt_template = mm_cfg.get("prompt_template", None)
    if prompt_template is None or "INSTRUCTION" not in prompt_template:
        raise RuntimeError("prompt_template with INSTRUCTION key is required in mmengine config")

    image_token = mm_cfg.get("image_token", "<image>")
    rag_token = getattr(model, "rag_token", "[RAG]")

    return model, tokenizer, prompt_template, image_processor, image_token, rag_token
