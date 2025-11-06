#!/usr/bin/env python3
"""Run an end-to-end demo combining FrozenLlavaSAM with Wikipedia RAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from xtuner.registry import BUILDER

from integration import RoutableFLMMPipeline, WikipediaRAGBridge


def _load_model(cfg: Config, checkpoint: str | None, device: str) -> Any:
    model = BUILDER.build(cfg.model)
    model = model.to(device)
    if checkpoint:
        from xtuner.model.utils import guess_load_checkpoint

        print(f"Loading checkpoint: {checkpoint}")
        state = guess_load_checkpoint(checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(
                f"Unexpected keys ({len(unexpected)}): {unexpected[:8]}"
                f"{' ...' if len(unexpected) > 8 else ''}"
            )
    model.eval()
    return model


def _load_tokenizer(cfg: Config, model: Any) -> Any:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    tok_cfg = cfg.tokenizer
    if tok_cfg is None:
        raise RuntimeError("Tokenizer config missing from F-LMM config file")
    tokenizer = BUILDER.build(tok_cfg)
    base = getattr(model, "llm", None) or getattr(model, "model", None) or model
    base.resize_token_embeddings(len(tokenizer))
    return tokenizer


def _load_image_processor(cfg: Config) -> Any:
    image_proc_cfg = cfg.image_processor
    if image_proc_cfg is None:
        raise RuntimeError("Image processor config missing from F-LMM config file")
    return BUILDER.build(image_proc_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="User question to answer")
    parser.add_argument("image", help="Path to the query image")
    parser.add_argument("--flmm-config", required=True, help="FrozenLlavaSAM mmengine config")
    parser.add_argument("--flmm-checkpoint", help="Path to a trained FrozenLlavaSAM checkpoint")
    parser.add_argument("--wikipedia-root", required=True, help="Path to the wikipedia project root")
    parser.add_argument(
        "--wikipedia-config",
        default="config.yaml",
        help="Relative or absolute path to Wikipedia RAG config YAML",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rag-threshold", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--disable-nli", action="store_true")
    parser.add_argument("--disable-vlm", action="store_true")
    parser.add_argument("--no-generate", action="store_true", help="Skip VLM answer generation")

    args = parser.parse_args()

    cfg = Config.fromfile(args.flmm_config)
    model = _load_model(cfg, args.flmm_checkpoint, args.device)
    tokenizer = _load_tokenizer(cfg, model)
    image_processor = _load_image_processor(cfg)

    prompt_template = cfg.prompt_template
    image_token = cfg.get("image_token", "<image>")
    rag_token = getattr(model, "rag_token", "[RAG]")

    bridge = WikipediaRAGBridge(
        wikipedia_root=args.wikipedia_root,
        config_path=args.wikipedia_config,
        enable_nli=not args.disable_nli,
        enable_vlm=not args.disable_vlm,
    )

    pipeline = RoutableFLMMPipeline(
        model=model,
        rag_bridge=bridge,
        tokenizer=tokenizer,
        image_processor=image_processor,
        prompt_template=prompt_template,
        image_token=image_token,
        rag_token=rag_token,
    )

    result = pipeline.route(
        question=args.question,
        image_path=Path(args.image),
        rag_threshold=args.rag_threshold,
        generate_answer=not args.no_generate,
        max_new_tokens=args.max_new_tokens,
    )

    retr = result["retrieval"]

    print("=== Router Decision ===")
    print(f"P(use_rag) = {result['router_prob']:.4f} (threshold {result['router_threshold']:.4f})")
    print(f"Using retrieval: {result['use_rag']}")
    print()

    if retr.sections:
        print("=== Retrieved Context ===")
        for idx, (meta, text) in enumerate(zip(retr.section_metadata, retr.sections), 1):
            title = meta.get("source_title") or meta.get("doc_title")
            section_title = meta.get("section_title") or meta.get("section_id")
            print(f"[{idx}] {title} :: {section_title}")
            print(text.strip()[:400])
            if len(text) > 400:
                print("...")
            print()
    else:
        print("No sections were selected for this query.")

    if retr.answer is not None:
        print("=== Generated Answer ===")
        print(retr.answer)

    summary = {
        "question": retr.question,
        "image_path": retr.image_path,
        "use_rag": retr.use_retrieval,
        "router_prob": result["router_prob"],
        "router_threshold": result["router_threshold"],
        "answer": retr.answer,
    }
    print()
    print("=== JSON Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
