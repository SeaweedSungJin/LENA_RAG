"""Integration utilities bridging F-LMM router outputs with the
Wikipedia RAG pipeline.

This module allows the FrozenLlavaSAM router (from the F-LMM project) to
invoke the encyclopedic RAG stack housed under ``wikipedia/src`` when the
model predicts that external retrieval is required.  It exposes two
primary abstractions:

* :class:`WikipediaRAGBridge` – a lightweight wrapper that loads the
  Wikipedia pipeline configuration, runs retrieval/NLI clustering, and
  (optionally) generates a final answer using the pipeline's VLM helper.
* :class:`RoutableFLMMPipeline` – an orchestrator that feeds a question
  and image through the FrozenLlavaSAM router to obtain the retrieval
  decision, then delegates to :class:`WikipediaRAGBridge` to fetch
  supporting context and produce an answer.

The implementation deliberately caches all heavyweight models (FAISS
index, rerankers, NLI, VLM) inside module-level singletons exposed by the
Wikipedia project so that multiple queries can be served efficiently
within the same Python process.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


@dataclass
class RAGRetrievalResult:
    """Container holding the outcome of a single Wikipedia RAG lookup."""

    question: str
    image_path: str
    use_retrieval: bool
    sections: List[str]
    section_metadata: List[dict]
    image_results: List[dict]
    search_elapsed: float
    nli_elapsed: float
    answer: Optional[str] = None


class WikipediaRAGBridge:
    """Adapter around the encyclopedic VQA RAG pipeline.

    Parameters
    ----------
    wikipedia_root:
        Path to the ``wikipedia`` project checkout (the directory that
        contains ``src/`` and ``config.yaml``).
    config_path:
        Path to the YAML configuration file controlling the RAG pipeline
        (typically ``wikipedia/config.yaml``).
    config_overrides:
        Optional mapping of attribute overrides applied after the config
        file is loaded (e.g. device placements or FAISS parameters).
    enable_nli:
        Whether to run the graph-based NLI clustering step to refine the
        retrieved sections.
    enable_vlm:
        Whether to load and use the pipeline's VLM (LLaVA 1.6 Mistral)
        to generate final answers.  When disabled, callers receive the
        retrieved context and may handle answer generation separately.
    """

    def __init__(
        self,
        wikipedia_root: str | Path,
        config_path: str | Path,
        *,
        config_overrides: Optional[Dict[str, Any]] = None,
        enable_nli: bool = True,
        enable_vlm: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.wikipedia_root = Path(wikipedia_root).resolve()
        if not self.wikipedia_root.exists():
            raise FileNotFoundError(f"Wikipedia root not found: {self.wikipedia_root}")

        self._ensure_import_path()

        from src.utils import download_nltk_data  # type: ignore
        from src.models import setup_cuda  # type: ignore
        from src.config import Config

        # One-time initialisation for tokenizers and CUDA heuristics.
        try:
            download_nltk_data()
        except Exception:
            pass
        try:
            setup_cuda()
        except Exception:
            pass

        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.wikipedia_root / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Wikipedia config not found: {config_path}")

        self._config_path = config_path
        self._base_cfg = Config.from_yaml(str(config_path))
        if config_overrides:
            for key, value in config_overrides.items():
                if not hasattr(self._base_cfg, key):
                    raise AttributeError(f"Config has no attribute '{key}'")
                setattr(self._base_cfg, key, value)

        self.enable_nli = enable_nli
        self.enable_vlm = enable_vlm

        self._clip_cache_handle = None
        self._clip_cache_flush_interval = max(1, int(os.getenv("WIKI_CLIP_CACHE_FLUSH", "32")))
        self._clip_cache_pending = 0
        self._kb_doc_count: Optional[int] = None

        self._nli_model = None
        self._nli_tokenizer = None
        self._vlm_model = None
        self._vlm_processor = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_import_path(self) -> None:
        """Insert the Wikipedia ``src`` directory into ``sys.path``."""

        src_path = self.wikipedia_root / "src"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Wikipedia src directory not found: {src_path}"  # pragma: no cover - defensive
            )
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def _preflight_config(self, cfg) -> None:
        """Validate critical paths and normalize image path.

        - Ensure KB JSON and FAISS index files exist; raise helpful errors if not.
        - If ``image_path`` is relative, try resolving it under ``wikipedia_root``.
        """
        from pathlib import Path as _P

        # Normalize image path
        img = getattr(cfg, "image_path", None)
        if isinstance(img, str) and img and not img.startswith("http"):
            p = _P(img)
            if not p.is_absolute():
                cand = self.wikipedia_root / img
                if cand.exists():
                    cfg.image_path = str(cand)

        # Validate KB files
        base = getattr(cfg, "base_path", None)
        kb_json_name = getattr(cfg, "kb_json_name", None)
        if base and kb_json_name:
            kb_json = _P(base) / kb_json_name
            if not kb_json.exists():
                raise FileNotFoundError(f"KB JSON not found: {kb_json}. Set wikipedia.base_path/kb_json_name correctly.")
            faiss_idx = _P(base) / "kb_index.faiss"
            faiss_ids = _P(base) / "kb_index_ids.pkl"
            if not faiss_idx.exists():
                raise FileNotFoundError(f"FAISS index not found: {faiss_idx}")
            if not faiss_ids.exists():
                raise FileNotFoundError(f"FAISS ids map not found: {faiss_ids}")

    def _clone_config(self, **kwargs: Any):
        """Return a shallow copy of the base config with overrides."""

        cfg = replace(self._base_cfg)
        for key, value in kwargs.items():
            setattr(cfg, key, value)
        return cfg

    def _ensure_clip_cache(self, cfg) -> Any:
        cache_dir = getattr(cfg, "clip_cache_dir", None)
        if not cache_dir:
            return None
        handle = self._clip_cache_handle
        clip_k_max = max(cfg.k_value, int(getattr(cfg, "clip_cache_kmax", cfg.k_value)))
        if handle is not None:
            handle.bump_kmax(clip_k_max)
            return handle

        from src import clip_cache  # type: ignore
        from src.utils import load_kb  # type: ignore

        if self._kb_doc_count is None:
            kb_list, _ = load_kb(cfg.kb_json_path)
            self._kb_doc_count = len(kb_list)

        meta = clip_cache.build_meta(
            dataset_csv=cfg.dataset_csv,
            dataset_start=cfg.dataset_start,
            dataset_end=cfg.dataset_end,
            kb_json_path=cfg.kb_json_path,
            kb_doc_count=self._kb_doc_count,
            image_encoder=getattr(cfg, "image_encoder_name", "BAAI/EVA-CLIP-8B"),
            base_path=cfg.base_path,
            clip_k_max=clip_k_max,
        )
        handle = clip_cache.open_cache(cache_dir, meta)
        handle.bump_kmax(clip_k_max)
        self._clip_cache_handle = handle
        self._clip_cache_pending = 0
        return handle

    def _maybe_flush_clip_cache(self, force: bool = False) -> None:
        handle = self._clip_cache_handle
        if handle is None or not handle.dirty:
            return

        from src import clip_cache  # type: ignore

        if force:
            clip_cache.save_cache(handle)
            self._clip_cache_pending = 0
            return

        self._clip_cache_pending += 1
        if self._clip_cache_pending >= self._clip_cache_flush_interval:
            clip_cache.save_cache(handle)
            self._clip_cache_pending = 0

    def flush_caches(self) -> None:
        """Flush any pending cache updates to disk."""
        self._maybe_flush_clip_cache(force=True)

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.flush_caches()
        except Exception:
            pass

    def _ensure_nli(self) -> None:
        if not self.enable_nli:
            return
        if self._nli_model is not None:
            return
        from src.models import load_nli_model, resolve_device

        device = resolve_device(self._base_cfg.nli_device)
        nli_choice = self._base_cfg.nli_models.get
        if nli_choice("deberta", False):
            model_name = self._base_cfg.deberta_nli_model
        elif nli_choice("roberta", False):
            model_name = self._base_cfg.roberta_nli_model
        elif nli_choice("deberta_v3", False):
            model_name = self._base_cfg.deberta_v3_nli_model
        else:
            model_name = self._base_cfg.deberta_nli_model
        self._nli_model, self._nli_tokenizer = load_nli_model(model_name, device)

    def _ensure_vlm(self) -> None:
        if not self.enable_vlm:
            return
        if self._vlm_model is not None and self._vlm_processor is not None:
            return
        from src.models import load_vlm_model

        self._vlm_model, self._vlm_processor = load_vlm_model(
            device_map=self._base_cfg.vlm_device
        )

    def _build_parent_uid(self, section: Dict[str, Any]) -> tuple:
        return (
            section.get("source_title", ""),
            section.get("section_idx"),
            section.get("section_title", ""),
        )

    def _effective_nli_max_length(self) -> int:
        max_len = int(getattr(self._base_cfg, "nli_max_length", 512))
        if self._nli_tokenizer is not None:
            tok_len = getattr(self._nli_tokenizer, "model_max_length", None)
            if isinstance(tok_len, int) and 0 < tok_len < 1_000_000:
                max_len = min(max_len, tok_len)
        if self._nli_model is not None:
            mdl_len = getattr(getattr(self._nli_model, "config", None), "max_position_embeddings", None)
            if isinstance(mdl_len, int) and mdl_len > 0:
                max_len = min(max_len, mdl_len)
        return max_len

    def _score_question_entailment(self, sentences: List[str], question: str) -> List[float]:
        if not sentences:
            return []
        if self._nli_model is None or self._nli_tokenizer is None:
            return [0.0] * len(sentences)
        # Optional: convert question to declarative form for better NLI scoring
        try:
            from src.utils import convert_question_to_statement  # type: ignore
            use_stmt = bool(getattr(self._base_cfg, "nli_convert_question_to_statement", False))
            hyp = convert_question_to_statement(question) if use_stmt else question
            if use_stmt and (not hyp or hyp.strip() == ""):
                hyp = question
        except Exception:
            hyp = question
        max_len = self._effective_nli_max_length()
        inputs = self._nli_tokenizer(
            sentences,
            [hyp] * len(sentences),
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        device = getattr(self._nli_model, "device", "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        entail_idx = 2  # [contradiction, neutral, entailment]
        return probs[:, entail_idx].tolist()

    def _prepare_sentence_candidates(
        self,
        sections: List[dict],
        question: str,
    ) -> List[dict]:
        from src.utils import get_spacy_sentencizer  # type: ignore

        if not sections:
            return []

        try:
            sentencizer = get_spacy_sentencizer()
        except ImportError:
            # If spaCy is unavailable fall back to original sections.
            return [dict(sec) for sec in sections]

        sentences: List[dict] = []
        for sec in sections:
            text = str(sec.get("section_text", "")).strip()
            if not text:
                continue
            doc = sentencizer(text)
            parent_uid = self._build_parent_uid(sec)
            for idx, span in enumerate(doc.sents):
                sent_text = span.text.strip()
                if not sent_text:
                    continue
                candidate = dict(sec)
                candidate["section_text"] = sent_text
                candidate["section_title"] = f"{sec.get('section_title', '')} [sent {idx}]"
                candidate["sentence_idx"] = idx
                candidate["parent_uid"] = parent_uid
                sentences.append(candidate)

        if not sentences:
            return []

        threshold = float(getattr(self._base_cfg, "nli_sentence_entail_threshold", 0.5))
        if self.enable_nli:
            self._ensure_nli()
            if self._nli_model is not None and self._nli_tokenizer is not None:
                entail_scores = self._score_question_entailment(
                    [s["section_text"] for s in sentences],
                    question,
                )
                for sent, score in zip(sentences, entail_scores):
                    sent["question_entailment"] = float(score)

                if threshold > 0.0:
                    filtered = [s for s in sentences if s.get("question_entailment", 0.0) >= threshold]
                    if filtered:
                        return filtered
                    # Fallback: keep top-M sentences even if none pass threshold
                    sentences.sort(key=lambda s: s.get("question_entailment", 0.0), reverse=True)
                    top_k = max(1, min(len(sentences), int(getattr(self._base_cfg, "m_value", len(sentences)))))
                    return sentences[:top_k]

        return sentences

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        image_path: str | Path,
        *,
        use_retrieval: bool = True,
        nli_override: Optional[bool] = None,
    ) -> RAGRetrievalResult:
        """Run the Wikipedia RAG pipeline and return retrieved sections."""

        from src.pipeline import search_rag_pipeline
        from src.nli_cluster import cluster_sections_consistency

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # If router decided NOT to use retrieval, short-circuit without search
        if not use_retrieval:
            return RAGRetrievalResult(
                question=str(question),
                image_path=str(image_path),
                use_retrieval=False,
                sections=[],
                section_metadata=[],
                image_results=[],
                search_elapsed=0.0,
                nli_elapsed=0.0,
            )

        cfg = self._clone_config(text_query=str(question), image_path=str(image_path))
        # Validate paths and normalize before search
        self._preflight_config(cfg)
        clip_handle = self._ensure_clip_cache(cfg)
        img_results, top_sections, final_sections, elapsed = search_rag_pipeline(
            cfg,
            clip_cache_handle=clip_handle,
            manage_clip_cache=clip_handle is None,
        )
        if clip_handle is not None:
            self._maybe_flush_clip_cache()

        top_sections = list(top_sections)
        final_sections = list(final_sections)

        sentence_candidates = self._prepare_sentence_candidates(top_sections, str(question))
        if final_sections:
            target_uids = {self._build_parent_uid(sec) for sec in final_sections}
            sentence_final = [s for s in sentence_candidates if s.get("parent_uid") in target_uids]
            if not sentence_final:
                sentence_final = sentence_candidates
        else:
            sentence_final = sentence_candidates

        if sentence_final:
            sections_to_use = sentence_final
            top_for_nli = sentence_candidates
        else:
            sections_to_use = final_sections if final_sections else top_sections
            top_for_nli = top_sections

        section_metadata = [dict(sec) for sec in sections_to_use]
        sections_text = [sec.get("section_text", "") for sec in sections_to_use]
        nli_elapsed = 0.0

        should_run_nli = (self.enable_nli if nli_override is None else nli_override) and bool(top_for_nli)
        if should_run_nli:
            self._ensure_nli()
            candidate_sections = [
                {
                    "doc_title": sec.get("source_title", ""),
                    "section_id": sec.get("section_idx"),
                    "section_title": sec.get("section_title", ""),
                    "section_text": sec.get("section_text", ""),
                    "similarity": sec.get("similarity", 0.0),
                    "nli_text": (
                        f"# Article: {sec.get('source_title', '')}\n"
                        f"## Section: {sec.get('section_title', sec.get('section_idx'))}\n"
                        f"{sec.get('section_text', '')}"
                    ),
                }
                for sec in top_for_nli
            ]
            if candidate_sections and self._nli_model is not None:
                import time

                start = time.time()
                clusters, _ = cluster_sections_consistency(
                    candidate_sections,
                    model=self._nli_model,
                    tokenizer=self._nli_tokenizer,
                    max_length=self._base_cfg.nli_max_length,
                    device=self._nli_model.device,  # type: ignore[attr-defined]
                    alpha=getattr(self._base_cfg, "nli_alpha", 1.0),
                    beta=getattr(self._base_cfg, "nli_beta", 1.0),
                    batch_size=self._base_cfg.nli_batch_size,
                    tau_edge=max(0.0, min(1.0, getattr(self._base_cfg, "nli_tau", 0.0))),
                    hybrid_lambda=getattr(self._base_cfg, "nli_hybrid_lambda", 0.5),
                    edge_rule=getattr(self._base_cfg, "nli_edge_rule", "avg"),
                    dir_margin=getattr(self._base_cfg, "nli_dir_margin", 0.0),
                    target_size=getattr(self._base_cfg, "nli_max_cluster", 3),
                    autocast=getattr(self._base_cfg, "nli_autocast", True),
                    autocast_dtype=getattr(self._base_cfg, "nli_autocast_dtype", "fp16"),
                    reduction_mode=getattr(self._base_cfg, "nli_reduction_mode", "greedy"),
                    reduction_epsilon=getattr(self._base_cfg, "nli_reduction_epsilon", 1e-6),
                )
                nli_elapsed = time.time() - start
                if clusters:
                    best_cluster = clusters[0]["sections"]
                    sections_text = [sec.get("section_text", "") for sec in best_cluster]
                    section_metadata = best_cluster

        return RAGRetrievalResult(
            question=str(question),
            image_path=str(image_path),
            use_retrieval=use_retrieval,
            sections=sections_text,
            section_metadata=section_metadata,
            image_results=img_results,
            search_elapsed=elapsed,
            nli_elapsed=nli_elapsed,
        )

    def generate_answer(
        self,
        question: str,
        image_path: str | Path,
        *,
        use_retrieval: bool = True,
        nli_override: Optional[bool] = None,
        max_new_tokens: int = 64,
    ) -> RAGRetrievalResult:
        """Retrieve context (optionally) and run VLM generation."""

        retrieval = self.retrieve(
            question,
            image_path,
            use_retrieval=use_retrieval,
            nli_override=nli_override,
        )
        if not self.enable_vlm:
            return retrieval

        self._ensure_vlm()
        from src.models import generate_vlm_answer

        answer = generate_vlm_answer(
            self._vlm_model,
            self._vlm_processor,
            question,
            str(image_path),
            retrieval.sections,
            max_new_tokens=max_new_tokens,
        )
        retrieval.answer = answer
        return retrieval


class RoutableFLMMPipeline:
    """Combine FrozenLlavaSAM routing with Wikipedia RAG retrieval."""

    def __init__(
        self,
        model: Any,
        *,
        rag_bridge: WikipediaRAGBridge,
        tokenizer: Any,
        image_processor: Any,
        prompt_template: Dict[str, str],
        image_token: str = "<image>",
        rag_token: str = "[RAG]",
    ) -> None:
        self.model = model
        self.rag_bridge = rag_bridge
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.prompt_template = prompt_template
        self.image_token = image_token
        self.rag_token = rag_token

    def _format_direct_prompt(self, question: str) -> str:
        template = self.prompt_template or {}
        system_prompt = template.get("SYSTEM", "")
        roles = template.get("ROLES", ("USER", "ASSISTANT"))
        if isinstance(roles, (list, tuple)) and len(roles) >= 2:
            user_role = str(roles[0])
            assistant_role = str(roles[1])
        else:
            user_role, assistant_role = "USER", "ASSISTANT"

        instruction_template = template.get("INSTRUCTION")
        if isinstance(instruction_template, str) and "{input}" in instruction_template:
            user_content = instruction_template.format(input=question)
        else:
            user_content = question

        user_line = user_content.strip()
        user_prefix = f"{user_role}:"
        if not user_line.lower().startswith(user_prefix.lower()):
            user_line = f"{user_role}: {user_line}".strip()

        assistant_label = template.get("ASSISTANT", assistant_role)
        assistant_label = assistant_label.strip()
        if not assistant_label:
            assistant_label = assistant_role
        if not assistant_label.endswith(":"):
            assistant_label = f"{assistant_label}:"

        pieces: List[str] = []
        if system_prompt:
            pieces.append(system_prompt.strip())
        if self.image_token:
            pieces.append(self.image_token)
        pieces.append(user_line)
        pieces.append(assistant_label)
        return "\n".join(part for part in pieces if part)

    def _generate_direct_answer(
        self,
        question: str,
        image_path: str | Path,
        *,
        max_new_tokens: int = 64,
    ) -> str:
        base_model = getattr(self.model, "llm", None)
        if base_model is None:
            base_model = (
                getattr(self.model, "language_model", None)
                or getattr(self.model, "model", None)
                or getattr(self.model, "llava", None)
                or self.model
            )
        if not hasattr(base_model, "generate"):
            raise AttributeError("Router model does not expose a generate() method")

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for direct generation")

        prompt = self._format_direct_prompt(question)
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids))

        try:
            device = next(base_model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(base_model, "dtype", torch.float16)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with Image.open(str(image_path)) as raw_image:
            image = raw_image.convert("RGB")
            processed = self.image_processor.preprocess(image, return_tensors="pt")

        pixel_values = processed.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device=device, dtype=dtype)
        else:
            pixel_values = torch.tensor(pixel_values, device=device, dtype=dtype)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            output_ids = base_model.generate(  # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
            )

        generated = output_ids[:, input_ids.shape[1]:]
        answer = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return answer.strip()

    def _build_router_sample(self, question: str, image_path: str | Path) -> Dict[str, torch.Tensor]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        instruction = self.prompt_template["INSTRUCTION"].format(input=question)
        text_parts: List[str] = []
        if self.image_token:
            text_parts.append(self.image_token)
        text_parts.append(self.rag_token)
        text_parts.append(instruction)
        text = " ".join(text_parts)

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        bos_id = getattr(self.tokenizer, "bos_token_id", None) or getattr(
            self.tokenizer, "cls_token_id", None
        )
        if bos_id is not None:
            token_ids = [bos_id] + token_ids

        # Locate the underlying HF module to query device/dtype
        base = (
            getattr(self.model, "llm", None)
            or getattr(self.model, "llava", None)
            or getattr(self.model, "language_model", None)
            or getattr(self.model, "model", None)
            or None
        )
        if base is not None and hasattr(base, "device"):
            device = base.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(base, "dtype", torch.float16) if base is not None else torch.float16

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        image = Image.open(str(image_path)).convert("RGB")
        processed = self.image_processor.preprocess(image)
        pixel_values = torch.from_numpy(processed["pixel_values"][0])
        pixel_values = pixel_values.unsqueeze(0).to(device=device, dtype=dtype)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }

    def route(
        self,
        question: str,
        image_path: str | Path,
        *,
        rag_threshold: Optional[float] = None,
        force_use_rag: Optional[bool] = None,
        generate_answer: bool = True,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Run router + Wikipedia RAG and return a structured result."""

        sample = self._build_router_sample(question, image_path)
        # Ensure TF32 tweaks elsewhere do not change router behaviour
        orig_tf32_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
        orig_tf32_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
        try:
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
        except Exception:
            pass
        # Make router numerics invariant to global TF32/cudnn autotune toggles
        orig_bench = getattr(torch.backends.cudnn, "benchmark", None)
        try:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
        with torch.no_grad():
            router_out = self.model(sample, mode="tensor")
        # Restore previous TF32 settings
        try:
            if orig_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul  # type: ignore[attr-defined]
            if orig_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if orig_bench is not None:
                torch.backends.cudnn.benchmark = orig_bench  # type: ignore[attr-defined]
        except Exception:
            pass
        if not isinstance(router_out, dict):
            raise RuntimeError("FrozenLlavaSAM did not return a dict in tensor mode")

        prob_tensor = router_out.get("router_prob")
        if prob_tensor is None:
            logit_tensor = router_out.get("router_logit")
            if logit_tensor is None:
                raise RuntimeError("Router output missing both 'router_prob' and 'router_logit'")
            prob_tensor = torch.sigmoid(logit_tensor)
        prob_value = prob_tensor.detach().reshape(-1)[0].item()
        router_prob = float(prob_value)
        router_threshold = float(
            self.model.router_threshold if rag_threshold is None else rag_threshold
        )
        use_rag = bool(router_prob >= router_threshold)
        if force_use_rag is not None:
            use_rag = bool(force_use_rag)

        if generate_answer:
            if use_rag:
                retrieval = self.rag_bridge.generate_answer(
                    question,
                    image_path,
                    use_retrieval=True,
                    max_new_tokens=max_new_tokens,
                )
            else:
                # If the router exposes a direct generation helper, use it.
                retrieval = self.rag_bridge.retrieve(
                    question,
                    image_path,
                    use_retrieval=False,
                )
                if hasattr(self.model, "generate_answer"):
                    from dataclasses import replace as _replace

                    try:
                        answer = self.model.generate_answer(
                            question,
                            str(image_path),
                            max_new_tokens=max_new_tokens,
                        )
                    except Exception:
                        answer = None
                    retrieval.answer = answer
                else:
                    try:
                        answer = self._generate_direct_answer(
                            question,
                            image_path,
                            max_new_tokens=max_new_tokens,
                        )
                    except Exception as exc:
                        print(f"[Router] direct generation failed ({exc}); using bridge VLM fallback.")
                        fallback = self.rag_bridge.generate_answer(
                            question,
                            image_path,
                            use_retrieval=False,
                            max_new_tokens=max_new_tokens,
                        )
                        retrieval = fallback
                    else:
                        retrieval.answer = answer
        else:
            retrieval = self.rag_bridge.retrieve(
                question,
                image_path,
                use_retrieval=use_rag,
            )

        return {
            "router_prob": router_prob,
            "router_threshold": router_threshold,
            "use_rag": use_rag,
            "retrieval": retrieval,
        }
