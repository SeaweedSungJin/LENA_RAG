# hooks.py
import os               # ← 추가
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS   # ← 추가
from peft import get_peft_model, prepare_model_for_kbit_training

@HOOKS.register_module()              # ← 추가
class EnsureRagTokensHook(Hook):
    def __init__(self, tokens=("[RAG]",), verbose=True):
        self.tokens = list(tokens)
        self.verbose = verbose

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        tok = getattr(model, "tokenizer", None)
        if tok is None:
            runner.logger.warning("[EnsureRagTokensHook] tokenizer not found; skip.")
            return
        unk = tok.unk_token_id if tok.unk_token_id is not None else -1
        need_add = []
        for t in self.tokens:
            tid = tok.convert_tokens_to_ids(t)
            if tid in (-1, None) or tid == unk:
                need_add.append(t)
        if not need_add:
            if self.verbose:
                runner.logger.info(f"[EnsureRagTokensHook] tokens already present: {self.tokens}")
            return
        added = tok.add_special_tokens({"additional_special_tokens": need_add})
        if self.verbose:
            runner.logger.info(f"[EnsureRagTokensHook] added {added} tokens: {need_add}")
        if added > 0:
            llm = getattr(model, "llm", model)
            try:
                llm.resize_token_embeddings(len(tok))
                runner.logger.info(f"[EnsureRagTokensHook] resized embeddings to {len(tok)}")
            except Exception as e:
                runner.logger.warning(f"[EnsureRagTokensHook] resize_token_embeddings failed: {e}")

@HOOKS.register_module()              # ← 추가
class SaveTokenizerHook(Hook):
    def __init__(self, subdir="tokenizer", save_once=True, verbose=True):
        self.subdir = subdir
        self.save_once = save_once
        self.verbose = verbose
        self._saved = False

    def before_train(self, runner):
        if self.save_once and self._saved:
            return
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        tok = getattr(model, "tokenizer", None)
        if tok is None:
            runner.logger.warning("[SaveTokenizerHook] tokenizer not found; skip.")
            return
        out_dir = os.path.join(runner.work_dir, self.subdir)
        os.makedirs(out_dir, exist_ok=True)
        tok.save_pretrained(out_dir)
        self._saved = True
        if self.verbose:
            runner.logger.info(f"[SaveTokenizerHook] tokenizer saved to: {out_dir}")
