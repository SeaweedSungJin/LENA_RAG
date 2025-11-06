"""NLI-based sentence selector for EchoSight (LENA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import NLIConfig
from .question_rewrite import convert_question_to_statement
from .segmenter import segment_texts


@dataclass
class SectionCandidate:
    text: str
    doc_title: str
    section_title: str
    similarity: float = 0.0
    section_index: Optional[int] = None


@dataclass
class SentenceCandidate:
    text: str
    doc_title: str
    section_title: str
    section_index: int
    sentence_index: int
    parent_score: float
    question_entailment: float = 0.0


class NLISelector:
    def __init__(self, config: NLIConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    def select(self, question: str, sections: Sequence[SectionCandidate]) -> List[SentenceCandidate]:
        if not sections:
            return []
        sentences = self._segment(sections)
        if not sentences:
            return []
        hyp = convert_question_to_statement(question) if self.config.convert_question else question
        ent = self._score_entail([s.text for s in sentences], hyp)
        for s, sc in zip(sentences, ent):
            s.question_entailment = sc
        thresh = float(self.config.question_entail_threshold)
        filtered = [s for s in sentences if s.question_entailment >= thresh] or sorted(
            sentences, key=lambda x: x.question_entailment, reverse=True
        )[: max(1, int(self.config.keep_top_k_on_empty))]
        # Pairwise clustering & Shapley-like pruning (reuse classifier probs)
        trimmed = self._prune_pairwise(filtered)
        return trimmed or filtered

    def _segment(self, sections: Sequence[SectionCandidate]) -> List[SentenceCandidate]:
        out: List[SentenceCandidate] = []
        segs = segment_texts([s.text for s in sections])
        for si, parts in enumerate(segs):
            meta = sections[si]
            for chunk in parts:
                out.append(
                    SentenceCandidate(
                        text=chunk.text,
                        doc_title=meta.doc_title,
                        section_title=meta.section_title,
                        section_index=meta.section_index if meta.section_index is not None else si,
                        sentence_index=chunk.sentence_index,
                        parent_score=float(meta.similarity),
                    )
                )
        return out

    def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
        tok = AutoTokenizer.from_pretrained(self.config.model_name)
        model.to(self.config.device)
        model.eval()
        self._model, self._tokenizer = model, tok
        return model, tok

    @torch.no_grad()
    def _score_entail(self, sentences: Sequence[str], hypothesis: str) -> List[float]:
        if not sentences:
            return []
        model, tok = self._ensure_model()
        dev = model.device
        max_len = _effective_max_length(tok, model, self.config.max_length)
        out: List[float] = []
        bs = max(1, int(self.config.sentence_batch_size))
        for i in range(0, len(sentences), bs):
            batch = sentences[i : i + bs]
            inputs = tok(batch, [hypothesis] * len(batch), return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(dev)
            probs = model(**inputs).logits.softmax(dim=-1)
            out.extend(probs[:, 2].tolist())
        return out

    @torch.no_grad()
    def _prune_pairwise(self, sents: List[SentenceCandidate]) -> List[SentenceCandidate]:
        # Simple, fast pruning using global consistency increases (Shapley-like recompute)
        n = len(sents)
        if n <= 2:
            return sents
        # Build weights using averaged entailment/contradiction
        model, tok = self._ensure_model()
        dev = model.device
        max_len = self.config.max_length
        weights = [[0.0] * n for _ in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        bs = max(1, int(self.config.batch_size))
        for k in range(0, len(pairs), bs):
            batch = pairs[k : k + bs]
            p1 = [sents[i].text for i, _ in batch]
            p2 = [sents[j].text for _, j in batch]
            inp1 = tok(p1, p2, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(dev)
            inp2 = tok(p2, p1, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(dev)
            pr1 = model(**inp1).logits.softmax(dim=-1)
            pr2 = model(**inp2).logits.softmax(dim=-1)
            ent = ((pr1[:, 2] + pr2[:, 2]) / 2).tolist()
            con = ((pr1[:, 0] + pr2[:, 0]) / 2).tolist()
            for (i, j), e, c in zip(batch, ent, con):
                w = max(0.0, self.config.alpha * float(e) - self.config.beta * float(c))
                if w >= self.config.tau:
                    weights[i][j] = weights[j][i] = w
        # Greedy recompute removal
        cur = list(range(n))
        def score(idx: List[int]) -> float:
            m = len(idx)
            if m < 2:
                return 0.0
            tot = 0.0
            for a in range(m):
                for b in range(a + 1, m):
                    tot += weights[idx[a]][idx[b]]
            return (2.0 / (m * (m - 1))) * tot
        base = score(cur)
        improved = True
        while improved and len(cur) > max(1, int(self.config.target_size)):
            improved = False
            best_gain = 0.0
            best_k = None
            for k in list(cur):
                nxt = [x for x in cur if x != k]
                new_score = score(nxt)
                gain = new_score - base
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_k = k
            if best_k is not None and best_gain > 0.0:
                cur.remove(best_k)
                base = score(cur)
                improved = True
            else:
                break
        return [sents[i] for i in cur]


def _effective_max_length(tok, model, requested: int) -> int:
    cand = [int(requested)]
    tm = getattr(tok, "model_max_length", None)
    if isinstance(tm, int) and 0 < tm < 1_000_000:
        cand.append(tm)
    mm = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(mm, int) and mm > 0:
        cand.append(mm)
    return max(32, min(cand))

