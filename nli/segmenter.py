"""Sentence segmentation helpers with spaCy or NLTK fallbacks (LENA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import nltk

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

_SPACY_SENTENCIZER = None


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _ensure_spacy_sentencizer():
    global _SPACY_SENTENCIZER
    if _SPACY_SENTENCIZER is not None:
        return _SPACY_SENTENCIZER
    if spacy is None:
        return None
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _SPACY_SENTENCIZER = nlp
    return _SPACY_SENTENCIZER


@dataclass
class SentenceChunk:
    text: str
    section_index: int
    sentence_index: int


def segment_texts(texts: Iterable[str]) -> List[List[SentenceChunk]]:
    sentencizer = _ensure_spacy_sentencizer()
    chunks: List[List[SentenceChunk]] = []
    if sentencizer is not None:
        for sec_idx, text in enumerate(texts):
            doc = sentencizer(text or "")
            segs: List[SentenceChunk] = []
            for sent_idx, span in enumerate(doc.sents):
                sentence = span.text.strip()
                if sentence:
                    segs.append(SentenceChunk(sentence, sec_idx, sent_idx))
            chunks.append(segs)
        return chunks
    _ensure_punkt()
    for sec_idx, text in enumerate(texts):
        sentences = nltk.sent_tokenize(text or "")
        segs = [
            SentenceChunk(sentence.strip(), sec_idx, sent_idx)
            for sent_idx, sentence in enumerate(sentences)
            if sentence.strip()
        ]
        chunks.append(segs)
    return chunks

