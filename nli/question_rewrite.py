"""Heuristic helpers to convert questions into declarative statements (LENA)."""
from __future__ import annotations

import re
from typing import List, Tuple

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

_QUESTION_PRONOUNS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
_DROP_AUX = {"do", "does", "did"}
_KEEP_AUX = {
    "is", "are", "am", "was", "were", "be",
    "can", "could", "will", "would", "should",
    "shall", "may", "might", "must", "has", "have", "had",
}
_NEGATIONS = {"not", "never"}
_IRREG_3RD = {"be": "is", "am": "is", "are": "is", "have": "has", "do": "does", "go": "goes"}
_IRREG_PAST = {
    "be": "was", "am": "was", "are": "were", "begin": "began", "bring": "brought", "buy": "bought",
    "come": "came", "cut": "cut", "dig": "dug", "do": "did", "eat": "ate", "fall": "fell", "find": "found",
    "fly": "flew", "get": "got", "give": "gave", "go": "went", "grow": "grew", "have": "had", "hear": "heard",
    "hold": "held", "keep": "kept", "leave": "left", "let": "let", "lose": "lost", "make": "made", "meet": "met",
    "pay": "paid", "put": "put", "run": "ran", "say": "said", "see": "saw", "sell": "sold", "send": "sent",
    "set": "set", "sit": "sat", "speak": "spoke", "spend": "spent", "spread": "spread", "stand": "stood",
    "take": "took", "teach": "taught", "tell": "told", "think": "thought", "understand": "understood", "write": "wrote",
}

_SPACY_MODEL = None


def _q_preserve_case(src: str, dst: str) -> str:
    if not src:
        return dst
    if src.isupper():
        return dst.upper()
    if src[0].isupper() and src[1:].islower():
        return dst.capitalize()
    if src.islower():
        return dst.lower()
    return dst


def _q_conj_3rd(verb: str) -> str:
    low = verb.lower()
    if low in _IRREG_3RD:
        return _q_preserve_case(verb, _IRREG_3RD[low])
    if low.endswith("y") and len(verb) > 1 and verb[-2].lower() not in "aeiou":
        return _q_preserve_case(verb, verb[:-1] + "ies")
    if low.endswith(tuple("osxz")) or low.endswith(("sh", "ch")):
        return _q_preserve_case(verb, verb + "es")
    return _q_preserve_case(verb, verb + "s")


def _q_conj_past(verb: str) -> str:
    low = verb.lower()
    if low in _IRREG_PAST:
        return _q_preserve_case(verb, _IRREG_PAST[low])
    if low.endswith("e"):
        return _q_preserve_case(verb, verb + "d")
    if low.endswith("y") and len(verb) > 1 and verb[-2].lower() not in "aeiou":
        return _q_preserve_case(verb, verb[:-1] + "ied")
    return _q_preserve_case(verb, verb + "ed")


def _q_adjust_main_verb(verb: str, removed_aux: List[str]) -> str:
    out = verb
    for aux in removed_aux:
        a = aux.lower()
        if a == "does":
            out = _q_conj_3rd(out)
        elif a == "did":
            out = _q_conj_past(out)
    return out


def _q_cleanup(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _q_terminal(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else text + "."


def _ensure_spacy():
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL
    if spacy is None:
        return None
    try:
        _SPACY_MODEL = spacy.load("en_core_web_sm")
    except Exception:
        _SPACY_MODEL = None
    return _SPACY_MODEL


def _convert_question_with_spacy(question: str) -> str:
    nlp = _ensure_spacy()
    if nlp is None:
        return ""
    doc = nlp(question)
    q_tokens = {t for t in doc if t.tag_.startswith("W") or t.lower_ in _QUESTION_PRONOUNS}
    drop_aux = {t for t in doc if t.lower_ in _DROP_AUX and t.dep_ in {"aux", "auxpass"}}
    content = [t for t in doc if not t.is_space and t.text != "?" and t not in q_tokens and t not in drop_aux]
    if not content:
        return ""
    subj = next((t for t in content if "subj" in t.dep_), None)
    subj_tokens: List = []
    if subj is not None:
        subj_tokens = [t for t in sorted(subj.subtree, key=lambda x: x.i) if t in content]
    if not subj_tokens:
        chunk = next((c for c in doc.noun_chunks if all(t in content for t in c)), None)
        if chunk:
            subj_tokens = [t for t in chunk if t in content]
    if not subj_tokens:
        subj_tokens = [t for t in content if t.pos_ in {"NOUN", "PROPN", "PRON"}][:1]
    if not subj_tokens:
        return ""
    s_start = subj_tokens[0].i
    s_end = subj_tokens[-1].i
    subj_text = doc[s_start : s_end + 1].text
    subj_set = set(subj_tokens)
    rest = [t for t in content if t not in subj_set]
    removed_aux_text = [t.text for t in sorted(drop_aux, key=lambda x: x.i)]
    verbs = sorted([t for t in rest if t.pos_ in {"VERB", "AUX"}], key=lambda x: x.i)
    others = sorted([t for t in rest if t not in verbs], key=lambda x: x.i)
    if removed_aux_text:
        main = next((t for t in verbs if t.pos_ == "VERB"), None) or (verbs[0] if verbs else None)
        if main is not None:
            adj = _q_adjust_main_verb(main.text, removed_aux_text)
            verbs = [adj if t == main else t.text for t in verbs]
        else:
            verbs = [t.text for t in verbs]
    else:
        verbs = [t.text for t in verbs]
    other_texts = [t.text for t in others]
    parts = [subj_text]
    if verbs:
        parts.append(" ".join(verbs))
    if other_texts:
        parts.append(" ".join(other_texts))
    stmt = _q_cleanup(" ".join(p for p in parts if p))
    if not stmt:
        return ""
    stmt = stmt[0].upper() + stmt[1:]
    return _q_terminal(stmt)


def _is_likely_verb(token: str) -> bool:
    low = token.lower()
    if not low:
        return False
    if low in {
        "is", "are", "was", "were", "be", "being", "been", "do", "does", "did", "have", "has", "had",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    }:
        return True
    if low.endswith(("ed", "ing", "en")):
        return True
    if low.endswith(("ify", "ise", "ize", "ate", "fy")):
        return True
    if low.endswith("s") and len(low) > 2 and low[-2] != low[-1]:
        return True
    return False


def _split_subject_predicate(tokens: List[str]) -> Tuple[List[str], List[str]]:
    subj: List[str] = []
    pred: List[str] = []
    verb_found = False
    for tok in tokens:
        low = tok.lower()
        if not verb_found:
            if low in _NEGATIONS:
                verb_found = True
            elif _is_likely_verb(tok):
                verb_found = True
        (pred if verb_found else subj).append(tok)
    if not pred and subj:
        pred.append(subj.pop())
    return subj, pred


def _convert_question_without_spacy(question: str) -> str:
    words = question.split()
    if not words:
        return ""
    idx = 0
    while idx < len(words) and words[idx].lower() in _QUESTION_PRONOUNS:
        idx += 1
    if idx >= len(words):
        return ""
    removed_aux: List[str] = []
    kept_aux: List[str] = []
    remain = words[idx:]
    while remain and remain[0].lower() in (_DROP_AUX | _KEEP_AUX):
        lead = remain.pop(0)
        if lead.lower() in _DROP_AUX:
            removed_aux.append(lead)
        else:
            kept_aux.append(lead)
    if not remain:
        return ""
    subj, pred = _split_subject_predicate(remain)
    subj_text = " ".join(subj).strip()
    pred_text = " ".join(pred).strip()
    if removed_aux:
        parts = pred_text.split()
        if parts:
            parts[0] = _q_adjust_main_verb(parts[0], removed_aux)
            pred_text = " ".join(parts)
    aux_text = " ".join(k.lower() for k in kept_aux)
    parts_out = []
    if subj_text:
        parts_out.append(subj_text)
    if aux_text:
        parts_out.append(aux_text)
    if pred_text:
        parts_out.append(pred_text)
    stmt = _q_cleanup(" ".join(parts_out))
    if not stmt:
        return ""
    stmt = stmt[0].upper() + stmt[1:]
    return _q_terminal(stmt)


def convert_question_to_statement(question: str) -> str:
    q = (question or "").strip().rstrip("?")
    if not q:
        return ""
    stmt = _convert_question_with_spacy(q)
    if not stmt:
        stmt = _convert_question_without_spacy(q)
    return stmt or ""

