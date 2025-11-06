import spacy
import torch
from transformers import AutoTokenizer

# 1. spaCy 로딩
# nlp = spacy.load("en_core_web_sm")


# 2. 명사구 전처리
def process_noun_chunks(noun_chunks):
    new_chunks = []
    for i, noun in enumerate(noun_chunks):
        noun_lower = noun.lower()
        if "image" in noun_lower:
            continue
        if noun_lower in [
            "it",
            "this",
            "that",
            "those",
            "these",
            "them",
            "he",
            "she",
            "you",
            "i",
            "they",
            "me",
            "her",
            "him",
            "a",
            "what",
            "which",
            "whose",
            "who",
        ]:
            continue
        keep = True
        for j, other in enumerate(noun_chunks):
            if i != j and noun in other:
                if len(noun) < len(other) or i > j:
                    keep = False
                    break
        if keep:
            new_chunks.append(noun)
    return new_chunks


# 3. 명사구 추출 함수
def extract_noun_phrases(output_text: str, nlp):
    doc = nlp(output_text)
    raw_chunks = list(set(chunk.text for chunk in doc.noun_chunks))
    if not raw_chunks:
        return [output_text]
    noun_chunks = process_noun_chunks(raw_chunks)
    noun_chunks = sorted(noun_chunks, key=lambda x: output_text.find(x))
    return noun_chunks


# 4. 명사구 span (start, end) 위치 얻기
def get_spans(output_text, noun_phrases):
    spans = []
    last_end = 0
    for phrase in noun_phrases:
        start = output_text.find(phrase, last_end)
        if start == -1:
            continue
        end = start + len(phrase)
        if start < last_end:
            continue
        last_end = end
        spans.append((start, end))
    return spans


# 5. spaCy span → tokenizer token index 변환
def find_interval(offsets, char_idx):
    for i, (start, end) in enumerate(offsets):
        if start <= char_idx < end:
            return i
    return len(offsets) - 1


def get_token_spans(text: str, tokenizer, spans: list[tuple[int, int]]):
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    token_spans = []
    for start_char, end_char in spans:
        start_token = find_interval(offsets, start_char)
        end_token = max(start_token + 1, find_interval(offsets, end_char))
        token_spans.append((start_token, end_token))
    return token_spans


def extract_noun_token_ranges(output_text, output_ids, tokenizer, nlp):
    doc = nlp(output_text)
    noun_chunks = list(set(chunk.text for chunk in doc.noun_chunks))

    encoded = tokenizer(
        output_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded["offset_mapping"][0]  # shape: (seq_len, 2)
    token_ranges = []

    for chunk in noun_chunks:
        start_char = output_text.find(chunk)
        end_char = start_char + len(chunk)
        matched_token_indices = [
            i
            for i, (s, e) in enumerate(offsets.tolist())
            if not (e <= start_char or s >= end_char)
        ]
        if matched_token_indices:
            token_ranges.append(
                (matched_token_indices[0], matched_token_indices[-1] + 1)
            )
    return token_ranges
