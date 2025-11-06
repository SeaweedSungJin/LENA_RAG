import os

os.environ["HF_HOME"] = "/data/hf_cache"

import pandas as pd
import torch
from rag_flmm.utils.hf_auth import login_if_available
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# ========================
# 설정
# ========================
login_if_available()
INPUT_CSV = "/data/dataset/evqa/evqa_generated_answers.csv"
LLM_JUDGE_MODEL = "NousResearch/Hermes-2-Pro-Llama-3-8B"
DEBUG_SAMPLES = 20
DEVICE = "cuda:0"


# ========================
# Judge 모델 로드
# ========================
def load_judge(device):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_JUDGE_MODEL, quantization_config=quant_config, device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=8,
        do_sample=False,
    )
    return pipe


# ========================
# Judge 함수
# ========================
import re


def llm_judge(judge, question, answer, pred):
    prompt = f"""
You are a strict evaluator. Your job is to decide if the predicted answer has the same core meaning as the ground truth answer.

Guidelines:
- Synonyms, paraphrases, and additional explanations are allowed as long as the core meaning matches.
- If the predicted answer changes the core meaning or contradicts the ground truth, respond NO.

Question: {question}
Ground Truth Answer: {answer}
Predicted Answer: {pred}

Respond ONLY with "YES" if the predicted answer conveys the same meaning, otherwise respond "NO".
"""
    full_output = judge(prompt)[0]["generated_text"]
    clean_output = full_output.replace(prompt, "").strip()
    # YES/NO 만 추출 (대소문자 무관)
    match = re.search(r"\b(yes|no)\b", clean_output.lower())
    answer = match.group(1).upper() if match else "NO"  # 못 찾으면 NO로 처리
    return answer, (answer == "YES")


# ========================
# 디버깅 실행
# ========================
def main():
    df = pd.read_csv(INPUT_CSV).head(DEBUG_SAMPLES)
    judge = load_judge(DEVICE)

    for _, row in df.iterrows():
        q, a, pred = row["question"], str(row["answer"]), str(row.get("generated", ""))
        raw_result, is_match = llm_judge(judge, q, a, pred)
        print("=" * 50)
        print(f"[Q] {q}")
        print(f"[GT] {a}")
        print(f"[Pred] {pred}")
        print(f"[Judge Answer Only] {raw_result}")  # 모델이 낸 대답만 출력
        print(f"[Judge Parsed] {'MATCH' if is_match else 'NO MATCH'}")


if __name__ == "__main__":
    main()
