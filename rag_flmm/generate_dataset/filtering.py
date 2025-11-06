import csv
import re

import pandas as pd
from tqdm import tqdm

# ========================
# Settings
# ========================
INPUT = "/data/dataset/evqa/evqa_judged_answers.csv"
OUTPUT = "/data/dataset/evqa/evqa_judged_answers_filtered.csv"
expected_cols = 18  # 정상 컬럼 수

# ========================
# Pattern definitions
# ========================
# 객관식 패턴: 알파벳/숫자 + ., ), :
choice_pattern = re.compile(r"(?:[A-Za-z0-9][\.\):]\s*.+?){2,}", re.DOTALL)
how_tall_pattern = re.compile(r"^\s*how tall", re.IGNORECASE)
number_pattern = re.compile(r"\d+")
valid_labels = {"YES", "NO"}


# ========================
# RAG_LABEL update function
# ========================
def fix_rag_label(row):
    label = row["rag_label"]
    if label == "YES":
        generated = str(row["generated"]) if pd.notna(row["generated"]) else ""
        question = str(row["question"]) if pd.notna(row["question"]) else ""
        answer = str(row["answer"]) if pd.notna(row["answer"]) else ""

        # 조건 1: 객관식 / How tall / ? 포함 → NO
        if (
            choice_pattern.search(generated)
            or how_tall_pattern.match(question)
            or "?" in generated
        ):
            return "NO"

        # 조건 2: answer에 숫자가 있는데 generated에 하나라도 없으면 → NO
        answer_numbers = number_pattern.findall(answer)
        if answer_numbers:
            missing_number = any(num not in generated for num in answer_numbers)
            if missing_number:
                return "NO"

    return label if label in valid_labels else ""


# ========================
# 1) CSV 직접 읽어서 18컬럼 맞추기
# ========================
rows = []
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        if not any(cell.strip() for cell in row):  # 완전 빈 행 스킵
            continue
        if len(row) < expected_cols:
            row += [""] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        rows.append(row)

# ========================
# 2) DataFrame 변환
# ========================
df = pd.DataFrame(rows, columns=headers)

# ========================
# 3) rag_label 업데이트 (tqdm 진행률)
# ========================
df["rag_label"] = [
    fix_rag_label(row)
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Updating rag_label", dynamic_ncols=True
    )
]

# ========================
# 4) 저장
# ========================
df.to_csv(OUTPUT, index=False)
print(f"[INFO] Processing completed: {OUTPUT}")
