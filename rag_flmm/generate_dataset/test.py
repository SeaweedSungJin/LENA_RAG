import re

import pandas as pd
from tqdm import tqdm

ORIGIN_CSV = "/data/dataset/evqa/train_full_image_cleaned.csv"
NEW_CSV = "/data/dataset/evqa/evqa_judged_answers_fixed_2.csv"
OUTPUT_CSV = "/data/dataset/evqa/evqa_judged_answers_fixed_3.csv"
LOG_CSV = "/data/dataset/evqa/replaced_log.csv"
UNMATCHED_LOG = "/data/dataset/evqa/unmatched_log.csv"

# 허용 문자 패턴
allowed_pattern = re.compile(r'^[A-Za-z0-9\s.,;:!?\'"()\-_/&%|]*$')

# 로드
df_origin = pd.read_csv(ORIGIN_CSV)
df_new = pd.read_csv(NEW_CSV)

# 교체할 컬럼
check_columns = [
    "answer",
    "evidence",
    "evidence_section_title",
    "question",
    "question_original",
    "wikipedia_title",
    "wikipedia_url",
]

# 매칭용 컬럼: 교체 대상 컬럼은 키에서 제외
exclude_cols = set(check_columns) | {"generated", "reason", "image_path", "rag_label"}
match_columns = [
    c for c in df_new.columns if c in df_origin.columns and c not in exclude_cols
]

# === 1. 허용문자 벡터화 검사 ===
for col in check_columns:
    df_new[f"{col}_valid"] = df_new[col].astype(str).str.match(allowed_pattern)

# === 2. 병합용 키 생성 (교체 대상 제외)
df_new["merge_key"] = df_new[match_columns].astype(str).agg("|".join, axis=1)
df_origin["merge_key"] = df_origin[match_columns].astype(str).agg("|".join, axis=1)

# 중복 키 제거
df_origin = df_origin.drop_duplicates(subset=["merge_key"], keep="first")
origin_map = df_origin.set_index("merge_key")[check_columns].to_dict(orient="index")

# === 3. 교체 및 로그 저장 ===
log = []
unmatched = []
broken_rows = 0
replaced_count = 0

for idx, row in tqdm(df_new.iterrows(), total=len(df_new), desc="Fixing rows"):
    key = row["merge_key"]
    if key not in origin_map:
        unmatched.append({"row_idx": idx, "merge_key": key})
        continue
    origin_row = origin_map[key]
    broken = False
    for col in check_columns:
        if not row[f"{col}_valid"]:
            old_val = df_new.at[idx, col]
            new_val = origin_row[col]
            if old_val != new_val:
                df_new.at[idx, col] = new_val
                log.append(
                    {
                        "row_idx": idx,
                        "merge_key": key,
                        "column": col,
                        "old_value": old_val,
                        "new_value": new_val,
                    }
                )
                replaced_count += 1
                broken = True
    if broken:
        broken_rows += 1

# === 4. 저장 ===
pd.DataFrame(log).to_csv(LOG_CSV, index=False)
pd.DataFrame(unmatched).to_csv(UNMATCHED_LOG, index=False)
df_new.drop(columns=[f"{c}_valid" for c in check_columns] + ["merge_key"], inplace=True)
df_new.to_csv(OUTPUT_CSV, index=False)

print(f"Fixed file saved to {OUTPUT_CSV}")
print(f"Detected {broken_rows} broken rows, replaced {replaced_count} values.")
print(f"Unmatched rows: {len(unmatched)} (saved to {UNMATCHED_LOG})")
