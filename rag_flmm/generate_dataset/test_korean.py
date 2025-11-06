import re
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

INPUT_CSV = "/data/dataset/evqa/evqa_judged_answers_fixed_3.csv"
OUTPUT_CSV = "/data/dataset/evqa/non_english_issues.csv"

# 데이터 로드
df = pd.read_csv(INPUT_CSV)

# 허용 문자 패턴 (영문, 숫자, 공백, 구두점, 파이프 추가)
allowed_pattern = re.compile(r'^[A-Za-z0-9\s.,;:!?\'"()\-_/&%|]*$')


# 각 행 검사 함수
def check_row(idx_row):
    idx, row = idx_row
    issues = []
    for col, val in row.items():
        val = str(val)
        if not allowed_pattern.match(val):
            issues.append((idx, col, val))
    return issues


# 병렬 처리
with Pool(processes=cpu_count()) as pool:
    results = list(
        tqdm(pool.imap(check_row, df.iterrows(), chunksize=100), total=len(df))
    )

# 결과 합치기
issues = [item for sublist in results for item in sublist]
issues_df = pd.DataFrame(issues, columns=["row", "column", "value"])
issues_df.to_csv(OUTPUT_CSV, index=False)

print(f"Found {len(issues)} issues. Saved to {OUTPUT_CSV}")
