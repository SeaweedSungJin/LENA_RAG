import pandas as pd

INPUT_CSV = "/data/dataset/evqa/evqa_judged_answers_fixed_3.csv"
OUTPUT_CSV = "/data/dataset/evqa/train_sample100.csv"

# CSV 읽기
df = pd.read_csv(INPUT_CSV)

# 랜덤 100개 추출 (random_state로 재현성 유지)
df_sample = df.sample(n=100, random_state=42)

# 저장
df_sample.to_csv(OUTPUT_CSV, index=False)

print(f"Random 100 rows saved to: {OUTPUT_CSV}")
