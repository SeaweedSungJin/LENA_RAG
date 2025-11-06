import pandas as pd

# 파일 경로
A_PATH = "/data/dataset/infoseek/infoseek_train_with_rag_label.csv"
B_PATH = "/data/dataset/infoseek/infoseek_train_generated_answers.csv"
OUT_PATH = "/data/dataset/infoseek/merged.csv"

# 불러오기
df_a = pd.read_csv(A_PATH)
df_b = pd.read_csv(B_PATH)

# A의 컬럼 중 B에 없는 것만 선택 (data_id 제외)
cols_to_add = [c for c in df_a.columns if c != "data_id" and c not in df_b.columns]

print("👉 A에서 B로 옮길 컬럼:", cols_to_add)

# merge (B 기준 유지)
df_merged = df_b.merge(df_a[["data_id"] + cols_to_add], on="data_id", how="left")

# 저장
df_merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {OUT_PATH}")