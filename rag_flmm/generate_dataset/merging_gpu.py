import os, glob
import pandas as pd

# ===== 경로 설정 =====
OUTPUT_PATH = "/data/dataset/infoseek/infoseek_train_generated_answers.csv"
RANKED_PATTERN = OUTPUT_PATH.replace(".csv", ".rank*.csv")

def merge_ranked_csv(output_path=OUTPUT_PATH, ranked_pattern=RANKED_PATTERN):
    # GPU별 CSV 모으기
    files = sorted(glob.glob(ranked_pattern))
    if not files:
        print(f"❌ No files found matching pattern: {ranked_pattern}")
        return

    print(f"🔍 Found {len(files)} shard files:")
    for f in files:
        print("   ", f)

    # CSV 읽어서 concat
    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)

    # data_id 기준 중복 제거
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["data_id"], keep="first").reset_index(drop=True)
    after = len(df_all)

    # 최종 저장
    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Merged {len(files)} shards into {output_path}")
    print(f"   Rows before dedup: {before}, after dedup: {after}")

if __name__ == "__main__":
    merge_ranked_csv()
