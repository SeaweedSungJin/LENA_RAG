# import csv
# from collections import Counter

# INPUT = "/data/dataset/evqa/evqa_judged_answers.csv"

# col_counts = Counter()
# with open(INPUT, "r", encoding="utf-8") as f:
#     reader = csv.reader(f)
#     headers = next(reader)
#     for row in reader:
#         col_counts[len(row)] += 1

# print("Column count distribution:", col_counts)
# print("Most common (likely correct) column count:", col_counts.most_common(1)[0][0])

import pandas as pd

# CSV 불러오기 (인코딩 문제 대비)
try:
    df = pd.read_csv("/data/dataset/evqa/single_gpu_augmented_data.csv")
except UnicodeDecodeError:
    # UTF-8로 안 열리면 latin1이나 cp949로 시도
    df = pd.read_csv("/data/dataset/evqa/single_gpu_augmented_data.csv", encoding="latin1")

# rag_label 칼럼의 고유 값 출력
print(df["rag_label"].unique())

# 고유 값 개수도 확인하려면
print("개수:", df["rag_label"].nunique())

print(df["rag_label"].value_counts())

