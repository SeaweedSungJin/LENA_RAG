import csv
import re
import pandas as pd

# ========= Settings =========
INPUT = "/data/dataset/infoseek/infoseek_train_filtered.csv"
OUTPUT = "/data/dataset/infoseek/infoseek_train_with_rag_label.csv"

# Detect any digit (dates, heights, counts → treat as numeric)
number_pattern = re.compile(r"\d")

# ========= 1) Read original CSV + count rows/cols =========
rows = []
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)
    expected_cols = len(headers)
    orig_row_count = 0
    for row in reader:
        orig_row_count += 1
        # Normalize column length (pad/truncate to header length)
        if len(row) < expected_cols:
            row += [""] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        rows.append(row)

print(f"[INFO] Original columns: {expected_cols}, original data rows: {orig_row_count}")

# ========= 2) To DataFrame =========
df = pd.DataFrame(rows, columns=headers)
proc_row_count = len(df)
assert proc_row_count == orig_row_count, \
    f"Row count mismatch after processing: {proc_row_count} != {orig_row_count}"

# ========= 3) Create rag_label (answer has number → NO, else YES) =========
if "answer" not in df.columns:
    raise ValueError("CSV is missing 'answer' column.")

has_number = df["answer"].fillna("").astype(str).str.contains(number_pattern)
df["rag_label"] = has_number.map({True: "NO", False: "YES"})

# Validate values
valid = {"YES", "NO"}
bad = df.loc[~df["rag_label"].isin(valid)]
assert bad.empty, f"Found invalid rag_label values:\n{bad['rag_label'].value_counts()}"

# ========= 4) Column order/count checks =========
# Ensure rag_label is appended as the last column
assert list(df.columns) == headers + ["rag_label"], "rag_label is not the last column."
assert len(df.columns) == expected_cols + 1, \
    f"Column count mismatch: {len(df.columns)} != {expected_cols + 1}"

# ========= 5) Save =========
df.to_csv(OUTPUT, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"[INFO] Saved to: {OUTPUT}")

# ========= 6) Re-verify saved file (rows/cols) =========
with open(OUTPUT, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    out_headers = next(reader)
    out_cols = len(out_headers)
    out_row_count = sum(1 for _ in reader)

assert out_cols == expected_cols + 1, \
    f"Saved file column count mismatch: {out_cols} != {expected_cols + 1}"
assert out_row_count == orig_row_count, \
    f"Saved file row count mismatch: {out_row_count} != {orig_row_count}"

print(f"[OK] Row/column verification passed: rows={out_row_count}, cols={out_cols}")
