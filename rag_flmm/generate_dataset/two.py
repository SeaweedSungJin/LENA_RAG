import os
import re
import json
import pandas as pd
import torch
import torch.multiprocessing as mp
import transformers
from rag_flmm.utils.hf_auth import login_if_available
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

transformers.utils.logging.set_verbosity_error()

# ========================
# 설정
# ========================
# 요청 이력: 앞으로 제공하는 스크립트에서는 HF_HOME을 설정하지 않도록 함
# os.environ["HF_HOME"] = "/data/hf_cache"  # 제거

# HF 로그인 (사용자가 제공한 토큰 그대로 사용)
login_if_available()

INPUT_CSV = "/data/dataset/infoseek/merged.csv"  # 전체 데이터 (YES/NO 모두 포함, llava_answer 포함)
OUTPUT_DIR = "/data/dataset/infoseek/judged_chunks"                   # GPU별 분할 저장 디렉토리
FINAL_OUTPUT = "/data/dataset/infoseek/infoseek_test_judged_answers.csv"  # 최종 병합 파일 (전체 행 포함)
LLM_JUDGE_MODEL = "NousResearch/Hermes-2-Pro-Llama-3-8B"

BATCH_SIZE = 64
SAVE_INTERVAL = 10          # 배치 단위로 저장(append); GPU 파일은 덮어쓰지 않고 이어 붙임
DEBUG_SAMPLES = None        # None이면 전체, 정수면 N개만 처리

YES_SET = {"yes", "y", "1", "true", "use", "use_rag"}

def is_yes(x):
    return str(x).strip().lower() in YES_SET

# ========================
# Judge 모델 로드
# ========================
def load_judge():
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
        batch_size=None,   # 파이프라인 내부 배치 대신 우리 쪽 BATCH_SIZE로 컨트롤
        truncation=False,
    )
    return pipe

# ========================
# Judge 함수 (배치)
# ========================
def llm_judge_batch(judge, questions, answers, preds):
    prompts = [
        f"""You are a strict evaluator. Decide if the predicted answer has the same core meaning as the ground truth answer.

Guidelines:
- Synonyms, paraphrases, and additional explanations are allowed as long as the core meaning matches.
- If the predicted answer changes the core meaning or contradicts the ground truth, respond NO.

Question: {q}
Ground Truth Answer: {a}
Predicted Answer: {p}

Respond ONLY with "YES" or "NO"."""
        for q, a, p in zip(questions, answers, preds)
    ]
    try:
        outputs = judge(prompts)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[FATAL] OOM detected. Terminating all processes.", flush=True)
            os._exit(1)
        raise e

    results = []
    for out, prompt in zip(outputs, prompts):
        # transformers pipeline은 각 입력당 리스트[dict] 형태 반환
        text = out[0]["generated_text"].replace(prompt, "").strip()
        m = re.search(r"\b(yes|no)\b", text.lower())
        results.append(m.group(1).upper() if m else "NO")
    return results

# ========================
# 워커: YES & 비교가능 행만 처리
# ========================
def worker(rank, data_chunk):
    """
    data_chunk: df_yes_only의 부분집합 (이미 rag_label==YES이며 llava_answer가 있는 행만)
    파일: gpu_{rank}.csv (row_id, judged_label)
    """
    judge = load_judge()
    gpu_file = os.path.join(OUTPUT_DIR, f"gpu_{rank}.csv")

    buf_rows = []
    since_save = 0

    for i in tqdm(range(0, len(data_chunk), BATCH_SIZE), position=rank, desc=f"GPU{rank}"):
        batch = data_chunk.iloc[i : i + BATCH_SIZE]

        # 입력 구성
        row_ids  = batch["__row_id"].tolist()
        questions = batch["question"].astype(str).tolist()
        answers   = batch["answer"].astype(str).tolist()
        preds     = batch["llava_answer"].astype(str).tolist()

        # 배치 판정
        try:
            judged = llm_judge_batch(judge, questions, answers, preds)
        except SystemExit:
            os._exit(1)

        # 결과 누적 (원래 YES였던 것을 유지/반전)
        for rid, label in zip(row_ids, judged):
            out_label = "YES" if label == "YES" else "NO"
            buf_rows.append({"__row_id": rid, "judged_label": out_label})
            since_save += 1

        # SAVE_INTERVAL마다 저장 (append)
        if since_save >= SAVE_INTERVAL:
            df_buf = pd.DataFrame(buf_rows)
            df_buf.to_csv(
                gpu_file,
                mode="a",
                index=False,
                header=not os.path.exists(gpu_file),
                encoding="utf-8-sig",
            )
            buf_rows = []
            since_save = 0

    # 잔여 버퍼 저장
    if buf_rows:
        df_buf = pd.DataFrame(buf_rows)
        df_buf.to_csv(
            gpu_file,
            mode="a",
            index=False,
            header=not os.path.exists(gpu_file),
            encoding="utf-8-sig",
        )

# ========================
# 메인
# ========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 원본 전체 로드
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    input_count = len(df)
    print(f"[INFO] Loaded {input_count} rows from INPUT_CSV.")

    if DEBUG_SAMPLES:
        df = df.head(DEBUG_SAMPLES).copy()
        input_count = len(df)
        print(f"[INFO] DEBUG mode: restricted to {input_count} rows.")

    # 고유 식별자 부여 (병합 안전)
    df["__row_id"] = df.index

    # YES 후보: rag_label == YES and 비교 가능한 데이터(질문/정답/예측 존재)
    mask_yes = df["rag_label"].apply(is_yes)
    cmp_mask = mask_yes & df["question"].notna() & df["answer"].notna() & df["llava_answer"].notna() & (df["llava_answer"].astype(str).str.len() > 0)
    df_yes_only = df.loc[cmp_mask].copy()
    judge_count = len(df_yes_only)
    print(f"[INFO] Candidates for judging (rag_label==YES & comparable): {judge_count}")

    # YES가 하나도 없으면 바로 저장
    if judge_count == 0:
        final = df.drop(columns=["__row_id"])
        final.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"[INFO] No YES rows to judge. Saved original to {FINAL_OUTPUT}")
        print(f"[INFO] Input rows: {input_count}, Final rows: {len(final)}")
        return

    # 멀티-GPU 분할
    num_gpus = max(torch.cuda.device_count(), 1)
    idx_tensor = torch.arange(judge_count)
    chunks = torch.chunk(idx_tensor, num_gpus)
    data_chunks = [df_yes_only.iloc[ch.tolist()] for ch in chunks if len(ch) > 0]

    # 프로세스 실행
    processes = []
    for rank, chunk_df in enumerate(data_chunks):
        p = mp.Process(target=worker, args=(rank, chunk_df))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # GPU별 결과 병합
    all_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.startswith("gpu_")]
    if not all_files:
        print("[WARN] No GPU chunk files found; keeping original labels.")
        combined = pd.DataFrame(columns=["__row_id", "judged_label"])
    else:
        combined = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in all_files], ignore_index=True)

    # 중복 제거 (row_id 기준)
    combined.drop_duplicates(subset=["__row_id"], keep="last", inplace=True)
    print(f"[INFO] Judged rows collected: {len(combined)} / requested: {judge_count}")

    # 원본에 반영: 해당 row_id만 업데이트
    df = df.merge(combined, on="__row_id", how="left")
    # 기존 YES 행 중 판정 결과가 존재하면 그 결과로 덮어씀, 없으면 기존 값 유지
    df["rag_label"] = df.apply(
        lambda r: r["judged_label"] if (is_yes(r["rag_label"]) and pd.notna(r["judged_label"])) else r["rag_label"],
        axis=1
    )
    # 보조 컬럼 제거
    df.drop(columns=["__row_id", "judged_label"], inplace=True)

    # 최종 저장
    df.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved final merged file: {FINAL_OUTPUT}")
    print(f"[INFO] Input rows: {input_count}, Final rows: {len(df)}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
