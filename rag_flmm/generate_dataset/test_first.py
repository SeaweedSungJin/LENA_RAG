import os, json
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ===== 경로 설정 =====
CSV_PATH = "/data/dataset/infoseek/infoseek_train_with_rag_label.csv"
INFOSEEK_MANIFEST = "/data/dataset/infoseek/infoseek_manifest.json"
OUTPUT_PATH = "/data/dataset/infoseek/infoseek_train_generated_answers.csv"
BASE_MODEL = "llava-hf/llava-1.5-7b-hf"

# ===== 파라미터 =====
BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
SAVE_INTERVAL = 10000   # 진행과 무관(append 방식), 진행 로그 목적으로만 유지

with open(INFOSEEK_MANIFEST) as f:
    INF_M = json.load(f)    

def is_yes(x):
    return str(x).strip().lower() == "yes"

def setup_dist():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    return rank, world, f"cuda:{rank}"

def build_inputs(processor, images, questions):
    convs = []
    for q in questions:
        convs.append([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}])
    prompts = processor.apply_chat_template(convs, add_generation_prompt=True)
    return processor(images=images, text=prompts, padding=True, return_tensors="pt")

def decode_batch(processor, input_ids, generated_ids):
    outs = []
    new_tokens = generated_ids[:, input_ids.shape[1]:]
    for i in range(new_tokens.size(0)):
        outs.append(processor.decode(new_tokens[i], skip_special_tokens=True).strip())
    return outs

def append_rows(path, df_chunk):
    """헤더는 파일이 없을 때만 쓰고, 이어서 append."""
    exists = os.path.exists(path)
    df_chunk.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8-sig")

def main():
    rank, world, device = setup_dist()
    out_rank = OUTPUT_PATH.replace(".csv", f".rank{rank}.csv")

    # === 데이터 로드 ===
    usecols = ["data_id", "dataset_name", "dataset_image_ids", "question", "rag_label"]
    df = pd.read_csv(CSV_PATH, usecols=usecols, encoding="utf-8-sig")
    # infoseek만
    df = df[df["dataset_name"].astype(str).str.lower() == "infoseek"].reset_index(drop=True)

    # === 샤드 분할 (YES/NO 모두 동일하게 분할) ===
    df_shard = df.iloc[rank::world].reset_index(drop=True)

    # === 재시작 대비: 이미 기록된 data_id는 스킵 ===
    written_ids = set()
    if os.path.exists(out_rank):
        try:
            prev = pd.read_csv(out_rank, usecols=["data_id"])
            written_ids = set(prev["data_id"].astype(str).tolist())
        except Exception:
            pass

    df_shard["data_id"] = df_shard["data_id"].astype(str)
    df_shard = df_shard[~df_shard["data_id"].isin(written_ids)].reset_index(drop=True)

    # === NO 샘플(생성 안 함) 먼저 기록 (llava_answer="") ===
    df_no = df_shard[~df_shard["rag_label"].apply(is_yes)].copy()
    if not df_no.empty:
        df_no["llava_answer"] = ""
        append_rows(out_rank, df_no.drop(columns=[], errors="ignore"))
        # 생성 대상에서 제외
        df_shard = df_shard[df_shard["rag_label"].apply(is_yes)].reset_index(drop=True)

    # === 이미지 경로 매핑 (YES만 필요) ===
    if not df_shard.empty:
        ids = df_shard["dataset_image_ids"].astype(str)
        path1 = ids.map(lambda k: INF_M.get(k, None))
        path2_key = ids.map(lambda k: (k.split("_",1)[1] if k.startswith("oven_") else f"oven_{k}"))
        path2 = path2_key.map(lambda k: INF_M.get(k, None))
        paths = path1.fillna(path2)
        paths = paths.map(lambda p: p if (isinstance(p, str) and os.path.exists(p)) else None)
        df_shard = df_shard.assign(__image_path=paths)
        df_shard = df_shard[df_shard["__image_path"].notnull()].reset_index(drop=True)

    # === 모델 준비 (생성할 게 없으면 스킵) ===
    if df_shard.empty:
        print(f"[rank {rank}] nothing to generate; only NO rows were appended. File: {out_rank}")
        return

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()

    total = len(df_shard)
    pbar = tqdm(total=total, desc=f"[rank {rank}] generating", position=rank)

    # === 배치별로 생성 후 즉시 append ===
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df_shard.iloc[start:end]

        # 이미지 로딩(안전)
        imgs = []
        for p in batch["__image_path"].tolist():
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (2, 2), (0, 0, 0)))  # fallback

        qs = batch["question"].astype(str).tolist()
        inputs = build_inputs(processor, imgs, qs)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        batch_answers = decode_batch(processor, inputs["input_ids"], out_ids)

        # append
        part = batch.drop(columns=["__image_path"], errors="ignore").copy()
        part["llava_answer"] = batch_answers
        append_rows(out_rank, part)

        pbar.update(len(batch))

        # 진행 로그 (선택)
        if (start // BATCH_SIZE + 1) * BATCH_SIZE % SAVE_INTERVAL == 0:
            print(f"[rank {rank}] appended up to row {end} -> {out_rank}")

    pbar.close()
    print(f"[rank {rank}] done. Appended YES + earlier NO to {out_rank}")

if __name__ == "__main__":
    main()
