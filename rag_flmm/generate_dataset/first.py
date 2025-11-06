import os, re, json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from functools import lru_cache

# ===== 경로 설정 =====
CSV_PATH = "/data/dataset/infoseek/infoseek_train_with_rag_label.csv"
INATURALIST_ROOT = "/dataset/inaturalist"
GOOGLELANDMARK_ROOT = "/dataset/landmarks/train"
INFOSEEK_IMAGE_ROOT = "/data/dataset/infoseek/oven/images"
OVEN_UNPACKED_ROOT = "/data/dataset/infoseek/oven/unpacked"
INFOSEEK_MANIFEST = "/data/dataset/infoseek/infoseek_manifest.json"
ID2NAME_PATH = "/dataset/inaturalist/val_id2name.json"
OUTPUT_PATH = "/data/dataset/infoseek/infoseek_train_generated_answers.csv"
INFOSEEK_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
GEN_COL = "llava_answer"
SAVE_INTERVAL = 100000 

with open(INFOSEEK_MANIFEST) as f:
    _INF_M = json.load(f)

# ===== 이미지 경로 유틸 =====
# ---------- infoseek 이미지 경로 ----------
def _try_paths(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def _infoseek_path(image_id_str: str):
    # 1) 매니페스트 즉시 조회
    p = _INF_M.get(image_id_str)
    if p and os.path.exists(p):
        return p

    # 2) 접두사 보정하여 재시도
    if image_id_str.startswith("oven_"):
        key = image_id_str.split("_", 1)[1]
    else:
        key = f"oven_{image_id_str}"
    p = _INF_M.get(key)
    if p and os.path.exists(p):
        return p

    return None

# def _infoseek_path(image_id_str: str):
#     """infoseek 이미지는 dataset_image_ids 기반 탐색 (+ unpacked 폴백)"""
#     prefix = image_id_str.split("_")[0] if "_" in image_id_str else None
#     roots = [
#         INFOSEEK_IMAGE_ROOT,
#         os.path.join(os.path.dirname(INFOSEEK_IMAGE_ROOT), "images"),
#         os.path.dirname(INFOSEEK_IMAGE_ROOT),
#     ]
#     roots = list(dict.fromkeys([r for r in roots if r]))  # 중복 제거

#     # 1) 기본 위치들에서 직접 매칭
#     cands = []
#     for root in roots:
#         for ext in INFOSEEK_EXTS:
#             # <root>/<image_id>.<ext>
#             cands.append(os.path.join(root, f"{image_id_str}{ext}"))
#             # <root>/<prefix>/<image_id>.<ext>
#             if prefix:
#                 cands.append(os.path.join(root, prefix, f"{image_id_str}{ext}"))

#     found = _try_paths(cands)
#     if found:
#         return found

#     # 2) 폴백: /data/dataset/infoseek/oven/unpacked/<앞 두자리>/ 에서 탐색
#     #    - dataset_image_id: 예) "oven_02312789"
#     #    - "oven_" 제거 → "02312789"
#     #    - 앞 두 자리("02") 디렉토리로 진입하여 파일 탐색
#     #    - 파일명은 보통 "02312789.jpg" 형태 가정 + 유연하게 "*02312789*.(ext)" 재귀 탐색
#     img_key = image_id_str.split("_", 1)[1] if image_id_str.startswith("oven_") else image_id_str
#     if len(img_key) >= 2:
#         subdir = img_key[:2]
#         base_dir = os.path.join(OVEN_UNPACKED_ROOT, subdir)

#         if os.path.isdir(base_dir):
#             # 2-1) 정밀 매칭: <base_dir>/<img_key>.<ext>
#             cands = [os.path.join(base_dir, f"{img_key}{ext}") for ext in INFOSEEK_EXTS]
#             found = _try_paths(cands)
#             if found:
#                 return found

#             # 2-2) 유연 매칭: 재귀적으로 "*<img_key>*.<ext>" 검색
#             for root, _, files in os.walk(base_dir):
#                 for fname in files:
#                     lower = fname.lower()
#                     # 확장자 허용 여부
#                     if not any(lower.endswith(ext) for ext in INFOSEEK_EXTS):
#                         continue
#                     # 키워드 포함 여부
#                     if img_key in fname:
#                         cand = os.path.join(root, fname)
#                         if os.path.exists(cand):
#                             return cand

#     # 못 찾으면 None
#     return None

# ---------- dataset별 경로 ----------
def get_image_path(dataset_name, image_id, id2name=None, row=None):
    image_id_str = str(image_id)

    if dataset_name == "inaturalist":
        if id2name and image_id_str in id2name:
            rel_path = id2name[image_id_str]
            full_path = os.path.join(INATURALIST_ROOT, rel_path)
            if os.path.exists(full_path):
                return full_path

    elif dataset_name == "landmarks":
        if len(image_id_str) >= 3:
            path = os.path.join(
                GOOGLELANDMARK_ROOT,
                image_id_str[0], image_id_str[1], image_id_str[2],
                f"{image_id_str}.jpg",
            )
            if os.path.exists(path):
                return path

    elif dataset_name == "infoseek":
        return _infoseek_path(image_id_str)

    return None


def load_llava(device):
    processor = LlavaProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]}
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()
    return processor, model


def clean_answer(pred, question):
    pred = pred.strip()
    if pred.lower().startswith(question.lower()):
        pred = pred[len(question) :].strip()
    pred = pred.lstrip(".:;- \n")
    return pred


def generate_batch(processor, model, batch_data, device, max_new_tokens=128):
    images = [Image.open(p).convert("RGB") for p in batch_data["image_paths"]]
    prompts = [f"{DEFAULT_IMAGE_TOKEN}\n{q}" for q in batch_data["questions"]]
    inputs = processor(
        images=images, text=prompts, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    preds = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_preds = [clean_answer(p, q) for p, q in zip(preds, batch_data["questions"])]
    return cleaned_preds


def worker(rank, data_chunk, id2name, output_dir):
    device = f"cuda:{rank}"
    processor, model = load_llava(device)
    results = []
    gpu_file = os.path.join(output_dir, f"gpu_{rank}.csv")

    for i in tqdm(
        range(0, len(data_chunk), BATCH_SIZE), position=rank, desc=f"GPU{rank}"
    ):
        batch = data_chunk.iloc[i : i + BATCH_SIZE]
        valid_rows, image_paths, questions = [], [], []
        for _, row in batch.iterrows():
            dataset_name = row["dataset_name"].strip().lower()
            image_id = str(row["dataset_image_ids"])
            if dataset_name == "infoseek":
                image_path = get_image_path(dataset_name, image_id)
            else:
                image_path = get_image_path(dataset_name, image_id, id2name)
            if image_path:
                valid_rows.append(row)
                image_paths.append(image_path)
                questions.append(row["question"])
        if not valid_rows:
            continue

        try:
            preds = generate_batch(
                processor,
                model,
                {"image_paths": image_paths, "questions": questions},
                device,
            )
            for row, pred, img_path in zip(valid_rows, preds, image_paths):
                row_data = row.to_dict()
                row_data.update(
                    {"generated": pred, "reason": "", "image_path": img_path}
                )
                results.append(row_data)
        except Exception as e:
            # OOM 등 치명적 오류 시 종료
            if "out of memory" in str(e).lower():
                print(f"[GPU{rank}] OOM 발생 - 프로세스 종료")
                os._exit(1)
            for row, img_path in zip(valid_rows, image_paths):
                row_data = row.to_dict()
                row_data.update(
                    {"generated": "", "reason": f"fail:{e}", "image_path": img_path}
                )
                results.append(row_data)

        # N 배치마다 저장
        if (i // BATCH_SIZE + 1) % SAVE_INTERVAL == 0:
            pd.DataFrame(results).to_csv(
                gpu_file,
                mode="a",
                index=False,
                header=not os.path.exists(gpu_file),
                encoding="utf-8-sig",
            )
            results = []

    # 남은 데이터 저장
    if results:
        pd.DataFrame(results).to_csv(
            gpu_file,
            mode="a",
            index=False,
            header=not os.path.exists(gpu_file),
            encoding="utf-8-sig",
        )


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    id2name = json.load(open(ID2NAME_PATH))
    num_gpus = torch.cuda.device_count()
    print(
        f"[INFO] Found {num_gpus} GPUs. Batch size: {BATCH_SIZE}, Save every {SAVE_INTERVAL} batches"
    )

    chunks = torch.chunk(torch.arange(len(df)), num_gpus)
    data_chunks = [df.iloc[chunk.tolist()] for chunk in chunks]

    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=worker, args=(rank, data_chunks[rank], id2name, output_dir)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("[INFO] GPU별 CSV 병합 중...")
    all_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("gpu_")
    ]
    combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 최종 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()