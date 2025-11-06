# build_manifest.py
import os, json
from pathlib import Path
from tqdm import tqdm   # ✅ tqdm 추가

INFOSEEK_IMAGE_ROOT = "/data/dataset/infoseek/oven/images"
OVEN_UNPACKED_ROOT  = "/data/dataset/infoseek/oven/unpacked"
OUT_PATH            = "/data/dataset/infoseek/infoseek_manifest.json"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def scan_dir(root):
    root = Path(root)
    all_files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in EXTS:
                all_files.append(p)
    # tqdm으로 감싸서 진행률 표시
    for p in tqdm(all_files, desc=f"Scanning {root}"):
        yield p.stem, str(p)

def main():
    manifest = {}
    # 1) 기본 images
    for k, v in scan_dir(INFOSEEK_IMAGE_ROOT):
        manifest[k] = v
    # 2) unpacked/*
    for k, v in scan_dir(OVEN_UNPACKED_ROOT):
        manifest[k] = v

    # 접두사 유무 모두 접근 가능하게 별칭 추가
    extra = {}
    for k, v in manifest.items():
        if k.startswith("oven_"):
            extra[k.split("_", 1)[1]] = v       # "02312789" → path
        else:
            extra[f"oven_{k}"] = v             # "oven_02312789" → path
    manifest.update(extra)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f)
    print(f"saved: {OUT_PATH} (keys={len(manifest)})")

if __name__ == "__main__":
    main()
