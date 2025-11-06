import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RAGVQADataset(Dataset):
    def __init__(
        self,
        csv_paths,  # {"infoseek": "...csv", "evqa": "...csv"}
        kb_paths,  # {"infoseek": "...json", "evqa": "...json"}
        image_processor,
        tokenizer,
        prompt_template,
        image_dir_inat,
        image_dir_gld,
        inat_id2name_path,
        image_dir_oven=None,
        selected_sources="infoseek|evqa",
    ):
        super().__init__()
        self.data = []
        self.kb = {}
        self.kb_per_item = []

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.image_dir_inat = image_dir_inat
        self.image_dir_gld = image_dir_gld
        self.image_dir_oven = image_dir_oven
        self.inat_id2name = json.load(open(inat_id2name_path))
        self.selected_sources = selected_sources.split("|")

        for src in self.selected_sources:
            assert src in csv_paths, f"{src} is missing in csv_paths"
            assert src in kb_paths, f"{src} is missing in kb_paths"

            df = pd.read_csv(csv_paths[src])
            df["source"] = src  # ← source를 따로 저장 (infoseek / evqa)
            self.kb[src] = json.load(open(kb_paths[src]))
            self.data.extend(df.to_dict(orient="records"))
            self.kb_per_item.extend([self.kb[src]] * len(df))

    def __len__(self):
        return len(self.data)

    def resolve_image_path(self, source, dataset_name, image_id):
        if source == "evqa":
            if dataset_name == "inaturalist":
                if str(image_id) not in self.inat_id2name:
                    print(f"[Missing] {image_id} not found in inaturalist mapping")
                    return None
                return os.path.join(
                    self.image_dir_inat, self.inat_id2name[str(image_id)]
                )
            elif dataset_name == "landmarks":
                hex_id = image_id[:3]
                return os.path.join(
                    self.image_dir_gld,
                    hex_id[0],
                    hex_id[1],
                    hex_id[2],
                    image_id + ".jpg",
                )
            else:
                print(f"[Error] Unknown evqa dataset_name: {dataset_name}")
                return None

        elif source == "infoseek":
            return os.path.join(self.image_dir_oven, image_id + ".JPEG")

        else:
            print(f"[Error] Unknown source: {source}")
            return None

    def __getitem__(self, index):
        for _ in range(len(self.data)):  # 전체 데이터 길이만큼 순회 시도
            sample = self.data[index]
            kb = self.kb[sample["source"]]

            question = sample["question"]
            answer = sample.get("answer", "")
            dataset_name = sample["dataset_name"]
            source = sample["source"]

            # 이미지 ID 처리
            raw_image_ids = sample.get("dataset_image_ids")
            if isinstance(raw_image_ids, str):
                image_ids = raw_image_ids.split("|")
            else:
                image_ids = [str(raw_image_ids)]
            image_id = image_ids[0]

            image_path = self.resolve_image_path(source, dataset_name, image_id)
            if image_path is None or not os.path.exists(image_path):
                index = (index + 1) % len(self.data)
                continue  # 다음 샘플로 건너뜀

            # 이미지 로딩 및 전처리
            image = Image.open(image_path).convert("RGB")
            image_data = self.image_processor.preprocess(image)
            pixel_values = torch.from_numpy(image_data["pixel_values"][0])

            # 토큰화
            prompt = self.prompt_template["INSTRUCTION"].format(input=question)
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt, add_special_tokens=True), dtype=torch.long
            )

            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "pixel_values": pixel_values,
                "image": image,
                "question": question,
                "answer": answer,
                "data_id": sample.get("data_id", f"{source}_{index:06d}"),
                "image_id": image_id,
                "dataset_name": dataset_name,
                "kb": kb,
            }

        raise RuntimeError("No valid image found in dataset.")
