import json
import os
import pickle

import faiss
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


@BUILDER.register_module()
class InfoSeekRAGDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_root,
        faiss_index_path,
        faiss_id_map_path,
        doc_json_path,
        image_processor=None,
        tokenizer=None,
        prompt_template=None,
        image2tensor=True,
        add_image_token=False,
        image_token=DEFAULT_IMAGE_TOKEN,
        top_k=1,
    ):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.tokenizer = (
            BUILDER.build(tokenizer) if isinstance(tokenizer, dict) else tokenizer
        )
        self.image_processor = (
            BUILDER.build(image_processor)
            if isinstance(image_processor, dict)
            else image_processor
        )
        self.prompt_template = prompt_template
        self.image2tensor = image2tensor
        self.add_image_token = add_image_token
        self.image_token = image_token
        self.top_k = top_k

        with open(doc_json_path, "r") as f:
            self.doc_dict = json.load(f)

        self.index = faiss.read_index(faiss_index_path)
        with open(faiss_id_map_path, "rb") as f:
            self.index_id_map = pickle.load(f)

        if add_image_token:
            special_tokens_dict = {"additional_special_tokens": [self.image_token]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.image_token_idx = self.tokenizer.encode(
            self.image_token, add_special_tokens=False
        )[-1]

    def __len__(self):
        return len(self.data)

    def retrieve_docs(self, question):
        # naive embedding by tokenizer (교체 필요)
        input_ids = self.tokenizer.encode(question, add_special_tokens=True)
        question_emb = torch.tensor(input_ids, dtype=torch.float32).unsqueeze(0).numpy()
        _, indices = self.index.search(question_emb, self.top_k)
        doc_ids = [self.index_id_map[i] for i in indices[0]]
        docs = [self.doc_dict[doc_id] for doc_id in doc_ids if doc_id in self.doc_dict]
        return docs

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = row["question"]
        answer = row.get("answer", "")
        image_id = row["dataset_image_ids"]
        data_id = row.get("data_id", f"infoseek_{index:08d}")
        image_path = os.path.join(self.image_root, image_id + ".JPEG")

        docs = self.retrieve_docs(question)
        context = "\n".join(docs)

        full_prompt = self.prompt_template["INSTRUCTION"].format(
            input=question + "\n" + context
        )
        input_ids = torch.tensor(
            self.tokenizer.encode(full_prompt, add_special_tokens=True),
            dtype=torch.long,
        )
        labels = torch.ones_like(input_ids) * IGNORE_INDEX
        labels[:] = input_ids[:]

        if self.add_image_token:
            input_ids[input_ids == self.image_token_idx] = IMAGE_TOKEN_INDEX

        image = Image.open(image_path).convert("RGB")
        image_data = self.image_processor.preprocess(image)
        pixel_values = image_data["pixel_values"][0]
        if self.image2tensor:
            pixel_values = torch.from_numpy(pixel_values)

        return dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=torch.tensor(image_data["image_sizes"][0]),
            meta_data=image_data["meta_datas"][0],
            image=image,
            file_name=image_id,
            question=question,
            answer=answer,
            data_id=data_id,
            context=context,
        )
