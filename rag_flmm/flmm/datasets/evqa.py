# import json
# import linecache
# import os

# import pandas as pd
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from xtuner.registry import BUILDER
# from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX


# @BUILDER.register_module()
# class EncyclopedicVQADataset(Dataset):
#     def __init__(
#         self,
#         csv_file,
#         inaturalist_root,
#         googlelandmark_root,
#         image_processor=None,
#         tokenizer=None,
#         prompt_template=None,
#         add_rag_token=False,                 # True면 [RAG] 앵커를 입력에 삽입
#         image_token=DEFAULT_IMAGE_TOKEN,
#         kb_file=None,
#         id2name_file=None,
#         rag_anchor_token="[RAG]",            # 단일 라우팅 앵커 토큰
#     ):
#         super().__init__()
#         self.csv_file = csv_file
#         self.inaturalist_root = inaturalist_root
#         self.googlelandmark_root = googlelandmark_root
#         self.prompt_template = prompt_template
#         self.add_rag_token = add_rag_token
#         self.kb_file = kb_file
#         self.rag_anchor_token = rag_anchor_token

#         with open(csv_file, "r") as f:
#             self.header = f.readline().strip().split(",")
#         self.total_lines = sum(1 for _ in open(csv_file)) - 1

#         self.tokenizer = BUILDER.build(tokenizer) if isinstance(tokenizer, dict) else tokenizer
#         self.image_processor = BUILDER.build(image_processor) if isinstance(image_processor, dict) else image_processor

#         # Special tokens: <image>, [RAG]
#         self.image_token = image_token or None
#         if self.image_token:
#             self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})
#             self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
#         else:
#             self.image_token_id = None

#         self.rag_token_id = None
#         if self.add_rag_token:
#             self.tokenizer.add_special_tokens({"additional_special_tokens": [self.rag_anchor_token]})
#             self.rag_token_id = self.tokenizer.convert_tokens_to_ids(self.rag_anchor_token)

#         self.id2name = None
#         if id2name_file:
#             with open(id2name_file, "r") as f:
#                 self.id2name = json.load(f)

#         # BOS/EOS/PAD
#         self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
#         self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
#         self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

#     def __len__(self):
#         return self.total_lines

#     # 존재 여부와 무관하게 "예상 경로"를 만들어 주는 함수
#     def _guess_image_path(self, dataset_name, image_id):
#         image_id_str = str(image_id)
#         if dataset_name == "inaturalist":
#             if self.id2name and image_id_str in self.id2name:
#                 rel_path = self.id2name[image_id_str]
#                 return os.path.join(self.inaturalist_root, rel_path)
#             return os.path.join(self.inaturalist_root, f"[missing_id2name]/{image_id_str}.jpg")
#         elif dataset_name == "landmarks":
#             if len(image_id_str) >= 3:
#                 return os.path.join(
#                     self.googlelandmark_root,
#                     image_id_str[0], image_id_str[1], image_id_str[2],
#                     f"{image_id_str}.jpg",
#                 )
#             return os.path.join(self.googlelandmark_root, f"[bad_id]/{image_id_str}.jpg")
#         return None

#     # 실제 파일이 있을 때만 경로 반환
#     def _get_image_path(self, dataset_name, image_id):
#         guess = self._guess_image_path(dataset_name, image_id)
#         if guess and os.path.exists(guess):
#             return guess
#         return None

#     def _load_kb(self, doc_id):
#         if not self.kb_file:
#             return None
#         with open(self.kb_file, "r") as f:
#             for line in f:
#                 item = json.loads(line)
#                 if item.get("doc_id") == doc_id:
#                     return item.get("text", "")
#         return None

#     def __getitem__(self, idx):
#         max_retry = self.__len__()
#         attempts = 0

#         while attempts < max_retry:
#             dataset_name = None
#             image_id = None
#             image_path_guess = None

#             try:
#                 line = linecache.getline(self.csv_file, idx + 2)
#                 row = dict(zip(self.header, line.strip().split(",")))

#                 dataset_name = row.get("dataset_name", "").strip().lower()
#                 image_id = str(row.get("dataset_image_ids", "")).strip()
#                 image_path_guess = self._guess_image_path(dataset_name, image_id)

#                 if dataset_name not in ["inaturalist", "landmarks"]:
#                     idx = (idx + 1) % len(self)
#                     attempts += 1
#                     continue

#                 question = row["question"]
#                 answer = row.get("answer", "")
#                 doc_id = row.get("doc_id", None)

#                 # rag_label: YES(1=검색필요X) / NO(0=검색필요)
#                 rag_label_str = row.get("rag_label", "NO").strip().upper()
#                 rag_label = 1 if rag_label_str == "YES" else 0

#                 # --- Prompt 텍스트 구성 ---
#                 instr = self.prompt_template["INSTRUCTION"].format(input=question)
#                 parts = []
#                 if self.image_token:
#                     parts.append(self.image_token)
#                 if self.add_rag_token and self.rag_token_id is not None:
#                     parts.append(self.rag_anchor_token)  # [RAG] 앵커
#                 parts.append(instr)
#                 prompt_text = " ".join(parts)

#                 # Tokenize
#                 prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
#                 answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)

#                 # Input & Label
#                 input_ids = [self.bos_token_id] + prompt_tokens + answer_tokens + [self.eos_token_id]
#                 labels = [IGNORE_INDEX] * (1 + len(prompt_tokens)) + answer_tokens + [self.eos_token_id]
#                 # 주의: [RAG], <image>, INSTRUCTION 모두 prompt 영역이므로 IGNORE_INDEX 처리됨.

#                 # 이미지 로드
#                 image_path = self._get_image_path(dataset_name, image_id)
#                 if image_path is None or not os.path.exists(image_path):
#                     raise FileNotFoundError(f"Image not found: {image_path}")
#                 image = Image.open(image_path).convert("RGB")
#                 image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

#                 kb_text = self._load_kb(doc_id) if doc_id else None

#                 return {
#                     "input_ids": torch.tensor(input_ids, dtype=torch.long),
#                     "labels": torch.tensor(labels, dtype=torch.long),
#                     "pixel_values": image_tensor,
#                     "image_path": image_path,
#                     "question": question,
#                     "answer": answer,
#                     "data_id": f"evqa_{idx:08d}",
#                     "kb_text": kb_text,
#                     "rag_label": rag_label,             # 0/1
#                     "pad_token_id": self.pad_token_id,
#                 }

#             except Exception as e:
#                 exists = os.path.exists(image_path_guess) if image_path_guess else False
#                 print(
#                     f"[SKIP] idx={idx}, dataset={dataset_name}, image_id={image_id}, "
#                     f"guess_path={image_path_guess}, exists={exists}, error={e}"
#                 )
#                 idx = (idx + 1) % len(self)
#                 attempts += 1

#         raise RuntimeError("[ERROR] Too many invalid samples. Dataset may be corrupted.")

import json
import linecache
import os

import pandas as pd
import torch
import torch.nn as nn  # ★ 추가
from PIL import Image
from torch.utils.data import Dataset
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX


def _human(n: float) -> str:
    units = ["", "K", "M", "G", "T", "P"]
    k = 1000.0
    i = 0
    x = float(n)
    while i < len(units) - 1 and abs(x) >= k:
        x /= k
        i += 1
    return f"{x:.3f}{units[i]}"


@BUILDER.register_module()
class EncyclopedicVQADataset(Dataset):
    def __init__(self,
                 csv_file,
                 inaturalist_root,
                 googlelandmark_root,
                 image_processor=None,
                 tokenizer=None,
                 prompt_template=None,
                 add_rag_token=False,
                 image_token=DEFAULT_IMAGE_TOKEN,
                 kb_file=None,
                 id2name_file=None,
                 rag_anchor_token="[RAG]",
                 **kwargs):
        super().__init__()
        self.csv_file = csv_file
        self.inaturalist_root = inaturalist_root
        self.googlelandmark_root = googlelandmark_root
        self.prompt_template = prompt_template
        self.add_rag_token = add_rag_token
        self.kb_file = kb_file
        self.rag_anchor_token = rag_anchor_token

        with open(csv_file, "r") as f:
            self.header = f.readline().strip().split(",")
        self.total_lines = sum(1 for _ in open(csv_file)) - 1

        self.tokenizer = BUILDER.build(tokenizer) if isinstance(tokenizer, dict) else tokenizer
        self.image_processor = BUILDER.build(image_processor) if isinstance(image_processor, dict) else image_processor

        # ---- rag_label 전처리 (CSV 읽어서 미리 리스트 저장) ----
        self.rag_labels = []
        with open(csv_file, "r") as f:
            header = f.readline().strip().split(",")
            for line in f:
                row = dict(zip(header, line.strip().split(",")))
                rag_label_str = row.get("rag_label", "NO").strip().upper()
                # NO → 검색필요 → USE_RAG=1, YES → 검색불필요 → NO_RAG=0
                rag_label = 1 if rag_label_str == "NO" else 0
                self.rag_labels.append(rag_label)

        # Special tokens: <image>, [RAG]
        self.image_token = image_token or None
        if self.image_token:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token_id = None

        self.rag_token_id = None
        if self.add_rag_token:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.rag_anchor_token]})
            self.rag_token_id = self.tokenizer.convert_tokens_to_ids(self.rag_anchor_token)

        self.id2name = None
        if id2name_file:
            with open(id2name_file, "r") as f:
                self.id2name = json.load(f)

        # BOS/EOS/PAD
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # ---- 비전 연산량 출력 설정 ----
        self.vision_encoder = None
        self.measure_vision_ops = False
        self.print_vision_ops_once = True
        self.verbose_vision_ops = False
        self._vision_ops_printed = False  # 내부 플래그

    def __len__(self):
        return self.total_lines

    # 존재 여부와 무관하게 "예상 경로"를 만들어 주는 함수
    def _guess_image_path(self, dataset_name, image_id):
        image_id_str = str(image_id)
        if dataset_name == "inaturalist":
            if self.id2name and image_id_str in self.id2name:
                rel_path = self.id2name[image_id_str]
                return os.path.join(self.inaturalist_root, rel_path)
            return os.path.join(self.inaturalist_root, f"[missing_id2name]/{image_id_str}.jpg")
        elif dataset_name == "landmarks":
            if len(image_id_str) >= 3:
                return os.path.join(
                    self.googlelandmark_root,
                    image_id_str[0], image_id_str[1], image_id_str[2],
                    f"{image_id_str}.jpg",
                )
            return os.path.join(self.googlelandmark_root, f"[bad_id]/{image_id_str}.jpg")
        return None

    # 실제 파일이 있을 때만 경로 반환
    def _get_image_path(self, dataset_name, image_id):
        guess = self._guess_image_path(dataset_name, image_id)
        if guess and os.path.exists(guess):
            return guess
        return None

    def _load_kb(self, doc_id):
        if not self.kb_file:
            return None
        with open(self.kb_file, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("doc_id") == doc_id:
                    return item.get("text", "")
        return None

    # ---- 연산량 훅: Conv2d / Linear / Attention 근사 MACs ----
    def _register_vision_hooks(self, module: nn.Module):
        handles = []
        self._macs_counter = 0  # 리셋

        def add(macs: int):
            self._macs_counter += int(macs)

        def h_linear(mod: nn.Linear, inputs, output):
            if not inputs: return
            x = inputs[0]
            if not torch.is_tensor(x) or x.numel() == 0: return
            in_f, out_f = mod.in_features, mod.out_features
            elems = x.numel() // max(1, x.shape[-1])
            add(elems * in_f * out_f)

        def h_conv2d(mod: nn.Conv2d, inputs, output):
            if not inputs: return
            x = inputs[0]
            if (not torch.is_tensor(x)) or (x.dim() != 4) or (not torch.is_tensor(output)): return
            Bout, Cout, Hout, Wout = output.shape
            Cin = mod.in_channels
            kh, kw = mod.kernel_size
            groups = max(1, mod.groups)
            add(Bout * Cout * Hout * Wout * (Cin // groups) * kh * kw)

        def h_attn_general(mod: nn.Module, inputs, output):
            if not inputs: return
            x = inputs[0]
            if not torch.is_tensor(x) or x.dim() != 3: return
            if x.shape[0] <= 8 and x.shape[1] > 8:
                L, B, E = x.shape
            else:
                B, L, E = x.shape
            H = getattr(mod, "num_heads", 1)
            D = getattr(mod, "head_dim", E if H == 1 else (E // max(1, H)))
            add(2 * B * H * L * L * D)  # QK^T + AV

        for m in module.modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(h_linear))
            elif isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(h_conv2d))
            else:
                cls = m.__class__.__name__.lower()
                if "attention" in cls or "attn" in cls:
                    handles.append(m.register_forward_hook(h_attn_general))
        return handles

    def _maybe_print_vision_ops(self, image_tensor: torch.Tensor, image_path: str):
        if not self.measure_vision_ops:
            return
        if self.print_vision_ops_once and self._vision_ops_printed:
            return
        # 분산 시 rank 0만 출력
        try:
            rank = int(os.environ.get("RANK", "0"))
        except Exception:
            rank = 0
        if rank != 0:
            self._vision_ops_printed = True
            return

        enc = self.vision_encoder
        if not isinstance(enc, nn.Module):
            return

        # 디바이스 추정
        try:
            dev = next(enc.parameters()).device
        except StopIteration:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pixel = image_tensor.unsqueeze(0).to(dev)  # [1,3,H,W]

        # 훅 등록 -> 비전 인코더만 직접 호출
        handles = self._register_vision_hooks(enc)
        was_training = enc.training
        enc.eval()
        with torch.no_grad():
            try:
                try:
                    _ = enc(pixel_values=pixel)
                except TypeError:
                    _ = enc(pixel)
            except Exception as e:
                # SAM 래퍼 안쪽 시도
                inner = getattr(enc, "vision_model", None)
                if isinstance(inner, nn.Module):
                    try:
                        _ = inner(pixel_values=pixel)
                    except TypeError:
                        _ = inner(pixel)
                else:
                    # 실패해도 훅 제거 후 리턴
                    for h in handles: h.remove()
                    if was_training: enc.train()
                    print(f"[VisionOps] forward 실패: {type(enc).__name__} | path={image_path} | err={e}")
                    self._vision_ops_printed = True
                    return
        # 훅 해제
        for h in handles:
            h.remove()
        if was_training:
            enc.train()

        macs = self._macs_counter
        flops = macs * 2
        print("\n[Vision-only Complexity] (dataset-internal, single image)")
        print(f"  image_path     : {image_path}")
        print(f"  Vision Encoder : {_human(macs)} MACs  (~{_human(flops)} FLOPs)")

        if self.verbose_vision_ops:
            lin_cnt = sum(1 for m in enc.modules() if isinstance(m, nn.Linear))
            conv_cnt = sum(1 for m in enc.modules() if isinstance(m, nn.Conv2d))
            print(f"  [dbg] Linear: {lin_cnt}, Conv2d: {conv_cnt}, enc={enc.__class__.__name__}")

        self._vision_ops_printed = True

    def __getitem__(self, idx):
        max_retry = self.__len__()
        attempts = 0

        while attempts < max_retry:
            dataset_name = None
            image_id = None
            image_path_guess = None

            try:
                line = linecache.getline(self.csv_file, idx + 2)
                row = dict(zip(self.header, line.strip().split(",")))

                dataset_name = row.get("dataset_name", "").strip().lower()
                image_id = str(row.get("dataset_image_ids", "")).strip()
                image_path_guess = self._guess_image_path(dataset_name, image_id)

                if dataset_name not in ["inaturalist", "landmarks"]:
                    idx = (idx + 1) % len(self)
                    attempts += 1
                    continue

                question = row["question"]
                answer = row.get("answer", "")
                doc_id = row.get("doc_id", None)

                # rag_label: YES(1=검색필요X) / NO(0=검색필요)
                rag_label_str = row.get("rag_label", "NO").strip().upper()
                rag_label = 1 if rag_label_str == "NO" else 0   # NO→1(검색필요), YES→0(검색불필요)

                # --- Prompt 텍스트 구성 ---
                instr = self.prompt_template["INSTRUCTION"].format(input=question)
                parts = []
                if self.image_token:
                    parts.append(self.image_token)
                if self.add_rag_token and self.rag_token_id is not None:
                    parts.append(self.rag_anchor_token)  # [RAG] 앵커
                parts.append(instr)
                prompt_text = " ".join(parts)

                # Tokenize
                prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)

                # Input & Label
                input_ids = [self.bos_token_id] + prompt_tokens + answer_tokens + [self.eos_token_id]
                labels = [IGNORE_INDEX] * (1 + len(prompt_tokens)) + answer_tokens + [self.eos_token_id]
                # 주의: [RAG], <image>, INSTRUCTION 모두 prompt 영역이므로 IGNORE_INDEX 처리됨.

                # 이미지 로드
                image_path = self._get_image_path(dataset_name, image_id)
                if image_path is None or not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

                # ★ 여기서 1회만 비전 인코더 연산량 출력
                self._maybe_print_vision_ops(image_tensor, image_path)

                kb_text = self._load_kb(doc_id) if doc_id else None

                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "pixel_values": image_tensor,
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "data_id": f"evqa_{idx:08d}",
                    "kb_text": kb_text,
                    "rag_label": rag_label,             # 0/1
                    "pad_token_id": self.pad_token_id,
                }

            except Exception as e:
                exists = os.path.exists(image_path_guess) if image_path_guess else False
                print(
                    f"[SKIP] idx={idx}, dataset={dataset_name}, image_id={image_id}, "
                    f"guess_path={image_path_guess}, exists={exists}, error={e}"
                )
                idx = (idx + 1) % len(self)
                attempts += 1

        raise RuntimeError("[ERROR] Too many invalid samples. Dataset may be corrupted.")
