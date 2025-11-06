import os

import spacy
import torch

# SDPA 비활성화(안전하게 eager path 강제)
import torch.backends.cuda as sdp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from mmengine.model import BaseModel
from torchvision.ops import masks_to_boxes
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

from flmm.models.word_utils import (
    extract_noun_phrases,
    extract_noun_token_ranges,
    get_spans,
    get_token_spans,
)
from flmm.utils import compute_mask_IoU
from flmm.xtuner_model.llava import LLaVAModel

sdp.enable_flash_sdp(False)
sdp.enable_mem_efficient_sdp(False)
sdp.enable_math_sdp(True)


class FrozenLlavaSAM(LLaVAModel):
    def __init__(self, sam, tokenizer=None, *args, **kwargs):
        pretrained = kwargs.pop("pretrained", None)
        super().__init__(*args, **kwargs)
        self.sam = BUILDER.build(sam)
        self.text_proj = nn.Linear(
            self.llm.config.hidden_size, self.sam.model.prompt_encoder.embed_dim
        )

        self.tokenizer = BUILDER.build(tokenizer)
        if self.tokenizer is not None:
            special_tokens_dict = {
                "additional_special_tokens": ["<image>", "<use_rag>", "<no_rag>"]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            # === 중요: 토큰 인덱스와 pad_token 동기화 ===
            self.llm.config.image_token_index = self.tokenizer.convert_tokens_to_ids(
                "<image>"
            )
            if self.llm.config.pad_token_id is None:
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.config.ignore_index = -100

        # --- attention 구현을 eager로 고정 ---
        self.llm.config.attn_implementation = "eager"
        if hasattr(self.llm, "language_model") and hasattr(
            self.llm.language_model, "config"
        ):
            self.llm.language_model.config.attn_implementation = "eager"

        # --- attention/hidden/caching 설정 ---
        self.llm.config.use_cache = False
        self.llm.config.output_attentions = True
        self.llm.config.output_hidden_states = True
        if hasattr(self.llm, "language_model") and hasattr(
            self.llm.language_model, "config"
        ):
            self.llm.language_model.config.use_cache = False
            self.llm.language_model.config.output_attentions = True
            self.llm.language_model.config.output_hidden_states = True

        print(
            "[cfg] llm.attn_impl =",
            getattr(self.llm.config, "attn_implementation", None),
        )
        if hasattr(self.llm, "language_model") and hasattr(
            self.llm.language_model, "config"
        ):
            print(
                "[cfg] llm.language_model.attn_impl  =",
                getattr(self.llm.language_model.config, "attn_implementation", None),
            )

        # === text layer weight 파라미터 ===
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llm.config.num_hidden_layers)
        )

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

        self.rag_router = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 128), nn.ReLU(), nn.Linear(128, 2)
        )

        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llm.config.num_hidden_layers)
        )
        self.nlp = spacy.load("en_core_web_sm")

        # merge 방식 (mean/max)
        self.merge = getattr(self, "merge", "mean")

    def apply_merge(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.merge == "mean":
            return x.float().mean(dim=dim)
        elif self.merge == "max":
            return x.float().max(dim=dim).values
        else:
            raise NotImplementedError(f"merge={self.merge}")

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def _generate_black_image(self, image_tensor, pred_masks, max_images=3):
        os.makedirs("debug_outputs", exist_ok=True)
        if image_tensor.dtype != torch.uint8:
            image_tensor = (image_tensor * 255).byte()
        image_tensor = image_tensor.cpu()
        black_images = []
        for i in range(min(max_images, pred_masks.shape[0])):
            mask = pred_masks[i] > 0.5
            box = masks_to_boxes(mask[None])[0].int().tolist()
            x1, y1, x2, y2 = box
            boxed_img = image_tensor.clone()
            for c in range(boxed_img.shape[0]):
                boxed_img[c, :y1, :] = 0
                boxed_img[c, y2:, :] = 0
                boxed_img[c, y1:y2, :x1] = 0
                boxed_img[c, y1:y2, x2:] = 0
            img_pil = F.to_pil_image(boxed_img)
            img_pil.save(f"debug_outputs/mask_with_box_{i}.png")
            black_images.append(img_pil)
        return black_images

    def _predict_masks(self, outputs, data_sample):
        text_layer_weights = self.get_text_layer_weights()
        meta_data = data_sample["meta_data"]
        mask_ids = outputs["mask_ids"][0]
        attentions = [
            attn[0, ..., outputs["image_to_overwrite"][0]]
            for attn in outputs.attentions
        ]
        hidden_states = outputs.hidden_states[-self.llm.config.num_hidden_layers :]
        labels = outputs.labels[0]

        hidden_states = torch.stack(
            [hs[0] for hs in hidden_states]
        )  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(
            0
        )  # seq_len, dim

        padded_h, padded_w = (
            meta_data["padded_shape"]["height"],
            meta_data["padded_shape"]["width"],
        )
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size
        attentions = [
            attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions
        ]

        masks = data_sample["masks"]
        mask_attentions = []
        text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            mask_attn = torch.cat(
                [self.apply_merge(attn[:, matched], dim=1) for attn in attentions],
                dim=0,
            )
            mask_attentions.append(mask_attn)
            text_embeds.append(self.text_proj(hidden_states[matched]))
        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)

        with torch.no_grad():
            pred_masks = self.mask_head(mask_attentions)[:, 0]

        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        before_height = int(
            meta_data["padding"]["before_height"] * padded_mask_h / padded_h
        )
        before_width = int(
            meta_data["padding"]["before_width"] * padded_mask_w / padded_w
        )
        mask_h = int(
            meta_data["image_shape"]["height"] * padded_mask_h / padded_h + 0.5
        )
        mask_w = int(meta_data["image_shape"]["width"] * padded_mask_w / padded_w + 0.5)
        pred_masks = pred_masks[
            :,
            before_height : before_height + mask_h,
            before_width : before_width + mask_w,
        ].contiguous()

        return pred_masks, mask_ids, hidden_states

    def _forward(self, data_sample, use_rag, mode="loss"):
        text_layer_weights = self.get_text_layer_weights()
        device = self.llm.device

        # Step 1: prepare inputs
        input_ids = data_sample["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.llm.device)

        pixel_values = data_sample["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pass
        elif (
            pixel_values.ndim == 5
            and pixel_values.shape[0] == 1
            and pixel_values.shape[1] == 1
        ):
            pixel_values = pixel_values.squeeze(0)
        pixel_values = pixel_values.to(device=self.llm.device, dtype=self.llm.dtype)

        labels = data_sample["labels"]
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        # ✅ labels는 Long 유지
        labels = labels.to(self.llm.device, dtype=torch.long)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if mode == "predict":
            generate_outputs = self.llm.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
            )
            output_ids = generate_outputs[0]
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            llm_loss = None
        else:
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.llm(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_rag=use_rag,
                )
            llm_loss = outputs.loss
            output_ids = input_ids[0]
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # === 명사 span → mask_ids 생성 (1D) ===
        noun_ranges = extract_noun_token_ranges(
            output_text, output_ids, self.tokenizer, self.nlp
        )
        mask_ids = torch.full_like(output_ids, fill_value=-100)
        for idx, (start, end) in enumerate(noun_ranges):
            mask_ids[start:end] = idx

        # === 2차 forward: model이 확장한 mask_ids 사용 ===
        orig_input_ids = data_sample["input_ids"]
        if orig_input_ids.ndim == 1:
            orig_input_ids = orig_input_ids.unsqueeze(0)
        orig_input_ids = orig_input_ids.to(device)

        # 이미지 토큰 위치와 패치 개수
        img_token_id = getattr(self.llm.config, "image_token_index", -1)
        img_pos = (orig_input_ids[0] == img_token_id).nonzero(as_tuple=True)[0].item()

        meta = data_sample.get("meta_data", {})
        pH = meta.get("padded_shape", {}).get("height", 336)
        pW = meta.get("padded_shape", {}).get("width", 336)
        llh, llw = pH // self.patch_size, pW // self.patch_size
        num_patches = llh * llw

        batch_size = input_ids.shape[0]
        if mask_ids is not None and mask_ids.ndim == 1:
            mask_ids = mask_ids.unsqueeze(0).expand(batch_size, -1)

        second_outputs = self.llm(
            input_ids=orig_input_ids,
            mask_ids=mask_ids,  # (B, T)
            pixel_values=pixel_values,
            labels=labels,
            attention_mask=torch.ones_like(orig_input_ids, dtype=torch.bool),
            output_hidden_states=True,
            output_attentions=True,
        )

        # 디버그
        print(
            "[DBG] att returns:",
            [None if a is None else a.shape for a in second_outputs.attentions],
        )

        # === mask 예측 준비 ===
        meta_data = data_sample.get("meta_data", {})
        text_layer_weights = self.get_text_layer_weights()

        # 이미지 패치 구간만 선택
        img_slice = slice(img_pos, img_pos + num_patches)
        attentions = [
            attn[0, :, img_slice].view(
                attn.shape[1], -1, llh, llw
            )  # [heads, seq_len, H, W]
            for attn in second_outputs.attentions
        ]

        # (중요) 모델이 확장해서 돌려준 mask_ids 사용 (1D, 길이 L')
        expanded_mask_ids = second_outputs.mask_ids[0]  # shape: [L']

        hidden_states = second_outputs.hidden_states[
            -self.llm.config.num_hidden_layers :
        ]
        hidden_states = torch.stack(
            [hs[0] for hs in hidden_states]
        )  # [num_layers, L', D]
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(
            0
        )  # [L', D]

        padded_h = meta_data.get("padded_shape", {}).get("height", 336)
        padded_w = meta_data.get("padded_shape", {}).get("width", 336)
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size
        # attentions는 이미 [heads, L', H, W] 형태

        mask_attns, text_embeds = [], []
        for m_id in range(len(noun_ranges)):
            matched = (
                expanded_mask_ids == m_id
            )  # ✅ 1D boolean, length L' (attn의 seq_len과 정합)
            if matched.sum() == 0:
                continue
            # 각 레이어(attention map)에서 [heads, L', H, W] → L'에서 matched 선택 → dim=1로 merge
            merged_per_layer = [
                self.apply_merge(attn[:, matched], dim=1) for attn in attentions
            ]  # list of [heads, H, W]
            mask_attns.append(
                torch.cat(merged_per_layer, dim=0)
            )  # [heads*layers, H, W]
            text_embeds.append(
                self.text_proj(hidden_states[matched])
            )  # [#tokens_of_span, D'] after proj

        if len(mask_attns) > 0:
            mask_attns = torch.stack(mask_attns).to(
                self.mask_head.dtype
            )  # [N_spans, C_in, H, W]
            pred_masks = self.mask_head(mask_attns)[:, 0]
        else:
            pred_masks = torch.zeros((1, padded_h, padded_w), device=device)

        # === SAM refinement ===
        sam_pred_masks = (
            self.sam(data_sample["image"], pred_masks, text_embeds)
            if len(text_embeds) > 0
            else pred_masks
        )

        return {
            "pred_masks": pred_masks,
            "sam_pred_masks": sam_pred_masks,
            "labels": labels,
            "mask_ids": expanded_mask_ids,  # ✅ 확장된 mask_ids 반환
            "hidden_states": hidden_states,
            "llm_loss": llm_loss,  # ✅ forward에서 사용
            "output_text": output_text,
        }

    def forward(self, *args, **kwargs):
        mode = kwargs.pop("mode", "loss")
        data = kwargs

        # === 입력 차원 보정 ===
        input_ids = data["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=self.llm.device)  # int64 유지

        labels = data["labels"]
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        labels = labels.to(device=self.llm.device, dtype=torch.long)  # ✅ long

        pixel_values = data["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 5:
            pixel_values = pixel_values.squeeze(0)
        pixel_values = pixel_values.to(device=self.llm.device, dtype=self.llm.dtype)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # 디버깅 출력
        print(f"[forward] input_ids dtype: {input_ids.dtype}")
        print(f"[forward] pixel_values dtype: {pixel_values.dtype}")
        print(f"[forward] labels dtype: {labels.dtype}")
        print(f"[forward] model param dtype: {next(self.llm.parameters()).dtype}")

        # 이미지 토큰 검증
        if (input_ids == getattr(self.llm.config, "image_token_index", -1)).sum() == 0:
            raise ValueError(
                f"No image token found in input_ids. Got shape {input_ids.shape}"
            )

        # RAG 라우터 (그라디언트 필요 없으면 detach)
        router_outputs = self.llm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_rag=False,
        )
        first_token_hidden = router_outputs.hidden_states[-1][0, 0]
        router_logits = self.rag_router(first_token_hidden)
        use_rag = torch.argmax(router_logits, dim=-1).item() == 1

        # 메인 forward
        data["input_ids"] = input_ids
        data["labels"] = labels
        data["pixel_values"] = pixel_values

        out = self._forward(data_sample=data, use_rag=use_rag)

        if mode == "predict":
            return out

        # === Loss 계산 ===
        llm_loss = out["llm_loss"]  # ✅ 키 수정
        rag_label = torch.tensor(
            [data.get("rag_label", int(use_rag))], device=self.llm.device
        )
        rag_loss = F.cross_entropy(router_logits.unsqueeze(0), rag_label)

        return {"loss": llm_loss + rag_loss, "llm_loss": llm_loss, "rag_loss": rag_loss}

    @torch.no_grad()
    def predict(self, data_sample):
        return self._forward(data_sample, use_rag=False, mode="predict")[
            "sam_pred_masks"
        ]

    def compute_loss(self, data, data_samples=None):
        out = self._forward(data, use_rag=False, mode="loss")
        loss = out["llm_loss"]
        return {"loss": loss}

    def train_step(self, data, optimizer):
        return self.compute_loss(data)
