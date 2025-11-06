import os
import math

import spacy
import torch

# SDPA로는 attentions 수집 불가 → eager로 강제
import torch.backends.cuda as sdp
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchvision.ops import masks_to_boxes
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

from flmm.models.word_utils import extract_noun_token_ranges
from flmm.xtuner_model.llava import LLaVAModel

sdp.enable_flash_sdp(False)
sdp.enable_mem_efficient_sdp(True)
sdp.enable_math_sdp(True)

IGNORE_INDEX = -100

class RouterNetV2(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.1, temperature=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1  = nn.Linear(in_dim, hidden)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden, 1)
        # 검증에서 스윕만 할 거면 requires_grad=False 권장
        self.tau  = nn.Parameter(torch.tensor(float(temperature)), requires_grad=False)

    def forward(self, x):              # x: [B, in_dim]
        h = self.drop(self.act(self.fc1(self.norm(x))))
        logit = self.fc2(h).squeeze(-1)  # [B]
        return logit / self.tau
    
class RouterNetV3(nn.Module):
    def __init__(self, txt_dim, vis_dim, base_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.txt_proj = nn.Linear(txt_dim, 128, bias=False)
        self.vis_proj = nn.Linear(vis_dim, 128, bias=False)
        self.bilin    = nn.Bilinear(128, 128, 64, bias=True)

        self._hidden = hidden
        self._dropout = dropout
        self.mlp = None
        self._mlp_in = None

    @staticmethod
    def _entropy_like(x):
        p = torch.softmax(x.float(), dim=-1)
        return -(p * (p.clamp_min(1e-9).log())).sum(dim=-1, keepdim=True)

    def _ensure_mlp(self, in_dim: int, device: torch.device):
        if (self.mlp is None) or (self._mlp_in != in_dim):
            self._mlp_in = in_dim
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, self._hidden), nn.GELU(),
                nn.Dropout(self._dropout),
                nn.Linear(self._hidden, 1)
            ).to(device)  # ★ 여기!

    def forward(self, base_feat, txt_feat, vis_feat, txt_logits=None, vis_logits=None, ret_stats=None):
        target_dtype = self.txt_proj.weight.dtype
        txt_feat = txt_feat.to(target_dtype)
        vis_feat = vis_feat.to(self.vis_proj.weight.dtype)
        base_feat = base_feat.to(target_dtype)

        t = self.txt_proj(txt_feat)
        v = self.vis_proj(vis_feat)
        inter = self.bilin(t, v)

        pieces = [txt_feat.norm(dim=-1, keepdim=True).to(target_dtype),
                  vis_feat.norm(dim=-1, keepdim=True).to(target_dtype)]
        if txt_logits is not None: pieces.append(self._entropy_like(txt_logits))
        if vis_logits is not None: pieces.append(self._entropy_like(vis_logits))
        if ret_stats is not None:
            if 'max'  in ret_stats: pieces.append(ret_stats['max'])
            if 'mean' in ret_stats: pieces.append(ret_stats['mean'])

        # ret_stats가 CPU일 수 있으니 장치 맞추기
        dev = base_feat.device
        pieces = [p.to(device=dev, dtype=target_dtype) for p in pieces]

        scalars = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        x = torch.cat([base_feat, inter, scalars], dim=-1)

        self._ensure_mlp(x.size(-1), x.device)  # ★ 여기!
        return self.mlp(x).squeeze(-1)


class FrozenLlavaSAM(BaseModel):
    def __init__(
        self,
        model,
        sam,
        tokenizer=None,
        mask_head=None,
        merge: str = "mean",
        alpha=0.5,
        router_threshold: float | None = None,   # or 0.70 (세트 D)
        rag_pos_weight: float = 0.65,     # or 0.33
        rag_focal_gamma:float =0.43,     # or 2.0
        rag_focal_alpha: float | None = None,    # or 0.40
        llm_pos_weight:float | None = None,
        *args,
        **kwargs,
    ):
        pretrained = kwargs.pop("pretrained", None)
        llm_lora = kwargs.pop("llm_lora", None)
        lora_pretrained_path = kwargs.pop("lora_pretrained_path", None)
        super().__init__(*args, **kwargs)

        self.llm = BUILDER.build(model)
        if llm_lora is not None:
            try:
                self.llm = prepare_model_for_kbit_training(self.llm, use_gradient_checkpointing=True)
            except Exception:
                pass

            lora_cfg = BUILDER.build(llm_lora) if isinstance(llm_lora, dict) else llm_lora
            if isinstance(lora_cfg, dict):
                lora_cfg = LoraConfig(**lora_cfg)

            self.llm = get_peft_model(self.llm, lora_cfg)

            if lora_pretrained_path:
                try:
                    self.llm.load_adapter(lora_pretrained_path, adapter_name="default")
                except Exception:
                    sd = torch.load(lora_pretrained_path, map_location="cpu")
                    missing, unexpected = self.llm.load_state_dict(sd, strict=False)
                    print("[LoRA] missing:", missing, "unexpected:", unexpected)

            trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llm.parameters())
            print(f"[LoRA] trainable params: {trainable}/{total} ({100 * trainable / total:.2f}%)")

        # --- external modules ---
        self.sam = BUILDER.build(sam)

        # --- tokenizer / special tokens ---
        self.tokenizer = BUILDER.build(tokenizer)
        self.rag_token = "[RAG]"
        if self.tokenizer is not None:
            special_tokens_dict = {"additional_special_tokens": ["<image>", self.rag_token]}
            added = self.tokenizer.add_special_tokens(special_tokens_dict)
            if added > 0:
                self.llm.resize_token_embeddings(len(self.tokenizer))
            self.llm.config.image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
            self.rag_token_id = self.tokenizer.convert_tokens_to_ids(self.rag_token)
            if self.llm.config.pad_token_id is None:
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.config.ignore_index = IGNORE_INDEX
        else:
            self.rag_token_id = None

        # --- attention impl ---
        self.llm.config.attn_implementation = "eager"
        if hasattr(self.llm, "language_model") and hasattr(self.llm.language_model, "config"):
            self.llm.language_model.config.attn_implementation = "eager"

        # --- outputs/caches ---
        self.llm.config.use_cache = False
        self.llm.config.output_hidden_states = False
        if hasattr(self.llm, "language_model"):
            self.llm.language_model.config.output_hidden_states = True
            self.llm.language_model.config.use_cache = False

        print("[cfg] llm.attn_impl =", getattr(self.llm.config, "attn_implementation", None))
        if hasattr(self.llm, "language_model") and hasattr(self.llm.language_model, "config"):
            print("[cfg] llm.language_model.attn_impl =", getattr(self.llm.language_model.config, "attn_implementation", None))

        # --- projection for SAM text prompt ---
        config = self.llm.config

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                hidden_size = getattr(text_cfg, "hidden_size", None)
        if hidden_size is None:
            lang_cfg = getattr(getattr(self.llm, "language_model", None), "config", None)
            if lang_cfg is not None:
                hidden_size = getattr(lang_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.llm, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Could not resolve hidden_size from Llava config")

        self.text_proj = nn.Linear(int(hidden_size), self.sam.model.prompt_encoder.embed_dim)

        # --- patch size from vision tower ---
        self.patch_size = getattr(getattr(self.llm.config, "vision_config", None), "patch_size", 14)

        # --- merge strategy ---
        assert merge in ("mean", "max")
        self.merge = merge

        # Resolve layer/head metadata up front
        resolved_num_layers = getattr(config, "num_hidden_layers", None)
        if resolved_num_layers is None:
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                resolved_num_layers = getattr(text_cfg, "num_hidden_layers", None)
        if resolved_num_layers is None:
            lang_cfg = getattr(getattr(self.llm, "language_model", None), "config", None)
            if lang_cfg is not None:
                resolved_num_layers = getattr(lang_cfg, "num_hidden_layers", None)
        if resolved_num_layers is None:
            resolved_num_layers = getattr(self.llm, "num_hidden_layers", None)
        if resolved_num_layers is None:
            raise AttributeError("Could not resolve num_hidden_layers from Llava config")

        # --- text layer weights ---
        self.text_layer_weights = nn.Parameter(torch.ones(int(resolved_num_layers)))

        # --- mask_head ---
        assert mask_head is not None, "mask_head config must be provided"
        num_layers = resolved_num_layers

        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                num_heads = getattr(text_cfg, "num_attention_heads", None)
        if num_heads in (None, 0) and hasattr(self.llm, "language_model"):
            num_heads = getattr(self.llm.language_model.config, "num_attention_heads", num_heads)
        if num_heads is None:
            num_heads = getattr(self.llm, "num_attention_heads", None)
        if num_heads is None:
            raise AttributeError("Could not resolve num_attention_heads from Llava config")

        in_channels = int(num_layers) * int(num_heads)
        mh_cfg = dict(mask_head)
        mh_cfg.update(in_channels=in_channels)
        self.mask_head = BUILDER.build(mh_cfg)

        # --- RAG router head: single logit ---
        H = int(hidden_size)
        # vision hidden size 추출 (없으면 텍스트와 동일 차원으로 fallback)
        H_v = None
        vcfg = getattr(self.llm.config, 'vision_config', None)
        if vcfg is not None:
            H_v = getattr(vcfg, 'hidden_size', None)
        if H_v is None:
            vt = getattr(self.llm, 'vision_tower', None)
            if vt is not None and hasattr(vt, 'config'):
                H_v = getattr(vt.config, 'hidden_size', None)
        if H_v is None:
            H_v = H  # 마지막 안전장치

        self._router_in_dim = 3 * H  # [h_rag || h_bos || mean]
        self.rag_router = RouterNetV3(txt_dim=H, vis_dim=H_v, base_dim=self._router_in_dim, hidden=256, dropout=0.1)


        self.alpha = float(alpha)
        self.router_threshold = float(0.5 if router_threshold is None else router_threshold)
        self.rag_pos_weight = rag_pos_weight
        self.rag_focal_gamma = rag_focal_gamma
        self.rag_focal_alpha = rag_focal_alpha

        # --- NLP ---
        self.nlp = spacy.load("en_core_web_sm")

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

    # ------------------------ utils ------------------------
    def apply_merge(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.merge == "mean":
            return x.float().mean(dim=dim)
        elif self.merge == "max":
            return x.float().max(dim=dim).values
        else:
            raise NotImplementedError(f"merge={self.merge}")

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def _select_rag_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """hidden: [B,T,H], input_ids: [B,T] -> [RAG] 위치 히든(없으면 BOS)"""
        if self.rag_token_id is None:
            return hidden[:, 0, :]
        with torch.no_grad():
            rag_mask = (input_ids == self.rag_token_id)
            if rag_mask.any():
                idx = rag_mask.float().argmax(dim=1)
            else:
                idx = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
        B = input_ids.size(0)
        return hidden[torch.arange(B, device=hidden.device), idx, :]

    def _build_labels_rag_only(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        labels를 전부 -100로 마스킹하고, [RAG] 위치만 정답으로 남긴다.
        HF의 shift 로스 규칙상 label[t]는 logits[t-1]에 대응하므로
        [RAG]가 시퀀스의 첫 위치(=t=0)라면 유효 감독이 되지 않는다(경고만 출력).
        """
        assert input_ids.dim() == 2, "input_ids must be [B,T]"
        dev = input_ids.device
        labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX, device=dev)

        if self.rag_token_id is None:
            return labels  # 감독 없음

        # 배치별 첫 [RAG] 위치
        rag_mask = (input_ids == self.rag_token_id)        # [B,T]
        has_rag = rag_mask.any(dim=1)                      # [B]
        rag_pos = rag_mask.float().argmax(dim=1)           # [B]

        # t=0인 경우는 유효 감독 불가(HF shift 규칙). 경고 로그(1회성) 정도만.
        # 여기선 그냥 무시.
        valid = (has_rag & (rag_pos > 0))
        if valid.any():
            b_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            t_idx = rag_pos[b_idx]
            labels[b_idx, t_idx] = self.rag_token_id

        return labels

    # ------------------------ forward paths ------------------------
    def _forward(self, data_sample, mode="loss"):
        """
        Router: [RAG] 히든 → 단일 로짓
        LLM:   [RAG] 토큰만 supervise하는 CE loss
        """
        dev = next(self.llm.parameters()).device

        # --- inputs normalize ---
        input_ids = data_sample["input_ids"]
        input_ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
        pixel_values = data_sample["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim == 5:
            pixel_values = pixel_values.squeeze(0)
        input_ids = input_ids.to(dev)
        pixel_values = pixel_values.to(dev, dtype=getattr(self.llm, "dtype", torch.float16))

        # attention_mask
        attn = data_sample.get("attention_mask", None)
        if attn is None:
            pad_id = getattr(self.llm.config, "pad_token_id", getattr(self, "pad_token_id", 0))
            attn = input_ids.ne(pad_id)
        else:
            attn = attn.to(dev)
            if attn.ndim == 1:
                attn = attn.unsqueeze(0)
        attention_mask = attn.to(dtype=torch.bool)

        # ----- LLM labels: [RAG]만 supervise -----
        labels = None
        if mode != "predict":
            labels = self._build_labels_rag_only(input_ids)

        # ===== Router/LLM pass =====
        with torch.cuda.amp.autocast(enabled=False):
            router_out = self.llm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,                       # <-- [RAG]만 CE에 포함
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                use_rag=False,
            )

        last_hidden = router_out.hidden_states[-1]             # [B,T,H]
        h_rag  = self._select_rag_hidden(last_hidden, input_ids)  # [B,H]
        h_bos  = last_hidden[:, 0, :]                          # [B,H]
        h_mean = last_hidden.mean(dim=1)                       # [B,H]
        base_feat = torch.cat([h_rag, h_bos, h_mean], dim=-1)  # [B, 3H]

        # 텍스트 대표 피처(권장: mean). BOS 써도 무방.
        txt_feat = h_mean  # [B,H]

        # 비전 피처: 현재 라우트 경로에서 직접 노출되지 않는 경우가 많으므로
        # 우선 0-텐서로 안전하게 채워 차원만 맞춤. (추후 실제 vision feature 연결 가능)
        B = last_hidden.size(0)
        H_v = self.rag_router.vis_proj.in_features
        vis_feat = txt_feat.new_zeros(B, H_v)  # [B, H_v]  (fallback)

        # 엔트로피용 텍스트 로짓(선택): 마지막 토큰 vocab logits 사용
        txt_logits = getattr(router_out, 'logits', None)
        if txt_logits is not None and txt_logits.dim() == 3:
            txt_logits = txt_logits[:, -1, :]  # [B, V]
        else:
            txt_logits = None

        router_logit = self.rag_router(
            base_feat=base_feat,
            txt_feat=txt_feat,
            vis_feat=vis_feat,
            txt_logits=txt_logits,
            vis_logits=None,
            ret_stats=None,
        )
        router_prob = torch.sigmoid(router_logit)
        use_rag = (router_prob >= self.router_threshold).long()

        out = {
            "router_logit": router_logit,
            "router_prob": router_prob,
            "use_rag_pred": use_rag,
            "llm_loss": getattr(router_out, "loss", None),       # [RAG] 감독 CE
        }

        # Non-RAG 분기에서 디코딩은 선택
        return out

    def forward(self, *args, **kwargs):
        mode = kwargs.pop("mode", "loss")

        if kwargs:
            data = kwargs
        elif args:
            data = args[0]
        else:
            raise ValueError("No inputs provided to forward().")

        if mode == "loss":
            return self.compute_loss(data)
        elif mode == "predict":
            batch = data if isinstance(data, (list, tuple)) else [data]
            return [self._forward(sample, mode="predict") for sample in batch]
        elif mode == "tensor":
            sample = data[0] if isinstance(data, (list, tuple)) else data
            return self._forward(sample, mode="predict")
        else:
            raise NotImplementedError(f"mode={mode}")

    def _rag_router_loss(self, router_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        router_logit: [B], target: [B]∈{0.,1.}
        포컬 지정 시 포컬, 아니면 BCE(pos_weight).
        """
        logits = router_logit.view_as(target)
        if self.rag_focal_gamma is not None:
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p = torch.sigmoid(logits)
            pt = target * p + (1 - target) * (1 - p)
            if self.rag_focal_alpha is not None:
                a = self.rag_focal_alpha
                w = target * a + (1 - target) * (1 - a)
                return (w * (1 - pt).pow(self.rag_focal_gamma) * bce).mean()
            return ((1 - pt).pow(self.rag_focal_gamma) * bce).mean()
        if self.rag_pos_weight is not None:
            pw = torch.tensor(self.rag_pos_weight, device=target.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target)
        return nn.BCEWithLogitsLoss()(logits, target)
    
    def compute_loss(self, data):
        dev = next(self.llm.parameters()).device

        # --- 배치 정규화 ---
        if isinstance(data, dict):
            B = data["input_ids"].size(0)
            batch = []
            for i in range(B):
                sample = {}
                for k, v in data.items():
                    if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == B and k not in ("pad_token_id",):
                        sample[k] = v[i]
                    else:
                        sample[k] = v
                batch.append(sample)
        else:
            batch = data if isinstance(data, (list, tuple)) else [data]

        loss_list, llm_list, rag_list = [], [], []
        bce = nn.BCEWithLogitsLoss()

        for sample in batch:
            out = self._forward(sample, mode="loss")

            # LLM loss: [RAG] CE
            llm_loss = out.get("llm_loss", None)
            llm_loss_t = torch.zeros((), device=dev, dtype=torch.float32) if llm_loss is None else llm_loss.float()

            # Router loss (BCE)
            router_logit = out["router_logit"]    # [B]
            rag_label = sample.get("rag_label", None)
            if rag_label is not None:
                if torch.is_tensor(rag_label):
                    target = rag_label.to(router_logit.device).float().view(-1)
                else:
                    target = torch.tensor([float(rag_label)], device=router_logit.device)
                rag_loss_t = self._rag_router_loss(router_logit.view_as(target), target)
            else:
                rag_loss_t = torch.zeros((), device=dev, dtype=torch.float32)

            total_t = llm_loss_t + self.alpha * rag_loss_t

            loss_list.append(total_t)
            llm_list.append(llm_loss_t.detach())
            rag_list.append(rag_loss_t.detach())

        loss_mean = torch.stack(loss_list).mean()
        llm_mean  = torch.stack(llm_list).mean()
        rag_mean  = torch.stack(rag_list).mean()

        return {
            "loss": loss_mean,
            "llm_loss": llm_mean,
            "rag_loss": rag_mean,
        }

    @torch.no_grad()
    def predict(self, data_sample):
        out = self._forward(data_sample, mode="predict")
        return out.get("sam_pred_masks", None)

# import os
# import math

# import spacy
# import torch
# import torch.backends.cuda as sdp
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModel
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from xtuner.model.utils import guess_load_checkpoint
# from xtuner.registry import BUILDER

# IGNORE_INDEX = -100

# # SDPA → eager
# sdp.enable_flash_sdp(False)
# sdp.enable_mem_efficient_sdp(True)
# sdp.enable_math_sdp(True)


# class FrozenLlavaSAM(BaseModel):
#     def __init__(
#         self,
#         model,
#         sam,
#         tokenizer=None,
#         mask_head=None,
#         merge: str = "mean",
#         alpha: float = 0.5,              # total = llm_loss + alpha * rag_loss
#         router_threshold: float = 0.5,   # sigmoid 임계값
#         # 불균형 대응 옵션
#         rag_pos_weight: float | None = None,   # BCE pos_weight(≈ p_neg/p_pos)
#         rag_focal_gamma: float | None = None,  # 지정 시 포컬 사용(예: 2.0)
#         rag_focal_alpha: float | None = None,  # 포컬 알파(예: 0.25)
#         llm_pos_weight: float | None = None,   # LLM BCE용 pos_weight(기본은 rag_pos_weight)
#         *args,
#         **kwargs,
#     ):
#         pretrained = kwargs.pop("pretrained", None)
#         llm_lora = kwargs.pop("llm_lora", None)
#         lora_pretrained_path = kwargs.pop("lora_pretrained_path")
#         super().__init__(*args, **kwargs)

#         # ----- LLM -----
#         self.llm = BUILDER.build(model)
#         if llm_lora is not None:
#             try:
#                 self.llm = prepare_model_for_kbit_training(self.llm, use_gradient_checkpointing=True)
#             except Exception:
#                 pass
#             lora_cfg = BUILDER.build(llm_lora) if isinstance(llm_lora, dict) else llm_lora
#             if isinstance(lora_cfg, dict):
#                 lora_cfg = LoraConfig(**lora_cfg)
#             self.llm = get_peft_model(self.llm, lora_cfg)
#             if lora_pretrained_path:
#                 try:
#                     self.llm.load_adapter(lora_pretrained_path, adapter_name="default")
#                 except Exception:
#                     sd = torch.load(lora_pretrained_path, map_location="cpu")
#                     missing, unexpected = self.llm.load_state_dict(sd, strict=False)
#                     print("[LoRA] missing:", missing, "unexpected:", unexpected)

#         # ----- external modules -----
#         self.sam = BUILDER.build(sam)

#         # ----- tokenizer / special tokens -----
#         self.tokenizer = BUILDER.build(tokenizer)
#         self.rag_token = "[RAG]"
#         if self.tokenizer is not None:
#             added = self.tokenizer.add_special_tokens(
#                 {"additional_special_tokens": ["<image>", self.rag_token]}
#             )
#             if added > 0:
#                 self.llm.resize_token_embeddings(len(self.tokenizer))
#             self.llm.config.image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
#             self.rag_token_id = self.tokenizer.convert_tokens_to_ids(self.rag_token)
#             if self.llm.config.pad_token_id is None:
#                 self.llm.config.pad_token_id = self.tokenizer.pad_token_id
#             self.llm.config.ignore_index = IGNORE_INDEX
#         else:
#             self.rag_token_id = None

#         # ----- attention impl / outputs -----
#         self.llm.config.attn_implementation = "eager"
#         if hasattr(self.llm, "language_model") and hasattr(self.llm.language_model, "config"):
#             self.llm.language_model.config.attn_implementation = "eager"

#         self.llm.config.use_cache = False
#         self.llm.config.output_hidden_states = False
#         if hasattr(self.llm, "language_model"):
#             self.llm.language_model.config.output_hidden_states = True
#             self.llm.language_model.config.use_cache = False

#         print("[cfg] llm.attn_impl =", getattr(self.llm.config, "attn_implementation", None))
#         if hasattr(self.llm, "language_model") and hasattr(self.llm.language_model, "config"):
#             print("[cfg] llm.language_model.attn_impl =", getattr(self.llm.language_model.config, "attn_implementation", None))

#         # ----- router head: concat([RAG],[BOS],mean) → LN → Linear(3H→H) → Linear(H→1) -----
#         H = self.llm.config.hidden_size
#         self._router_in_dim = 3 * H
#         self.rag_router = nn.Sequential(
#             nn.LayerNorm(self._router_in_dim),
#             nn.Linear(self._router_in_dim, H, bias=True),
#             nn.Linear(H, 1, bias=True),
#         )

#         self.alpha = float(alpha)
#         self.router_threshold = float(router_threshold)

#         # 불균형 옵션
#         self.rag_pos_weight = rag_pos_weight
#         self.rag_focal_gamma = rag_focal_gamma
#         self.rag_focal_alpha = rag_focal_alpha
#         self.llm_pos_weight = llm_pos_weight if llm_pos_weight is not None else rag_pos_weight

#         # (다른 구성요소들; 유지)
#         self.patch_size = getattr(getattr(self.llm.config, "vision_config", None), "patch_size", 14)
#         self.merge = merge
#         self.text_layer_weights = nn.Parameter(torch.ones(self.llm.config.num_hidden_layers))
#         self.text_proj = nn.Linear(self.llm.config.hidden_size, self.sam.model.prompt_encoder.embed_dim)

#         self.nlp = spacy.load("en_core_web_sm")

#         if pretrained is not None:
#             _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

#     # ------------------------ utils ------------------------
#     def apply_merge(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
#         return x.float().mean(dim=dim) if self.merge == "mean" else x.float().max(dim=dim).values

#     def get_text_layer_weights(self):
#         return torch.softmax(self.text_layer_weights, dim=0)

#     def _select_rag_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
#         """hidden: [B,T,H], input_ids: [B,T] -> [RAG] 위치 히든(없으면 BOS)"""
#         if self.rag_token_id is None:
#             return hidden[:, 0, :]
#         with torch.no_grad():
#             rag_mask = (input_ids == self.rag_token_id)
#             if rag_mask.any():
#                 idx = rag_mask.float().argmax(dim=1)
#             else:
#                 idx = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
#         B = input_ids.size(0)
#         return hidden[torch.arange(B, device=hidden.device), idx, :]

#     # ---- vocab projection (핵심 수정) ----
#     def _project_to_vocab(self, hidden_last: torch.Tensor) -> torch.Tensor:
#         """
#         hidden_last: [B,T,H] → lm_head → logits [B,T,V]
#         다양한 래퍼/구현을 대비해 lm_head 탐색을 일반화.
#         """
#         # 순서대로 찾기: self.llm.lm_head → self.llm.language_model.lm_head → get_output_embeddings()
#         proj = getattr(self.llm, "lm_head", None)
#         if proj is None and hasattr(self.llm, "language_model"):
#             proj = getattr(self.llm.language_model, "lm_head", None)
#         if proj is None:
#             try:
#                 proj = self.llm.get_output_embeddings()
#             except Exception:
#                 proj = None
#         if proj is None and hasattr(self.llm, "language_model"):
#             try:
#                 proj = self.llm.language_model.get_output_embeddings()
#             except Exception:
#                 proj = None
#         if proj is None:
#             raise RuntimeError("lm_head(get_output_embeddings)가 없어 vocab logits을 만들 수 없습니다.")
#         # dtype 정렬
#         if hidden_last.dtype != proj.weight.dtype:
#             hidden_last = hidden_last.to(proj.weight.dtype)
#         return proj(hidden_last)  # [B,T,V]

#     # ------------------------ losses ------------------------
#     def _llm_loss_rag_binary(self, logits: torch.Tensor, rag_label) -> torch.Tensor:
#         """
#         logits: [B,T,V] (vocab 차원)
#         마지막 스텝의 [RAG] logit로 BCE. rag_label ∈ {0,1}.
#         """
#         if self.rag_token_id is None:
#             return torch.zeros((), device=logits.device, dtype=torch.float32)

#         last_logits = logits[:, -1, :]                  # [B,V]
#         # 안전 가드: vocab 크기 확인
#         V = last_logits.size(-1)
#         if self.rag_token_id >= V:
#             raise IndexError(f"[RAG] id {self.rag_token_id} >= vocab size {V}")
#         rag_logit = last_logits[:, self.rag_token_id]   # [B]

#         if torch.is_tensor(rag_label):
#             target = rag_label.to(logits.device).float().view(-1)
#         else:
#             target = torch.tensor([float(rag_label)], device=logits.device)

#         if self.llm_pos_weight is not None:
#             pw = torch.tensor(self.llm_pos_weight, device=logits.device)
#             crit = nn.BCEWithLogitsLoss(pos_weight=pw)
#         else:
#             crit = nn.BCEWithLogitsLoss()
#         return crit(rag_logit, target)

#     def _rag_router_loss(self, router_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         router_logit: [B], target: [B]∈{0.,1.}
#         포컬 지정 시 포컬, 아니면 BCE(pos_weight).
#         """
#         logits = router_logit.view_as(target)
#         if self.rag_focal_gamma is not None:
#             bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
#             p = torch.sigmoid(logits)
#             pt = target * p + (1 - target) * (1 - p)
#             if self.rag_focal_alpha is not None:
#                 a = self.rag_focal_alpha
#                 w = target * a + (1 - target) * (1 - a)
#                 return (w * (1 - pt).pow(self.rag_focal_gamma) * bce).mean()
#             return ((1 - pt).pow(self.rag_focal_gamma) * bce).mean()
#         if self.rag_pos_weight is not None:
#             pw = torch.tensor(self.rag_pos_weight, device=target.device)
#             return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target)
#         return nn.BCEWithLogitsLoss()(logits, target)

#     # ------------------------ forward paths ------------------------
#     def _forward(self, data_sample, mode="loss"):
#         dev = next(self.llm.parameters()).device

#         # inputs normalize
#         input_ids = data_sample["input_ids"]
#         input_ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
#         pixel_values = data_sample["pixel_values"]
#         if pixel_values.ndim == 3:
#             pixel_values = pixel_values.unsqueeze(0)
#         elif pixel_values.ndim == 5:
#             pixel_values = pixel_values.squeeze(0)
#         input_ids = input_ids.to(dev)
#         pixel_values = pixel_values.to(dev, dtype=getattr(self.llm, "dtype", torch.float16))

#         attn = data_sample.get("attention_mask", None)
#         if attn is None:
#             pad_id = getattr(self.llm.config, "pad_token_id", getattr(self, "pad_token_id", 0))
#             attn = input_ids.ne(pad_id)
#         else:
#             attn = attn.to(dev)
#             if attn.ndim == 1:
#                 attn = attn.unsqueeze(0)
#         attention_mask = attn.to(dtype=torch.bool)

#         # LLM forward (labels=None; logits는 직접 만들 수 있음)
#         with torch.cuda.amp.autocast(enabled=False):
#             out = self.llm(
#                 input_ids=input_ids,
#                 pixel_values=pixel_values,
#                 labels=None,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True,
#                 output_attentions=False,
#                 use_rag=False,
#                 return_dict=True,
#             )

#         last_hidden = out.hidden_states[-1]     # [B,T,H]
#         logits = getattr(out, "logits", None)

#         # 안전 가드: logits가 없거나, vocab 차원이 너무 작으면 lm_head로 재계산
#         need_project = (
#             (logits is None)
#             or (logits.size(-1) <= last_hidden.size(-1))  # 4096 같은 히든 크기 케이스
#             or (self.rag_token_id is not None and logits.size(-1) <= self.rag_token_id)
#         )
#         if need_project:
#             logits = self._project_to_vocab(last_hidden)  # [B,T,V]

#         # Router features: [RAG], BOS, mean
#         h_rag = self._select_rag_hidden(last_hidden, input_ids)     # [B,H]
#         h_bos = last_hidden[:, 0, :]                                 # [B,H]
#         h_txt = last_hidden.mean(dim=1)                              # [B,H]
#         router_feat = torch.cat([h_rag, h_bos, h_txt], dim=-1)      # [B,3H]

#         router_logit = self.rag_router(router_feat).squeeze(-1)     # [B]
#         router_prob = torch.sigmoid(router_logit)                   # [B]
#         use_rag = (router_prob >= self.router_threshold).long()     # [B]

#         # LLM loss: 마지막 스텝 [RAG] BCE(불균형 가중 가능)
#         llm_loss = self._llm_loss_rag_binary(logits, data_sample.get("rag_label", 0))

#         return {
#             "router_logit": router_logit,
#             "router_prob": router_prob,
#             "use_rag_pred": use_rag,
#             "llm_loss": llm_loss,
#         }

#     def forward(self, *args, **kwargs):
#         mode = kwargs.pop("mode", "loss")
#         data = kwargs if kwargs else (args[0] if args else None)
#         if data is None:
#             raise ValueError("No inputs provided to forward().")
#         if mode == "loss":
#             return self.compute_loss(data)
#         elif mode == "predict":
#             batch = data if isinstance(data, (list, tuple)) else [data]
#             return [self._forward(sample, mode="predict") for sample in batch]
#         elif mode == "tensor":
#             sample = data[0] if isinstance(data, (list, tuple)) else data
#             return self._forward(sample, mode="predict")
#         else:
#             raise NotImplementedError(f"mode={mode}")

#     def compute_loss(self, data):
#         dev = next(self.llm.parameters()).device

#         # 배치 정규화
#         if isinstance(data, dict):
#             B = data["input_ids"].size(0)
#             batch = []
#             for i in range(B):
#                 sample = {}
#                 for k, v in data.items():
#                     if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == B and k not in ("pad_token_id",):
#                         sample[k] = v[i]
#                     else:
#                         sample[k] = v
#                 batch.append(sample)
#         else:
#             batch = data if isinstance(data, (list, tuple)) else [data]

#         loss_list, llm_list, rag_list = [], [], []

#         for sample in batch:
#             out = self._forward(sample, mode="loss")

#             # LLM loss (graph 유지)
#             llm_loss_t = out["llm_loss"].float()

#             # Router loss
#             router_logit = out["router_logit"]
#             rag_label = sample.get("rag_label", None)
#             if rag_label is not None:
#                 if torch.is_tensor(rag_label):
#                     target = rag_label.to(router_logit.device).float().view(-1)
#                 else:
#                     target = torch.tensor([float(rag_label)], device=router_logit.device)
#                 rag_loss_t = self._rag_router_loss(router_logit.view_as(target), target)
#             else:
#                 rag_loss_t = torch.zeros((), device=dev, dtype=torch.float32)

#             total_t = llm_loss_t + self.alpha * rag_loss_t

#             loss_list.append(total_t)
#             llm_list.append(llm_loss_t.detach())
#             rag_list.append(rag_loss_t.detach())

#         loss_mean = torch.stack(loss_list).mean()
#         llm_mean = torch.stack(llm_list).mean()
#         rag_mean = torch.stack(rag_list).mean()

#         return {
#             "loss": loss_mean,
#             "llm_loss": llm_mean,
#             "rag_loss": rag_mean,
#         }

#     @torch.no_grad()
#     def predict(self, data_sample):
#         out = self._forward(data_sample, mode="predict")
#         return out.get("sam_pred_masks", None)
