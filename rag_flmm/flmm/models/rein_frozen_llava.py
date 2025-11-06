# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from xtuner.registry import BUILDER
# from mmengine.model import BaseModel
# from xtuner.model.utils import guess_load_checkpoint
# from flmm.utils import compute_mask_IoU
# from ppo.ppo_trainer import RolloutBuffer, PPOTrainer

# class FrozenLlava(BaseModel):
#     def __init__(self,
#                  model,
#                  mask_head,
#                  merge='mean',
#                  loss_mask=None,
#                  loss_dice=None,
#                  pretrained=None,
#                  **kwargs):
#         super().__init__()
#         self.llava = model
#         self.llava.requires_grad_(False)
#         in_channels = (self.llava.config.text_config.num_attention_heads *
#                        self.llava.config.text_config.num_hidden_layers)

#         mask_head.in_channels = in_channels
#         self.mask_head = mask_head

#         self.patch_size = self.llava.config.vision_config.patch_size
#         self.merge = merge
#         assert merge in ['mean', 'max']

#         self.loss_mask = loss_mask
#         self.loss_dice = loss_dice

#         self.text_layer_weights = nn.Parameter(
#             torch.ones(self.llava.config.text_config.num_hidden_layers))

#         if pretrained is not None:
#             _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

#     def get_text_layer_weights(self):
#         return torch.softmax(self.text_layer_weights, dim=0)

#     def apply_merge(self, x, dim=1):
#         if self.merge == 'mean':
#             return x.mean(dim=dim)
#         elif self.merge == 'max':
#             return x.max(dim=dim).values
#         else:
#             raise NotImplementedError

#     def init_weights(self):
#         pass

#     def train(self, mode=True):
#         super().train(mode=mode)
#         self.llava.train(mode=False)
#         self.training = mode
#         return self

#     def forward(self, data, data_samples=None, mode='loss'):
#         if mode == 'loss':
#             return self.compute_loss(data)
#         elif mode == 'predict':
#             return self.predict(data)
#         elif mode == 'tensor':
#             return self._forward(data)
#         else:
#             raise NotImplementedError

#     def _compute(self, pred_masks, gt_masks):
#         mask_cnt = pred_masks.shape[0]
#         loss_dice = self.loss_dice(
#             pred_masks.view(mask_cnt, -1), gt_masks.view(mask_cnt, -1),
#             avg_factor=mask_cnt)
#         loss_mask = self.loss_mask(
#             pred_masks.view(-1),
#             gt_masks.view(-1),
#             avg_factor=pred_masks.numel())
#         accuracy = torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks),
#                             gt_masks).to(gt_masks).mean()
#         aiou = compute_mask_IoU((pred_masks.detach().sigmoid() > 0.5).to(gt_masks).view(mask_cnt, -1),
#                                 gt_masks.view(mask_cnt, -1)).mean()

#         return loss_dice, loss_mask, accuracy, aiou

# class FrozenLlavaSAM_RL(FrozenLlava):
#     def __init__(self, sam, rollout_buffer=None, *args, **kwargs):
#         pretrained = kwargs.pop('pretrained', None)
#         super().__init__(*args, **kwargs)

#         self.sam = sam
#         self.text_proj = nn.Linear(
#             self.llava.config.text_config.hidden_size,
#             self.sam.model.prompt_encoder.embed_dim
#         )

#         self.num_layers = self.llava.config.text_config.num_hidden_layers
#         self.use_attn = True

#         self.feature_dim = self.llava.config.text_config.hidden_size + 512
#         if self.use_attn:
#             self.feature_dim += self.num_layers

#         self.policy_net = nn.Sequential(
#             nn.Linear(8227, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.num_layers)
#         )
#         self.value_net = nn.Sequential(
#             nn.Linear(8227, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#         # PPO components
#         self.rollout_buffer = rollout_buffer if rollout_buffer else RolloutBuffer()
#         self.ppo_trainer = PPOTrainer(self.policy_net, self.value_net)

#         if pretrained is not None:
#             _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

#     def compute_dice_reward(self, pred_mask, gt_mask, threshold=0.85):
#         dice = self.compute_dice(pred_mask, gt_mask)
#         reward = 1.0 if dice >= threshold else -1.0
#         return reward, dice

#     def _forward(self, data_sample):
#         text_layer_weights = self.get_text_layer_weights()
#         input_ids = data_sample['input_ids'].to(self.llava.device)
#         pixel_values = data_sample['images_clip'].to(device=self.llava.device, dtype=self.llava.dtype)
#         # === 이미지 개수와 텍스트 개수 확인 후 맞춰주기 ===
#         # if input_ids.shape[0] != pixel_values.shape[0]:
#         #     assert input_ids.shape[0] % pixel_values.shape[0] == 0, \
#         #         f"Incompatible batch sizes: input_ids={input_ids.shape[0]}, pixel_values={pixel_values.shape[0]}"
#         #     repeat_factor = input_ids.shape[0] // pixel_values.shape[0]
#         #     pixel_values = pixel_values.repeat_interleave(repeat_factor, dim=0)

#         labels = data_sample['labels'][None].to(self.llava.device)
#         attention_mask = torch.ones_like(input_ids)

#         with torch.no_grad():
#             outputs = self.llava(
#                 input_ids=input_ids,
#                 pixel_values=pixel_values,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True,
#                 output_attentions=True
#             )

#         attentions = outputs.attentions  # list of [1, num_heads, seq, seq]
#         hidden_states = outputs.hidden_states[-self.llava.config.text_config.num_hidden_layers:]
#         labels = outputs.labels[0]

#         hidden_states = torch.stack([hs[0] for hs in hidden_states])  # [num_layers, seq, dim]
#         hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # [seq, dim]

#         # === padded size 가져오기 ===
#         padded_h, padded_w = data_sample['resize'][0], data_sample['resize'][1]
#         llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size

#         # === attention layer를 평균해서 spatial map 만들기 ===
#         attention_maps = []
#         for layer_attn in attentions:
#             layer_attn = layer_attn[0].mean(dim=0)  # [seq, seq]
#             image_token_attn = layer_attn[:, -llava_h * llava_w:]  # image tokens가 뒤에 있다고 가정
#             spatial_attn = image_token_attn.mean(dim=0).view(llava_h, llava_w)
#             attention_maps.append(spatial_attn)

#         attention_maps = torch.stack(attention_maps).to(self.mask_head.dtype)  # [L, H, W]

#         # === state vector 만들기 ===
#         image_feat = data_sample['image'].to(self.llava.device).mean(dim=(1, 2))  # [C]
#         attn_summary = attention_maps.mean(dim=(1, 2))  # [L]
#         text_output_embed = outputs.hidden_states[-1][0].mean(dim=0)  # [D]

#         state_vec = torch.cat([image_feat, attn_summary, text_output_embed], dim=0)

#         # === PPO policy로 attention layer 선택 ===
#         action_logits = self.policy_net(state_vec)
#         action_dist = Categorical(logits=action_logits)
#         selected_action = action_dist.sample()

#         selected_attn = attention_maps[selected_action].unsqueeze(0)  # [1, H, W]
#         pred_mask = self.mask_head(selected_attn.unsqueeze(0))[:, 0]  # [1, H, W]

#         # === mask crop 생략 === (meta_data 없음 → 그대로 사용)
#         sam_out = self.sam(data_sample['image'], pred_mask, text_output_embed)
#         sam_pred_mask = sam_out['masks'][0]  # [1, H, W]

#         # === reward 계산 ===
#         gt_mask = data_sample['masks'].to(sam_pred_mask.device)  # [H, W]
#         reward, _ = self.compute_dice_reward(sam_pred_mask, gt_mask)

#         return {
#             "pred_masks": pred_mask,
#             "sam_pred_masks": sam_pred_mask,
#             "labels": labels,
#             "actions": selected_action,
#             "log_probs": action_dist.log_prob(selected_action),
#             "state_vecs": state_vec,
#             "rewards": reward
#         }

#     @torch.no_grad()
#     def predict(self, data_sample):
#         return self._forward(data_sample)['sam_pred_masks']

#     def compute_loss(self, data):
#         mask_cnts = 0

#         loss_dice = 0
#         loss_mask = 0
#         accuracy = 0
#         aiou = 0

#         sam_loss_dice = 0
#         sam_loss_mask = 0
#         sam_accuracy = 0
#         sam_aiou = 0

#         for data_sample in data:
#             forward_output = self._forward(data_sample)
#             pred_masks, sam_pred_masks = forward_output['pred_masks'], forward_output['sam_pred_masks']
#             masks = data_sample['masks'].to(self.llava.device)
#             gt_masks = F.interpolate(masks[None].float(),
#                                      size=pred_masks.shape[-2:])[0].to(pred_masks)
#             sam_gt_masks = F.interpolate(masks[None].float(),
#                                          size=sam_pred_masks.shape[-2:])[0].to(sam_pred_masks)

#             mask_cnt = pred_masks.shape[0]
#             assert pred_masks.shape == gt_masks.shape
#             mask_cnts += mask_cnt

#             loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(pred_masks, gt_masks)
#             loss_dice += loss_dice_ * mask_cnt
#             loss_mask += loss_mask_ * mask_cnt
#             accuracy += accuracy_ * mask_cnt
#             aiou += aiou_ * mask_cnt

#             sam_loss_dice_, sam_loss_mask_, sam_accuracy_, sam_aiou_ = self._compute(sam_pred_masks, sam_gt_masks)
#             sam_loss_dice += sam_loss_dice_ * mask_cnt
#             sam_loss_mask += sam_loss_mask_ * mask_cnt
#             sam_accuracy += sam_accuracy_ * mask_cnt
#             sam_aiou += sam_aiou_ * mask_cnt

#         assert mask_cnts > 0

#         loss_dict = {'loss_mask': loss_mask / mask_cnts,
#                      'loss_dice': loss_dice / mask_cnts,
#                      'accuracy': accuracy / mask_cnts,
#                      'aiou': aiou / mask_cnts,
#                      'sam_loss_mask': sam_loss_mask / mask_cnts,
#                      'sam_loss_dice': sam_loss_dice / mask_cnts,
#                      'sam_accuracy': sam_accuracy / mask_cnts,
#                      'sam_aiou': sam_aiou / mask_cnts,
#                      }

#         return loss_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from ppo.ppo_trainer import PPOTrainer, RolloutBuffer
from torch.distributions import Categorical
from transformers.modeling_outputs import CausalLMOutputWithPast
from xtuner.model.utils import guess_load_checkpoint

from flmm.models.llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaMetaModel,
)
from flmm.models.mask_head.mask_refiner import SAMWrapper
from flmm.utils import compute_mask_IoU


class FrozenLlava(BaseModel):
    def __init__(
        self,
        model: LlavaLlamaForCausalLM,
        mask_head,
        merge="mean",
        loss_mask=None,
        loss_dice=None,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()
        self.llava: LlavaLlamaForCausalLM = model
        self.llava.requires_grad_(False)

        self.sam = SAMWrapper(
            use_text=True,
            use_mask=True,
            multimask_output=False,
            model_name="vit_l",
            checkpoint="/home/hrkim/f-lmm/checkpoints/sam_vit_l_0b3195.pth",
        )

        text_config = self.llava.config  # 전체 모델 config
        vision_config = (
            self.llava.model.get_vision_tower().config
        )  # vision tower의 config

        self.patch_size = vision_config.patch_size
        self.num_layers = text_config.num_hidden_layers

        self.text_proj = nn.Linear(
            text_config.hidden_size, self.sam.model.prompt_encoder.embed_dim
        )

        in_channels = text_config.num_attention_heads * text_config.num_hidden_layers
        mask_head.in_channels = in_channels
        self.mask_head = mask_head

        self.patch_size = vision_config.patch_size
        self.merge = merge
        assert merge in ["mean", "max"]

        self.loss_mask = loss_mask
        self.loss_dice = loss_dice

        self.text_layer_weights = nn.Parameter(
            torch.ones(text_config.num_hidden_layers)
        )

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def apply_merge(self, x, dim=1):
        if self.merge == "mean":
            return x.mean(dim=dim)
        elif self.merge == "max":
            return x.max(dim=dim).values
        else:
            raise NotImplementedError

    def init_weights(self):
        pass

    def train(self, mode=True):
        super().train(mode=mode)
        self.llava.train(mode=False)
        self.training = mode
        return self

    def forward(self, data, data_samples=None, mode="loss"):
        if mode == "loss":
            return self.compute_loss(data)
        elif mode == "predict":
            return self.predict(data)
        elif mode == "tensor":
            return self._forward(data)
        else:
            raise NotImplementedError

    def _compute(self, pred_masks, gt_masks):
        mask_cnt = pred_masks.shape[0]
        loss_dice = self.loss_dice(
            pred_masks.view(mask_cnt, -1),
            gt_masks.view(mask_cnt, -1),
            avg_factor=mask_cnt,
        )
        loss_mask = self.loss_mask(
            pred_masks.view(-1), gt_masks.view(-1), avg_factor=pred_masks.numel()
        )
        accuracy = (
            torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks), gt_masks)
            .float()
            .mean()
        )
        aiou = compute_mask_IoU(
            (pred_masks.detach().sigmoid() > 0.5).to(gt_masks).view(mask_cnt, -1),
            gt_masks.view(mask_cnt, -1),
        ).mean()
        return loss_dice, loss_mask, accuracy, aiou


class FrozenLlavaSAM_RL(FrozenLlava):
    def __init__(self, rollout_buffer=None, *args, **kwargs):
        pretrained = kwargs.pop("pretrained", None)
        super().__init__(*args, **kwargs)

        text_config = self.llava.config

        self.text_proj = nn.Linear(
            text_config.hidden_size, self.sam.model.prompt_encoder.embed_dim
        )

        self.num_layers = text_config.num_hidden_layers
        self.use_attn = True

        self.feature_dim = text_config.hidden_size + 512
        if self.use_attn:
            self.feature_dim += self.num_layers

        self.policy_net = nn.Sequential(
            nn.Linear(8227, 128), nn.ReLU(), nn.Linear(128, self.num_layers)
        )
        self.value_net = nn.Sequential(
            nn.Linear(8227, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.rollout_buffer = rollout_buffer if rollout_buffer else RolloutBuffer()
        self.ppo_trainer = PPOTrainer(self.policy_net, self.value_net)

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)

    def compute_dice_reward(self, pred_mask, gt_mask, threshold=0.85):
        dice = self.compute_dice(pred_mask, gt_mask)
        reward = 1.0 if dice >= threshold else -1.0
        return reward, dice

    def _forward(self, data_sample):
        text_config = self.llava.model.language_model.config
        text_layer_weights = self.get_text_layer_weights()

        input_ids = data_sample["input_ids"].to(self.llava.device)
        pixel_values = data_sample["images_clip"].to(
            device=self.llava.device, dtype=self.llava.dtype
        )
        labels = data_sample["labels"][None].to(self.llava.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.llava(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        attentions = outputs.attentions
        hidden_states = outputs.hidden_states[-text_config.num_hidden_layers :]
        labels = outputs.labels[0]

        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # [L, S, D]
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(
            0
        )  # [S, D]

        padded_h, padded_w = data_sample["resize"][0], data_sample["resize"][1]
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size

        attention_maps = []
        for layer_attn in attentions:
            layer_attn = layer_attn[0].mean(dim=0)  # [S, S]
            image_token_attn = layer_attn[:, -llava_h * llava_w :]
            spatial_attn = image_token_attn.mean(dim=0).view(llava_h, llava_w)
            attention_maps.append(spatial_attn)

        attention_maps = torch.stack(attention_maps).to(
            self.mask_head.dtype
        )  # [L, H, W]

        image_feat = data_sample["image"].to(self.llava.device).mean(dim=(1, 2))  # [C]
        attn_summary = attention_maps.mean(dim=(1, 2))  # [L]
        text_output_embed = outputs.hidden_states[-1][0].mean(dim=0)  # [D]

        state_vec = torch.cat([image_feat, attn_summary, text_output_embed], dim=0)

        action_logits = self.policy_net(state_vec)
        action_dist = Categorical(logits=action_logits)
        selected_action = action_dist.sample()

        selected_attn = attention_maps[selected_action].unsqueeze(0)  # [1, H, W]
        pred_mask = self.mask_head(selected_attn.unsqueeze(0))[:, 0]  # [1, H, W]

        sam_out = self.sam(data_sample["image"], pred_mask, text_output_embed)
        sam_pred_mask = sam_out["masks"][0]

        gt_mask = data_sample["masks"].to(sam_pred_mask.device)
        reward, _ = self.compute_dice_reward(sam_pred_mask, gt_mask)

        return {
            "pred_masks": pred_mask,
            "sam_pred_masks": sam_pred_mask,
            "labels": labels,
            "actions": selected_action,
            "log_probs": action_dist.log_prob(selected_action),
            "state_vecs": state_vec,
            "rewards": reward,
        }

    @torch.no_grad()
    def predict(self, data_sample):
        return self._forward(data_sample)["sam_pred_masks"]

    def compute_loss(self, data):
        mask_cnts = 0

        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0

        sam_loss_dice = 0
        sam_loss_mask = 0
        sam_accuracy = 0
        sam_aiou = 0

        for data_sample in data:
            forward_output = self._forward(data_sample)
            pred_masks, sam_pred_masks = (
                forward_output["pred_masks"],
                forward_output["sam_pred_masks"],
            )
            masks = data_sample["masks"].to(self.llava.device)
            gt_masks = F.interpolate(masks[None].float(), size=pred_masks.shape[-2:])[
                0
            ].to(pred_masks)
            sam_gt_masks = F.interpolate(
                masks[None].float(), size=sam_pred_masks.shape[-2:]
            )[0].to(sam_pred_masks)

            mask_cnt = pred_masks.shape[0]
            assert pred_masks.shape == gt_masks.shape
            mask_cnts += mask_cnt

            loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(
                pred_masks, gt_masks
            )
            loss_dice += loss_dice_ * mask_cnt
            loss_mask += loss_mask_ * mask_cnt
            accuracy += accuracy_ * mask_cnt
            aiou += aiou_ * mask_cnt

            sam_loss_dice_, sam_loss_mask_, sam_accuracy_, sam_aiou_ = self._compute(
                sam_pred_masks, sam_gt_masks
            )
            sam_loss_dice += sam_loss_dice_ * mask_cnt
            sam_loss_mask += sam_loss_mask_ * mask_cnt
            sam_accuracy += sam_accuracy_ * mask_cnt
            sam_aiou += sam_aiou_ * mask_cnt

        assert mask_cnts > 0

        return {
            "loss_mask": loss_mask / mask_cnts,
            "loss_dice": loss_dice / mask_cnts,
            "accuracy": accuracy / mask_cnts,
            "aiou": aiou / mask_cnts,
            "sam_loss_mask": sam_loss_mask / mask_cnts,
            "sam_loss_dice": sam_loss_dice / mask_cnts,
            "sam_accuracy": sam_accuracy / mask_cnts,
            "sam_aiou": sam_aiou / mask_cnts,
        }
