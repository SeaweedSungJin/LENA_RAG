from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import ModelOutput

# from transformers.models.llava.modeling_llava import (_CONFIG_FOR_DOC,
#                                                       LLAVA_START_DOCSTRING, LLAVA_INPUTS_DOCSTRING,
#                                                       LlavaForConditionalGeneration)
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    image_to_overwrite: Optional[Tuple[torch.BoolTensor]] = None
    mask_ids: Optional[Tuple[torch.LongTensor]] = None
    labels: Optional[Tuple[torch.LongTensor]] = None


# @add_start_docstrings(
#     """The LLAVA model which consists of a vision backbone and a language model.""",
#     LLAVA_START_DOCSTRING,
# )
class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def _merge_input_ids_with_image_features(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels=None,
        mask_ids=None,
    ):
        """
        단일 이미지 전용: image_token 1개를 image_features로 치환
        Args:
            image_features: [B, num_patches, D]
            inputs_embeds: [B, L, D]
            input_ids: [B, L]
            attention_mask: [B, L]
            labels: [B, L] or None
            mask_ids: [B, L] or None
        """
        B, L, D = inputs_embeds.shape
        _, num_patches, _ = image_features.shape
        if mask_ids is not None and mask_ids.ndim == 1:
            mask_ids = mask_ids.unsqueeze(0).expand(input_ids.size(0), -1)

        special_image_token_mask = input_ids == self.config.image_token_index
        image_positions = [
            mask.nonzero(as_tuple=True)[0][0].item()
            for mask in special_image_token_mask
        ]

        new_L = L + (num_patches - 1)
        device = inputs_embeds.device

        # final_embeddings = torch.zeros(B, new_L, D, device=device)
        # final_attention_mask = torch.zeros(B, new_L, device=device)
        # final_labels = torch.full((B, new_L), self.config.ignore_index, device=device) if labels is not None else None
        # final_mask_ids = torch.full((B, new_L), -1, device=device) if mask_ids is not None else None
        emb_dtype = inputs_embeds.dtype
        attn_dtype = attention_mask.dtype
        lbl_dtype = labels.dtype if labels is not None else torch.long
        mid_dtype = mask_ids.dtype if mask_ids is not None else torch.long

        final_embeddings = torch.zeros(B, new_L, D, device=device, dtype=emb_dtype)
        final_attention_mask = torch.zeros(B, new_L, device=device, dtype=attn_dtype)
        final_labels = (
            torch.full(
                (B, new_L), self.config.ignore_index, device=device, dtype=lbl_dtype
            )
            if labels is not None
            else None
        )
        final_mask_ids = (
            torch.full((B, new_L), -1, device=device, dtype=mid_dtype)
            if mask_ids is not None
            else None
        )

        for b in range(B):
            img_pos = image_positions[b]
            # 앞부분 텍스트
            final_embeddings[b, :img_pos] = inputs_embeds[b, :img_pos]
            final_attention_mask[b, :img_pos] = attention_mask[b, :img_pos]
            if labels is not None:
                final_labels[b, :img_pos] = labels[b, :img_pos]
            if mask_ids is not None:
                final_mask_ids[b, :img_pos] = mask_ids[b, :img_pos]

            # 이미지 패치 삽입
            final_embeddings[b, img_pos : img_pos + num_patches] = image_features[b]
            final_attention_mask[b, img_pos : img_pos + num_patches] = 1

            # 뒷부분 텍스트
            tail_len = L - (img_pos + 1)
            if tail_len > 0:
                final_embeddings[
                    b, img_pos + num_patches : img_pos + num_patches + tail_len
                ] = inputs_embeds[b, img_pos + 1 :]
                final_attention_mask[
                    b, img_pos + num_patches : img_pos + num_patches + tail_len
                ] = attention_mask[b, img_pos + 1 :]
                if labels is not None:
                    final_labels[
                        b, img_pos + num_patches : img_pos + num_patches + tail_len
                    ] = labels[b, img_pos + 1 :]
                if mask_ids is not None:
                    final_mask_ids[
                        b, img_pos + num_patches : img_pos + num_patches + tail_len
                    ] = mask_ids[b, img_pos + 1 :]

        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill(
            final_attention_mask == 0, 1
        )

        return (
            final_embeddings,
            final_attention_mask,
            final_labels,
            position_ids,
            final_mask_ids,
        )

    # @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask_ids: Optional[torch.LongTensor] = None,
        image_to_overwrite: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "\nUSER: What's the content of the image?\nASSISTANT: The image features a stop sign on a street corner"
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(
                    pixel_values, output_hidden_states=True
                )
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[
                    vision_feature_layer
                ]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
                inputs_embeds, attention_mask, labels, position_ids, mask_ids = (
                    self._merge_input_ids_with_image_features(
                        image_features,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        labels,
                        mask_ids=mask_ids,
                    )
                )
                if labels is None:
                    labels = torch.full_like(
                        attention_mask, self.config.ignore_index
                    ).to(torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif (
                past_key_values is not None
                and pixel_values is not None
                and input_ids.shape[1] == 1
            ):
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        with torch.cuda.amp.autocast(enabled=False):
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        logits = outputs[0]

        loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     if attention_mask is not None:
        #         shift_attention_mask = attention_mask[..., 1:]
        #         shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
        #         shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        #     else:
        #         shift_logits = logits[..., :-1, :].contiguous()
        #         shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(
        #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        #     )

        assert return_dict, "Use dict in our implementation"

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_to_overwrite=image_to_overwrite,
            mask_ids=mask_ids,
            labels=labels,
        )
