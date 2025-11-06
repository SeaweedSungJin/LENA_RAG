import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN

from scripts.demo.utils import colors


def build_data_sample(
    image_path, prompt, tokenizer_model_name, image_processor_config, prompt_template
):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    size = (
        image_processor_config.get("size")
        or image_processor_config.get("image_size")
        or 384
    )

    mean = image_processor_config.get("mean", [0.5, 0.5, 0.5])
    std = image_processor_config.get("std", [0.5, 0.5, 0.5])

    image_processor = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    image_tensor = image_processor(image)
    pixel_values = image_tensor.unsqueeze(0)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    if isinstance(prompt_template, str):
        formatted_prompt = (
            prompt_template.replace("<image>", DEFAULT_IMAGE_TOKEN) + prompt
        )
    else:
        formatted_prompt = DEFAULT_IMAGE_TOKEN + prompt

    tokenized = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = tokenized.input_ids[0]
    mask_ids = torch.tensor([0])

    data_sample = {
        "input_ids": input_ids,
        "pixel_values": pixel_values.squeeze(0),
        "image": image,
        "mask_ids": mask_ids,
        "masks": [0],
        "meta_data": {
            "padded_shape": {"height": orig_h, "width": orig_w},
            "padding": {"before_height": 0, "before_width": 0},
            "image_shape": {"height": orig_h, "width": orig_w},
        },
    }
    return data_sample


def save_output(image, pred_masks, sam_masks, mask_attn, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    image_np = np.array(image).astype(np.float32)
    conv_image = image_np.copy()
    sam_image = image_np.copy()
    attn_image = image_np.copy()

    for color_id, (cnn_mask, sam_mask, attn_mask) in enumerate(
        zip(pred_masks, sam_masks, mask_attn)
    ):
        cnn_mask = cnn_mask > 0
        sam_mask = sam_mask > 0
        attn_mask = attn_mask > 0

        conv_image[cnn_mask] = (
            conv_image[cnn_mask] * 0.2
            + np.array(colors[color_id % len(colors)]).reshape((1, 1, 3)) * 0.8
        )
        sam_image[sam_mask] = (
            sam_image[sam_mask] * 0.2
            + np.array(colors[color_id % len(colors)]).reshape((1, 1, 3)) * 0.8
        )
        attn_image[attn_mask] = (
            attn_image[attn_mask] * 0.2
            + np.array(colors[color_id % len(colors)]).reshape((1, 1, 3)) * 0.8
        )

    all_in_one = np.concatenate([image_np, attn_image, conv_image, sam_image], axis=1)

    Image.fromarray(conv_image.astype(np.uint8)).save(
        os.path.join(save_dir, "conv.png")
    )
    Image.fromarray(sam_image.astype(np.uint8)).save(os.path.join(save_dir, "sam.png"))
    Image.fromarray(attn_image.astype(np.uint8)).save(
        os.path.join(save_dir, "attn.png")
    )
    Image.fromarray(all_in_one.astype(np.uint8)).save(os.path.join(save_dir, "all.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/hrkim/f-lmm/configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/hrkim/f-lmm/checkpoints/flmm_checkpoints/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.pth",
        type=str,
    )
    parser.add_argument(
        "--image",
        default="/home/hrkim/dataset/refer_seg/images/mscoco/images/train2017/000000325647.jpg",
    )
    parser.add_argument(
        "--prompt", default="What objects in this room are used to light the room?"
    )
    parser.add_argument("--save_dir", default="output")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)

    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda().eval()

    tokenizer_model_name = cfg.tokenizer.get(
        "pretrained_model_name_or_path", "bert-base-uncased"
    )
    image_processor_config = cfg.image_processor
    prompt_template = cfg.get("prompt_template", DEFAULT_IMAGE_TOKEN + " ")

    data_sample = build_data_sample(
        args.image,
        args.prompt,
        tokenizer_model_name,
        image_processor_config,
        prompt_template,
    )

    for key in ["pixel_values", "input_ids", "mask_ids"]:
        data_sample[key] = data_sample[key].cuda()

    with torch.no_grad():
        output = model._forward(data_sample)
        if "text_output" in output:
            print("\n[Model Text Output]\n", output["text_output"])

    save_output(
        data_sample["image"],
        output["pred_masks"].cpu().numpy(),
        output["sam_pred_masks"].cpu().numpy(),
        output["mask_attentions"].cpu().numpy(),
        args.save_dir,
    )

    print("\n처리가 완료되었습니다. 결과는", args.save_dir, "에 저장되었습니다.")
