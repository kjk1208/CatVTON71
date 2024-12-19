import os
import argparse
import gradio as gr
from datetime import datetime

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.flux.pipeline_flux_tryon import FluxTryOnPipeline
from utils import resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX Try-On Demo")
    parser.add_argument(
        "--base_model_path",
        type=str,
        # default="black-forest-labs/FLUX.1-Fill-dev",
        default="Models/FLUX.1-Fill-dev",
        help="The path to the base model to use for evaluation."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="The Path to the checkpoint of trained tryon model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Whether or not to allow TF32 on Ampere GPUs."
    )
    return parser.parse_args()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    resolution,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    # 根据分辨率设置高度和宽度
    height = resolution
    width = int(height * 0.75)
    args.width = width
    args.height = height

    # 处理图像编辑器输入
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    # 准备输出文件夹
    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    # 设置随机种子
    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    # 处理输入图像
    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    
    # 调整图像大小
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    # 处理遮罩
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # 推理
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]

    # 后处理
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 2, 2)
    save_result_image.save(result_save_path)

    # 根据显示类型返回结果
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)
        
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
        return new_result_image

def app_gradio():
    with gr.Blocks(title="CatVTON with FLUX.1-Fill-dev") as demo:
        gr.Markdown("# CatVTON with FLUX.1-Fill-dev")
        
        with gr.Row():
            with gr.Column(scale=1):
                person_image = gr.ImageEditor(
                    interactive=True, label="Person Image", type="filepath"
                )
                cloth_image = gr.Image(
                    interactive=True, label="Condition Image", type="filepath"
                )
                
                cloth_type = gr.Radio(
                    label="Try-On Cloth Type",
                    choices=["upper", "lower", "overall"],
                    value="upper",
                )

                resolution = gr.Radio(
                    label="Resolution",
                    choices=[1024, 1280],
                    value=1024,
                )

                submit = gr.Button("Submit")
                
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
                    guidance_scale = gr.Slider(
                        label="CFG Strength", minimum=0.0, maximum=50.0, step=1.0, value=30.0
                    )
                    seed = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )

            with gr.Column(scale=2):
                result_image = gr.Image(interactive=False, label="Result")

        submit.click(
            submit_function,
            [
                person_image,
                cloth_image,
                cloth_type,
                resolution,
                num_inference_steps,
                guidance_scale,
                seed,
                show_type
            ],
            result_image
        )
    
    demo.queue().launch(share=True, show_error=True)

# 解析参数
args = parse_args()

# 加载模型
repo_path = snapshot_download(repo_id=args.resume_path)
pipeline = FluxTryOnPipeline.from_pretrained(args.base_model_path)
pipeline.load_lora_weights(
    os.path.join(repo_path, "flux-lora"), 
    weight_name='pytorch_lora_weights.safetensors'
)
pipeline.to("cuda", torch.bfloat16)

# 初始化 AutoMasker
mask_processor = VaeImageProcessor(
    vae_scale_factor=8, 
    do_normalize=False, 
    do_binarize=True, 
    do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda'
)

if __name__ == "__main__":
    app_gradio()
