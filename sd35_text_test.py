#!/usr/bin/env python3

# fixed_sd35_txt2img.py
# SD 3.5 Large 모델의 품질 개선을 위한 수정 버전

import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as FM

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def ensure_resolution_compatibility(height: int, width: int) -> Tuple[int, int]:
    """SD 3.5에 최적화된 해상도 조정 (64의 배수)"""
    def snap_to_64(v: int) -> int:
        return max(64, ((v + 31) // 64) * 64)
    
    return snap_to_64(height), snap_to_64(width)

def get_optimal_scheduler_config():
    """SD 3.5에 최적화된 스케줄러 설정"""
    return {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "prediction_type": "epsilon",
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096
    }

@torch.no_grad()
def txt2img_sd35_optimized(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    use_cpu_offload: bool = False,
) -> torch.Tensor:
    """SD 3.5에 최적화된 텍스트-투-이미지 생성"""
    
    # 1. 해상도 최적화 (64의 배수)
    height, width = ensure_resolution_compatibility(height, width)
    
    # 2. 스케줄러 최적화
    scheduler_config = get_optimal_scheduler_config()
    pipe.scheduler = FM.from_config(scheduler_config)
    
    # CPU 오프로드로 VRAM 절약
    if use_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    
    # 3. 시드 고정
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # 4. 최적화된 파라미터로 생성
    with torch.inference_mode():
        # negative_prompt가 빈 문자열이면 None으로 처리
        neg_prompt = negative_prompt if negative_prompt.strip() else None
        
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed) if seed is not None else None,
            output_type="pt",  # PyTorch tensor로 출력
            return_dict=False
        )
        
        # result는 튜플이므로 이미지만 추출
        if isinstance(result, tuple):
            image_tensor = result[0]
        else:
            image_tensor = result.images
        
        return image_tensor.clamp(0, 1)

@torch.no_grad()
def img2img_refine_sd35(
    pipe: StableDiffusion3Pipeline,
    image: torch.Tensor,
    prompt: str,
    strength: float = 0.3,
    steps: int = 20,
    guidance_scale: float = 5.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """SD 3.5 최적화된 이미지 리파인"""
    
    # PIL 이미지로 변환 (diffusers img2img는 PIL을 기대)
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # [0,1] -> [0,255] -> PIL
    image_pil = Image.fromarray(
        (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=image_pil,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            return_dict=False
        )
        
        if isinstance(result, tuple):
            refined = result[0]
        else:
            refined = result.images
            
        return refined.clamp(0, 1)

def main():
    parser = argparse.ArgumentParser(description="Fixed SD 3.5 Large Text-to-Image Generator")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--prompt_a", type=str, required=True)
    parser.add_argument("--prompt_b", type=str, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)  # SD 3.5 최적값
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--negative_prompt", type=str, 
                       default="worst quality, low quality, blurry, lowres, artifacts, watermark, text")
    
    # 리파인 옵션
    parser.add_argument("--enable_refine", action="store_true")
    parser.add_argument("--refine_steps", type=int, default=15)
    parser.add_argument("--refine_strength", type=float, default=0.25)
    parser.add_argument("--refine_guidance", type=float, default=5.0)
    
    # 최적화 옵션
    parser.add_argument("--cpu_offload", action="store_true", 
                       help="Enable CPU offload to save VRAM")
    parser.add_argument("--use_float32", action="store_true",
                       help="Use float32 instead of bfloat16/float16")
    
    parser.add_argument("--output_dir", type=str, default="./sd35_output")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 디바이스 및 데이터 타입 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_float32:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # 모델 로드
    print("Loading SD 3.5 Large model...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if not args.use_float32 else None
    )
    
    if not args.cpu_offload:
        pipe = pipe.to(device)
    
    # 메모리 최적화
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"XFormers not available: {e}")
    
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_tiling()
    
    # A 이미지 생성
    print(f"Generating image A with prompt: {args.prompt_a}")
    img_a = txt2img_sd35_optimized(
        pipe=pipe,
        prompt=args.prompt_a,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        device=device,
        dtype=dtype,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        use_cpu_offload=args.cpu_offload,
    )
    
    # B 이미지 생성
    print(f"Generating image B with prompt: {args.prompt_b}")
    img_b = txt2img_sd35_optimized(
        pipe=pipe,
        prompt=args.prompt_b,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        device=device,
        dtype=dtype,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        use_cpu_offload=args.cpu_offload,
    )
    
    # 베이스 이미지 저장
    save_image(img_a, os.path.join(args.output_dir, "img_a_base.png"))
    save_image(img_b, os.path.join(args.output_dir, "img_b_base.png"))
    
    # 선택적 리파인
    if args.enable_refine:
        print("Refining images...")
        img_a = img2img_refine_sd35(
            pipe=pipe,
            image=img_a,
            prompt=args.prompt_a,
            strength=args.refine_strength,
            steps=args.refine_steps,
            guidance_scale=args.refine_guidance,
            device=device,
            dtype=dtype,
        )
        
        img_b = img2img_refine_sd35(
            pipe=pipe,
            image=img_b,
            prompt=args.prompt_b,
            strength=args.refine_strength,
            steps=args.refine_steps,
            guidance_scale=args.refine_guidance,
            device=device,
            dtype=dtype,
        )
    
    # 최종 이미지 저장
    save_image(img_a, os.path.join(args.output_dir, "img_a.png"))
    save_image(img_b, os.path.join(args.output_dir, "img_b.png"))
    
    # 차이 이미지 계산 및 저장
    diff = torch.abs(img_a - img_b)
    save_image(diff, os.path.join(args.output_dir, "diff_abs.png"))
    
    # 메트릭 계산
    l1_diff = float(diff.mean())
    mse_diff = float(((img_a - img_b) ** 2).mean())
    
    # 로그 저장
    log_path = os.path.join(args.output_dir, "generation_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"SD 3.5 Large Generation Results\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Prompt A: {args.prompt_a}\n")
        f.write(f"Prompt B: {args.prompt_b}\n")
        f.write(f"Resolution: {args.height}x{args.width}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Guidance Scale: {args.guidance_scale}\n")
        f.write(f"Negative Prompt: {args.negative_prompt}\n")
        f.write(f"Refine Enabled: {args.enable_refine}\n")
        if args.enable_refine:
            f.write(f"Refine Steps: {args.refine_steps}\n")
            f.write(f"Refine Strength: {args.refine_strength}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"L1 Difference: {l1_diff:.6f}\n")
        f.write(f"MSE Difference: {mse_diff:.6f}\n")
    
    print(f"Generation complete!")
    print(f"Images saved to: {args.output_dir}")
    print(f"L1 Difference: {l1_diff:.6f}")
    print(f"MSE Difference: {mse_diff:.6f}")

if __name__ == "__main__":
    main()