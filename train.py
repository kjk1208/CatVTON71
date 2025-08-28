# ~/catvton/train.py
import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import itertools

# ------------------------------------------------------------
# Diffusers (SD3.5 / SD3)
# ------------------------------------------------------------
try:
    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
except Exception as e:
    raise ImportError(
        "This training script requires diffusers with SD3 support. "
        "Install/upg: pip install -U 'diffusers>=0.34.0' 'transformers>=4.41.0' 'safetensors>=0.4.3'"
    ) from e

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
except Exception as e:
    raise ImportError(
        "FlowMatchEulerDiscreteScheduler not found. Please update diffusers (>=0.29, preferably >=0.34)."
    ) from e

# HuggingFace
from huggingface_hub import HfApi, snapshot_download

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def from_uint8(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    img = img.convert("RGB").resize(size_hw[::-1], Image.BICUBIC)
    x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                          .view(img.size[1], img.size[0], 3)
                          .numpy()).astype("float32"))
    x = x.permute(2, 0, 1) / 255.0
    x = x * 2.0 - 1.0
    return x


def load_mask(p: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    m = Image.open(p).convert("L").resize(size_hw[::-1], Image.NEAREST)
    t = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(m.tobytes()))
                          .view(m.size[1], m.size[0])
                          .numpy()).astype("float32")) / 255.0
    t = (t > 0.5).float().unsqueeze(0)
    return t


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class PairListDataset(Dataset):
    def __init__(self, list_file: str, size_hw: Tuple[int, int], mask_based: bool = True):
        super().__init__()
        self.items = []
        if list_file.endswith(".jsonl"):
            with open(list_file, "r") as f:
                for line in f:
                    self.items.append(json.loads(line))
        elif list_file.endswith(".json"):
            self.items = json.load(open(list_file))
        else:
            with open(list_file, "r") as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) == 2:
                        self.items.append({"person": parts[0], "garment": parts[1]})
                    elif len(parts) >= 3:
                        self.items.append({"person": parts[0], "garment": parts[1], "mask": parts[2]})

        self.H, self.W = size_hw
        self.mask_based = mask_based

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        person = Image.open(meta["person"])
        garment = Image.open(meta["garment"])

        x_p = from_uint8(person, (self.H, self.W))
        x_g = from_uint8(garment, (self.H, self.W))

        if self.mask_based:
            assert "mask" in meta, "mask_based=True requires mask path per line"
            m = load_mask(meta["mask"], (self.H, self.W))
            x_p_in = x_p * m
        else:
            m = torch.zeros(1, self.H, self.W, dtype=x_p.dtype)
            x_p_in = x_p

        x_concat_in = torch.cat([x_p_in, x_g], dim=2)    # [3,H,2W]
        x_concat_gt = torch.cat([x_p, x_g], dim=2)       # [3,H,2W]
        m_concat = torch.cat([m, torch.zeros_like(m)], dim=2)  # [1,H,2W]

        return {"x_concat_in": x_concat_in, "x_concat_gt": x_concat_gt, "m_concat": m_concat}


# ------------------------------------------------------------
# SD3.5 components + helpers
# ------------------------------------------------------------

def load_sd3_components(model_id: str, device, dtype, token: str = None, revision: str = None):
    """
    SD3/SD3.5를 허브에서 받아올 때 간혹 diffusers의 from_pretrained(token=...)
    경로에서 403이 나는 환경이 있어, 허브의 snapshot_download로 먼저 로컬 캐시에
    내려받고(토큰 확실히 적용), 그 로컬 디렉토리에서 파이프라인을 로드한다.
    """
    # 1) 권한/토큰 디버그(선택)
    print(f"[HF] token prefix: {(token or '')[:8]}")
    try:
        who = HfApi(token=token).whoami()
        print(f"[HF] whoami: {who.get('name') or who.get('email')}")
    except Exception as e:
        print("[HF] whoami failed:", e)

    # 2) 먼저 파일을 로컬로 다운로드 (토큰 강제 적용)
    #    allow_patterns으로 최소 파일만 받아도 되지만,
    #    SD3.5는 추가 가중치가 필요할 수 있어 기본값(None)로 둔다.
    local_dir = snapshot_download(
        repo_id=model_id,
        token=token,
        revision=revision,
        resume_download=True,
        local_files_only=False,   # 실제 다운로드
    )
    print(f"[HF] snapshot at: {local_dir}")

    # 3) 로컬 디렉토리에서 파이프라인 로드(여기서는 토큰 불필요)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        local_files_only=True,    # 반드시 로컬에서만 읽기
        use_safetensors=True,
    ).to(device)

    vae = pipe.vae
    transformer: SD3Transformer2DModel = pipe.transformer
    encode_prompt = pipe.encode_prompt
    sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return vae, transformer, encode_prompt, sched


@torch.no_grad()
def to_latent_sd3(vae, x_bchw: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(x_bchw).latent_dist
    latents = posterior.sample()
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    latents = (latents - sh) * sf
    return latents


class ConcatProjector(nn.Module):
    def __init__(self, in_ch=33, out_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


def freeze_all_but_attn_sd3(transformer: SD3Transformer2DModel):
    for p in transformer.parameters():
        p.requires_grad = False
    train_count = 0
    for name, module in transformer.named_modules():
        if ".attn" in name:
            for p in module.parameters():
                p.requires_grad = True
            train_count += sum(p.numel() for p in module.parameters())
    return train_count


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

@dataclass
class TrainConfig:
    list_file: str
    output_dir: str = "ckpts/catvton_sd35"
    sd3_model: str = "stabilityai/stable-diffusion-3.5-large"
    size_h: int = 512
    size_w: int = 384
    mask_based: bool = True
    lr: float = 1e-5
    batch_size: int = 4
    grad_accum: int = 1
    max_steps: int = 16000
    save_every: int = 2000
    log_every: int = 50
    cond_dropout_p: float = 0.1
    mixed_precision: str = "fp16"
    seed: int = 1337
    num_workers: int = 4
    loss_sigma_weight: bool = False
    hf_token: str = None


class CatVTON_SD3_Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        seed_everything(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if cfg.mixed_precision == "fp16" and self.device.type == "cuda" else torch.float32

        # Resolve token in the caller scope (env -> cfg.hf_token)
        env_token = (os.environ.get("HUGGINGFACE_TOKEN")
                     or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                     or os.environ.get("HF_TOKEN"))
        token = cfg.hf_token or env_token
        if token is None:
            print("[warn] No HF token found. If the repo is gated, set HUGGINGFACE_TOKEN or pass --hf_token.")
        # Debug: show token prefix and whoami (safe)
        print(f"[HF] token prefix: {(token or '')[:8]}")
        try:
            who = HfApi(token=token).whoami()
            print(f"[HF] whoami: {who.get('name') or who.get('email')}")
        except Exception as e:
            print("[HF] whoami failed:", e)

        self.vae, self.transformer, self.encode_prompt, self.scheduler = load_sd3_components(
            cfg.sd3_model, self.device, self.dtype, token=token
        )

        self.projector = ConcatProjector(in_ch=33, out_ch=16).to(self.device, dtype=self.dtype)

        trainable_tf = freeze_all_but_attn_sd3(self.transformer)
        proj_params = sum(p.numel() for p in self.projector.parameters())
        print(f"Trainable params: transformer-attn={trainable_tf/1e6:.2f}M, projector={proj_params/1e6:.2f}M")

        self.dataset = PairListDataset(
            cfg.list_file, size_hw=(cfg.size_h, cfg.size_w), mask_based=cfg.mask_based
        )
        self.loader = DataLoader(
            self.dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )

        optim_params = itertools.chain(
            (p for p in self.transformer.parameters() if p.requires_grad),
            self.projector.parameters()
        )
        self.optimizer = torch.optim.AdamW(optim_params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _encode_prompts(self, bsz: int):
        pe, _, ppe, _ = self.encode_prompt(
            prompt=[""] * bsz,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return pe.to(self.dtype), ppe.to(self.dtype)

    def step(self, batch, global_step: int):
        H, W = self.cfg.size_h, self.cfg.size_w

        x_concat_in = batch["x_concat_in"].to(self.device, self.dtype)
        x_concat_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m_concat = batch["m_concat"].to(self.device, self.dtype)

        with torch.no_grad():
            Xi = to_latent_sd3(self.vae, x_concat_in)           # [B,16,H/8,2W/8]
            z0 = to_latent_sd3(self.vae, x_concat_gt)           # [B,16,H/8,2W/8]
            Mi = F.interpolate(m_concat, size=(H // 8, (2 * W) // 8), mode="nearest")

        B = Xi.shape[0]

        if self.cfg.cond_dropout_p > 0:
            drop = (torch.rand(B, device=self.device) < self.cfg.cond_dropout_p).float().view(B, 1, 1, 1)
            Xi = Xi * (1.0 - drop)
            Mi = Mi * (1.0 - drop)

        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        sigmas = self.scheduler.sigmas[timesteps].view(B, 1, 1, 1).to(dtype=self.dtype)

        noise = torch.randn_like(z0)
        noisy = sigmas * noise + (1.0 - sigmas) * z0

        hidden_in = self.projector(torch.cat([noisy, Xi, Mi], dim=1))  # [B,16,h,w]

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(B)

        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16)):
            out = self.transformer(
                hidden_states=hidden_in,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            ).sample  # [B,16,h,w]

            target = noise - z0
            if self.cfg.loss_sigma_weight:
                w = (sigmas ** 2).detach()
                loss = ((out - target) ** 2 * w).mean()
            else:
                loss = F.mse_loss(out, target, reduction="mean")

        return loss

    def save(self, step: int):
        path = os.path.join(self.cfg.output_dir, f"catvton_sd35_step{step:06d}.pt")
        payload = {
            "transformer": self.transformer.state_dict(),
            "projector": self.projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": vars(self.cfg),
            "step": step,
        }
        torch.save(payload, path)
        print(f"[save] {path}")

    def train(self):
        global_step = 0
        self.transformer.train()
        self.projector.train()
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))

        data_iter = itertools.cycle(self.loader)

        while global_step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(self.cfg.grad_accum):
                batch = next(data_iter)
                with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16)):
                    loss = self.step(batch, global_step)

                if self.dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += float(loss.detach().cpu())

            if self.dtype == torch.float16:
                scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.transformer.parameters() if p.requires_grad), 1.0
            )
            torch.nn.utils.clip_grad_norm_(self.projector.parameters(), 1.0)

            if self.dtype == torch.float16:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            global_step += 1

            if (global_step % self.cfg.log_every) == 0 or global_step == 1:
                print(f"[{global_step}/{self.cfg.max_steps}] loss={loss_accum/self.cfg.grad_accum:.4f}")

            if (global_step % self.cfg.save_every) == 0:
                self.save(global_step)

        self.save(global_step)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--list_file", type=str, required=True,
                   help="JSONL/JSON/CSV with person,garment[,mask] per line")
    p.add_argument("--output_dir", type=str, default="ckpts/catvton_sd35")
    p.add_argument("--sd3_model", type=str, default="stabilityai/stable-diffusion-3.5-large",
                   help="HF repo id for SD3/SD3.5 (e.g., stabilityai/stable-diffusion-3-medium)")
    p.add_argument("--size_h", type=int, default=512)
    p.add_argument("--size_w", type=int, default=384)
    p.add_argument("--mask_based", action="store_true", help="Stage-1 (use mask & masked person)")
    p.add_argument("--mask_free", action="store_true", help="Stage-2 (no mask; Ii = Ip)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=16000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--cond_dropout_p", type=float, default=0.1)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--loss_sigma_weight", action="store_true", help="weight MSE by sigma^2")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HF access token. If omitted, read from env HUGGINGFACE_TOKEN/HUGGINGFACE_HUB_TOKEN/HF_TOKEN")
    args = p.parse_args()

    assert not (args.mask_based and args.mask_free), "Choose exactly one of --mask_based or --mask_free."
    if not args.mask_based and not args.mask_free:
        args.mask_based = True

    return args


def main():
    args = parse_args()
    cfg = TrainConfig(
        list_file=args.list_file,
        output_dir=args.output_dir,
        sd3_model=args.sd3_model,
        size_h=args.size_h,
        size_w=args.size_w,
        mask_based=args.mask_based,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        cond_dropout_p=args.cond_dropout_p,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        num_workers=args.num_workers,
        loss_sigma_weight=args.loss_sigma_weight,
        hf_token=args.hf_token,
    )
    trainer = CatVTON_SD3_Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
