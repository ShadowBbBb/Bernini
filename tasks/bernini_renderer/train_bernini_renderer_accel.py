# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the Bernini Renderer with accelerate -- no VeOmni dependency.

Replaces ``tasks/bernini_renderer/train_bernini_renderer.py``'s VeOmni
plumbing (arguments, dataloader/collator, checkpointer, optimizer, FSDP2) with
accelerate, while reusing the veomni-free ``BerniniRendererModel.forward`` and
``bernini.training.data.process_renderer_sample`` transform unchanged.

Runs on CUDA (nccl) and Ascend 910B NPU (hccl) via DDP + bf16; Ulysses
sequence parallel stays disabled (size=1), so the model's SP ops are no-ops.

Usage:
    accelerate launch tasks/bernini_renderer/train_bernini_renderer_accel.py \\
        configs/bernini_renderer_train/train_cfg/bernini_renderer_high_accel.yaml \\
        --train.lr 1e-5
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import time
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import trange

# Make `bernini` importable regardless of how the entry is launched
# (accelerate puts the script's dir, not the repo root, on sys.path).
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from bernini.models.transformer_wan import WanRotaryPosEmbed
from bernini.training.args import as_plain_dict, parse_cli
from bernini.training.collator import RendererPackingCollator
from bernini.training.data import NoiseScheduler, process_renderer_sample
from bernini.training.dataset import build_dataset_from_args


def _setup_hardware():
    """Detect NPU and redirect cuda->npu (mirrors run_all_tests.py).

    Returns the DDP backend to use ("hccl" on NPU, "nccl" on CUDA).
    """
    if importlib.util.find_spec("torch_npu") is not None:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        torch.npu.config.allow_internal_format = False
        return "hccl"
    return "nccl"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(model: torch.nn.Module, module_name: str, detach: bool = False) -> None:
    wrapped = getattr(model, "module", model)
    module = getattr(wrapped, module_name)
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    if detach:
        forward = module.forward

        def no_grad_forward(*args, **kwargs):
            with torch.no_grad():
                return forward(*args, **kwargs)

        module.forward = no_grad_forward  # type: ignore[assignment]


def reduce_diff_loss(diff_loss: torch.Tensor, target_lens: torch.Tensor, accelerator) -> torch.Tensor:
    """Token-weighted reduction of the per-patch diff loss.

    Reproduces VeOmni's per-target mean + global token normalization, adapted
    to DDP gradient averaging: ``loss = (rank_target_sum / global_tokens) * W``
    so that after DDP all-reduce-mean the gradient is proportional to the
    global token-weighted average.
    """
    sample_losses = diff_loss.mean(dim=-1).squeeze(0)  # [N_tgt]
    target_lens = target_lens.squeeze(0)
    target_lens = target_lens[target_lens > 0]
    chunks = torch.split(sample_losses, target_lens.tolist())
    per_target = torch.stack([c.mean() for c in chunks])
    rank_target_sum = per_target.sum()
    rank_count = torch.tensor(float(per_target.numel()), device=sample_losses.device, dtype=torch.float32)
    global_count = accelerator.reduce(rank_count, reduction="sum")
    world_size = accelerator.num_processes
    return (rank_target_sum / global_count.clamp_min(1.0)) * float(world_size)


def build_lr_scheduler(optimizer, name: str, max_steps: int, warmup_ratio: float):
    from transformers import get_scheduler

    mapping = {
        "constant": "constant_with_warmup",
        "constant_with_warmup": "constant_with_warmup",
        "cosine": "cosine",
        "linear": "linear",
    }
    sched_name = mapping.get(name, "constant_with_warmup")
    num_warmup = max(1, int(warmup_ratio * max_steps))
    return get_scheduler(sched_name, optimizer, num_warmup_steps=num_warmup, num_training_steps=max_steps)


def move_to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def save_checkpoint(accelerator, model, tokenizer, args, global_step: int) -> None:
    save_dir = args.train.checkpoint.save_path or args.train.checkpoint.output_dir
    ckpt_dir = os.path.join(save_dir, f"global_step_{global_step}")
    # HF weights are replicated under DDP: write on the main process only.
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
        try:
            tokenizer.save_pretrained(ckpt_dir)
        except Exception:
            pass
        with open(os.path.join(ckpt_dir, "step.json"), "w") as f:
            json.dump({"global_step": global_step}, f)
    # accelerate state (optimizer/scheduler/dataloader): every rank calls this
    # so the collective-based save does not hang.
    accelerator.save_state(os.path.join(ckpt_dir, "accel_state"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"saved checkpoint -> {ckpt_dir}")


def main() -> None:
    backend = _setup_hardware()
    args = parse_cli()
    # Empty ddp_backend => keep the auto-detected backend (hccl on NPU, nccl on CUDA).
    # accelerate launch inits the process group from its config; on NPU the
    # transfer_to_npu import above maps nccl->hccl, so the default works there too.
    if args.train.ddp_backend:
        backend = args.train.ddp_backend

    seed_everything(args.train.seed)
    from accelerate import Accelerator

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = args.train.gradient_accumulation_steps
    if grad_accum is None:
        denom = max(1, args.train.micro_batch_size * world_size)
        grad_accum = max(1, args.train.global_batch_size // denom)
    accelerator = Accelerator(
        mixed_precision=args.train.mixed_precision,
        gradient_accumulation_steps=grad_accum,
    )
    accelerator.wait_for_everyone()
    accelerator.print(f"[accel] backend={backend} world_size={world_size} grad_accum={grad_accum} "
                      f"mixed_precision={args.train.mixed_precision}")
    if accelerator.is_main_process:
        os.makedirs(args.train.checkpoint.output_dir, exist_ok=True)
        with open(os.path.join(args.train.checkpoint.output_dir, "accel_args.json"), "w") as f:
            json.dump(as_plain_dict(args), f, indent=2, default=str)

    # ---- model ----
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel

    model_config = dict(args.model.model_config)
    model_config.setdefault("dtype", torch.bfloat16)
    model_config.setdefault("use_src_id_rotary_emb", args.train.use_src_id_rotary_emb)
    renderer_config = BerniniRendererConfig.from_pretrained(args.model.config_path, **model_config)
    model = BerniniRendererModel(renderer_config)
    model.train()
    if args.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    for module_name in args.train.freeze_modules:
        freeze_module(model, module_name, detach=False)
    for module_name in args.train.detach_modules:
        freeze_module(model, module_name, detach=True)

    # ---- tokenizer / rope / vae mean+std / noise scheduler / transform ----
    from transformers import AutoTokenizer

    model_config_obj = model.config
    tokenizer_path = getattr(model_config_obj, "wan22_base", None) or args.model.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, subfolder="tokenizer", padding_side="right", trust_remote_code=True
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=args.train.use_src_id_rotary_emb)
    vae_config_path = args.model.vae_config_path or os.path.join(model_config_obj.wan22_base, "vae", "config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
    vae_latent_mean = torch.tensor(vae_config["latents_mean"], device="cpu").view(vae_config["z_dim"], 1, 1, 1)
    vae_latent_std = torch.tensor(vae_config["latents_std"], device="cpu").view(vae_config["z_dim"], 1, 1, 1)
    noise_scheduler = NoiseScheduler(**args.data.noise_scheduler_config)
    transform = partial(
        process_renderer_sample,
        tokenizer=tokenizer,
        vae_rope_func=rope,
        vae_latent_mean=vae_latent_mean,
        vae_latent_std=vae_latent_std,
        noise_scheduler=noise_scheduler,
        text_dropout_rate=args.data.text_dropout_rate,
        img_dropout_rate=args.data.img_dropout_rate,
        video_dropout_rate=args.data.video_dropout_rate,
        max_vae_frames=args.data.max_vae_frames,
    )

    # ---- dataset / collator / dataloader ----
    dataset = build_dataset_from_args(args.data, transform, seed=args.train.seed)
    collator = RendererPackingCollator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train.micro_batch_size,
        collate_fn=collator,
        num_workers=args.data.dataloader.num_workers,
        drop_last=args.data.dataloader.drop_last,
        prefetch_factor=args.data.dataloader.prefetch_factor if args.data.dataloader.num_workers > 0 else None,
    )

    # ---- optimizer / scheduler ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.train.optimizer.type.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=args.train.optimizer.lr, weight_decay=args.train.optimizer.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.train.optimizer.lr, weight_decay=args.train.optimizer.weight_decay)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        args.train.optimizer.lr_decay_style,
        args.train.max_steps,
        args.train.optimizer.lr_warmup_ratio,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    global_step = 0
    if args.train.checkpoint.load_path:
        load_dir = args.train.checkpoint.load_path
        accel_state_dir = os.path.join(load_dir, "accel_state")
        if os.path.isdir(accel_state_dir):
            accelerator.load_state(accel_state_dir)
        step_file = os.path.join(load_dir, "step.json")
        if os.path.exists(step_file):
            with open(step_file) as f:
                global_step = json.load(f).get("global_step", 0)
        accelerator.print(f"resumed from {load_dir} at step {global_step}")

    # ---- training loop ----
    device = accelerator.device
    max_steps = args.train.max_steps
    pbar = trange(max_steps, initial=global_step, disable=not accelerator.is_main_process)
    start = global_step
    for epoch in range(args.train.num_train_epochs):
        for batch in dataloader:
            if global_step >= max_steps:
                break
            with accelerator.accumulate(model):
                batch = move_to_device(batch, device)
                batch.pop("source_names", None)
                output = model(**batch, use_cache=False)
                loss = reduce_diff_loss(output.diff_loss, batch["target_lens"], accelerator)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model, args.train.optimizer.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    global_step += 1
                    if accelerator.is_main_process and args.train.wandb.enable:
                        import wandb

                        if global_step == start + 1:
                            wandb.init(project=args.train.wandb.project, name=args.train.wandb.name, config=as_plain_dict(args))
                        wandb.log(
                            {
                                "train/loss": float(loss.item()),
                                "train/lr": float(lr_scheduler.get_last_lr()[0]),
                                "train/step": global_step,
                            },
                            step=global_step,
                        )
                    pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{float(lr_scheduler.get_last_lr()[0]):.2e}")
                    pbar.update(1)
                    if args.train.checkpoint.save_steps and global_step % args.train.checkpoint.save_steps == 0:
                        save_checkpoint(accelerator, model, tokenizer, args, global_step)
                    if args.train.empty_cache_steps and global_step % args.train.empty_cache_steps == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        if global_step >= max_steps:
            break
    pbar.close()
    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, model, tokenizer, args, global_step)
    if accelerator.is_main_process and args.train.wandb.enable:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
