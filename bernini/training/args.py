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

"""Accelerate-friendly training arguments for the Bernini Renderer.

Replaces VeOmni's ``parse_args(VeOmniArguments)`` with plain dataclasses fed
from a YAML file and overridden by ``--<group>.<key> value`` CLI flags, so the
existing ``noise_scheduler_config`` / dropout / optimizer sections of the
VeOmni configs can be reused verbatim.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import MISSING, dataclass, field, is_dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ModelArgs:
    config_path: str = "configs/bernini_renderer_wan22"
    vae_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLoaderArgs:
    num_workers: int = 1
    prefetch_factor: int = 2
    drop_last: bool = True


@dataclass
class FakeArgs:
    """Random-tensor dataset config for end-to-end plumbing tests."""

    task_type: str = "v2v"          # t2v / v2v / t2i
    num_frames: int = 81            # output clip length in frames
    height: int = 480
    width: int = 832
    z_dim: int = 16                 # Wan VAE latent channels
    prompt: str = "a sample clip for smoke test"
    num_samples: int = 0            # 0 = infinite


@dataclass
class DataArgs:
    train_path: str = ""
    multisource: Optional[Dict[str, Any]] = None
    datasets_type: str = "iterable"
    data_type: str = "diffusion"
    max_seq_len: int = 150000
    noise_scheduler_config: Dict[str, Any] = field(default_factory=dict)
    max_vae_frames: int = 61
    text_dropout_rate: float = 0.1
    img_dropout_rate: float = 0.1
    video_dropout_rate: float = 0.1
    dataloader: DataLoaderArgs = field(default_factory=DataLoaderArgs)
    fake_dataset: bool = False
    fake: FakeArgs = field(default_factory=FakeArgs)


@dataclass
class OptimizerArgs:
    type: str = "adamw"
    lr: float = 1.0e-5
    lr_start: float = 7.1e-7
    lr_min: float = 0.0
    lr_warmup_ratio: float = 0.004
    lr_decay_style: str = "constant"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class CheckpointArgs:
    output_dir: str = "bernini_renderer_train_accel"
    save_steps: int = 1000
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    save_hf_weights: bool = True
    save_async: bool = False


@dataclass
class WandbArgs:
    enable: bool = False
    project: str = "bernini_renderer"
    name: str = "accel"


@dataclass
class TrainArgs:
    micro_batch_size: int = 1
    global_batch_size: int = 1
    gradient_accumulation_steps: Optional[int] = None
    num_train_epochs: int = 1
    max_steps: int = 100000
    seed: int = 42
    init_device: str = "cpu"
    freeze_modules: List[str] = field(default_factory=lambda: ["t5_text_encoder"])
    detach_modules: List[str] = field(default_factory=list)
    use_src_id_rotary_emb: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    ddp_backend: str = ""   # empty = auto (hccl on NPU, nccl on CUDA)
    pad_to_length: bool = False
    empty_cache_steps: int = 1
    optimizer: OptimizerArgs = field(default_factory=OptimizerArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    wandb: WandbArgs = field(default_factory=WandbArgs)


@dataclass
class AccelerateArgs:
    model: ModelArgs = field(default_factory=ModelArgs)
    data: DataArgs = field(default_factory=DataArgs)
    train: TrainArgs = field(default_factory=TrainArgs)


_GROUP_CLASSES = {
    "model": ModelArgs,
    "data": DataArgs,
    "train": TrainArgs,
}

_SUBGROUP_CLASSES = {
    ("data", "dataloader"): DataLoaderArgs,
    ("data", "fake"): FakeArgs,
    ("train", "optimizer"): OptimizerArgs,
    ("train", "checkpoint"): CheckpointArgs,
    ("train", "wandb"): WandbArgs,
}


def _coerce(value: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    lv = value.strip()
    if lv.lower() in ("true", "yes", "on"):
        return True
    if lv.lower() in ("false", "no", "off"):
        return False
    if lv.lower() in ("none", "null"):
        return None
    try:
        return int(lv)
    except ValueError:
        pass
    try:
        return float(lv)
    except ValueError:
        pass
    return value


def _set_nested(args: AccelerateArgs, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) == 2:
        group, key = parts
        if group not in _GROUP_CLASSES:
            raise ValueError(f"unknown argument group '{group}'")
        obj = getattr(args, group)
        if not hasattr(obj, key):
            raise ValueError(f"unknown argument '{group}.{key}'")
        setattr(obj, key, value)
    elif len(parts) == 3:
        group, sub, key = parts
        if (group, sub) not in _SUBGROUP_CLASSES:
            raise ValueError(f"unknown argument group '{group}.{sub}'")
        obj = getattr(getattr(args, group), sub)
        if not hasattr(obj, key):
            raise ValueError(f"unknown argument '{group}.{sub}.{key}'")
        setattr(obj, key, value)
    else:
        raise ValueError(
            f"override key must be '<group>.<key>' or '<group>.<subgroup>.<key>', got '{dotted_key}'"
        )


def _populate(group_cls, raw: Dict[str, Any]):
    if raw is None:
        raw = {}
    init = {}
    for name, fdef in group_cls.__dataclass_fields__.items():
        if name not in raw:
            continue
        val = raw[name]
        sub = None
        if fdef.default_factory is not MISSING:
            try:
                sub = fdef.default_factory()
            except Exception:
                sub = None
        elif fdef.default is not MISSING and is_dataclass(fdef.default):
            sub = fdef.default
        if sub is not None and is_dataclass(sub) and isinstance(val, dict):
            init[name] = _populate(type(sub), val)
        else:
            init[name] = val
    return group_cls(**init)


def load_config(yaml_path: Optional[str], cli_overrides: List[Tuple[str, str]]) -> AccelerateArgs:
    raw: Dict[str, Any] = {}
    if yaml_path:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}

    args = AccelerateArgs()
    if "model" in raw:
        args.model = _populate(ModelArgs, raw["model"])
    if "data" in raw:
        args.data = _populate(DataArgs, raw["data"])
    if "train" in raw:
        args.train = _populate(TrainArgs, raw["train"])

    for dotted_key, raw_value in cli_overrides:
        _set_nested(args, dotted_key, _coerce(raw_value))
    return args


def parse_cli(argv: Optional[List[str]] = None) -> AccelerateArgs:
    parser = argparse.ArgumentParser(
        description="Train the Bernini Renderer with accelerate (no VeOmni).",
        usage="%(prog)s <config.yaml> [--<group>.<key> value ...]",
    )
    parser.add_argument("config", nargs="?", default=None, help="YAML config file")
    known, rest = parser.parse_known_args(argv)

    overrides: List[Tuple[str, str]] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.startswith("--") and "." in token[2:]:
            key = token[2:]
            if i + 1 >= len(rest):
                parser.error(f"missing value for '{token}'")
            overrides.append((key, rest[i + 1]))
            i += 2
        else:
            parser.error(f"unexpected argument '{token}'")
    return load_config(known.config, overrides)


def as_plain_dict(args: AccelerateArgs) -> Dict[str, Any]:
    return asdict(args)
