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

"""Packing collator that reproduces VeOmni's ``MainCollator`` semantics for
the renderer, without the sequence-parallel padding/slicing (Ulysses is
disabled on the single-DDP path).

Per-sample dicts come from ``process_renderer_sample``; this collator packs
N samples into one batch whose tensors match the ``BerniniRendererModel.forward``
contract (squeeze(0)/unsqueeze(0) entry points).

  PACK  keys: cat along the last (token) dim, then ``unsqueeze(0)``  -> [1, total]
  CONCAT keys: cat along dim 0, no batch dim added                    -> [total, ...]
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

# Token/length fields packed along the last dim into a [1, total] tensor.
PACK_KEYS = (
    "input_ids",
    "attention_mask",
    "t5_input_lens",
    "vae_latents_mask",
    "vae_seqlen",
    "timesteps",
    "target_lens",
    "num_tokens",
    "vlm_seqlen",
)

# Tensor fields concatenated along dim 0 (the packed token axis), no batch dim.
CONCAT_KEYS = (
    "input_vae_latents",
    "input_vae_rope",
    "target_velocity",
)


class RendererPackingCollator:
    """Pack a list of per-sample dicts into one model-ready batch."""

    def __init__(self, return_source_names: bool = True):
        self.return_source_names = return_source_names

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("cannot pack an empty batch")

        batch: Dict[str, Any] = {}
        present_pack = [k for k in PACK_KEYS if k in samples[0]]
        present_concat = [k for k in CONCAT_KEYS if k in samples[0]]

        for key in present_pack:
            tensors = [s[key] for s in samples if key in s]
            if not tensors:
                continue
            batch[key] = torch.cat(tensors, dim=-1).unsqueeze(0).contiguous()

        for key in present_concat:
            tensors = [s[key] for s in samples if key in s]
            if not tensors:
                continue
            batch[key] = torch.cat(tensors, dim=0).contiguous()

        if self.return_source_names:
            batch["source_names"] = [
                s.get("source_name", "") for s in samples if "source_name" in s
            ]
        return batch
