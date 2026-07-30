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

"""Parquet datasets for accelerate-based Bernini Renderer training.

Each parquet row carries the pre-extracted columns produced by
``tools/preprocess_data.py`` (``image_embeds`` / ``image_vae_latents`` /
``video_embeds`` / ``video_vae_latents`` as ``torch.save`` blobs, plus the
``inputs`` conversation JSON). The per-sample transform
``process_renderer_sample`` (veomni-free) turns one row into the token dict the
packing collator consumes.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info


def _list_parquet_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    files += sorted(glob.glob(os.path.join(path, "*.parquet")))
    # deduplicate preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _row_to_sample(row) -> Dict[str, Any]:
    """Normalize a pyarrow row (dict-like) into a plain dict of python values."""
    sample: Dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if hasattr(value, "as_py"):
            value = value.as_py()
        sample[key] = value
    return sample


def _inject_source_name(transform, source_name):
    """Wrap a transform so the returned dict carries its source name for logging."""

    def _t(sample):
        out = transform(sample, source_name=source_name)
        if out:
            out[0]["source_name"] = source_name
        return out

    return _t


class FakeRendererDataset(IterableDataset):
    """Yield random-tensor rows through the real renderer transform.

    Produces a fake raw sample (a generated ``inputs`` conversation JSON plus
    random Wan-VAE ``latent_dist.parameters`` blobs) and runs it through the
    real ``process_renderer_sample`` transform, so RoPE / noise scheduling /
    VAE-latent packing all happen on the real code path. This guarantees the
    exact per-sample contract the packing collator and ``model.forward``
    expect, with no parquet / VAE / Qwen2.5-VL dependency.

    The ``inputs`` JSON is built by ``generate_unified_inputs`` (no file I/O);
    the random blobs have shape ``[1, 2*z_dim, T_lat, H//8, W//8]`` matching
    ``DiagonalGaussianDistribution.parameters``.
    """

    def __init__(
        self,
        transform: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        task_type: str = "v2v",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        z_dim: int = 16,
        prompt: str = "a sample clip for smoke test",
        num_samples: int = 0,
        seed: int = 42,
    ):
        self.transform = transform
        self.task_type = task_type
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.z_dim = z_dim
        self.prompt = prompt
        self.num_samples = num_samples
        self.seed = seed

    def _build_row(self, g: torch.Generator) -> Dict[str, Any]:
        from bernini.data_utils import generate_unified_inputs, tensor_to_bytes

        h_lat = self.height // 8
        w_lat = self.width // 8
        t_lat = (self.num_frames - 1) // 4 + 1
        if h_lat % 2 or w_lat % 2:
            raise ValueError(
                f"height//8={h_lat} and width//8={w_lat} must be even "
                f"(height/width must be divisible by 16)"
            )

        if self.task_type == "t2i":
            inputs = generate_unified_inputs(
                self.prompt, input_image_paths=[], input_video_paths=[],
                has_video_input=False, output_t=1,
                output_h=self.height, output_w=self.width,
            )
            blob = torch.randn(1, 2 * self.z_dim, 1, h_lat, w_lat, generator=g)
            return {"inputs": inputs, "image_vae_latents": [tensor_to_bytes(blob)]}

        if self.task_type == "t2v":
            inputs = generate_unified_inputs(
                self.prompt, input_image_paths=[], input_video_paths=[],
                has_video_input=False, output_t=self.num_frames,
                output_h=self.height, output_w=self.width,
            )
            blob = torch.randn(1, 2 * self.z_dim, t_lat, h_lat, w_lat, generator=g)
            return {"inputs": inputs, "video_vae_latents": [tensor_to_bytes(blob)]}

        if self.task_type == "v2v":
            inputs = generate_unified_inputs(
                self.prompt, input_image_paths=[], input_video_paths=["fake_src"],
                has_video_input=True, output_t=self.num_frames,
                output_h=self.height, output_w=self.width,
            )
            src_blob = torch.randn(1, 2 * self.z_dim, t_lat, h_lat, w_lat, generator=g)
            tgt_blob = torch.randn(1, 2 * self.z_dim, t_lat, h_lat, w_lat, generator=g)
            return {"inputs": inputs, "video_vae_latents": [tensor_to_bytes(src_blob), tensor_to_bytes(tgt_blob)]}

        raise ValueError(f"unsupported fake task_type '{self.task_type}' (use t2v/v2v/t2i)")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        g = torch.Generator().manual_seed(self.seed + rank * 7919 + worker_id)
        n = 0
        while self.num_samples == 0 or n < self.num_samples:
            row = self._build_row(g)
            try:
                out = self.transform(row)
            except Exception as exc:
                print(f"WARNING: fake transform failed: {exc}")
                n += 1
                continue
            if out:
                yield out[0]
            n += 1


class ParquetRendererDataset(IterableDataset):
    """Stream rows from parquet files and apply the renderer transform.

    The transform must return a one-element list (the contract of
    ``process_renderer_sample``); this dataset yields the single dict.
    """

    def __init__(
        self,
        data_path: str,
        transform: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        seed: int = 42,
        shuffle: bool = True,
        shuffle_buffer: int = 200,
        infinite: bool = True,
    ):
        self.data_path = data_path
        self.transform = transform
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.infinite = infinite

    def _iter_rows(self, epoch: int) -> Iterator[Dict[str, Any]]:
        files = _list_parquet_files(self.data_path)
        if not files:
            raise FileNotFoundError(f"no .parquet files under {self.data_path}")
        rng = random.Random(self.seed + epoch)
        if self.shuffle:
            rng.shuffle(files)

        import pyarrow.parquet as pq

        for path in files:
            try:
                pf = pq.ParquetFile(path)
            except Exception as exc:
                print(f"WARNING: skip unreadable parquet {path}: {exc}")
                continue
            for batch in pf.iter_batches():
                # batch is a pyarrow RecordBatch; convert to rows
                num_rows = batch.num_rows
                cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
                row_order = list(range(num_rows))
                if self.shuffle:
                    rng.shuffle(row_order)
                for r in row_order:
                    row = {name: cols[name][r] for name in cols}
                    sample = _row_to_sample(row)
                    try:
                        transformed = self.transform(sample)
                    except Exception as exc:
                        print(f"WARNING: transform failed on a row in {path}: {exc}")
                        continue
                    if not transformed:
                        continue
                    yield transformed[0]

    def _shuffle_iter(self, base: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        buf: List[Dict[str, Any]] = []
        for item in base:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                idx = random.randint(0, len(buf) - 1)
                yield buf.pop(idx)
        while buf:
            idx = random.randint(0, len(buf) - 1)
            yield buf.pop(idx)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        epoch = 0
        rng = random.Random(self.seed + worker_id)
        while True:
            rows = self._iter_rows(epoch + worker_id)
            # shard files across workers is approximated by skipping rows
            if num_workers > 1:
                rows = (row for i, row in enumerate(rows) if (i % num_workers) == worker_id)
            if self.shuffle and self.shuffle_buffer > 1:
                random.seed(self.seed + worker_id + epoch)
                rows = self._shuffle_iter(rows)
            for row in rows:
                yield row
            epoch += 1
            if not self.infinite:
                break


class WeightedMultiSourceDataset(IterableDataset):
    """Interleave multiple sources by weight, never exhausting.

    Mirrors VeOmni's ``weighted_multisource`` with ``stopping_strategy:
    never_exhausted``. At each step a source is drawn by its weight; when a
    source's iterator is exhausted it is restarted (cycled), so short sources
    are sampled more often than their weight implies only until others catch
    up -- acceptable for finetuning-style training.
    """

    def __init__(
        self,
        datasets: Sequence[ParquetRendererDataset],
        weights: Sequence[float],
        seed: int = 42,
    ):
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")
        if len(datasets) == 0:
            raise ValueError("at least one source is required")
        self.datasets = list(datasets)
        self.weights = torch.tensor([float(w) for w in weights], dtype=torch.float64)
        self.weights = self.weights / self.weights.sum()
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + worker_id)
        iters = [iter(ds) for ds in self.datasets]
        probs = self.weights.tolist()
        while True:
            src_idx = rng.choices(range(len(iters)), weights=probs, k=1)[0]
            try:
                yield next(iters[src_idx])
            except StopIteration:
                iters[src_idx] = iter(self.datasets[src_idx])
                yield next(iters[src_idx])


def build_dataset_from_args(
    data_args,
    transform: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    seed: int,
):
    """Build a fake / single-source / weighted-multi-source dataset from DataArgs."""
    if getattr(data_args, "fake_dataset", False):
        from dataclasses import asdict

        return FakeRendererDataset(transform=transform, seed=seed, **asdict(data_args.fake))

    ms = data_args.multisource
    if ms and ms.get("sources"):
        sources = ms["sources"]
        names = ms.get("names", [f"src{i}" for i in range(len(sources))])
        schedule = ms.get("schedule") or [{"schedule_type": "const", "weights": [1.0] * len(sources)}]
        weights = schedule[0].get("weights", [1.0] * len(sources))
        datasets = [
            ParquetRendererDataset(
                data_path=sources[i],
                transform=_inject_source_name(transform, names[i]),
                seed=seed + i,
            )
            for i in range(len(sources))
        ]
        return WeightedMultiSourceDataset(datasets, weights, seed=seed)
    return ParquetRendererDataset(
        data_path=data_args.train_path,
        transform=transform,
        seed=seed,
    )
