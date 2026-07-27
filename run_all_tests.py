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

"""Run every Bernini-R test case in a single process, loading the model once.

Single-GPU only. The default --config targets the 1.3B diffusers layout so
the whole suite runs on one GPU; pass --config to switch to the 14B release
(checkpoints/bernini_14b) on a card with enough memory.

Usage:
    python run_all_tests.py
    python run_all_tests.py --config checkpoints/bernini_14b
    python run_all_tests.py --seed 123 --num_inference_steps 50
"""

import argparse
import json
import logging
import os
import time

import torch

from bernini.cli import (
    add_common_args,
    build_pipeline,
    generation_kwargs,
    resolve_system_prompt,
    setup_logging,
)
from bernini.pipeline import BerniniPipeline


logger = logging.getLogger("bernini.run_all")


# (case_path, guidance_mode, per-task overrides on top of the common kwargs).
# guidance_mode and the rv2v_case2 720p/24fps override mirror scripts/bernini_r/*.sh;
# the remaining sampling params use the CLI defaults from add_common_args
# (num_inference_steps=40, flow_shift=5.0, seed=42, fps=16, omega_*, ...).
TASKS = [
    ("assets/testcases/t2i/t2i.json",        "t2v_apg", {"num_frames": 1}),
    ("assets/testcases/i2i/i2i.json",         "v2v",     {"num_frames": 1}),
    ("assets/testcases/t2v/t2v.json",         "t2v_apg", {}),
    ("assets/testcases/v2v/v2v_case1.json",   "v2v_apg", {}),
    ("assets/testcases/v2v/v2v_case2.json",   "v2v_apg", {}),   # case sets task_type=mv2v
    ("assets/testcases/v2v/v2v_case3.json",   "v2v_apg", {}),
    ("assets/testcases/r2v/r2v.json",          "r2v_apg", {}),
    ("assets/testcases/r2v/r2v_case2.json",   "r2v_apg", {}),
    ("assets/testcases/rv2v/rv2v_case1.json", "rv2v",    {}),
    # rv2v_case2 is the 720p / 24fps ads-insertion example from run_rv2v.sh;
    # case sets task_type=ads2v.
    ("assets/testcases/rv2v/rv2v_case2.json", "rv2v",    {"num_frames": 121, "fps": 24, "max_image_size": 1280}),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all Bernini-R test cases, loading the model once")
    add_common_args(parser)
    # Default to the 1.3B diffusers layout so the suite runs on a single GPU.
    parser.set_defaults(config="checkpoints/bernini_1.3b")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Pre-flight: fail fast on missing case files before the slow model load.
    missing = [cp for cp, _, _ in TASKS if not os.path.exists(cp)]
    if missing:
        raise SystemExit(f"missing case files: {missing}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pipeline = build_pipeline(args, device)
    if isinstance(pipeline, BerniniPipeline):
        raise SystemExit(
            "run_all_tests.py targets Bernini-R only; the resolved --config "
            f"'{args.config}' loaded the full Bernini pipeline. "
            "Pass a Bernini-R directory (e.g. checkpoints/bernini_1.3b "
            "or checkpoints/bernini_14b)."
        )

    rewriter = None
    if args.use_pe:
        from bernini.prompt_enhancer import PromptEnhancer

        rewriter = PromptEnhancer(model=args.pe_model)

    common = generation_kwargs(args)
    results = []

    for case_path, guidance_mode, overrides in TASKS:
        t0 = time.time()
        try:
            with open(case_path) as f:
                case = json.load(f)
            task_type = case.get("task_type", args.task_type)
            prompt = case["prompt"]
            if rewriter is not None:
                prompt = rewriter(
                    task_type,
                    prompt,
                    video=case.get("video"),
                    image=case.get("image"),
                    images=case.get("images"),
                ) or prompt
            per_task = dict(common, guidance_mode=guidance_mode, **overrides)
            pipeline(
                prompt,
                video=case.get("video"),
                image=case.get("image"),
                images=case.get("images"),
                output_path=case.get("output", args.output),
                system_prompt=resolve_system_prompt(case, args),
                **per_task,
            )
            status = "OK"
        except Exception as e:  # noqa: BLE001
            status = "FAIL"
            logger.error("[FAIL] %s: %s", case_path, e)
        dt = time.time() - t0
        results.append((case_path, status, dt))
        logger.info("[%s] %s (%.1fs)", status, case_path, dt)

    logger.info("=== Summary ===")
    for case_path, status, dt in results:
        logger.info("  [%s] %s  %.1fs", status, case_path, dt)


if __name__ == "__main__":
    main()
