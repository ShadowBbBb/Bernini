#!/usr/bin/env bash
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

# Accelerate-based Bernini Renderer training (no VeOmni). Launches the entry
# script via `accelerate launch` and forwards extra args to the arg parser.
#
# CUDA : bash scripts/bernini_r_train/train_bernini_renderer_accel.sh <config.yaml>
# NPU  : ASCEND_RT_VISIBLE_DEVICES=0,1 bash .../train_bernini_renderer_accel.sh <config.yaml> \
#          --train.ddp_backend hccl
#
# Override topology via env: NNODES / NPROC_PER_NODE / MASTER_ADDR / MASTER_PORT.

set -euo pipefail
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MODELING_BACKEND=hf

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

CONFIG=${1:-configs/bernini_renderer_train/train_cfg/bernini_renderer_high_accel.yaml}
shift || true

# Ascend 910B: torch_npu.contrib.transfer_to_npu is imported inside the entry,
# which redirects cuda->npu and maps nccl->hccl. Set the visible devices here.
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export ACL_OP_SELECT_IMPL_MODE=high_precision
  echo "[accel] NPU mode: ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
else
  echo "[accel] CUDA mode"
fi

python - <<'PY'
import torch, importlib.util
print(f"torch={torch.__version__}")
if importlib.util.find_spec("torch_npu") is not None:
    print("torch_npu=available")
else:
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
PY

LAUNCH_ARGS="--num_cpu_threads_per_process 8 --num_processes ${NPROC_PER_NODE}"
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  LAUNCH_ARGS="${LAUNCH_ARGS} --multi_gpu"
fi

exec accelerate launch ${LAUNCH_ARGS} \
  -- \
  tasks/bernini_renderer/train_bernini_renderer_accel.py "${CONFIG}" "$@"
