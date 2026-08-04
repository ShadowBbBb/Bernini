# 服务器执行（1.3B 训练冒烟）

> 每次重写本文件，只放本次要执行的命令。服务器侧 `git pull origin main` 后 `cat todo.md` 复制执行。

```bash
# 1. checkpoint 已落本地盘则跳过（首次才 cp）
mkdir -p /home/ma-user/work/x50055359/bernini_1.3b
cp -rn /data/jijunxiang/Bernini/checkpoints/bernini_1.3b/* /home/ma-user/work/x50055359/bernini_1.3b/ 2>/dev/null || true

# 2. 拉最新（拿 dataloader 不进 prepare 的修复）
cd /data/jijunxiang/Bernini
git pull origin main

# 3. 跑（CLI 覆盖 wan22_base 指本地盘，不动 yaml）
ASCEND_RT_VISIBLE_DEVICES=6,7 NPROC_PER_NODE=2 \
  bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_1p3b_accel.yaml \
  --model.model_config.wan22_base /home/ma-user/work/x50055359/bernini_1.3b
```

## 预期
- 模型加载已验证（5/5 + 2/2 shard，~30s）。
- 本次 dataloader 不再 `accelerator.prepare`，跳过 batch 广播 → 不再 `Unsupported data type for HCCL`。
- 应进入训练循环：tqdm `0/20` 往前走，每步打印 loss/lr，到第 10 步存 ckpt。

## 若仍报错
- 若又冒 HCCL `Unsupported data type` → 说明还有别的 collective 走了 bool/complex，把日志发我。
- 若 `double dtype` 警告后真崩 → 找哪里 `.double()`，改成 `.float()`。
- 若 NPU 算子报错（如某个 aten 算子 NPU 没 impl）→ 发日志继续。
