# 服务器执行（1.3B 训练冒烟）

> 每次重写本文件，只放本次要执行的命令。服务器侧 `git pull origin main` 后 `cat todo.md` 复制执行。

```bash
# 1. checkpoint 已落本地盘则跳过（首次才 cp）
mkdir -p /home/ma-user/work/x50055359/bernini_1.3b
cp -rn /data/jijunxiang/Bernini/checkpoints/bernini_1.3b/* /home/ma-user/work/x50055359/bernini_1.3b/ 2>/dev/null || true

# 2. 拉最新（拿 clip_grad_norm_ 的 model.parameters() 修复）
cd /data/jijunxiang/Bernini
git pull origin main

# 3. 跑（CLI 覆盖 wan22_base 指本地盘，不动 yaml）
ASCEND_RT_VISIBLE_DEVICES=6,7 NPROC_PER_NODE=2 \
  bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_1p3b_accel.yaml \
  --model.model_config.wan22_base /home/ma-user/work/x50055359/bernini_1.3b
```

## 预期
- 模型加载 5/5 + 2/2（~30s，已验证）。
- dataloader 不广播、grad-ckpt `use_reentrant=False`（已修）→ forward/backward 跑通。
- 本次修 `clip_grad_norm_(model.parameters(), ...)`（原传 `model` 不可迭代）→ step 1 应能 clip + 出 loss + 继续跑。
- tqdm `0/20` 往前走，step 1 出 loss/lr，第 10 步存 ckpt。

## 若仍报错
- 又冒 `not iterable` / DDP 相关 → 发日志，多半是某 API 对 DDP 包装对象的用法。
- `double dtype` 警告后真崩 → 找 `.double()` 改 `.float()`。
- NPU 算子没实现 → 发日志继续。
- 跑通但有 loss=nan → 数值问题，把前几步 loss 发我。
