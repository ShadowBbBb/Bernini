# 服务器执行（1.3B 训练冒烟）

> 每次重写本文件，只放本次要执行的命令。服务器侧 `git pull origin main` 后 `cat todo.md` 复制执行。

```bash
# 1. checkpoint 已落本地盘则跳过（首次才 cp）
mkdir -p /home/ma-user/work/x50055359/bernini_1.3b
cp -rn /data/jijunxiang/Bernini/checkpoints/bernini_1.3b/* /home/ma-user/work/x50055359/bernini_1.3b/ 2>/dev/null || true

# 2. 拉最新（拿 use_reentrant=False 的 grad-ckpt 修复）
cd /data/jijunxiang/Bernini
git pull origin main

# 3. 跑（CLI 覆盖 wan22_base 指本地盘，不动 yaml）
ASCEND_RT_VISIBLE_DEVICES=6,7 NPROC_PER_NODE=2 \
  bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_1p3b_accel.yaml \
  --model.model_config.wan22_base /home/ma-user/work/x50055359/bernini_1.3b
```

## 预期
- 模型加载 5/5 + 2/2（已验证，~30s）。
- dataloader 不再 prepare（已修，跳过 batch 广播）。
- 梯度检查点改 `use_reentrant=False`（本次修），`**kwargs` 能进 `WanTransformerBlock.forward`。
- **应进入 forward/backward**，tqdm `0/20` 往前走，step 1 出 loss，第 10 步存 ckpt。

## 若仍报错
- 又冒 `Unexpected keyword arguments` → HF 版本不认 `gradient_checkpointing_kwargs`，退而关 grad-ckpt：`--train.gradient_checkpointing false`。
- `double dtype` 警告后真崩 → 找 `.double()` 改 `.float()`。
- NPU 算子没实现 → 发日志继续。
