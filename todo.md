# 服务器执行（1.3B 训练冒烟）

> 每次重写本文件，只放本次要执行的命令。服务器侧 `git pull origin main` 后 `cat todo.md` 复制执行。

```bash
# 1. 落 checkpoint 到本地盘（已存在则跳过，-r 递归 -n 不覆盖已有）
mkdir -p /home/ma-user/work/x50055359/bernini_1.3b
cp -rn /data/jijunxiang/Bernini/checkpoints/bernini_1.3b/* /home/ma-user/work/x50055359/bernini_1.3b/

# 2. 校验 shard 数一致
ls /data/jijunxiang/Bernini/checkpoints/bernini_1.3b/text_encoder | wc -l
ls /home/ma-user/work/x50055359/bernini_1.3b/text_encoder | wc -l

# 3. 拉最新
cd /data/jijunxiang/Bernini
git pull origin main

# 4. 跑（CLI 覆盖 wan22_base 指本地盘，不动 yaml）
ASCEND_RT_VISIBLE_DEVICES=6,7 NPROC_PER_NODE=2 \
  bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_1p3b_accel.yaml \
  --model.model_config.wan22_base /home/ma-user/work/x50055359/bernini_1.3b
```

## 预期
本地盘读 shard 不经 S3 FUSE，`Loading checkpoint shards` 能往前走 → 5/5 → 进训练循环。

## 若仍卡
- 进度一直 0 不动 + 服务器变卡 → 仍是 S3 FUSE 残留或真 collective，把日志发我，上方案 B（改 `__init__`：rank0 独占 load + broadcast）。
- 跑通但有新错（如 NPU 算子、shape）→ 发日志继续。
