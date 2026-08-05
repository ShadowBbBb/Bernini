# 变更历史（摘要，不记详细）

- 3a10d15 新增 run_all_tests.py（本地 checkpoint 路径）
- 40067f8 适配 Ascend 910B（torch_npu）+ 运行说明
- 2982327 修复 transfer_to_npu 是模块不可调用
- ba2cfa4 Qwen2.5-VL / DTensor 导入在 NPU 上可选
- b07920e RoPE float64→float32（complex64），NPU Cat 支持
- b2f5e41 新增推理结构与 rv2v 数据流分析文档
- f746643 新增 accelerate 训练框架（不依赖 VeOmni）+ FakeDataset
- ceea892 重写 rv2v 文档（可读性 + 训练章节）
- 6ca78fe 修 accelerate 子进程 import bernini（PYTHONPATH）
- c30f243 修训练入口：makedirs + 从 config 构建模型（非 from_pretrained）
- 88fbce1 模型加载改回全 rank 并发（试过错峰 fa67b94 已废；绕 from_pretrained collective 死锁；S3 FUSE 读盘卡死待本地盘验证）
- bb7d386 args.py 加 dict 子键 CLI 覆盖（`--model.model_config.wan22_base` 等）；建 todo.md/history.md 约定（WorkingPipeline 规则⑤⑥）
- 47a9e69 训练入口 dataloader 不进 accelerator.prepare（跳过 batch 广播，绕 HCCL bool/complex64/int64 dtype 报错）；重写 todo.md
- 本次：grad-ckpt 改 `use_reentrant=False`（HF 4.57.3 默认 reentrant 不接受 **kwargs，WanTransformerBlock 的 cu_seqlens_* 等 kwarg 被拒）
