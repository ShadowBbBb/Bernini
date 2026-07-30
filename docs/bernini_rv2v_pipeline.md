# Bernini 推理结构与 rv2v 数据流分析

> 基于 `run_all_tests.py`（Bernini-R renderer-only 路径）的源码追踪。
> 所有 shape 为说明性，实际跟随输入/配置与源媒体尺寸；关键处保留 `file:line` 引用便于跳转。

## 目录

**Part I · 推理**
1. [两条流水线](#1-两条流水线)
2. [Bernini-R 数据流总览](#2-bernini-r-数据流总览)
3. [典型 case shape 全链路](#3-典型-case-shape-全链路t2v-81帧--480×832)
4. [GEN_Wanx22.sample 逐步](#4-gen_wanx22sample-逐步)
5. [WanTransformer3DModel.forward 内部](#5-wantransformer3dmodelforward-内部)
6. [guidance_mode 前向组合与公式](#6-guidance_mode-前向组合与公式)
7. [10 个 test case](#7-10-个-test-case)
8. [双专家切换（1.3B / 14B）](#8-双专家切换13b--14b)
9. [source_id RoPE 与插值](#9-source_id-rope-与插值r2v_case2-的-8-图)
10. [rv2v 数据 pipeline 详解](#10-rv2v-数据-pipeline-详解)
11. [前向次数统计](#11-前向次数统计)

**Part II · 训练**
12. [accelerate 训练框架（不依赖 VeOmni）](#12-accelerate-训练框架不依赖-veomni)
13. [随机张量冒烟（FakeRendererDataset）](#13-随机张量冒烟fakerendererdataset)

**附录**
- [A. NPU 适配工作日志](#附录-anpuascend-910b适配工作日志)

---

# Part I · 推理

## 1. 两条流水线

`bernini/cli.py:133 build_pipeline` 按 `config.json` 的 `model_type` 分流：

| model_type | 类 | 加载什么 | run_all_tests 用 |
|---|---|---|---|
| `bernini` | `BerniniPipeline` (`pipeline.py:382`) | Qwen2.5-VL 规划器 + MLPConnector + UMT5 + 双专家 DiT + VAE | ✗（`run_all_tests.py:95` 显式拒绝）|
| `bernini_renderer` | `BerniniRendererPipeline` (`pipeline.py:206`) | UMT5 文本编码器 + 双专家 Wan2.2 DiT + VAE | ✓ |

- **1.3B**（`configs/bernini_renderer_wan21_1p3b/config.json`）：`skip_transformer_2=true, switch_dit_boundary=0` → 单专家（只用 transformer_1）。
- **14B**（`configs/bernini_renderer_wan22/config.json`）：双专家，边界 0.875。

---

## 2. Bernini-R 数据流总览

`BerniniRendererPipeline.__call__`（`pipeline.py:254`）六步：

```
case.json (prompt / video / image / images)
  │
  1. 文本 tokenize → input_ids, attn_mask        [1,512]   (pad=max_length=512)
  │
  2. VAE 编码视觉条件 (fp32 VAE, z_dim=16):
       video  → preprocess_video  [1,3,T,H,W] → _vae_encode → [1,16,T/4,H/8,W/8]
       image  → preprocess_image  [1,3,1,H,W] → _vae_encode → [1,16,1,H/8,W/8]
       images → 每张 _vae_encode → list of [1,16,1,H/8,W/8]
       归一化: (latent - mean) / std
  │
  3. make_divisible(h,16) / make_divisible(w,16)
  │
  4. model.sample() → BerniniRendererModel.sample (renderer.py:322)
       ├─ encode_prompt: UMT5 → prompt_embeds [1,512,4096]
       └─ diff_dec.sample() → GEN_Wanx22.sample → latent [1,16,F',H/8,W/8]
  │
  5. _vae_decode → [1,3,T,H,W] → VideoProcessor → [T,H,W,3] np [0,1]
  │
  6. save_output → .mp4 / .png
```

VAE 缩放（`wan_diffusion.py:201`）：`vae_scale_factor_temporal=4`，`vae_scale_factor_spatial=8`，`patch_size=(1,2,2)`。

---

## 3. 典型 case shape 全链路（t2v 81帧 @ 480×832）

> 480×832 为说明用标准 Wan 分辨率；CLI 默认实为 480×848，带源媒体时 h/w 跟随源。

**输入**：`num_frames=81, H=480, W=832`

**帧数对齐**（`wan_diffusion.py:316/334/335`）：
- `num_frames = 81//4*4+1 = 81`
- `num_latent_frames = (81-1)//4+1 = 21`
- `shape = (1, 16, 21, 60, 104)`（H//8=60, W//8=104）

**噪声初始化**：
- `noise = randn(shape)` float32，CPU generator 播种
- `noisy_vae_latent = rearrange "b c t (h ph)(w pw) -> b (t h w)(ph pw c)"`，ph=2 pw=2 → `[1, 32760, 64]`（每 token = 2×2×1 latent patch，21×30×52=32760）

**去噪循环（40 步，UniPC）**，每步 4 个动作：

1. **patch_embed 每个 source**
   `Conv3d(16→inner, k=(1,2,2), s=(1,2,2))`：`[1,16,F,60,104]` → `[1,inner,F,30,52]` → flatten → `[1, F*30*52, inner]`

2. **source_id + RoPE**
   `rope(hidden, source_id)` → `[1,1, F*30*52, head_dim/2]` complex；noisy target `source_id=0`

3. **拼接 cond+noisy → forward**
   `WanTransformer3DModel.forward` → `[1, total, 64]`

4. **切 mask + guidance + step**
   `noise_pred = pred[:, mask, :]` → `[1, 32760, 64]`；按 mode 合成；`scheduler.step` 更新 noisy

**收尾**：`_to_spatial` → `[1,16,21,60,104]` → `_vae_decode` → `[1,3,81,480,832]` → mp4

`inner_dim = num_attention_heads × attention_head_dim`（14B=40×128=5120；1.3B≈12×128=1536）。`text_dim=4096` 经 `text_embedder` 投影到 inner_dim。

---

## 4. GEN_Wanx22.sample 逐步

`wan_diffusion.py:274`：

1. **调度器**：`use_unipc` → `UniPCMultistepScheduler`；否则 `FlowMatchScheduler`（`shift` 控制 σ 映射）。
2. **帧数对齐**：`num_frames = num_frames//4*4+1`，`num_latent_frames=(num_frames-1)//4+1`。
3. **噪声初始化**：`shape=(1,16,F',H/8,W/8)`，CPU generator 播种 → rearrange 成 packed `[1, F'·(H/16)·(W/16), 64]`。
4. **source_id 分配**（`:416-471`）：
   - noisy target 恒为 `source_id=0`
   - VI 组（视频+图）共享轴：`vi_sids = _make_sids(num_videos+num_images)`
   - I 组（仅图）独立轴：`i_sids = _make_sids(num_images)`
   - `_make_sids(n)`：n≤5 → `[1..n]`；n>5 → `linspace(1,5,n)` 小数 id
5. **条件组装**（`:479-490`）：4 个 combo，每个 = cond_latents + [noisy_latent]，配 rotary（cat dim=2）与 mask（cond=False, noisy=True）：
   - `none` ∅：仅 noisy
   - `v`：第一个视频 + noisy（多视频仅第 1 个进 V 组）
   - `i`：所有图（独立 id 轴）+ noisy
   - `vi`：全部视频 + 全部图（共享 id 轴）+ noisy
6. **单步前向** `_fwd`（`:494`）→ `shared_step` → `WanTransformer3DModel`，返回 `[1,total,64]`，再 `[:, msk,:]` 切 noisy 段 `[1, noisy_len, 64]`。
7. **guidance 合成**（见 §6）。
8. **scheduler.step**：UniPC 返回 `[0]`；FlowMatch 返回 tensor。更新 packed noisy。
9. **收尾**：`_to_spatial` → `[1,16,F',H/8,W/8]`。

**双专家切换**（14B，`:367-381`）：`t >= boundary_timestep(=0.875*1000=875)` 用 transformer_1（高噪）；首次越过边界时 transformer_1→CPU、transformer_2→GPU，且 `omega_*` 全部 `*= omega_scale`。1.3B 因 `switch_dit_boundary=0` 且 `skip_transformer_2=true`，全程只用 transformer_1。

---

## 5. WanTransformer3DModel.forward 内部

`transformer_wan.py:530`。输入契约：hidden 已被 caller patch-embed 成 `[1, total_tokens, inner_dim]`。

1. **条件嵌入** `condition_embedder`（`WanTimeTextImageEmbedding`）：timestep → temb；`encoder_hidden_states`(T5 `[1,512,4096]`) → text_embedder → `[1,512,inner]`。
2. **时间步投影**：`timestep_proj` reshape 成 per-token `[1,total,6,inner]`；`temb` 按各样本 token 数 expand 成 `[1,total,inner]`。
3. **RoPE 转置**：`rotary_emb = rotary_emb.transpose(1,2)` → `[1,total,1,head_dim/2]` complex。
4. **SP 准备** `prepare_inputs_for_sp`：单卡 Ulysses no-op；建 `cu_seqlens_q`（VAE 段边界）与 `cu_seqlens_k_cross`（文本段边界）用于 varlen attention。
5. **Transformer 块** ×N：自注意力（RoPE + varlen）→ 交叉注意力（到 T5 文本）→ FFN；`scale_shift_table`+temb 做 adaLN 调制。
6. **输出投影** `proj_out`：`Linear(inner → 16*1*2*2=64)` → `[1,total,64]`（每 patch 的 latent 预测）。

**RoPE 维度切分**（`attention_head_dim=128` 为例）：
- `h_dim = w_dim = 2*(128//6) = 42`，`t_dim = 128-84 = 44`
- t 22 对、h 21 对、w 21 对 = 64 对 complex

**source_id 相位**（`:282-289`）：`get_1d_rotary_pos_embed(128,[sid])` → `[1,64]` complex，与空间 freqs 相乘 → 给每个 source 一个独立相位（id=0 → 单位 1，即 noisy target 无额外相位）。

---

## 6. guidance_mode 前向组合与公式

`wan_diffusion.py:509-600`。记号：∅=none, V=v, I=i, VI=vi，T/VTI=同 combo 但换 cond_text。`ε_X` = 该 combo 的单步预测。

| mode | 前向 | 公式 | source |
|---|---|---|---|
| `rv2v` | 4 (∅,V,VI,VTI) | `ε_∅ + ω_V(ε_V-ε_∅) + ω_I(ε_VI-ε_V) + ω_TI(ε_VTI-ε_VI)` | 视频+图 |
| `v2v` | 2 (VI_uncond, VTI) | `ε_VI_uncond + ω_TI(ε_VTI-ε_VI_uncond)` | 视频(+图) |
| `v2v_chain` | 3 (∅,V,VTI) | `ε_∅ + ω_V(ε_V-ε_∅) + ω_TI(ε_VTI-ε_V)` | 视频 |
| `t2v` | 2 (∅,T) | `ε_∅ + ω_TI(ε_T-ε_∅)` | 无 |
| `r2v_apg` | 3 (∅,I,TI) | chained APG（先转 x-pred 再投影） | 图 |
| `v2v_apg` | 2 (VI_uncond,VTI) | single APG ∅/VTI | 视频(+图) |
| `t2v_apg` | 2 (∅,T) | single APG ∅/T | 无 |

**APG**（`:91-106`）：`diff=cond-uncond` 投到 `base_pred` 的平行/正交分量 → `diff_orth + η·diff_parallel` → `uncond + scale·nd`；带 MomentumBuffer 与 norm_threshold 裁剪。APG 模式需 `sigma_apg` 把 v-pred 转 x-pred 再算（`:551-564`）。

---

## 7. 10 个 test case

`run_all_tests.py:59` TASKS：

| # | case | task_type | guidance | frames | 输入源 | 前向 | 关键行为 |
|---|---|---|---|---|---|---|---|
| 1 | t2i | t2i | t2v_apg | 1 | 无 | 2 | F'=1；∅/T 仅 noisy；输出 png |
| 2 | i2i | i2i | v2v | 1 | image | 2 | image→VAE latent 进 I 组；F'=1 |
| 3 | t2v | t2v | t2v_apg | 81 | 无 | 2 | ∅/T；F'=21；480×848 |
| 4 | v2v_case1 | v2v | v2v_apg | 81 | video×1 | 2 | 源视频进 VI（第1个也进 V）；h/w 跟源 |
| 5 | v2v_case2 | v2v | v2v_apg | 81 | video×1 | 2 | 同上 |
| 6 | v2v_case3 | v2v | v2v_apg | 81 | video×1 | 2 | 同上 |
| 7 | r2v | r2v | r2v_apg | 81 | images×5 | 3 | 5图≤5→`i_sids=[1..5]`；chained APG ∅/I/TI |
| 8 | r2v_case2 | r2v | r2v_apg | 81 | images×8 | 3 | 8图>5→`linspace(1,5,8)` 插值 |
| 9 | rv2v_case1 | rv2v | rv2v | 81 | video×1+img×1 | 4 | V=video；VI=video+img；链式 ∅/V/VI/VTI |
| 10 | rv2v_case2 | ads2v | rv2v | 121 | videos×2 | 4 | 仅 vid0 进 V，2 个都进 VI；F'=31；720p |

> 1.3B 下 `switch_dit_boundary=0` 但 `skip_transformer_2=true` → `transformer_2=None`，切换分支不触发（`:372` 守卫），全程 transformer_1。

---

## 8. 双专家切换（1.3B / 14B）

**1.3B**（`bernini_renderer_wan21_1p3b`）
- `skip_transformer_2=true` → `self.transformer_2=None`
- `switch_dit_boundary=0`；`switched` 永不置真，`cur_transformer` 始终 `self.transformer`
- 无 `omega_*` 二次缩放

**14B**（`bernini_renderer_wan22`）
- 高噪段(t≥875) → transformer_1，低噪段(t<875) → transformer_2
- 切换瞬间 `omega_{vid,img,txt} *= omega_scale`（默认 0.75）
- 两 transformer 共享 `rope`/`patch_embedding` 结构，权重不同（high/low-noise 双专家）

---

## 9. source_id RoPE 与插值（r2v_case2 的 8 图）

`WanRotaryPosEmbed.forward`（`transformer_wan.py:254`），`use_src_id_rotary_emb=True` 时：

1. 算空间 freqs（t/h/w 三段，`[1,1,ppf·pph·ppw,head_dim/2]` complex）
2. `pos=tensor([float(source_id)])` → `get_1d_rotary_pos_embed(head_dim, pos)` → `[1,head_dim/2]` complex，expand 到 `[1,1,N,head_dim/2]`
3. `freqs = freqs * freqs_visual_id` → 给该 source 所有 patch 一个随 source_id 变化的整体相位偏移

- `source_id=0`（noisy target）：`e^{i·0}=1` → 相位不变（基准）
- 整数 id 1..5：训练见过的相位
- **r2v_case2 8图 > max_trained=5**：`_make_sids(8)` → `linspace(1.0,5.0,8)=[1,1.571,…,5]`，小数 id 落训练流形内，避免外推未见相位（`interpolate_src_id`，`renderer.py:87`，CLI `--interpolate_src_id`）

---

## 10. rv2v 数据 pipeline 详解

rv2v = reference + video editing：给一段源视频 + 一张参考图，按 prompt 把参考内容融进视频，输出同尺寸视频。是 run_all_tests 里唯一用 4 次前向的 mode。

### 10.1 case1：1 视频 + 1 参考图（`rv2v_case1.json`）

**第一步：把输入变成三类 tensor**

```
输入             处理                     产物
────────────────────────────────────────────────────────
prompt   → tokenize (pad 512)      → input_ids [1,512]
负prompt → tokenize                → neg_ids   [1,512]

源视频mp4 → 抽帧+resize+归一化     → 像素 [1,3,T,H,W]
          → VAE encode (/4,/8)    → 视频latent [1,16,T/4,H/8,W/8]

参考图jpg → resize+归一化          → 像素 [1,3,1,H',W']
          → VAE encode           → 图latent [1,16,1,H'/8,W'/8]
```

输出尺寸 `T,H,W` 跟随源视频（非写死 81/480/832）。VAE 编完挪到 CPU 腾显存。

**第二步：文本走 T5，视频/图走「组合拼接」**

文本单独走 UMT5 编码成 `[1,512,4096]`，后面只做交叉注意力，**不参与组合拼接**。视频/图 latent 才是真正拼进噪声一起过 transformer 的。给每个源编 `source_id`：

```
源视频        → source_id = 1
参考图        → source_id = 2   (跟视频共享一条 id 轴)
噪声(要生成的)→ source_id = 0   (永远 0)
```

每个源过一次 `patch_vae_latent`（1×2×2 Conv3d）切成 token：

```
[1,16,T/4,H/8,W/8] → [1, N_视频, inner]
[1,16,1,H'/8,W'/8] → [1, N_图,   inner]
噪声 [1,16,F',h,w]  → [1, N_噪,   inner]
```

source_id 通过 RoPE 给每个源独立旋转相位，transformer 靠此区分「源视频/参考图/要生成的」。

**第三步：拼出 4 个组合（rv2v 的核心）**

每个组合 = 「哪些源 + 噪声」拼成一条序列，配 mask 标出哪段是噪声（要预测的）。

```
组合 ∅  ：[噪声]                            只生成，不给参考
组合 V  ：[视频]      + [噪声]               给源视频当参考
组合 VI ：[视频]+[图] + [噪声]               视频+参考图都给
组合 VTI：[视频]+[图] + [噪声] + 正向文本    同 VI，但文本换正向
```

（VI 和 VTI 视觉输入相同，唯一区别是文本：VI 用负 prompt，VTI 用正向。）

**第四步：每步去噪跑 4 次前向，做链式差分**

每个组合各跑一次 transformer，只取噪声段预测 `ε`：

```
ε_∅   = forward(组合∅)
ε_V   = forward(组合V)
ε_VI  = forward(组合VI, 负文本)
ε_VTI = forward(组合VI, 正向文本)
```

链式叠加——每项是「多加一个条件」的增量：

```
最终预测 = ε_∅
         + ω_vid × (ε_V  - ε_∅)    ← 加视频的增益
         + ω_img × (ε_VI - ε_V)    ← 再加参考图的增益
         + ω_txt × (ε_VTI- ε_VI)   ← 再加文本的增益
```

用这个预测过一遍 scheduler 更新噪声，循环 40 步。

> 为什么 4 次不是 2 次？普通 v2v 只对「文本」做一次 CFG 差分（2 次）；rv2v 把「视频」「图」「文本」拆成三级逐级看贡献，所以 4 次。这是它比 v2v 更精细也更贵的原因。

**第五步：解码出视频**

```
最终噪声 latent [1,16,F',h,w]
  → VAE decode → [1,3,T,H,W]
  → 存成 .mp4 (fps=16)
```

### 10.2 case2：2 视频（`rv2v_case2.json`，task_type=ads2v）

task_type 是 `ads2v`（往屏幕上贴另一段视频），输入 2 段视频、没有图：

```
组合 V ：[视频0]       + 噪声     # 只给主视频
组合 VI：[视频0]+[视频1]+ 噪声    # 再给待插入的 ref 视频
```

公式结构完全一样，只是「图」那一级换成「第二段视频」。所以 I 组为空、`i` 组合退化成 ∅，实际仍 4 次前向。

| | case1 (rv2v) | case2 (ads2v) |
|---|---|---|
| 源 | 1视频 + 1图 | 2视频 |
| V 组 | vid0(sid=1) | vid0(sid=1) |
| VI 组 | vid0(sid=1)+img(sid=2) | vid0(sid=1)+vid1(sid=2) |
| I 组 | img(sid=1) | 空（=none） |
| 前向 | 4 | 4 |
| 含义 | 编辑人物服装，用参考图替换 | 把 ref 视频贴到主视频屏幕（ads 插入） |

---

## 11. 前向次数统计

rv2v_case1 = **160 次** transformer 前向：

```
rv2v 每步 4 次前向 (∅ / V / VI / VTI)
× num_inference_steps = 40     (CLI 默认，无 override)
× 单专家 (1.3B: skip_transformer_2=true)
= 4 × 40 = 160，全部落在 transformer_1
```

各 mode 前向总数对比（均按 40 步）：

| guidance_mode | 每步前向 | 40 步总计 | 适用 case |
|---|---|---|---|
| t2v_apg | 2 | 80 | t2i, t2v |
| v2v / v2v_apg | 2 | 80 | i2i, v2v_case1/2/3 |
| r2v_apg | 3 | 120 | r2v, r2v_case2 |
| **rv2v** | **4** | **160** | **rv2v_case1/case2** |

补充：
1. 这 4 次前向是**串行**的 4 个独立 `shared_step` 调用，代码没把 4 个 combo 拼成一个大 batch 一次跑（VI 和 VTI 视觉输入相同、只差文本，理论上可批算但当前实现没做）。
2. 切 **14B**（`switch_dit_boundary=0.875`）仍 160 次，但按噪声分两段：高噪步(t≥875)走 transformer_1、低噪步(t<875)切 transformer_2，边界处做一次 GPU↔CPU 换卡。

---

# Part II · 训练

## 12. accelerate 训练框架（不依赖 VeOmni）

仓库提供两套**并行、互不干扰**的训练框架。原 VeOmni 框架完全未改，新 accelerate 框架独立命名（`_accel` 后缀）。

### 12.1 两套框架对照

| | 原框架（VeOmni） | 新框架（accelerate） |
|---|---|---|
| 入口 | `tasks/bernini_renderer/train_bernini_renderer.py` | `tasks/bernini_renderer/train_bernini_renderer_accel.py` |
| 脚本 | `scripts/bernini_r_train/train_bernini_renderer.sh` | `scripts/bernini_r_train/train_bernini_renderer_accel.sh` |
| 配置 | `configs/.../bernini_renderer_high.yaml` / `_low.yaml` | `..._high_accel.yaml` / `_low_accel.yaml` / `_1p3b_accel.yaml` |
| 分布式 | VeOmni（FSDP2 + Ulysses SP） | accelerate（DDP + bf16，ulysses 关） |
| 依赖 | veomni、Qwen2.5-VL（preprocess） | 仅 accelerate + torch |
| 数据 | VeOmni collator + dynamic batching | `ParquetRendererDataset` + `RendererPackingCollator` |

### 12.2 新增文件

| 文件 | 作用 |
|---|---|
| `tasks/bernini_renderer/train_bernini_renderer_accel.py` | accelerate 训练入口 |
| `bernini/training/args.py` | 数据类参数 + yaml 加载 + `--group.key` 覆盖 |
| `bernini/training/dataset.py` | `ParquetRendererDataset` + `WeightedMultiSourceDataset` + `FakeRendererDataset` |
| `bernini/training/collator.py` | `RendererPackingCollator`（复刻 veomni packing） |
| `scripts/bernini_r_train/train_bernini_renderer_accel.sh` | `accelerate launch` 包装 |
| `configs/.../*_accel.yaml` | high / low / 1.3b 配置 |

### 12.3 共用不改

- **模型**：`BerniniRendererModel.forward`（`renderer.py`）——两套都用
- **数据变换**：`process_renderer_sample` / `NoiseScheduler`（`bernini/training/data.py`）——两套都用
- **并行算子**：`bernini/parallel/*`（ulysses=1 时 no-op）——两套都用

### 12.4 训练数据流

```
parquet row
  │
  ├─ process_renderer_sample (复用, veomni-free)
  │    ├─ encode_renderer_messages: inputs JSON → tokenize → input_ids [L]
  │    └─ pack_vae_latents: *_vae_latents blob → .sample()+归一化 → pack
  │         → input_vae_latents, input_vae_rope, vae_latents_mask,
  │           vae_seqlen, target_velocity, target_lens, timesteps
  │
  ├─ RendererPackingCollator (N 样本拼一条)
  │    PACK  (cat 末维 + unsqueeze0): input_ids, attention_mask, t5_input_lens,
  │         vae_latents_mask, vae_seqlen, timesteps, target_lens, ...
  │    CONCAT(cat dim0): input_vae_latents [N,16,1,2,2],
  │         input_vae_rope [N,1,head_dim/2], target_velocity [N_tgt,16,1,2,2]
  │
  ├─ BerniniRendererModel.forward → diff_loss [1, N_tgt, 64]
  │
  ├─ reduce_diff_loss: token 加权归约 → scalar loss
  │
  └─ accelerator.backward → clip_grad_norm → optimizer.step → scheduler.step
```

### 12.5 loss 归约（token 加权，适配 DDP 均分）

```
sample_losses = diff_loss.mean(-1).squeeze(0)          # [N_tgt]
per_target    = split by target_lens → 各目标 mean
rank_target_sum = sum(per_target)
global_count  = accelerator.reduce(rank_count, "sum")  # 跨 DP 总 target 数
loss = (rank_target_sum / global_count) × world_size
```

DDP 对梯度做 all-reduce-mean（÷world_size）；乘 `world_size` 后还原为**全局 token 加权均值**，与 VeOmni FSDP 语义一致。

### 12.6 NPU 处理

- 入口顶部 `_setup_hardware()`：检测 `torch_npu` → `from torch_npu.contrib import transfer_to_npu` + `torch.npu.config.allow_internal_format=False`（同 `run_all_tests.py`）
- `ddp_backend=""` → 自动 hccl（CUDA 上 nccl）
- 不用 FSDP（NPU FSDP2 不可用），纯 DDP + bf16 autocast
- RoPE 已是 float32（commit `b07920e`），complex64 NPU 支持

### 12.7 checkpoint

- **HF 权重**（main-only）：`accelerator.unwrap_model(model).save_pretrained(dir, safe_serialization=True)`
- **训练状态**（全 rank，避免集体通信挂起）：`accelerator.save_state(dir/accel_state)`
- `step.json` 记 global_step；`checkpoint.load_path` 指向 ckpt 目录即可 resume

### 12.8 启动命令

```bash
# CUDA 8 卡
NPROC_PER_NODE=8 bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_high_accel.yaml \
  --train.optimizer.lr 1e-5

# NPU 8 卡（自动 hccl）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_high_accel.yaml

# CLI 覆盖示例
... --train.max_steps 1000 --data.fake.num_frames 5 --train.ddp_backend hccl
```

### 12.9 切真实数据

fake 模式关掉、指 parquet 目录即可，其余不动：

```bash
--data.fake_dataset false --data.train_path /path/to/preprocessed_parquet
```

---

## 13. 随机张量冒烟（FakeRendererDataset）

### 13.1 目的

无真实数据时验证训练全链路：forward → backward → optimizer → clip_grad → save_state → ckpt，以及 DDP 集体通信（all_reduce/save_state 不挂）。真跑 T5 + 真 transformer，只有 VAE latent 是随机的。

### 13.2 原理

`FakeRendererDataset` 产出「假原始行」`{inputs: <JSON>, video_vae_latents: [随机 bytes]}`，过**真实** `process_renderer_sample` transform → RoPE/noise/packing 全走真路径，契约自动对齐，NPU complex64 链路也覆盖。

- **inputs JSON**：用 `generate_unified_inputs` 造合法对话（不读任何文件）
- **假 VAE blob**：`torch.randn([1, 2*z_dim, T_lat, H//8, W//8])` = `DiagonalGaussianDistribution.parameters` 格式
- **真跑 T5**：input_ids 来自真实 prompt 分词，UMT5 前向真跑（冻结）
- **DDP 各 rank 种子**：`seed + RANK*7919 + worker_id`，避免各卡同 batch

### 13.3 配置

```yaml
data:
  fake_dataset: true
  fake:
    task_type: v2v       # t2v / v2v / t2i
    num_frames: 81
    height: 480
    width: 832
    z_dim: 16
    num_samples: 0       # 0 = 无限
```

CLI 覆盖：`--data.fake.num_frames 5 --data.fake.height 128 --data.fake.width 128`

### 13.4 启动

```bash
# 1.3B + fake，8 卡 DDP 冒烟
NPROC_PER_NODE=8 bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_1p3b_accel.yaml

# 快速冒烟（小尺寸 + 少步）
... --data.fake.num_frames 5 --data.fake.height 128 --data.fake.width 128 --train.max_steps 5

# NPU
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  bash scripts/bernini_r_train/train_bernini_renderer_accel.sh \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_1p3b_accel.yaml
```

切真实数据：`--data.fake_dataset false --data.train_path /path/to/parquet`。

### 13.5 1.3B vs 14B 训练差异

| | 1.3B | 14B |
|---|---|---|
| 专家 | 单（`skip_transformer_2=true`） | 双（high/low 各训一次） |
| noise 范围 | 0.0 ~ 1.0（整段） | high: 0.875~1.0；low: 0~0.875 |
| wan22_base | `checkpoints/bernini_1.3b`（本地） | Wan2.2-A14B diffusers |
| inner_dim | ≈1536 | 5120 |
| 冒烟推荐 | ✓（小，单卡可跑） | 需多卡 + grad ckpt |

---

## 附录 A：NPU（Ascend 910B）适配工作日志

将 Bernini-R fork 适配到 Ascend 910B（torch_npu），已提交推送到 `origin/main`：

| commit | 内容 |
|---|---|
| `3a10d15` | 新增 `run_all_tests.py`（本地 checkpoint 路径） |
| `40067f8` | 适配 Ascend 910B（torch_npu）+ 运行说明注释 |
| `2982327` | 修复：`transfer_to_npu` 是模块不可调用 |
| `ba2cfa4` | 让 Qwen2.5-VL 与 DTensor 导入在 NPU 上可选 |
| `b07920e` | RoPE 频率改 float32（complex64），NPU Cat 才支持 |

**RoPE complex128 → complex64 修复**（`bernini/models/transformer_wan.py`，4 处 `torch.float64`→`torch.float32`）：
- L54 `_apply_rotary_emb`：`x.to(torch.float32)`
- L246 `WanRotaryPosEmbed.__init__`：`freqs_dtype=torch.float32`
- L282/285 `forward`：source_id 的 `pos` 与 `freqs_dtype` 均改 float32

根因：NPU `aclnnCat` 支持 `DT_COMPLEX64` 但不支持 `DT_COMPLEX128`；RoPE 全链路复数张量由 complex128 改 complex64 后，L272 的 `torch.cat` 不再报 `aclnnCat` 错。

**环境要点**：
- `requirements.txt` pin `torch==2.7.1+cu126`（CUDA）——服务器是 NPU，**绝不** `pip install -r requirements.txt`（会覆盖 NPU torch），只选择性升级非 torch 依赖。
- `transfer_to_npu` 在该 torch_npu 是模块（不可调用），import 即自动把 cuda 调用重定向到 NPU。
- `run_all_tests.py` 顶部 `os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES","7")` + `from torch_npu.contrib import transfer_to_npu` + `torch.npu.config.allow_internal_format=False`。
