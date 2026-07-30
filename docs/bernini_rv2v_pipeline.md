# Bernini 推理结构与 rv2v 数据流分析

> 基于 `run_all_tests.py`（Bernini-R renderer-only 路径）的源码追踪。所有 shape 为说明性，实际跟随输入/配置与源媒体尺寸。
> 关键处保留 `file:line` 引用，便于跳转核对。

---

## 1. 两条流水线，run_all_tests 走哪条

`bernini/cli.py:133 build_pipeline` 按 `config.json` 的 `model_type` 分流：

| model_type | 类 | 加载什么 | run_all_tests 用 |
|---|---|---|---|
| `bernini` | `BerniniPipeline` (`pipeline.py:382`) | Qwen2.5-VL 规划器 + MLPConnector + UMT5 + 双专家 DiT + VAE | ✗（被 `run_all_tests.py:95` 显式拒绝）|
| `bernini_renderer` | `BerniniRendererPipeline` (`pipeline.py:206`) | UMT5 文本编码器 + 双专家 Wan2.2 DiT + VAE | ✓ |

- 1.3B 配置（`configs/bernini_renderer_wan21_1p3b/config.json`）：`skip_transformer_2=true, switch_dit_boundary=0` → **单专家**（只用 transformer_1）。
- 14B 配置（`configs/bernini_renderer_wan22/config.json`）：双专家，边界 0.875。

---

## 2. Bernini-R 数据流总览

`BerniniRendererPipeline.__call__`（`pipeline.py:254`）：

```
case.json (prompt / video / image / images)
  │
  ├─ 1. 文本 tokenize: _tokenize → input_ids, attn_mask  [1,512]  (padding=max_length=512)
  │
  ├─ 2. VAE 编码视觉条件（fp32 VAE）:
  │     video  → preprocess_video  → [1,3,T,H,W] → _vae_encode → [1,16,T/4,H/8,W/8] 归一化 latent
  │     image  → preprocess_image  → [1,3,1,H,W] → _vae_encode → [1,16,1,H/8,W/8]
  │     images → 每张 _vae_encode → list of [1,16,1,H/8,W/8]
  │     (VAE 归一化: (latent-mean)/std，z_dim=16)
  │
  ├─ 3. make_divisible(h,16) / make_divisible(w,16)
  │
  ├─ 4. model.sample(...) → BerniniRendererModel.sample (renderer.py:322)
  │     ├─ encode_prompt: UMT5 → prompt_embeds [1,512,4096]  (pad 到 max_sequence_length)
  │    └─ diff_dec.sample(...) → GEN_Wanx22.sample → 返回 latent [1,16,F',H/8,W/8]
  │
  ├─ 5. _vae_decode: vae.decode → [1,3,T,H,W] → VideoProcessor → [T,H,W,3] np float [0,1]
  └─ 6. save_output → .mp4 / .png
```

VAE 缩放因子（`wan_diffusion.py:201`）：`vae_scale_factor_temporal=4`，`vae_scale_factor_spatial=8`，patch_size=(1,2,2)。

---

## 3. 典型 case shape 全链路追踪（t2v 81帧 @ 480×832）

> 480×832 为说明用标准 Wan 分辨率；CLI 默认实为 480×848，带源媒体时 h/w 跟随源。

```
输入: num_frames=81, H=480, W=832
  │
  num_frames = 81//4*4+1 = 81                       (wan_diffusion.py:316)
  num_latent_frames = (81-1)//4+1 = 21               (:334)
  shape = (1, 16, 21, 60, 104)                       H//8=60, W//8=104 (:335)
  │
  noise = randn(shape) float32                       (CPU generator)
  noisy_vae_latent = rearrange "b c t (h ph)(w pw)->b (t h w)(ph pw c)"
                   ph=2,pw=2 → [1, 21*30*52=32760, 64]   (每个 token=2×2×1 latent patch)
  │
  ┌──────── 去噪循环 (40 步, UniPC) ────────┐
  │ 每个 source 经 patch_vae_latent:         │
  │   Conv3d(16→inner_dim,k=(1,2,2),s=(1,2,2))│
  │   [1,16,F,60,104]→[1,inner,F,30,52]       │
  │   →flatten→ [1, F*30*52, inner_dim]       │
  │   + rope(hidden,source_id):               │
  │     [1,1, F*30*52, head_dim/2] complex    │
  │ noisy target source_id=0: [1,32760,inner] │
  │ 拼接 cond+noisy: [1, total, inner_dim]    │
  │ _fwd→ WanTransformer3DModel.forward        │
  │   → [1, total, 16*4=64]                   │
  │ 切出 noisy mask → noise_pred [1,32760,64] │
  │ guidance 合成 (按 mode)                    │
  │ scheduler.step → 更新 noisy_vae_latent    │
  └──────────────────────────────────────────┘
  return _to_spatial → [1, 16, 21, 60, 104] latent
  │
  _vae_decode → [1,3,81,480,832] → [81,480,832,3] mp4
```

`inner_dim` = `num_attention_heads × attention_head_dim`，配置驱动：14B=40×128=5120；1.3B≈12×128=1536（Wan2.1-1.3B）。`text_dim=4096` 经 `text_embedder` 投影到 inner_dim。

---

## 4. GEN_Wanx22.sample 逐步

`wan_diffusion.py:274`：

1. **调度器**：`use_unipc`→`UniPCMultistepScheduler`；否则 `FlowMatchScheduler`（`shift` 控制 σ 映射）。
2. **帧数对齐**：`num_frames = num_frames//4*4+1`，`num_latent_frames=(num_frames-1)//4+1`。
3. **噪声初始化**：`shape=(1,16,F',H/8,W/8)`，CPU generator 播种 → `noise`，rearrange 成 packed `[1, F'·(H/16)·(W/16), 64]`。
4. **source_id 分配**（`:416-471`）：
   - noisy target 恒为 `source_id=0`。
   - VI 组（视频+图）共享一条 id 轴：`vi_sids = _make_sids(num_videos+num_images)`。
   - I 组（仅图）独立轴：`i_sids = _make_sids(num_images)`。
   - `_make_sids(n)`：n ≤ max_trained(5) → `[1..n]`；n>5 → `linspace(1,5,n)` 小数 id。
5. **条件组装**（`:479-490`）：4 个 combo，每个 = cond_latents + [noisy_latent]，对应 4 套 rotary（cat dim=2）与 mask（cond=False，noisy=True）：
   - `none`：∅（仅 noisy）
   - `v`：第一个视频 + noisy（多视频里仅第 1 个进 V 组）
   - `i`：所有图（独立 id 轴）+ noisy
   - `vi`：全部视频 + 全部图（共享 id 轴）+ noisy
6. **单步前向** `_fwd`（`:494`）→ `shared_step` → `WanTransformer3DModel`，返回 `[1,total,64]`，再 `[:, msk,:]` 切出 noisy 段 `[1, noisy_len, 64]`。
7. **guidance 合成**（见 §6）。
8. **scheduler.step**：UniPC 返回 `[0]`；FlowMatch 返回 tensor。更新 packed noisy。
9. 收尾 `_to_spatial` → `[1,16,F',H/8,W/8]`。

**双专家切换**（14B，`:367-381`）：`t >= boundary_timestep(=0.875*1000=875)` 用 transformer_1（高噪）；首次越过边界时 transformer_1→CPU、transformer_2→GPU，且 `omega_*` 全部 `*= omega_scale`。1.3B 因 `switch_dit_boundary=0` 且 `skip_transformer_2=true`，全程只用 transformer_1，无切换。

---

## 5. WanTransformer3DModel.forward 内部

`transformer_wan.py:530`。输入契约：hidden 已被 caller patch-embed 成 `[1, total_tokens, inner_dim]`。

1. `condition_embedder`（`WanTimeTextImageEmbedding`）：timestep→temb；`encoder_hidden_states`(T5, [1,512,4096]) → text_embedder → [1,512,inner]。
2. `timestep_proj` reshape 成 per-token `[1,total,6,inner]`；`temb` 按各样本 token 数 expand 成 `[1,total,inner]`。
3. `rotary_emb = rotary_emb.transpose(1,2)` → `[1,total,1,head_dim/2]` complex。
4. `prepare_inputs_for_sp`：单卡时 Ulysses no-op；建 `cu_seqlens_q`（VAE 段边界）与 `cu_seqlens_k_cross`（文本段边界）用于 varlen attention。
5. 每个 `WanTransformerBlock`：自注意力（RoPE + varlen）→ 交叉注意力（到 T5 文本）→ FFN；用 `scale_shift_table`+temb 做 adaLN 调制。
6. `proj_out`：`Linear(inner → 16*1*2*2=64)` → 输出 `[1,total,64]`（每个 patch 的 latent 预测）。

**RoPE 维度切分**（`attention_head_dim=128` 为例）：`h_dim=w_dim=2*(128//6)=42`，`t_dim=128-84=44` → t 22 对、h 21 对、w 21 对 = 64 对 complex。`source_id` 项（`:282-289`）：`get_1d_rotary_pos_embed(128,[sid])` 得 `[1,64]` complex，与空间 freqs 相乘 → 给每个 source 一个独立相位（id=0 → 单位 1，即 noisy target 无额外相位）。

---

## 6. 各 guidance_mode 前向组合与公式

`wan_diffusion.py:509-600`。记号：∅=none, V=v, I=i, VI=vi, T/VTI=同 combo 但换 cond_text。`ε_X` = 该 combo 的单步预测。

| mode | 前向 | 公式 | source |
|---|---|---|---|
| `rv2v` | 4 (∅,V,VI,VTI) | `ε_∅ + ω_V(ε_V-ε_∅) + ω_I(ε_VI-ε_V) + ω_TI(ε_VTI-ε_VI)` | 视频+图 |
| `v2v` | 2 (VI_uncond, VTI) | `ε_VI_uncond + ω_TI(ε_VTI-ε_VI_uncond)` | 视频(+图) |
| `v2v_chain` | 3 (∅,V,VTI) | `ε_∅ + ω_V(ε_V-ε_∅) + ω_TI(ε_VTI-ε_V)` | 视频 |
| `t2v` | 2 (∅,T) | `ε_∅ + ω_TI(ε_T-ε_∅)` | 无 |
| `r2v_apg` | 3 (∅,I,TI) | chained APG（先转 x-pred 再投影） | 图 |
| `v2v_apg` | 2 (VI_uncond,VTI) | single APG between ∅/VTI | 视频(+图) |
| `t2v_apg` | 2 (∅,T) | single APG between ∅/T | 无 |

**APG**（`:91-106`）：先把 `diff=cond-uncond` 投到 `base_pred` 的平行/正交分量，`diff_orth + η·diff_parallel`，再 `uncond + scale·nd`；带 MomentumBuffer 与 norm_threshold 裁剪。APG 模式需 `sigma_apg` 把 v-pred 转 x-pred 再算（`:551-564`）。

---

## 7. 10 个 test case 逐个分析

`run_all_tests.py:59` TASKS：

| # | case | task_type | guidance | num_frames | 输入源 | 前向 | 关键 shape/行为 |
|---|---|---|---|---|---|---|---|
| 1 | t2i | t2i | t2v_apg | 1 (override) | 无 | 2 | F'=1；∅/T combo 仅 noisy；输出 png |
| 2 | i2i | i2i | v2v | 1 (override) | image=source.png | 2 | image→VAE latent 进 I 组；F'=1；输出 png |
| 3 | t2v | t2v | t2v_apg | 81 | 无 | 2 | 无源，∅/T；F'=21；标准 480×848 |
| 4 | v2v_case1 | v2v | v2v_apg | 81 | video(1) | 2 | source 视频进 VI 组（第1个也进 V）；h/w 跟随源 |
| 5 | v2v_case2 | v2v | v2v_apg | 81 | video(1) | 2 | 同上 |
| 6 | v2v_case3 | v2v | v2v_apg | 81 | video(1) | 2 | 同上 |
| 7 | r2v | r2v | r2v_apg | 81 | images(5) | 3 | 5 图≤5→`i_sids=[1..5]`；无视频，VI 组=I 组；chained APG ∅/I/TI |
| 8 | r2v_case2 | r2v | r2v_apg | 81 | images(8) | 3 | 8 图>5→`i_sids=linspace(1,5,8)` 小数 id 插值；其余同 7 |
| 9 | rv2v_case1 | rv2v | rv2v | 81 | video(1)+images(1) | 4 | V 组=video；VI=video+img；`vi_sids=[1,2]`；链式 ∅/V/VI/VTI |
| 10 | rv2v_case2 | ads2v | rv2v | 121 (override,24fps,1280) | videos(2) | 4 | 2 视频：仅第1个进 V 组，2 个都进 VI 组；`vi_sids=[1,2]`；F'=31；720p |

注意：
- 1.3B 下 `switch_dit_boundary=0` ⇒ 所有 t 都 `<boundary`（除 t=0），但 `skip_transformer_2=true` ⇒ `transformer_2=None`，切换分支不触发（`:372` 守卫 `self.transformer_2 is not None`），全程 transformer_1。

---

## 8. 双专家切换（1.3B 单 / 14B 双）

- **1.3B**（`bernini_renderer_wan21_1p3b`）：`skip_transformer_2=true`→`self.transformer_2=None`；`switch_dit_boundary=0`。循环里 `model_id` 理论上按 t 选，但 `switched` 永不置真（`transformer_2 is None`），`cur_transformer` 始终 `self.transformer`。无 `omega_*` 二次缩放。
- **14B**（`bernini_renderer_wan22`）：高噪段(t≥875)用 transformer_1，低噪段(t<875)切到 transformer_2；切换瞬间 `omega_{vid,img,txt} *= omega_scale`(0.75 默认)。两个 transformer 共享同一 `rope`/`patch_embedding` 结构但权重不同（high-noise / low-noise 双专家）。

---

## 9. source_id RoPE 与插值（r2v_case2 的 8 图）

`WanRotaryPosEmbed.forward`（`transformer_wan.py:254`）当 `use_src_id_rotary_emb=True`：

1. 算空间 freqs（t/h/w 三段，`[1,1,ppf·pph·ppw,head_dim/2]` complex）。
2. `pos=tensor([float(source_id)])` → `get_1d_rotary_pos_embed(head_dim,pos)` → `[1,head_dim/2]` complex，expand 到 `[1,1,N,head_dim/2]`。
3. `freqs = freqs * freqs_visual_id` → 给该 source 的所有 patch 一个随 source_id 变化的整体相位偏移。

- `source_id=0`（noisy target）：`e^{i·0}=1` → 相位不变（基准）。
- 整数 id 1..5：训练见过的相位。
- **r2v_case2 8 图 > max_trained=5**：`_make_sids(8)` 走 `linspace(1.0,5.0,8)`=[1,1.571,2.143,…,5]，小数 id 落在训练流形内，避免外推到未见相位。这是 `interpolate_src_id` 的作用（`renderer.py:87`，CLI `--interpolate_src_id`）。

---

## 10. rv2v 数据 pipeline 详解

rv2v = reference + video editing：给一段源视频 + 一张参考图，按 prompt 把参考内容融进视频里，输出同尺寸视频。是 run_all_tests 里唯一用 4 次前向的 mode。

### 10.1 case1：1 视频 + 1 参考图（`rv2v_case1.json`）

**第一步：把输入变成三类 tensor**

```
输入                          处理                      产物
─────────────────────────────────────────────────────────────────
prompt 文本        →  tokenize (pad 到 512)    →  input_ids  [1, 512]
负 prompt          →  tokenize                 →  neg_ids    [1, 512]

源视频 mp4         →  抽帧+resize+归一化       →  像素  [1,3,T,H,W]
                   →  VAE encode (时间/4,空间/8) →  视频latent [1,16,T/4,H/8,W/8]

参考图 jpg         →  resize+归一化            →  像素  [1,3,1,H',W']
                   →  VAE encode              →  图latent [1,16,1,H'/8,W'/8]
```

输出尺寸 `T,H,W` 跟随源视频（不是写死的 81/480/832）。VAE 编完挪到 CPU 腾显存。

**第二步：文本走 T5，视频/图走「组合拼接」**

文本单独走 UMT5 编码成 `[1,512,4096]`，后面只做交叉注意力，**不参与组合拼接**。视频/图 latent 才是真正拼进噪声一起过 transformer 的。给每个源编一个 `source_id`：

```
源视频         →  source_id = 1
参考图         →  source_id = 2     (跟视频共享一条 id 轴)
噪声(要生成的) →  source_id = 0     (永远是 0)
```

每个源过一次 `patch_vae_latent`（1×2×2 的 Conv3d），把 latent 切成 token：

```
[1,16,T/4,H/8,W/8]  →  [1, N_视频, inner]
[1,16,1,H'/8,W'/8]   →  [1, N_图,   inner]
噪声 [1,16,F',h,w]    →  [1, N_噪,   inner]
```

source_id 通过 RoPE 给每个源一组独立旋转相位，transformer 靠这个区分「这是源视频、那是参考图、这是我要生成的」。

**第三步：拼出 4 个组合（rv2v 的核心）**

每个组合 = 「哪些源 + 噪声」拼成一条序列，配一个 mask 标出哪段是噪声（要预测的）。

```
组合 ∅   ：[噪声]                              只生成，不给任何参考
组合 V   ：[视频]        + [噪声]               给源视频当参考
组合 VI  ：[视频] + [图] + [噪声]               视频+参考图都给
组合 VTI ：[视频] + [图] + [噪声]  + 正向文本    同 VI，但文本换正向
```

（VI 和 VTI 是同一组视觉输入，唯一区别是文本：VI 用负 prompt，VTI 用正向 prompt。）

**第四步：每步去噪跑 4 次前向，做链式差分**

每个组合各跑一次 transformer，只取噪声段的预测 `ε`：

```
ε_∅   = forward(组合∅)
ε_V   = forward(组合V)
ε_VI  = forward(组合VI, 负文本)
ε_VTI = forward(组合VI, 正向文本)
```

链式叠加——每一项是「多加一个条件」带来的增量：

```
最终预测 = ε_∅
         + ω_vid × (ε_V  - ε_∅)     ← 加视频的增益
         + ω_img × (ε_VI - ε_V)     ← 再加参考图的增益
         + ω_txt × (ε_VTI- ε_VI)    ← 再加文本的增益
```

用这个预测过一遍 scheduler 更新噪声，循环 40 步。

> 为什么是 4 次而不是 2 次？普通 v2v 只对「文本」做一次 CFG 差分（2 次）；rv2v 把「视频」「图」「文本」拆成三级，逐级看各自贡献，所以 4 次。这也是它比 v2v 更精细、更贵的原因。

**第五步：解码出视频**

```
最终噪声 latent [1,16,F',h,w]
   → VAE decode → [1,3,T,H,W]
   → 存成 .mp4 (fps=16)
```

### 10.2 case2：2 视频（`rv2v_case2.json`，task_type=ads2v）

task_type 是 `ads2v`（往屏幕上贴另一段视频），输入是 2 段视频、没有图：

```
组合 V  ：[视频0]        + 噪声          # 只给主视频
组合 VI ：[视频0]+[视频1]+ 噪声          # 再给待插入的 ref 视频
```

公式结构完全一样，只是「图」那一级换成「第二段视频」。所以 I 组为空、`i` 组合退化成 ∅ 组合，实际还是 4 次前向。

| | case1 (rv2v) | case2 (ads2v) |
|---|---|---|
| 源 | 1 视频 + 1 图 | 2 视频 |
| V 组 | vid0(sid=1) | vid0(sid=1) |
| VI 组 | vid0(sid=1)+img(sid=2) | vid0(sid=1)+vid1(sid=2) |
| I 组 | img(sid=1) | 空（=none） |
| 前向 | 4 | 4 |
| 含义 | 编辑视频里人物服装，用参考图替换 | 把 ref 视频内容贴到主视频屏幕上（ads 插入） |

---

## 11. 前向次数统计

rv2v_case1 = **160 次** transformer 前向：

```
rv2v 每步 4 次前向 (∅ / V / VI / VTI)
× num_inference_steps = 40       (CLI 默认，无 override)
× 单专家 (1.3B: skip_transformer_2=true, switch_dit_boundary=0)
= 4 × 40 = 160 次，全部落在 transformer_1
```

各 mode 的前向总数对比（均按 40 步）：

| guidance_mode | 每步前向 | 40 步总计 | 适用 case |
|---|---|---|---|
| t2v_apg | 2 | 80 | t2i, t2v |
| v2v / v2v_apg | 2 | 80 | i2i, v2v_case1/2/3 |
| r2v_apg | 3 | 120 | r2v, r2v_case2 |
| **rv2v** | **4** | **160** | **rv2v_case1/case2** |

补充：
1. 这 4 次前向是**串行**的 4 个独立 `shared_step` 调用，代码没把 4 个 combo 拼成一个大 batch 一次跑（VI 和 VTI 视觉输入相同、只差文本，理论上可批算但当前实现没做）。
2. 若切 **14B**（`switch_dit_boundary=0.875`），仍是 160 次，但按噪声分两段路由：高噪步(t≥875)走 transformer_1、低噪步(t<875)切到 transformer_2，并在边界处做一次 GPU↔CPU 换卡。

---

## 附录 A：NPU（Ascend 910B）适配工作日志

将 Bernini-R fork 适配到 Ascend 910B（torch_npu）的过程，已提交并推送到 `origin/main`：

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
