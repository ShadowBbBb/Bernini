## 🤖 Agent Workspace & Workflow Protocol (Agent 初始化必读)

**【System Context / 系统上下文】**
你当前运行在一台**本地计算机**上，目标是协助开发 `bytedance/Bernini` 相关的代码。
本项目的核心算力（GPU、大模型环境）位于**远程云服务器**。本地计算机**无法通过 SSH 直接连接**远程服务器。
因此，我们采用基于 GitHub Fork 的非对称开发工作流。

**【Architecture / 架构说明】**

| 节点 | 环境描述 | 核心任务 | 硬件状态 |
|---|---|---|---|
| **Local (Here)** | 你的工作区 | 编写逻辑、修改代码、本地 Mock 测试 | 无强力 GPU (仅 CPU 测试) |
| **GitHub** | 中转枢纽 | 托管 Fork 的仓库，进行版本控制 | 仅限代码，禁止大文件 |
| **Server** | 目标执行区 | 拉取 GitHub 最新代码，执行真实训练/推理 | 拥有 GPU 及完整模型权重 |

**【Agent Standard Operating Procedure (SOP) / 标准开发流程】**

1. **Understand (理解需求)**: 明确当前要新增的 Feature 或修复的 Bug。
2. **Code (编写代码)**: 在本地直接修改仓库代码。务必注意设备兼容性（如 `device='cuda' if torch.cuda.is_available() else 'cpu'`）。
3. **Mock Test (本地沙盒测试)**: 使用微型数据集或 Dummy Tensors 在本地跑通 CPU 测试，确保无基础语法错误、张量维度匹配。
4. **Commit & Push (提交中转)**: 
   - 确认无误后，执行 `git add .` (注意排除大文件)
   - 编写规范的 Commit message
   - 执行 `git push origin main`（或当前工作分支）
5. **Wait for Feedback (等待反馈)**: 人类开发者将在云服务器执行 `git pull` 并运行真实环境测试。如果报错，人类会将云端报错日志发送给你，你需要根据日志进行 Debug 并重复上述流程。

**【🚫 STRICT RULES / 核心红线 (绝对禁止违反)】**

- **RULE 1: 严禁提交大型文件。** 任何情况下，都不允许将模型权重 (`*.safetensors`, `*.pt`, `*.bin`, `*.pth`)、大型数据集 push 到 GitHub。在执行 git 操作前，**必须**确保这些文件已被 `.gitignore` 规则拦截。
- **RULE 2: 隔离本地与远程路径。** 绝对不要在代码中硬编码本地计算机的绝对路径。请使用相对路径，或通过环境变量 (`os.environ.get()`) 来动态读取数据和模型路径。
- **RULE 3: 防御性硬件调用。** 由于本地没有 GPU 算力支持，严禁在初始化或测试代码中强制使用 `.cuda()`。所有 `.to('cuda')` 的调用必须有 Fallback 机制，否则本地测试将直接崩溃。
- **RULE 4: 依赖一致性。** 若在开发中引入了新的 Python 库，必须同步更新 `requirements.txt`，以便远程服务器更新环境。