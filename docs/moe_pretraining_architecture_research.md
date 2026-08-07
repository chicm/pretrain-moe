# 20–30B 总参数 MoE 预训练调研与架构设计

> **文档状态**：调研与设计稿，不包含模型实现代码
> **资料截止日期**：2026-08-05
> **设计目标**：从头训练约 20–30B **总参数（total parameters）**的 decoder-only MoE 基座模型
> **第一代范围**：纯文本主干；视觉编码器和多模态联合训练不计入参数、FLOP 或训练计划
> **版本**：v0.1
>
> 除非特别说明，本文候选的 total/active 均指**不含可选 MTP 层的主干参数**。若后续启用 MTP，必须重新报告整模 total/active，不能沿用主干数字。

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [范围、定义与证据标准](#2-范围定义与证据标准)
3. [官方开放状态核验](#3-官方开放状态核验)
4. [Kimi K3 架构与可迁移性](#4-kimi-k3-架构与可迁移性)
5. [DeepSeek-V4-Flash-0731 架构与可迁移性](#5-deepseek-v4-flash-0731-架构与可迁移性)
6. [Qwen3.8-Max 已知事实与 Qwen3.5 代理](#6-qwen38-max-已知事实与-qwen35-代理)
7. [其他与目标规模相关的开放 MoE](#7-其他与目标规模相关的开放-moe)
8. [技术迁移风险分级](#8-技术迁移风险分级)
9. [本项目设计假设](#9-本项目设计假设)
10. [参数计算方法](#10-参数计算方法)
11. [候选架构](#11-候选架构)
12. [参数量明细](#12-参数量明细)
13. [训练 FLOP、KV Cache 与通信估算](#13-训练-flopkv-cache-与通信估算)
14. [并行策略与硬件映射](#14-并行策略与硬件映射)
15. [路由与训练稳定性设计](#15-路由与训练稳定性设计)
16. [优化器、精度和上下文训练策略](#16-优化器精度和上下文训练策略)
17. [分阶段实验和消融计划](#17-分阶段实验和消融计划)
18. [验收门槛与风险登记](#18-验收门槛与风险登记)
19. [最终建议](#19-最终建议)
20. [官方来源](#20-官方来源)
21. [附录 A：推荐配置的关键不变量](#附录-a推荐配置的关键不变量)
22. [附录 B：报告更新规则](#附录-b报告更新规则)

---

## 1. 执行摘要

### 1.1 核心结论

1. **Kimi K3 和 DeepSeek-V4-Flash-0731 均已开放官方权重和架构资料，但均未开放完整训练栈。**
   - Kimi K3 公开了 HF custom model definition；DeepSeek V4 公开了 reference inference。
   - 可通过 config、模型定义、报告和权重索引理解其结构。
   - 无法仅凭公开仓库复现其大规模训练效率、专家并行、定制通信和数值稳定性。

2. **截至 2026-08-05，Qwen3.8-Max 仍没有公开权重、config 或精确结构。**
   - 官方博客确认 2.4T 总参数、95B 激活参数、基于 Qwen3.5 架构基础、原生多模态和 API 可用。
   - 截至该日期，Qwen 官方 Hugging Face 中没有可公开访问的 Qwen3.8 模型。
   - 官方博客中的“下周开放权重”承诺截至该日期尚未兑现；不能把 Qwen3.5 的层数、专家数和 Top-k 写成 Qwen3.8 的事实。

3. 对第一次从 8B dense 跨到 20–30B total MoE，优先级应为：
   - **成熟路由和专家并行；**
   - **稳定训练；**
   - **通信效率；**
   - 然后才是 KDA、mHC、AttnRes 等前沿机制。

4. 本报告推荐两个生产候选（均为不含可选 MTP 的主干口径）：
   - **R-Full：25.86B total / 3.07B active**，全 GQA，作为默认首个生产模型。
   - **R-Hybrid：26.49B total / 3.70B active**，3:1 Gated DeltaNet/full attention，用于明确要求 128K 以上长上下文、且 ROCm 内核验收通过的场景。

5. 不建议第一次正式训练同时采用：
   - KDA；
   - Stable LatentMoE；
   - Block AttnRes；
   - mHC；
   - CSA/HCA；
   - FP4 从头训练；
   - Quantile Balancing；
   - auxiliary-loss-free routing；
   - 新优化器和 MTP。

   这些机制应逐项消融，否则一旦训练异常，很难归因。

### 1.2 推荐并行映射

在 15 节点、每节点 8 张 MI300X-class GPU、共 120 GPU 的假设下，首选：

| 维度 | 建议 |
|---|---:|
| Expert Parallel | 8，限制在单节点内 |
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Data Parallel | 15 |

即：

$$
EP(8) \times TP(1) \times PP(1) \times DP(15)=120
$$

在基础 `CP=1` 映射中，专家 all-to-all 留在节点内高速互联，跨节点网络主要承担数据并行梯度同步；长上下文 CP 备选见 §14.2。

---

## 2. 范围、定义与证据标准

### 2.1 目标定义

本项目中的“20–30B 模型”明确指：

> **20–30B 总参数，而不是 20–30B 激活参数。**

每个候选均单独报告：

- 总参数；
- 每 token 激活参数；
- BF16 权重体积；
- 训练 FLOP 估算；
- KV cache；
- 专家 all-to-all 逻辑流量。

### 2.2 Active parameters 的统计口径

本报告采用 MoE 文献中常见的**结构口径**：将 embedding/LM head 的整组唯一参数计入 active；tied embedding/head 只计一份，untied 时计两份。严格执行层面，input embedding 对单 token 只是行查找，并不会触碰整个矩阵；本文采用结构口径是为了与公开的 “AxB” 模型命名做近似对齐，而不是宣称这些参数都执行了同等量的 GEMM。

计入每 token 激活参数：

- embedding 和 LM head；
- 所有 attention 或 linear-attention 参数；
- dense 层；
- router；
- shared expert；
- Top-k 被选中的 routed experts。

以下内容不计入：

- 未选中的 routed experts；
- optimizer states；
- KV cache；
- 训练激活；
- 可选 MTP head；
- 视觉塔。

不同厂商对 embedding、LM head 和 MTP 的口径可能略有差异，因此官方“AxB”命名和本文计算值允许存在小幅偏差。

### 2.3 证据等级

| 等级 | 定义 |
|---|---|
| A | 官方 config、权重索引、官方模型代码可交叉核验 |
| B | 官方技术报告或官方博客明确陈述 |
| C | 根据官方配置进行的参数或系统计算 |
| D | 本项目设计建议，需实验验证 |

设计报告必须区分：

- **官方事实；**
- **根据官方配置的推导；**
- **本项目建议。**

为避免逐句重复标签，本文按章节采用以下默认等级；章节内若改变口径会单独注明：

| 章节 | 默认证据类型 |
|---|---|
| §3–§7 | 官方事实 A/B；“风险/迁移性”段落属于分析 D |
| §10、§12、§13 | 根据明确假设计算的推导 C |
| §8、§9、§11、§14–§19 | 本项目建议或待验证假设 D |

所有动态网页和仓库状态均以 2026-08-05 为观察截点；后续发布不反向改变本文的历史结论。

### 2.4 “开源”术语

本报告严格区分：

1. API 可访问；
2. 开放权重；
3. 模型/推理代码；
4. 技术报告；
5. 完整训练代码与数据；
6. OSI 意义上的开源许可证。

只有开放权重和模型定义，并不等于完整训练过程可复现。

---

## 3. 官方开放状态核验

| 模型 | 官方参数 | 权重 | 模型/推理代码 | 报告 | 完整训练栈 | 许可证 |
|---|---:|---|---|---|---|---|
| Kimi K3 | 2.78T / 104.2B active | 有，96 shards | 官方 HF custom model definition；高性能推理仍依赖专用后端 | 有 | 无 | Kimi K3 自定义许可证 |
| DeepSeek-V4-Flash-0731 | 284B / 13B active | 有，48 shards | 官方 reference inference | 有 | 无 | MIT |
| Qwen3.8-Max | 2.4T / 95B active | 截至截点未公开 | 未公开模型/推理代码；API 可用 | 官方博客 | 无 | 尚不能确认 |
| Qwen3.5-35B-A3B-Base | 官方命名 35B-A3B | 有 | Transformers 多模态外壳及其中的 text backbone 代码 | 博客/config | 无完整预训练栈 | Apache-2.0 |
| Qwen3-Next-80B-A3B | 官方命名 80B-A3B | 有 | Transformers 模型代码 | 博客/config | 无完整预训练栈 | Apache-2.0 |
| Qwen3-30B-A3B-Base | 官方命名 30B-A3B | 有 | Transformers 模型代码 | 报告/config | 无完整预训练栈 | Apache-2.0 |
| OLMoE-1B-7B | 约 7B / 1B active | 有 | 有 | 有 | 训练代码、数据和中间 checkpoint 较完整 | Apache-2.0 |

### 3.1 Kimi K3 checkpoint 核验

官方权重索引包含：

- 497,220 个 tensor key；
- 96 个权重分片；
- `metadata.total_size = 1,560,860,324,864` bytes。

checkpoint 使用 MXFP4 等打包表示，不能用文件体积直接推断逻辑参数量。

Kimi K3 License 允许广泛使用、修改和分发，但带有显著的商业触发条件。官方文本包括：

- 若过去连续 12 个月 MaaS 业务及关联方合计收入超过 20M 美元，需要向 Moonshot AI 申请额外 commercial license；
- 若基于 K3 的商业产品月活超过 100M，或单月收入超过 20M 美元，也触发额外许可要求；
- 许可证同时列出了纯内部使用、Moonshot 官方产品和认证合作方等例外。

因此它应被描述为：

> **开放权重、source-available、自定义许可证；不是无条件 OSI 开源。**

以上仅是工程选型摘要，不构成法律意见；任何商业产品在采用其权重、代码或衍生模型之前都应由法务复核原始许可证。

### 3.2 DeepSeek-V4 checkpoint 核验

官方权重索引包含：

- 72,317 个 tensor key；
- 48 个分片；
- `metadata.total_size = 166,878,536,440` bytes。

其逻辑参数为 284B，但专家权重使用 FP4，配置中还包含 FP8 量化信息，所以 checkpoint 体积远小于 BF16 的 568GB。

### 3.3 Qwen3.8-Max 状态

截至 2026-08-05，Qwen 官方 Hugging Face 作者空间中：

- `Qwen3.8`：无公开模型；
- `Qwen3.8-Max`：无公开模型；
- `Qwen3.8-Max-Base`：无公开模型；
- `Qwen3.8-Max-Instruct`：无公开模型。

官方博客写明权重将在“下周”开放；截至 2026-08-05，该承诺尚未兑现。后续更新本文时应重新检查，但在官方 config 发布之前，下列字段必须保持未知：

- 隐藏维度；
- 层数；
- 专家数；
- Top-k；
- shared expert 数；
- Gated DeltaNet 与 full attention 的具体比例；
- 是否使用 MTP；
- 视觉塔规模；
- 精度与量化格式。

---

## 4. Kimi K3 架构与可迁移性

### 4.1 官方结构事实

| 字段 | Kimi K3 |
|---|---:|
| 总参数 | 2.78T |
| 激活参数 | 104.2B |
| decoder layers | 93 |
| hidden size | 7168 |
| KDA layers | 69 |
| Gated MLA layers | 24 |
| routed experts | 896 |
| Top-k | 16 |
| shared experts | 2 |
| latent MoE dimension | 3584 |
| expert FFN | 3072 |
| dense FFN | 33792 |
| dense layers | 首层 |
| AttnRes block size | 12 |
| context | 1,048,576 |
| vision encoder | MoonViT-V2，约 401M |

### 4.2 KDA

Kimi Delta Attention 将传统无限范围的 log-decay 改为有界衰减。报告中的完整实现包含 low-rank decay projection 和受约束参数化；下面只保留用于说明有界衰减思想的简化记号，而不是可直接实现的完整公式：

$$
g = g_{\min}\cdot\mathrm{Sigmoid}(\exp(A_h)z)
$$

其中：

- `g_min = -5`；
- `A_h` 初始化为 0。

目标是把对角和非对角 tile 都转化为适合 Tensor Core 的稠密矩阵运算，兼顾线性注意力的长上下文复杂度和硬件效率。

**迁移风险：高。**

原因：

- 官方模型定义不等于完整训练 kernel；
- 高性能 backward、chunking、状态精度和 checkpoint 重计算未完整开放；
- 在 ROCm 上的 Triton/CK/AITER 支持必须单独验证；
- 不能仅凭 PyTorch reference forward 预测真实吞吐。

### 4.3 Stable LatentMoE

Stable LatentMoE 在每个 MoE 层中增加共享的 down/up projection：

$$
h\rightarrow h_{\text{latent}}\rightarrow h
$$

专家 FFN 在 latent space 中执行，从而降低单专家参数成本，并允许增加专家数和 Top-k。

优点：

- 可在固定总参数下增加专家粒度；
- 可能提高专家专业化；
- 降低单个专家的计算和存储。

风险：

- 增加每个 MoE 层固定激活的 down/up projection；
- latent bottleneck 可能损伤通用表达；
- 更细专家意味着更多路由副本和 all-to-all；
- Kimi K3 的最终 artifact 是 2.78T；报告虽包含更小规模 ablation，但没有公开证据验证本报告组合在 25B 上同样成立，不能机械等比例迁移。

### 4.4 Block AttnRes

Kimi 每隔 12 层将历史 residual candidates 与当前 prefix sum 组合：

1. 对候选 residual 分别 RMS 归一化；
2. 学习一维投影；
3. softmax 得到混合权重；
4. 加权形成新的 block residual。

优点是改善深层 residual 信息选择；代价是：

- 多 residual stream 的激活内存；
- 额外 norm 和融合；
- 对 checkpoint/recompute、PP 和图编译有影响。

对 48 层、d=2048 的中型模型，标准 pre-norm 加 depth-scaled initialization 已经更成熟。因此 AttnRes 只应作为后续消融。

### 4.5 Quantile Balancing

对 `m` 个 token、`n` 个专家、Top-k=`k`，目标专家负载：

$$
q=\frac{mk}{n}
$$

Quantile Balancing 通过每个专家 margin 的分位点更新 routing bias，在不向主模型梯度注入 auxiliary balance loss 的情况下平衡负载。

潜在优点：

- 不用辅助损失干扰语言建模目标；
- 对超大专家数可能比固定 bias 步长更稳定。

风险：

- 需要正确的全局或并行组统计；
- 分位点同步开销和数值细节未形成成熟通用实现；
- 在中型模型和较小 global batch 下行为未知。

### 4.6 其他机制

- SiTU-GLU：K3 使用带参数约束的新激活，尚缺少多规模公开复现。
- Per-Head Muon：对注意力头分别做 Muon 更新，可能改善矩阵条件，但训练基础设施复杂。
- 原生多模态：视觉/文本从头联合 next-token prediction；数据、采样和视觉 token 预算是独立大型工程。
- MXFP4/MXFP8 QAT：需要硬件、kernel 和量化训练共同成熟，不适合第一版。

---

## 5. DeepSeek-V4-Flash-0731 架构与可迁移性

### 5.1 官方结构事实

| 字段 | DeepSeek-V4-Flash-0731 |
|---|---:|
| 总参数 | 284B |
| 激活参数 | 13B |
| layers | 43 |
| hidden size | 4096 |
| routed experts | 256 |
| Top-k | 6 |
| shared experts | 1 |
| expert FFN | 2048 |
| context | 1M |
| mHC multiplier | 4 |
| MTP | 1 层配置 |
| 预训练数据 | 超过 32T tokens |

### 5.2 CSA/HCA 混合注意力

前两层使用纯滑窗；其余层交替使用：

#### Compressed Sparse Attention（CSA）

- KV 压缩率 4；
- sparse index Top-512；
- 64 个 index heads；
- index head dim 128；
- 额外滑窗 128。

#### Hierarchical Compressed Attention（HCA）

- KV 压缩率 128；
- shared-KV multi-query attention；
- 面向超长上下文的粗粒度全局信息。

报告称在 1M 上下文时，KV cache 约为其对照基线——BF16、8 KV heads、head dim 128 的 GQA——的 2%。这不是相对于标准 MHA 的口径。该收益依赖：

- 压缩和索引 kernel；
- sparse gather；
- cache layout；
- 训练和推理解码的专用实现。

因此 CSA/HCA 不适合作为首个中型 MoE 的默认注意力。

### 5.3 mHC residual

mHC 将 residual stream 扩展到 `n_hc × d`：

$$
X_{l+1}=B_lX_l+C_lF_l(A_lX_l)
$$

DeepSeek V4 使用 `n_hc=4`，并通过流形约束和 Sinkhorn 迭代稳定映射。

对 20–30B 模型的主要问题：

- residual activation 接近 4 倍；
- 通信和重计算增加；
- 需要专用 fused kernel；
- 与 MoE、长上下文和 activation checkpointing 的组合复杂。

标准 pre-norm、QK-Norm、scaled initialization 和激活 clamp 的风险更低。

### 5.4 路由

DeepSeek V4 的 router 包括：

- 前若干层 token-ID hash routing；
- 后续 score routing；
- `sqrt(softplus)` 路由激活；
- 无辅助损失 correction bias；
- 轻量 sequence-wise balance；
- loss spike 时启用 Anticipatory Routing。

这些机制对 256 个专家和大规模 EP 有针对性。对于 96 个专家、EP8 的模型，成熟的 softmax Top-k + 小权重 auxiliary loss 更容易调试。

### 5.5 优化器和稳定性

DeepSeek 使用：

- Muon：大部分二维矩阵；
- AdamW：embedding、输出头、norm、mHC 等不适合正交化的参数；
- Hybrid Newton–Schulz；
- shape-aware RMS rescaling。

训练稳定措施包括：

- SwiGLU linear branch clamp 到 `[-10,10]`；
- gate branch 上限 clamp 为 `10`；
- 动态 Anticipatory Routing；
- router 统计监控。

其中激活 clamp 成本低、实现简单、可监控，适合纳入中型模型；Muon 则应先和 AdamW 做等 FLOP 消融。

---

## 6. Qwen3.8-Max 已知事实与 Qwen3.5 代理

### 6.1 Qwen3.8-Max 官方事实

截至 2026-08-05，官方博客只足以确认：

- 2.4T total parameters；
- 95B active parameters；
- 基于 Qwen3.5 架构基础；
- 原生多模态；
- API 已开放；
- 博客当时称权重将在“下周”开放，但截至观察截止点尚未兑现。

不能确认：

- 专家数和 Top-k；
- dense/MoE 层分布；
- 注意力层比例；
- residual 结构；
- 是否继续使用 Qwen3.5 的 output gate 和 MTP；
- 精度策略。

### 6.2 Qwen3.5-35B-A3B-Base 代理配置

官方 artifact 的外层是 `Qwen3_5MoeForConditionalGeneration` 多模态模型；下表只摘取其官方 `text_config`，作为 text-only backbone 的机制代理，不把视觉塔计入本项目。

| 字段 | 值 |
|---|---:|
| hidden size | 2048 |
| layers | 40 |
| full-attention interval | 4 |
| routed experts | 256 |
| Top-k | 8 |
| shared experts | 1 |
| expert FFN | 512 |
| full Q heads | 16 |
| full KV heads | 2 |
| full head dim | 256 |
| linear key heads | 16 × 128 |
| linear value heads | 32 × 128 |
| output gate | true |
| max positions | 262,144 |
| MTP layers | 1 |

其模式为：

> 3 层 Gated DeltaNet + 1 层 gated full attention，循环排列。

Qwen3.5 的官方 config 支持：

- 3:1 混合注意力；
- gated attention output；
- 更宽的 query projection；
- shared expert；
- 1 层 MTP 配置；
- `router_aux_loss_coef=0.001` 的辅助路由损失配置。

但它不支持把 35B 几何直接外推到 Qwen3.8，也不能仅凭 config 断言训练时一定采用了何种跨设备“全局”平衡实现。

### 6.3 Qwen3-Next-80B-A3B：开放的 hybrid-attention 先例

官方配置给出：

| 字段 | 值 |
|---|---:|
| hidden size | 2048 |
| layers | 48 |
| attention pattern | 3 Gated DeltaNet + 1 full attention |
| routed experts | 512 |
| Top-k | 10 |
| shared experts | 1 |
| expert/shared FFN | 512 / 512 |
| full Q/KV heads | 16 / 2 |
| full head dim | 256 |
| linear key heads | 16 × 128 |
| linear value heads | 32 × 128 |
| partial RoPE | 25% |
| max positions | 262,144 |

它证明官方开放 artifact 中存在 d=2048 的 3:1 hybrid attention 和相应张量几何；但 512 experts 是 80B 总参数规模的选择，不能机械缩放到 25B。对本项目最可迁移的是 attention pattern 和 kernel shape，而不是专家数。

### 6.4 Qwen3-30B-A3B：最接近目标规模的控制组

官方配置：

| 字段 | 值 |
|---|---:|
| hidden size | 2048 |
| layers | 48 |
| Q heads | 32 |
| KV heads | 4 |
| head dim | 128 |
| routed experts | 128 |
| Top-k | 8 |
| expert FFN | 768 |
| shared experts | 0 |
| vocab | 151,936 |
| tied embeddings | false |

根据统一公式计算：

- 总参数约 30.53B；
- 每 token 激活参数约 3.35B。

这与官方“30B-A3B”命名一致，因此它是本项目最有价值的参数和实现 sanity check。

---

## 7. 其他与目标规模相关的开放 MoE

### 7.1 DeepSeekMoE-16B

官方 config：

- hidden size 2048；
- 28 层；
- 首层 dense；
- 64 routed experts；
- Top-6；
- 2 shared experts；
- expert FFN 1408；
- 总参数约 16.4B；
- 激活参数约 2.8B。

它为本项目提供了已公开的中型设计先例：shared experts、细粒度 routed experts、Top-6 和首层 dense 可以在同一 artifact 中成立。至于 shared expert 是否承担更多通用知识、首层 dense 是否提升稳定性，则是 DeepSeekMoE 报告的动机及本项目待验证假设，不能只由 config 做因果推断。

### 7.2 OLMoE-1B-7B

官方 config：

- hidden size 2048；
- 16 层；
- 64 experts；
- Top-8；
- expert FFN 1024；
- dropless token-choice routing。

OLMoE 的最大价值不是绝对能力，而是：

- 训练代码；
- 数据；
- 中间 checkpoint；
- 评测；
- 路由分析；
- 较完整的可复现链路。

工程实现上应优先参考 OLMoE/MegaBlocks，而不是只参考超大模型的模型定义文件。

### 7.3 Mixtral 8×7B

官方 config 使用：

- hidden size 4096、32 层；
- 8 个 experts、Top-2；
- 每专家 FFN 14336；
- 32 Q heads / 8 KV heads；
- 32K max positions。

它提供了实现和部署都较成熟的粗粒度 Top-2 先例，但其张量几何对应更高的单 token 激活参数。对本项目而言，“细粒度专家可能更适合 20–30B total / 3–4B active 预算”只是设计判断，需要与 Mixtral-like 小专家数基线做等 FLOP 对照，不能把专家专业化程度仅由 expert 数量直接推断。

### 7.4 MegaBlocks

MegaBlocks 提供 dropless MoE 思路：

- 不按固定 capacity 丢弃 token；
- 使用 block-sparse/grouped GEMM；
- 通过动态 expert token 数处理负载不均衡。

对本项目的启示：

- dropless 是目标语义；
- 必须保留 OOM 安全阈值；
- 路由质量和 grouped GEMM 效率要同时监控；
- “没有 token drop”不代表“没有负载问题”。

---

## 8. 技术迁移风险分级

| 技术 | 公开证据 | 内核成熟度 | 训练风险 | 建议 |
|---|---|---|---|---|
| 细粒度 routed experts | 多模型 | 高 | 低 | 默认采用 |
| shared expert | DeepSeek/Qwen3.5 | 高 | 低 | 默认采用 |
| 前 1–2 层 dense | DeepSeek/Kimi | 高 | 低 | 默认采用 |
| softmax Top-k + auxiliary balance | Qwen/OLMoE | 高 | 低 | 默认采用 |
| dropless routing | OLMoE/MegaBlocks | 中高 | 中低 | 默认目标 |
| QK-Norm | 多模型 | 高 | 低 | 默认采用 |
| SwiGLU clamp | DeepSeek V4 | 高 | 低 | 建议采用并监控 |
| Gated DeltaNet 3:1 | Qwen3-Next/Qwen3.5 | 中 | 中 | 内核验收后采用 |
| attention output gate | Qwen3.5 | 中高 | 中低 | 跟随 hybrid 候选 |
| MTP | DeepSeek/Qwen | 中 | 中 | 主干稳定后加入 |
| Muon | Kimi/DeepSeek | 中 | 中高 | 与 AdamW 做 A/B |
| auxiliary-loss-free bias | DeepSeek | 中 | 中 | 第二阶段消融 |
| Quantile Balancing | Kimi K3 | 低 | 高 | 研究项 |
| KDA | Kimi K3 | 低 | 高 | 研究项 |
| Stable LatentMoE | Kimi K3 | 低 | 高 | 研究项 |
| Block AttnRes | Kimi K3 | 低 | 高 | 研究项 |
| mHC | DeepSeek V4 | 低 | 高 | 不用于第一版 |
| CSA/HCA | DeepSeek V4 | 低 | 高 | 不用于第一版 |
| hash routing | DeepSeek V4 | 低 | 中高 | 不用于第一版 |
| FP4 from-scratch/QAT | Kimi/DeepSeek | 低 | 极高 | 不用于第一版 |

---

## 9. 本项目设计假设

> 本节的集群规模、既有 dense 训练记录、tokenizer/EOT、已 tokenized 语料和训练预算均为**项目提供的输入**，不是本文通过公开来源独立核验的外部事实。

团队已有一条重要的工程基线：曾在同一 120×MI300X 规模上完成约 7.602B dense、998,244,352,000 tokens 的训练。它证明数据、dense 训练和长任务运维链路可用，但 MoE 仍新增了 EP、all-to-all、grouped GEMM、router 和 expert-aware checkpoint 风险；因此第一代架构只保留有限的新机制预算。

### 9.1 Tokenizer

默认沿用现有 Qwen3 tokenizer 和数据格式：

- `vocab_size = 151,936`；
- `<|endoftext|>` / EOT id 为 `151643`；
- tokenized 数据使用 `uint32`，可直接复用；
- 默认 tied input/output embeddings；
- 不因追随 Qwen3.5 的 248K 多模态词表而重新 tokenize 全量文本数据。

如果输出头改为 untied，会增加：

$$
151,936\times 2,048=311,164,928\approx0.311B
$$

参数。

### 9.2 模态

第一代生产范围固定为 text-only base model：

- 不包含视觉编码器；
- 不进行原生多模态联合预训练；
- 可以预留特殊 token；
- 多模态作为后续独立项目。

原因是视觉数据清洗、采样比例、patch/tokenizer、vision-language packing 和联合训练稳定性会显著扩大项目范围。

### 9.3 上下文

分两种目标：

1. **主要使用 ≤32K**：优先 R-Full；起始设计采用 `rope_theta=1,000,000`、完整 RoPE，并把 32,768 作为训练与验收上限，不声称零样本外推能力。
2. **明确要求 128K–256K**：考虑 R-Hybrid；起始设计参照 Qwen3.5 text config，采用 `rope_theta=10,000,000`、full-attention 层 25% partial RoPE，并将目标 max positions 明确设为 131,072 或 262,144。它仍必须先通过 Gated DeltaNet kernel 和长上下文数据验收。

上述 RoPE 参数是本项目建议 D，不是能力保证。若后续采用 YaRN、NTK-aware scaling 或其他扩展，必须把 scaling 类型、factor、原生训练长度和目标长度写入新版本 config，并重新做 perplexity、retrieval 和长代码评测。模型声明支持长上下文，并不等于只用短序列预训练后就能获得长上下文能力。

### 9.4 数据规模

已有 1T tokens 是有意义的第一阶段，但近期前沿 MoE 的公开数据规模明显更大：

- Qwen3-Next：约 15T tokens；
- DeepSeek V4：超过 32T tokens；
- Kimi K3：报告未明确披露最终预训练 token 总量。

因此：

- 1T 可以训练出可用的第一代模型；
- 不应期待仅凭架构接近训练数据在 15T 至超过 32T 量级的前沿模型；
- 如果目标是显著提升代码、数学、知识和长上下文，建议逐步扩展到 2–5T 高质量混合数据；
- 数据质量和去重优先于机械增加 token 数。

---

## 10. 参数计算方法

### 10.1 Tied embedding

$$
P_{\text{embedding}}=Vd
$$

Untied embedding/head：

$$
P_{\text{embedding+head}}=2Vd
$$

### 10.2 标准 SwiGLU

忽略 bias：

$$
P_{\text{SwiGLU}}=3df
$$

对应 gate、up 和 down 三个线性矩阵。

### 10.3 标准 MoE 层

设：

- routed experts：`N`；
- Top-k：`k`；
- routed expert FFN：`f_e`；
- shared expert FFN：`f_s`。

总参数：

$$
P_{\text{MoE,total}}=N\cdot3df_e+3df_s+dN
$$

每 token 激活参数：

$$
P_{\text{MoE,active}}=k\cdot3df_e+3df_s+dN
$$

router 需要为所有专家打分，因此 `dN` 始终激活。

### 10.4 GQA

设 query projection width 为 `D_q`，KV width 为 `D_{kv}`：

$$
P_{\text{GQA}}=dD_q+2dD_{kv}+D_qd
$$

若采用 Qwen3.5 式 elementwise output gate，并从 Q projection 同时产生 query 和 gate：

$$
P_{\text{gated GQA}}=2dD_q+2dD_{kv}+D_qd
$$

R-Hybrid 的单层 gated GQA 为：

$$
2\times2048\times4096+2\times2048\times512+4096\times2048
=27,262,976
$$

12 层合计 327,155,712 个核心矩阵参数。

### 10.5 Gated DeltaNet 参数

设 total key/query width 为 `D_k`、total value width 为 `D_v`、value head 数为 `H_v`、short-conv kernel 为 `c`。按 Qwen3.5 实现中的 q/k/v、z、a/b、output 和 depthwise conv 核心权重：

$$
P_{\text{GDN}}
=d(2D_k+D_v)+dD_v+2dH_v+D_vd+c(2D_k+D_v)
$$

代入 `d=2048`、`D_k=2048`、`D_v=4096`、`H_v=32`、`c=4`：

$$
P_{\text{GDN}}=33,718,272
$$

36 层合计 1,213,857,792。加上 12 层 gated GQA，R-Hybrid 的 attention/linear-attention 核心权重为 1,541,013,504；每头 norm、decay bias 等小向量统一计入表中的“Norm 等”舍入项。

### 10.6 Stable LatentMoE

设 latent dimension 为 `d_l`：

$$
P_{\text{LatentMoE,total}}
=N\cdot3d_lf_e+2dd_l+3df_s+dN
$$

每 token 激活参数中将 `N` 替换为 `k`。

### 10.7 训练 FLOP 近似

使用工程规划近似：

$$
F_{\text{token}}
\approx6P_{\text{active}}+12L_{\text{full}}Sd_q
$$

其中第二项为 full attention 的 QK 和 AV 前反向复杂度。

该公式不精确包含：

- softmax；
- norm；
- router top-k；
- recurrent-state update；
- padding；
- kernel utilization；
- 通信等待。

它适合架构间比较，不应替代 profiler。

---

## 11. 候选架构

### 11.1 C0：Qwen3-30B-A3B 超界控制组

> C0 的统一核算为 30.53B，略高于 20–30B 目标上界；它只用于对照和 bring-up，不是满足预算的候选。

用途：

- 验证参数计算；
- 验证 EP、router、grouped GEMM 和 checkpoint；
- 提供最接近目标规模的官方开放基线。

| 字段 | 值 |
|---|---:|
| vocab | 151,936 |
| tied embeddings | false |
| hidden size | 2048 |
| layers | 48 |
| attention | 全 full GQA |
| Q/KV heads | 32 / 4 |
| head dim | 128 |
| routed experts | 128 |
| Top-k | 8 |
| shared experts | 0 |
| expert FFN | 768 |
| total | 30.53B |
| active | 3.35B |

它是控制组，不是最终推荐，因为：

- 总参数略高于严格 30B；
- all-to-all 路由副本更多；
- 没有 shared expert；
- 对长上下文仍是全二次复杂度。

### 11.2 R-Full：生产默认候选

#### 主干

| 字段 | 值 |
|---|---:|
| vocab | 151,936 |
| tied embeddings | true |
| hidden size | 2048 |
| layers | 48 |
| dense layers | 前 2 层 |
| dense FFN | 5504 |
| MoE layers | 46 |
| routed experts | 96 |
| Top-k | 6 |
| shared experts | 1 |
| routed/shared FFN | 896 / 896 |
| total | 25.86B |
| active | 3.07B |

#### 注意力

- 48 层 full GQA；
- 32 query heads；
- 4 KV heads；
- head dim 128；
- query width 4096；
- QK-Norm；
- 完整 RoPE，`rope_theta=1,000,000`，原生训练/验收上限 32,768；
- 第一版不强制 elementwise output gate。

#### 设计理由

Qwen3-30B：

$$
\frac{k}{N}=\frac{8}{128}=\frac{1}{16}
$$

R-Full：

$$
\frac{k}{N}=\frac{6}{96}=\frac{1}{16}
$$

在相同 global tokens/step 下，每个专家的平均 token assignment 密度保持一致。

激活 FFN 宽度：

- Qwen3：`8 × 768 = 6144`；
- R-Full：`6 × 896 + 1 × 896 = 6272`。

因此 R-Full 在降低 routed copies 和总专家容量的同时，保持了接近 Qwen3-30B 的每 token FFN 计算宽度。

#### 适用场景

- 第一次生产 MoE；
- 主要训练序列 4K–8K；
- 目标推理上下文 ≤32K；
- 优先保证 kernel、checkpoint、路由和训练稳定性。

### 11.3 R-Hybrid：长上下文候选

MoE 主干与 R-Full 相同，仅替换 attention stack。

#### Layer pattern

- 36 层 Gated DeltaNet；
- 12 层 gated full GQA；
- 每 4 层一个 full attention；
- 模式为 3 linear + 1 full。

#### Gated DeltaNet

| 字段 | 值 |
|---|---:|
| key heads | 16 |
| key head dim | 128 |
| value heads | 32 |
| value head dim | 128 |
| short convolution | kernel 4 |
| recurrent state | FP32 优先 |

#### Full attention

| 字段 | 值 |
|---|---:|
| Q heads | 16 |
| KV heads | 2 |
| head dim | 256 |
| output gate | true |
| partial RoPE | full-attention 层 25% |
| rope theta | 10,000,000 |
| target max positions | 131,072 或 262,144，二选一冻结 |

#### 参数

以下均是不含可选 MTP 的主干口径：

- total：26.49B；
- active：3.70B；
- untied LM head 时：26.80B / 4.01B。

#### 优点

- full attention 层数减少 75%；
- 长上下文训练 FLOP 显著降低；
- KV cache 只在 12 个 full layers 中线性增长；
- recurrent state 不随序列长度增长；
- Qwen3-Next/Qwen3.5 已提供公开权重、config 与模型代码，可作为结构代理证据。

#### 风险

- Gated DeltaNet 的 active projection 参数比普通 GQA 多；
- 理论 FLOP 低不代表 ROCm tok/s 高；
- recurrent state 和短卷积的 backward 需 fused kernel；
- graph compile、activation checkpointing 和 sequence packing 更复杂；
- 长序列 retrieval 能力仍需 full attention 和专用数据。

### 11.4 X-K3：研究预算包络

> 本候选只是 K3-inspired 的**预算包络**，不是已经冻结、可直接实现的 config。24.79B / 3.08B 使用 0.796B 作为 KDA+Gated-MLA attention 核心权重预算；具体 head/rank/conv 和 AttnRes 参数尚未冻结，实施前必须重新做逐 tensor 参数表。它不应作为第一次正式训练配置。

| 字段 | 值 |
|---|---:|
| hidden size | 2048 |
| layers | 48 |
| dense layers | 2 |
| KDA layers | 36 |
| Gated MLA layers | 12 |
| routed experts | 128 |
| Top-k | 8 |
| latent dimension | 1024 |
| latent expert FFN | 1280 |
| shared expert | 1 × full-space FFN 896 |
| AttnRes block | 12 |
| total | ≈24.79B（包络） |
| active | ≈3.08B（包络） |

§13.3 的 X-K3 cache 数字另采用用于容量规划的示例几何：KDA 16 heads × 128 key/value dim；Gated MLA `kv_lora_rank=256`、`rope_dim=64`。这些 cache 假设不构成冻结实现。

正确的研究顺序应为：

1. 标准 MoE + KDA；
2. 标准 attention + LatentMoE；
3. 标准 backbone + AttnRes；
4. 每项单独成立后再考虑组合。

---

## 12. 参数量明细

### 12.1 主表

主表使用 `vocab=151,936`，R/X 候选 tied embeddings；所有数值均排除可选 MTP。C0 以 † 标出，因为它超过 30B 上界；X-K3 的 attention 部分仍是预算包络：

| 组成 | C0† | R-Full | R-Hybrid | X-K3 |
|---|---:|---:|---:|---:|
| Embedding/head | 0.622B | 0.311B | 0.311B | 0.311B |
| Dense FFN | 0 | 0.068B | 0.068B | 0.068B |
| Routed expert total | 28.991B | 24.310B | 24.310B | 23.153B |
| Shared expert | 0 | 0.253B | 0.253B | 0.253B |
| Router | 0.013B | 0.009B | 0.009B | 0.012B |
| Attention/linear attention | 0.906B | 0.906B | 1.541B | 0.796B |
| Latent projections | 0 | 0 | 0 | 0.193B |
| Norm 等 | <0.001B | <0.001B | <0.001B | <0.001B |
| **总参数** | **30.53B** | **25.86B** | **26.49B** | **≈24.79B** |
| **激活参数** | **3.35B** | **3.07B** | **3.70B** | **≈3.08B** |

按本文公式集并计入已经冻结语义的可学习小向量后，worksheet checksum 为：C0 与 R-Full 各计入 48 层 Q/K RMSNorm scales（12,288）；R-Hybrid 计入 36 层 GDN 的 `dt_bias`、`A_log`、gated RMSNorm scale（6,912）和 12 层 full-attention Q/K RMSNorm scales（6,144）。这里 Q/K RMSNorm 是逐 head 应用，但每层 Q 与 K 各自只存一个跨同类 heads 共享的 `d_h` 长度 scale，故计数为 `2L_full d_h`，不再乘 Q/K head 数。

| 候选 | total | active | untied head 后 total / active |
|---|---:|---:|---:|
| C0† | 30,532,122,624 | 3,353,032,704 | 已经 untied |
| R-Full | 25,857,439,744 | 3,066,640,384 | 26,168,604,672 / 3,377,805,312 |
| R-Hybrid | 26,492,484,352 | 3,701,684,992 | 26,803,649,280 / 4,012,849,920 |
| X-K3 包络 | ≈24,786,217,536 | ≈3,080,694,336 | ≈25,097,382,464 / ≈3,391,859,264 |

这些 checksum 用于发现预算表错误。X-K3 的 attention/状态小向量尚未冻结，因此即使显示到个位也仍是近似预算；其余候选也不等同于尚未实现的最终 `state_dict` tensor count，最终实现必须从模型实例重新枚举参数。

### 12.2 Active ratio

| 候选 | Active / Total |
|---|---:|
| C0 | 11.0% |
| R-Full | 11.9% |
| R-Hybrid | 14.0% |
| X-K3 | 12.4% |

### 12.3 MTP 参数处理

MTP 不计入主表，因为“一个 MTP layer”并没有唯一 tensor geometry：它可能复用或复制 embedding/head，可能只含轻量 dense block，也可能复用/复制 sparse decoder 组件；训练时是否每 token 执行、部署时是否保留，还会改变 active 口径。

建议第一轮主干稳定后，再把 MTP 作为独立设计冻结。届时必须列出每个新增/复用 tensor 的 shape、共享关系与执行路径，重新发布“主干 + MTP”的 total/active；在这些信息缺失时不提供推测性参数区间，也不能继续把 25.86B 或 26.49B 当作整模总参数。

---

## 13. 训练 FLOP、KV Cache 与通信估算

### 13.1 理论训练 FLOP

采用 §10.7 的一阶公式和不含 MTP 的 active 参数口径。

| 候选 | GFLOP/token @4K | @8K | @32K | @128K | @256K | ZFLOP / 1T @8K |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 29.78 | 39.45 | 97.43 | 329.36 | 638.59 | 39.45 |
| R-Full | 28.06 | 37.73 | 95.71 | 327.64 | 636.88 | 37.73 |
| R-Hybrid | ≥24.63 | ≥27.04 | ≥41.54 | ≥99.52 | ≥176.83 | ≥27.04 |
| X-K3 | ≥20.30 | ≥22.11 | ≥32.98 | ≥76.47 | ≥134.45 | ≥22.11 |

说明：

- R-Hybrid 的 `6P_active` 已包含参数化 projections，但未完整展开 recurrent-state update、short conv、chunk scan 和边界通信，因此列为下界；
- X-K3 尚未完整计入 KDA recurrent update 和 AttnRes overhead；
- embedding lookup 并不等同于 dense matmul，因此 `6P_active` 是近似；
- 1T ZFLOP 行只适用于全部 token 都按 8K 公式估算的情形；混入长序列时必须按实际长度分布重算；
- 实际 wall-clock 由 kernel MFU、序列 packing 和 all-to-all 决定；
- R-Hybrid 只有在 fused kernel 达标时才会兑现理论优势。

### 13.2 BF16 权重和训练状态

仍使用不含 MTP 的主干参数；GB 为十进制 `10^9` bytes，不能与后文 GiB 直接混用。

| 候选 | BF16 权重 | 约 16 bytes/param 的完整训练状态 |
|---|---:|---:|
| C0 | 61.1GB | 488.5GB |
| R-Full | 51.7GB | 413.7GB |
| R-Hybrid | 53.0GB | 423.9GB |
| X-K3 | 49.6GB | 396.6GB |

16 bytes/param 假设：

- BF16 weight：2 bytes；
- BF16 gradient：2 bytes；
- FP32 master weight：4 bytes；
- Adam first moment：4 bytes；
- Adam second moment：4 bytes。

Muon 或 distributed optimizer 会改变这个数字。

### 13.3 KV cache

以下均为 BF16、batch size 1、单 sequence 的**推理解码 cache**；不包括训练 activations、allocator padding、paged-cache block fragmentation 或 batch 维度。

#### C0 / R-Full

每层：

$$
2\times4\times128=1024
$$

个 BF16 元素/token。

48 层合计：

- 96 KiB/token；
- 32K：约 3 GiB/sequence；
- 128K：约 12 GiB/sequence。

#### R-Hybrid

只有 12 个 full layers 保存 KV：

- 24 KiB/token；
- 32K：约 0.75 GiB；
- 128K：约 3 GiB；
- 256K：约 6 GiB。

另外 Gated DeltaNet recurrent state 约：

- BF16：约 36 MiB/sequence；
- FP32：约 72 MiB/sequence；
- 另有少量 convolution state。

#### X-K3

如果使用真正的 absorbed MLA cache：

$$
(kv\_lora\_rank+rope\_dim)\times2\text{ bytes}\times12
$$

在 `kv_rank=256`、`rope_dim=64` 时约：

- 7.5 KiB/token；
- 32K：约 0.23 GiB；
- 128K：约 0.94 GiB。

在 §11.4 的示例 16-head KDA 几何下，FP32 recurrent state 另约 36 MiB/sequence；head 数或 state layout 改变时必须重算。

如果使用普通展开式 HF attention cache，则不会获得上述 MLA cache 收益。

### 13.4 专家 all-to-all

忽略 metadata 和 padding，dispatch + combine 的逻辑流量近似：

$$
B_{\text{A2A/token}}
=2L_{\text{MoE}}k d_{\text{dispatch}}\times2\text{ bytes}
$$

标准 MoE 的 `d_dispatch=d=2048`。Stable LatentMoE 先在本地做共享 down projection，再 dispatch latent representation，因此 X-K3 使用 `d_dispatch=d_latent=1024`。

| 候选 | 逻辑流量/token | EP8 下预计远端部分 |
|---|---:|---:|
| C0 | 3.00 MiB | 2.63 MiB |
| R-Full | 2.16 MiB | 1.89 MiB |
| R-Hybrid | 2.16 MiB | 1.89 MiB |
| X-K3 | 1.44 MiB | 1.26 MiB |

EP8 均匀路由时，约 `1 - 1/8 = 87.5%` 的 routed assignments 在远端 GPU。

这不是实际网络带宽读数，因为：

- collective 会批量传输；
- 通信可与专家计算重叠；
- padding 和负载不均衡会增加流量；
- shared expert 在本地执行，不计入 routed all-to-all。

### 13.5 参数梯度同步的逻辑 payload

A2A 不是唯一通信。按 EP8、未做 TP/PP、每参数 BF16 gradient 2 bytes 的逻辑 buffer 粗估；表中 GB 为十进制：

| 候选 | local routed params/GPU | replicated non-routed params/GPU | 每 GPU gradient buffer |
|---|---:|---:|---:|
| C0† | 3.624B | 1.541B | 10.33GB |
| R-Full | 3.039B | 1.547B | 9.17GB |
| R-Hybrid | 3.039B | 2.182B | 10.44GB |
| X-K3 | ≈2.894B | ≈1.634B | ≈9.06GB |

其中 routed expert gradients 在 15 个节点的相同 local EP rank 间同步；non-routed gradients 需要覆盖全部 replica，宜先节点内归约再跨节点同步。表中是待归约 tensor 的逻辑大小，不是某种 ring/tree 算法的实际 wire bytes；实际流量还取决于 all-reduce 或 reduce-scatter、层间 overlap、gradient dtype 和 gradient accumulation。若梯度以 FP32 通信，上表 payload 约翻倍。

---

## 14. 并行策略与硬件映射

### 14.1 120 GPU 推荐映射

对 R-Full/R-Hybrid：

- EP=8；
- TP=1；
- PP=1；
- DP=15。

96 routed experts 在 EP8 下：

$$
96/8=12\text{ experts/GPU}
$$

优点：

- all-to-all 完全在单节点；
- hidden size 2048 不被 TP 切碎；
- expert FFN 896 仍是 128 的整数倍，利于 GEMM；
- 无 pipeline bubble；
- routed expert 参数只在 15 个节点上相同 local rank 之间做 expert-data-parallel 同步；
- attention、dense、router 和 shared expert 等非路由参数在 EP ranks 上也有副本，需要分层 reduce：先节点内聚合，再由对应 rank 跨节点同步，最后节点内发布。跨节点网络不承载逐层 expert token all-to-all。

### 14.2 长上下文的 Context Parallel 备选

基础训练和 ≤32K 路径保持 `EP8 × CP1 × DP15`。若 128K/256K 的 activation 或单卡计算时间不可接受，可在仍保持节点内 EP8 的前提下，把 15 个节点分解为 CP×DP：

| 场景 | EP | CP | DP | TP | PP | 总 GPU |
|---|---:|---:|---:|---:|---:|---:|
| 基础/≤32K | 8 | 1 | 15 | 1 | 1 | 120 |
| 128K pilot | 8 | 3 | 5 | 1 | 1 | 120 |
| 256K 或更大 HBM 压力 | 8 | 5 | 3 | 1 | 1 | 120 |

映射方式：EP8 始终对应单节点 8 卡；CP3/CP5 由多个节点的相同 local rank 组成，DP 使用剩余节点组。代价是 full attention 的 context communication 跨节点，Gated DeltaNet 还需要正确的 chunk boundary/prefix-state scan；这两项都必须先做 correctness 和 throughput pilot。DP 降低后用 gradient accumulation 恢复目标 global tokens/step。

CP15/DP1 只适合作为极长序列诊断，不是首选生产映射，因为数据并行度和容错余量太低。若框架没有经过验证的 CP + EP 正交实现，则宁可缩短单序列或回退 32K，也不能在主训练中临时拼接未经验证的 collectives。

### 14.3 每卡模型状态粗估

#### R-Full

- local routed params：`24.310B / 8 = 3.039B`；
- replicated non-routed params：`25.857B - 24.310B = 1.547B`；
- 本地约 4.586B 参数；
- 未做 optimizer sharding 时约 73.4GB 完整训练状态/GPU。

#### R-Hybrid

- local routed params：约 3.039B；
- replicated non-routed params：`26.492B - 24.310B = 2.182B`；
- 本地约 5.221B 参数；
- 未做 optimizer sharding 时约 83.5GB/GPU。

MI300X-class 192GB HBM 容量上可行，但还需容纳：

- activations；
- all-to-all buffers；
- grouped GEMM workspace；
- attention/GDN temporary buffers；
- checkpoint/recompute metadata。

推荐使用 distributed optimizer：routed expert states 沿对应 local-rank 的 expert-data-parallel group 分片；非路由参数沿其完整 replica group 分片。实际 process-group 定义必须与 checkpoint shard metadata 一致。

### 14.4 为什么不建议大 TP

`d=2048`、`expert_ffn=896` 下：

- TP2 已会把 expert intermediate 切到 448；
- TP4 变为 224；
- TP8 变为 112。

这些小 GEMM 难以充分利用 GPU。除非实测显存不足，否则优先 TP1。

### 14.5 EP 跨节点

只有在单节点模型状态无法容纳时才考虑 EP16 或更大。跨节点 EP 会导致：

- 每层多次 RDMA all-to-all；
- tail latency；
- 更复杂的路由 group 限制；
- 网络拥塞放大；
- 训练故障恢复更复杂。

若被迫跨节点，可研究 group-limited routing；在 EP8 单节点方案中没有必要。

### 14.6 训练框架要求

正式训练框架至少需要：

- expert parallel process groups；
- dropless dispatch/combine；
- grouped expert GEMM；
- all-to-all 与 expert compute overlap；
- distributed optimizer；
- expert-aware checkpoint；
- topology-aware rank mapping；
- 长上下文路径所需的 CP attention 与 recurrent-state prefix scan；
- router statistics；
- 可恢复的数据游标；
- BF16/FP8 数值校验。

现有 dense FSDP2 路线适合作为 dense 基线，但不能假设仅加入一个 MoE layer 类就能获得可扩展 EP。

---

## 15. 路由与训练稳定性设计

### 15.1 第一版 router

推荐：

- softmax Top-6；
- selected routing weights 归一化；
- router logits FP32；
- dropless；
- global 或 EP-group load statistics；
- auxiliary load balance；
- router z-loss；
- no token drop 作为语义目标；
- OOM safety cap 作为异常保护，而不是正常 capacity policy。

### 15.2 Auxiliary loss

起始 sweep：

| 项 | 建议范围 |
|---|---:|
| load-balance coefficient | `3e-4`, `1e-3`, `3e-3` |
| router z-loss | `1e-4`–`1e-3` |

不能只看训练 loss，应同时观察：

- expert load coefficient of variation；
- max/mean load；
- dead expert 比例；
- router entropy；
- Top-1/Top-k margin；
- expert gradient norm；
- all-to-all tail latency。

### 15.3 Shared expert

shared expert 的执行方式和设计目标：

- 每 token 都执行；
- 不通过 routed all-to-all；
- 期望承担语法、通用知识和高频模式；
- 期望降低 routed experts 之间复制通用能力的压力。

需要监控：

- shared expert 输出范数；
- routed/shared contribution ratio；
- shared expert 是否过强导致 routed experts 退化；
- shared expert 梯度是否持续高于 routed experts。

### 15.4 Dense early layers

前两层 dense 的目的：

- 稳定 token/局部模式抽取；
- 避免极早层专家按 token ID 形成脆弱分区；
- 为后续 MoE 提供更连续的表示；
- 减少 router warmup 的压力。

对第一代训练，本文把 dense early layers 视为比引入 fixed hash routing 更易诊断的保守基线；其质量收益仍需 0/1/2 dense-layer 消融验证。

### 15.5 Activation clamp

可采用 DeepSeek V4 风格的保护：

- SwiGLU linear branch clamp `[-10,10]`；
- gate branch upper clamp `10`。

但必须记录：

- clamp hit count；
- clamp hit percentile；
- 分层命中率；
- 命中率是否随训练上升。

如果长期频繁命中，说明 clamp 只是掩盖优化器、初始化或路由问题。

### 15.6 Loss-free routing 的位置

DeepSeek bias balancing 和 Kimi Quantile Balancing 的目标合理，但第一版不应作为唯一方案。

建议顺序：

1. auxiliary baseline；
2. detached correction bias；
3. auxiliary coefficient 逐步减小的对照；
4. Quantile Balancing 单独研究。

在完整 run 中途切换路由算法会改变专家分工，不建议未经 pilot 直接进行。

---

## 16. 优化器、精度和上下文训练策略

### 16.1 优化器

#### 默认

第一版使用成熟 AdamW：

- embedding/head、norm、router 和所有矩阵统一处理，便于排查；
- 具体 LR、betas 和 weight decay 通过 proxy sweep 确定；
- router bias、norm 和其他向量参数通常不做 weight decay。

#### Muon 实验

在 AdamW 基线稳定后，测试：

- 仅二维矩阵使用 Muon；
- embedding、head、norm、router bias、卷积参数使用 AdamW；
- attention 按整矩阵 Muon 与 per-head Muon 分开对照；
- expert 矩阵是否使用 Muon单独验证。

比较必须按：

- 相同训练 FLOP；
- 相同 tokens；
- 相同数据顺序；
- 相同 seed；
- 实际 wall-clock；
- optimizer state memory。

不能只看前几个 billion tokens 的 loss。

### 16.2 精度

建议阶段：

1. BF16 reference；
2. BF16 + FP32 router/state；
3. FP8 GEMM 局部启用并和 BF16 数值对齐；
4. 稳定后才扩大 FP8 覆盖；
5. 第一代模型不使用 FP4 从头训练。

关键项保持 FP32 或高精度：

- router logits/statistics；
- GDN/KDA recurrent state；
- softmax reduction；
- norm statistics；
- loss reduction。

### 16.3 学习率日程

在 token budget 未最终确定前，不冻结单一 schedule。候选：

- 约 1% linear warmup；
- WSD，便于延长训练；或
- cosine，结构简单且 Kimi K3 有官方采用证据；
- 最后 10–20% token 独立低 LR 退火。

退火期是独立收益来源，不能用中段增长放缓提前宣称模型能力已饱和。

### 16.4 上下文阶段

#### R-Full

- 主训练：4K/8K packed sequences；
- 中期：16K/32K 数据混合；
- 128K 扩展成本高，应作为独立阶段。

#### R-Hybrid

- 主训练仍以 4K/8K 为主；
- 逐步加入 32K；
- 末期按冻结后的 native target 使用少量高质量 128K **或** 256K 数据；
- 需要 retrieval、multi-document、代码仓库和长推理数据，而不是简单拼接短文档。

### 16.5 MTP

MTP 的潜在价值：

- 辅助 backbone 学习未来 token；
- 若相关模块在部署时保留，且 serving stack 实现 draft、基础模型验证、接受/拒绝与 KV-cache 回滚/提交，可作为 speculative decoding 的组成部分；
- 可能改善数据效率。

风险：

- loss 权重；
- target shift；
- serving 支持；
- 额外参数和 activation；
- 与 hybrid attention 的交互。

建议只在主干稳定后评估 MTP，不进入第一轮架构 bring-up。加入前先冻结逐 tensor geometry、loss 权重与 serving contract；仅有辅助训练 branch 不自动等于可部署的 speculative decoder。

---

## 17. 分阶段实验和消融计划

### Phase 0：单 kernel 与通信基准

目标：在模型训练之前排除基础设施风险。

测试：

- expert grouped GEMM，不同 token/expert 分布；
- EP8 all-to-all；
- dispatch/combine overlap；
- 96 experts、Top-6 的真实 shape；
- GDN forward/backward；
- BF16 与 FP32 state；
- checkpoint 保存和恢复；
- rank failure 后的完整退出。

输出：

- tokens/s；
- all-to-all 带宽；
- grouped GEMM utilization；
- HBM peak；
- 数值误差；
- 通信占 step 比例。

### Phase 1：2–4B proxy

要求尽量保留：

- expert FFN shape；
- `k/N=1/16`；
- EP8；
- d=2048 或至少保持相同 head/expert kernel shape；
- router 实现。

训练 10–30B tokens，比较：

1. dense early layers 0/1/2；
2. shared expert 0/1；
3. Top-6/96E 与 Top-8/128E；
4. auxiliary loss 系数；
5. dropless buffer 行为。

### Phase 2：注意力消融

固定 MoE 和数据：

- R-Full；
- 3:1 Gated DeltaNet/full；
- output gate on/off；
- partial RoPE 比例；
- FP32/BF16 recurrent state。

比较：

- 相同 FLOP 的 validation loss；
- 4K、8K、32K tok/s；
- 长上下文 retrieval；
- HBM；
- kernel 稳定性。

### Phase 3：优化器与路由消融

逐项测试：

- AdamW vs Muon；
- auxiliary vs detached bias；
- 标准 softmax vs `sqrt(softplus)`；
- clamp on/off；
- MTP on/off。

禁止在一个 run 中同时切换多个变量。

### Phase 4：全尺寸 stability pilot

使用最终几何训练 25–50B tokens。

必须验证：

- 无 NaN/Inf；
- 无长期 dead experts；
- checkpoint 能原样恢复；
- resume 后路由和 loss 连续；
- all-to-all 无持续退化；
- 每卡 HBM 有足够安全余量；
- 评测和训练 loss 符合 proxy scaling。

全尺寸 pilot 通过后，正式 run 应从头开始，不应把发生过架构/优化器试验的 pilot 权重直接延长为生产模型。

### Phase 5：主训练与长上下文扩展

- 第一里程碑：1T tokens；
- 数据条件允许时扩展到 2–5T；
- 最后 10–20% 做明确退火；
- 长上下文作为单独阶段和 checkpoint 系列；
- 每阶段保留可回滚的完整 optimizer checkpoint。

---

## 18. 验收门槛与风险登记

### 18.1 工程门槛

建议在正式训练前设定以下门槛：

| 指标 | 建议门槛 |
|---|---|
| token drop | 正常训练为 0 |
| all-to-all 占 step 时间 | 目标 <20–25% |
| 单专家 max/mean load | 稳态不持续超过 1.25 |
| dead experts | 不允许长期存在 |
| resume loss | 与不中断轨迹连续 |
| BF16 reference 对齐 | FP8/GDN kernel 必须通过 |
| HBM 余量 | 峰值后至少保留 10–15% |
| kernel hang/timeout | 长时间 soak test 中为 0 |

阈值需要结合实际 global batch 和实现调整，但必须在训练开始前显式定义。

### 18.2 主要风险

| 风险 | 后果 | 缓解措施 |
|---|---|---|
| all-to-all 成为瓶颈 | active FLOP 低但 tok/s 差 | EP8 单节点、overlap、减少 Top-k |
| 专家负载失衡 | OOM、尾延迟、dead experts | auxiliary loss、全局统计、dropless guard |
| 小 expert GEMM 低效 | GPU 利用率低 | TP1、f=896、grouped GEMM |
| GDN ROCm kernel 不成熟 | 理论省 FLOP但更慢/不稳定 | R-Full fallback、先做 fused backward benchmark |
| 路由算法过多 | 训练异常无法归因 | 单变量消融 |
| Muon 超参数不适配 | loss spike 或收敛退化 | AdamW 基线、矩阵类别白名单 |
| 长上下文数据不足 | 形式支持但能力差 | 专用 long-context 数据阶段 |
| 1T 数据不足以匹配前沿 | 能力上限低于预期 | 扩展到 2–5T，提升代码/数学/多语种质量 |
| checkpoint/EP 恢复错误 | 长 run 损失 | expert-aware 原子 checkpoint 与 resume soak test |
| FP4/FP8 数值错误 | 隐性质量损失 | BF16 reference、逐模块启用 |
| 多模态范围膨胀 | 主项目失控 | 第一代 text-first，多模态另立阶段 |

### 18.3 需要在架构冻结前确认或记录的事项

1. 目标上下文是 32K、128K 还是 256K？
2. 推理目标是单请求延迟、批量吞吐还是长上下文服务？
3. 正式 token budget 是 1T、2–5T 还是更高？
4. 第一代原生多模态范围已关闭；若未来重启，必须另立参数、数据和训练计划。
5. 训练框架选择及 ROCm expert-parallel 支持程度如何？
6. 第一代已冻结 tied embedding；untied head 只有在受控消融证明收益后才重新讨论，并增加 0.311B total/active。
7. 是否需要 MTP/speculative decoding？
8. 目标训练周期和允许的 kernel 开发时间是多少？

---

## 19. 最终建议

### 19.1 默认生产选择

如果主要上下文不超过 32K：

> **选择 R-Full：25.86B total / 3.07B active（不含可选 MTP）。**

核心配置：

- d=2048；
- 48 层；
- 前 2 层 dense，FFN 5504；
- 后 46 层 MoE；
- 96 routed experts，Top-6；
- 1 shared expert；
- expert/shared FFN 896；
- 32Q/4KV、head dim 128 的 full GQA；
- QK-Norm、RMSNorm、SwiGLU；
- softmax routing、small auxiliary balance、router z-loss、dropless；
- BF16；
- EP8/TP1/PP1/DP15。

### 19.2 长上下文选择

如果目标明确为 128K–256K，并且 GDN ROCm pilot 达标：

> **选择 R-Hybrid：26.49B total / 3.70B active（不含可选 MTP）。**

采用：

- 36 Gated DeltaNet；
- 12 gated full GQA；
- 3:1 pattern；
- 在 128K 与 256K 之间冻结一个 native max-position target；
- FP32 recurrent state；
- 专用长上下文数据阶段；
- HBM 或单卡计算时间超标时采用 §14.2 的 CP3/DP5 或 CP5/DP3 pilot 映射。

若 kernel 吞吐、数值或恢复不达标，立即回退 R-Full，不在生产 run 中边训练边修 attention 栈。

### 19.3 研究路线

K3/DeepSeek V4 技术按以下顺序研究：

1. auxiliary-loss-free bias；
2. Muon；
3. lightweight MTP；
4. KDA；
5. LatentMoE；
6. AttnRes；
7. CSA/HCA 或 mHC。

每项先在 proxy 中单独成立，不能把单点超大模型结果直接视为中型模型定论。

### 19.4 数据结论

继续使用现有 Qwen3 tokenizer 和已 tokenized 语料，避免无必要的全量重新分词。1T 是第一代模型的合理里程碑，但架构不能替代数据规模：若目标是接近近期先进 MoE，应规划更高质量的 2–5T 数据，而不是单纯增加专家数。

---

## 20. 官方来源

以下链接均按 2026-08-05 的官方页面/仓库核验。Qwen 博客是动态 SPA，长期复核应同时保存带访问日期的内部快照；架构数字优先以可版本化的官方 config、模型代码和报告为准。

### 20.1 Kimi K3

- GitHub：<https://github.com/MoonshotAI/Kimi-K3>
- Hugging Face：<https://huggingface.co/moonshotai/Kimi-K3>
- Config：<https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json>
- Kimi K3 Technical Report（官方仓库 PDF）：<https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf>
- License：<https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/LICENSE>
- 官方 API 文档：<https://platform.kimi.ai/docs/guide/kimi-k3-quickstart>

### 20.2 DeepSeek V4

- Hugging Face：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- Config：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json>
- Technical report（官方模型卡指向）：<https://arxiv.org/abs/2606.19348>
- Reference inference：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference>
- License：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE>

### 20.3 Qwen

- Qwen3.8-Max 官方博客：<https://qwen.ai/blog?id=qwen3.8>
- Qwen3.5 官方博客：<https://qwen.ai/blog?id=qwen3.5>
- Qwen3.5-35B-A3B-Base：<https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base>
- Qwen3.5-35B config：<https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base/raw/main/config.json>
- Qwen3-Next 官方博客：<https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd>
- Qwen3-Next-80B-A3B：<https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>
- Qwen3-Next config：<https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json>
- Qwen3-30B-A3B-Base：<https://huggingface.co/Qwen/Qwen3-30B-A3B-Base>
- Qwen3-30B config：<https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/raw/main/config.json>

### 20.4 其他 MoE 与基础方法

- DeepSeekMoE paper：<https://arxiv.org/abs/2401.06066>
- DeepSeekMoE-16B config：<https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/raw/main/config.json>
- OLMoE GitHub：<https://github.com/allenai/OLMoE>
- OLMoE paper：<https://arxiv.org/abs/2409.02060>
- OLMoE config：<https://huggingface.co/allenai/OLMoE-1B-7B-0924/raw/main/config.json>
- Mixtral-8x7B-v0.1：<https://huggingface.co/mistralai/Mixtral-8x7B-v0.1>
- Mixtral config：<https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json>
- MegaBlocks：<https://github.com/databricks/megablocks>
- MegaBlocks paper：<https://arxiv.org/abs/2211.15841>
- ST-MoE：<https://arxiv.org/abs/2202.08906>
- Switch Transformer：<https://arxiv.org/abs/2101.03961>
- GShard：<https://arxiv.org/abs/2006.16668>

---

## 附录 A：推荐配置的关键不变量

为了降低消融和扩缩容时的混淆，优先保持：

1. `hidden_size=2048`；
2. routed expert FFN 维度是高效 GEMM 的整数倍；
3. `k/N=1/16`；
4. active routed FFN width 约 5.4K–6.1K；
5. shared expert active width约 896；
6. EP8 单节点；
7. TP1；
8. router FP32；
9. dropless；
10. 每次实验只改变一个主要机制。

## 附录 B：报告更新规则

当 Qwen3.8-Max 权重发布后，应首先更新：

1. 开放状态和许可证；
2. 官方 config；
3. 层数、hidden size、专家数和 Top-k；
4. attention pattern；
5. MTP 和多模态配置；
6. 权重索引和逻辑参数核算；
7. 本报告中以 Qwen3.5 为代理的所有段落。

在官方 config 出现之前，不使用第三方量化仓库、反向工程猜测或 API 定价信息填补未知结构。
