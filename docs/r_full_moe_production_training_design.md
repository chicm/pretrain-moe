# R-Full 25.86B MoE：120×MI300X 生产预训练设计

> 文档状态：**生产设计基线 v0.1，尚未完成 ROCm 端实现验收，禁止据此直接启动 1T 正式训练**
> 设计日期：2026-08-09
> 修订状态：corrected-design / implementation-pending / launch-blocked
> 审计基线 SHA-256：`e410a07b8916c58b6c0d981b834cb2bbef6365184aa99a2764b2a2fd82c6125a`
> 适用范围：从零预训练、纯文本、BF16、15 节点 × 8 张 MI300X、约 1T 训练 token
> 模型口径：25.857439744B 总参数，3.066640384B 单 token 激活参数；不含 MTP、视觉塔和任何推理量化副本

本文把 [MoE 架构调研报告](./moe_pretraining_architecture_research.md) 中的 R-Full 候选收敛为一份可实施、可审计、可恢复的生产训练设计。它不是新的前沿结构提案，也不是把教学代码直接扩展到 120 张 GPU；教学实现的边界参见 [Dense-to-MoE 系统教程](./moe_from_dense_to_modern_moe_tutorial.md)。

本文中的“冻结”表示正式 1T 作业开始后不得再变；“启动阻断项”表示在获得实测证据前不能声称生产就绪。所有实现、环境和数据清单都必须生成哈希并写入 checkpoint。若 pilot 迫使架构、路由语义、优化器或数据计划发生变化，应提升设计版本并从头开始正式 token 计数，不能在同一次正式训练中静默切换。

---

## 目录

1. [执行摘要与冻结决策](#executive-summary)
2. [范围、目标与非目标](#scope)
3. [R-Full 冻结配置](#frozen-spec)
4. [逐张量模型结构](#tensor-graph)
5. [精确参数账本](#parameter-ledger)
6. [路由、负载均衡与数值稳定](#router-stability)
7. [精度、初始化、优化器与学习率](#optimization)
8. [既有 1T 语料的复用](#data-reuse)
9. [样本调度器与无快进恢复](#data-scheduler)
10. [上下文课程与精确 token 计划](#sequence-plan)
11. [Megatron-Core/Megatron-LM 实现边界](#megatron)
12. [120×MI300X 并行组](#parallel-layout)
13. [ROCm 内核与环境锁](#rocm-stack)
14. [显存、计算与通信预算](#budgets)
15. [Checkpoint、恢复与保留策略](#checkpointing)
16. [可观测性与告警](#observability)
17. [验证、评测与发布判据](#validation)
18. [分阶段上线计划](#rollout)
19. [故障处理与运行纪律](#failure-handling)
20. [长上下文独立路径](#long-context)
21. [机器可读配置与四阶段工件](#resolved-config)
22. [启动与阶段切换 Runbook](#runbook)
23. [最终验收矩阵](#acceptance-matrix)
24. [未关闭的启动阻断项](#blockers)
25. [参考资料](#references)

---

<a id="executive-summary"></a>

## 1. 执行摘要与冻结决策

### 1.1 一页结论

| 项目 | 冻结结论 |
|---|---|
| 生产候选 | R-Full；普通全注意力 GQA，不采用混合线性注意力 |
| 模型宽深 | `d_model=2048`，48 层 |
| 层类型 | 第 0、1 层为 Dense SwiGLU；第 2–47 层为稀疏 MoE |
| 注意力 | 32 个 Q 头、4 个 KV 头、`head_dim=128`，全头维 RoPE，QK-RMSNorm |
| Dense FFN | 中间宽度 5504 |
| MoE | 96 个 routed experts，Top-6，1 个始终激活的 shared expert；专家宽度均为 896 |
| 路由 | FP32 logits；先在 logits 上 Top-6，再仅对入选 logits 做 softmax；dropless |
| 稳定项 | 标准辅助负载均衡、router z-loss、QK-Norm、受限 SwiGLU、全局梯度裁剪 |
| 词表 | tokenizer 原生词表 `151669`；模型 padded rows `151936`；EOT `151643`；输入 embedding 与 LM head 严格共享 |
| 参数量 | `25,857,439,744` total；`3,066,640,384` per-token active unique union |
| 训练精度 | BF16 参数；FP32 router compute、loss/reduction、gradient buffer 与 reduce-scatter；FP32 AdamW states；BF16 parameter all-gather |
| 生产框架 | Megatron-Core/Megatron-LM；FSDP2 只保留作参考与小规模正确性路径 |
| 初始并行 | `TP=1, PP=1, CP=1, EP=8`；EP 组固定在单节点 8 张 GPU 内 |
| 数据并行语义 | 120 条普通 dense-data lanes；每个 expert shard 的 expert-data-parallel 复制数为 15 |
| 单卡专家 | 每个 MoE 层每张 GPU 持有 12 个 routed experts |
| 语料 | 复用既有八源 provisional indexed payload；`938,008,272,770` 是待 manifest 复算的 reported count，不宣称 unique documents/tokens |
| 训练预算 | `254313` 个 successful updates；stop/replay、无 committed skip；`999,999,406,080` 个已提交 target tokens |
| 上下文课程 | 约 80% @4K、15% @8K、4% @16K、1% @32K |
| 全局 token batch | 每步固定 3,932,160 tokens；不同长度仅改变 sequence batch 和累积次数 |
| 优化器 | AdamW，峰值 LR `2.0e-4`，WSD：约 10B warmup、至约 900B 稳定、最后约 100B cosine decay |
| Checkpoint | Megatron distributed checkpoint；模型、优化器、scheduler、RNG、数据 ID 与拓扑共同保存 |
| 首版排除 | MTP、FP8/FP4、DeepSeek V4 新路由、Stable LatentMoE、AttnRes、KDA/MLA、CSA/HCA、mHC、视觉 |

### 1.2 为什么不是“25.86B 模型已经可以直接训练”

模型几何和参数账本已经冻结，但生产系统仍有四类必须以 MI300X 实测关闭的风险：

1. **ROCm grouped GEMM**：不能让 12 个本地专家退化为 Python 循环或 12 次低效小 GEMM。
2. **EP All-to-All**：必须证明 RCCL 在节点内的 dispatch/combine 正确、稳定且不会跨节点绕行。
3. **长序列注意力与词表损失**：32K 阶段不得构造平方注意力矩阵，也不应长期物化完整 FP32 `[S,V]` logits。
4. **分布式 checkpoint**：expert ownership、distributed optimizer shard 和数据位置必须能原样恢复。

因此，正式训练的定义不是“能启动 120 个进程”，而是：所有 120 张 GPU 有真实计算与显存占用、EP/EDP 组被打印并核对、首个优化器更新成功、checkpoint-resume 等价测试通过、2K-step burn-in 达到本文阈值。

### 1.3 设计中的强制分层

- **架构层**：冻结 R-Full，不在框架迁移期间叠加研究特性。
- **语义层**：路由、归一化、shared expert 加法和 loss 系数都有单一含义。
- **系统层**：Megatron process groups、内核、数据顺序、checkpoint 都必须显式。
- **运营层**：监控的是 token、专家负载、通信、GPU 和可恢复性，而不只是训练 loss。
- **研究层**：DeepSeek/Kimi 机制只能从稳定 baseline checkpoint 分叉做隔离消融。

---

<a id="scope"></a>

## 2. 范围、目标与非目标

### 2.1 目标

1. 在 15×8 MI300X 上，从随机初始化完成一条近似十进制 1T token 的纯文本预训练。
2. 模型总参数严格落在 20–30B 目标内，并把 active 参数、训练 FLOP、训练状态和通信分别核算。
3. 使用成熟的 full-attention + sparse-MoE baseline，为后续长上下文与新路由提供可信对照组。
4. 复用已经生成的 Qwen3 tokenizer token 文件，避免重新清洗、分词和写入数百 TB 中间数据。
5. 在节点故障、作业重启和阶段切换后，不做与已消费样本数成正比的数据快进。
6. 输出可以被审计的模型配置、数据 manifest、环境 lock、checkpoint manifest 和评测记录。

### 2.2 首版明确不做

- 不训练图像、音频或视频塔，不声明原生多模态。
- 不加入 MTP；因此 MTP 参数、loss 和推理收益均不进入本模型口径。
- 不启用 FP8、FP4、MXFP、量化感知训练或低精度 optimizer state。
- 不加入 Gated DeltaNet、KDA、MLA、CSA、HCA、mHC 或滑窗注意力。
- 不加入 Stable LatentMoE、AttnRes 或额外 latent projection。
- 不使用 DeepSeek V4 的 `sqrt(softplus)`、hash/anticipatory routing 或 sigmoid gate。
- 不使用 token dropping 作为吞吐优化手段。
- 不声称 32K 以外的外推能力；128K–256K 是独立项目。
- 不把现有 `DenseDecoderLM`、`ReferenceSparseMoE` 或 FSDP2 教学代码称为生产实现。

### 2.3 成功标准的优先级

优先级从高到低为：

1. 数学和分布式正确性；
2. 可恢复性；
3. 数值稳定与数据一致性；
4. 可观测性；
5. 吞吐与利用率；
6. 研究性收益。

任何为了吞吐而改变路由语义、丢 token、跳过 checkpoint 状态或隐藏 fallback 的做法都不接受。

---

<a id="frozen-spec"></a>

## 3. R-Full 冻结配置

### 3.1 模型主配置

| 字段 | 值 | 说明 |
|---|---:|---|
| tokenizer native vocab | `151669` | payload ID 合法上界，不等于 embedding rows |
| model padded vocab | `151936` | 必须 explicit override 并断言 embedding/head shape |
| `hidden_size` | 2048 | 残差流宽度 |
| `num_layers` | 48 | Transformer blocks |
| `dense_layers` | `[0,1]` | 0-based；只有前两层是 Dense FFN |
| `moe_layers` | `[2,...,47]` | 共 46 层 |
| `num_attention_heads` | 32 | Q 头数 |
| `num_query_groups` | 4 | KV 头数 |
| `kv_channels` | 128 | 单头维度 |
| Q projection width | 4096 | `32×128`，大于 residual width |
| K/V projection width | 各 512 | `4×128` |
| `ffn_hidden_size` | 5504 | Dense SwiGLU |
| `num_experts` | 96 | routed experts，不含 shared expert |
| `moe_router_topk` | 6 | 每 token 选 6 个 routed experts |
| `moe_ffn_hidden_size` | 896 | routed expert 宽度 |
| shared expert count | 1 | 每个 MoE 层一个 |
| shared expert width | 896 | 与 routed expert 相同 |
| norm | RMSNorm | pre-norm，`eps=1e-6` |
| QK-Norm | RMSNorm | Q/K 各一个共享的 128 维 scale 向量/层 |
| position | full RoPE | 覆盖全部 128 维头维度 |
| `rope_theta` | 1,000,000 | baseline 原生训练最大长度 32768 |
| `max_position_embeddings` | 32768 | continuation 只能走单独、受测的 checkpoint-argument override |
| linear bias | false | 所有主线性层与 router 无 bias |
| embedding tie | true | 输入 embedding 与 LM head 同一 Parameter |
| hidden/attention dropout | 0 | 基础预训练不使用 dropout |
| attention | causal full GQA | 无 sliding window |

### 3.2 结构不变量

实现必须逐项断言：

- `96 % 8 == 0`，每个 EP rank 恰有 12 个 routed experts。
- Dense 层恰为 2 层，MoE 层恰为 46 层；不能把第 0 层改为 MoE。
- shared expert 不进入 routed dispatch，不占 96 个 router logits 中的槽位。
- Q/K RMSNorm scale 的形状是 `[128]` 和 `[128]`，不是每头一套 `[H,128]`。
- tokenizer 原生词表固定为 `151669`，model rows 固定为 `151936`；二者不可混为同一字段。
- LM head 与 token embedding 指向同一参数存储；state dict 不能重复持有一份独立权重。
- Top-6 的 6 个 expert ID 对同一 token 不重复。
- 所有正常训练 token 都得到 6 个 routed outputs 和 1 个 shared output。
- 模型不含 learned absolute position embedding、MTP head 或额外 output bias。

### 3.3 首版激活函数

Dense、routed 和 shared FFN 使用同一受限 SwiGLU。对归一化后的输入 `x`：

$$
\begin{aligned}
g &= W_{\mathrm{gate}}x,\
u &= W_{\mathrm{up}}x,\
\tilde g &= \min(g, 10),\
\tilde u &= \mathrm{clip}(u,-10,10),\
\mathrm{FFN}(x) &= W_{\mathrm{down}}\left(\mathrm{SiLU}(\tilde g)\odot \tilde u\right).
\end{aligned}
$$

注意：gate 分支只做上界裁剪，不做对称下界裁剪；linear/up 分支做 `[-10,10]` 对称裁剪。若内核无法保持这一精确语义，则不能以普通 SwiGLU 静默替代。每层、每分支的命中率必须被监控。

---

<a id="tensor-graph"></a>

## 4. 逐张量模型结构

### 4.1 输入、标签与输出

输入 token IDs 的形状为 `[B,S]`，标签是同一 source-local token stream 向右移动一位后的 `[B,S]`。每个样本从 token 文件读取 `S+1` 个 `uint32` ID，并产生 `S` 个训练 target tokens。

- 不注入 BOS。
- 原始 EOT `151643` 保留并参与预测；EOT 后第一个 token 也参与 CE；baseline 不在 EOT 重置 attention/position。
- 每个样本读取同一 physical shard 的 `[a,a+S+1)`；input 为 `[a,a+S)`、label 为 `[a+1,a+S+1)`。
- stride 固定为 `S`，相邻窗口只共享一个 payload ID、label spans 不重叠；禁止跨 physical shard 拼接。
- 不在样本尾人工 padding。
- 训练 loss 对全部 `S` 个标签求全局 token 加权平均。
- 输出投影使用共享矩阵 `E∈R^{151936×2048}`：`logits=hE^T`。

32K 时完整 BF16 logits 约为 9.96 GB/序列，完整 FP32 logits 约为 19.91 GB/序列。因此生产 loss 路径必须验证 fused/chunked linear cross entropy；不允许由于“显存尚能勉强容纳”而默认长期物化完整 FP32 logits。

### 4.2 每个 Transformer block

设层输入为 `h_l`。所有层都采用相同的 pre-norm 残差拓扑：

$$
\begin{aligned}
a_l &= h_l + \mathrm{Attention}_l\left(\mathrm{RMSNorm}_{l,\mathrm{attn}}(h_l)\right),\
h_{l+1} &= a_l + \mathrm{FFNOrMoE}_l\left(\mathrm{RMSNorm}_{l,\mathrm{ffn}}(a_l)\right).
\end{aligned}
$$

不得把它改成 post-norm、parallel-attention-MLP 或多残差流。

### 4.3 GQA 注意力张量

对归一化后的 `x∈R^{S×2048}`：

| 张量/权重 | 逻辑形状 | 参数数/层 |
|---|---:|---:|
| `W_q` | `[4096,2048]` | 8,388,608 |
| `W_k` | `[512,2048]` | 1,048,576 |
| `W_v` | `[512,2048]` | 1,048,576 |
| `W_o` | `[2048,4096]` | 8,388,608 |
| Q RMS scale | `[128]` | 128 |
| K RMS scale | `[128]` | 128 |

运行顺序冻结为：

1. 线性投影产生 32 个 Q heads 和 4 个 K/V heads。
2. 分别按最后一个 128 维 head channel 对 Q、K 做 RMSNorm；scale 在头之间共享。
3. 对归一化后的 Q、K 应用全 head-dimension RoPE。
4. 每 8 个 Q heads 共享一组 K/V。
5. 使用缩放 `1/sqrt(128)`、严格 causal mask 和 full attention。
6. 拼接 32 个 Q outputs，经 `W_o` 回到 2048。

QK-Norm 必须位于 RoPE 之前。attention kernel 必须是线性显存的 fused/flash 路径；任何显式生成 `[B,H,S,S]` 矩阵的 fallback 都是启动阻断错误。

### 4.4 Dense FFN 层

第 0、1 层的第二个子层是单个 Dense SwiGLU：

| 权重 | 逻辑形状 |
|---|---:|
| `W_gate` | `[5504,2048]` |
| `W_up` | `[5504,2048]` |
| `W_down` | `[2048,5504]` |

每层参数为 `3×2048×5504=33,816,576`。

### 4.5 MoE 层

第 2–47 层接收 FFN pre-norm 后的 `x`，包含：

- 一个 `W_router∈R^{96×2048}`；
- 96 个 routed experts，每个是宽度 896 的受限 SwiGLU；
- 一个宽度 896 的 shared expert；
- shared expert 输出直接与 gated routed sum 相加，无额外 learned gate、无固定 `1/2` 缩放。

第 `l` 层的输出为：

$$
y_l(x_t)=E_{l,\mathrm{shared}}(x_t)+\sum_{i\in I_{l,t}}g_{l,t,i}E_{l,i}(x_t),
$$

其中 `|I_{l,t}|=6` 且 `sum_i g_{l,t,i}=1`。shared expert 本地执行，不经过 EP All-to-All；routed experts 必须先路由、再仅计算被选中的 token-expert pairs。

### 4.6 最终归一化

48 个 blocks 后再施加一个 2048 维 RMSNorm，然后进入共享 LM head。不存在额外 decoder output projection。

---

<a id="parameter-ledger"></a>

## 5. 精确参数账本

### 5.1 基本公式

对 bias-free SwiGLU expert：

$$
P_{\mathrm{expert}}=3df_e.
$$

每个 MoE 层的总参数为：

$$
P_{\mathrm{MoE,total/layer}}=N\cdot 3df_e+3df_s+dN.
$$

单 token 的 active routed expert 参数只计 Top-k：

$$
P_{\mathrm{routed,active}}=L_{\mathrm{MoE}}\cdot k\cdot 3df_e.
$$

### 5.2 分项精确值

| 组件 | 计算 | 总参数 | active 参数口径 |
|---|---:|---:|---:|
| 共享 embedding / LM head | `151936×2048` | 311,164,928 | 311,164,928 |
| 48 层 GQA projections | `48×18,874,368` | 905,969,664 | 905,969,664 |
| 2 层 Dense FFN | `2×3×2048×5504` | 67,633,152 | 67,633,152 |
| 46×96 routed experts | `46×96×3×2048×896` | 24,310,185,984 | — |
| Top-6 routed experts | `46×6×3×2048×896` | — | 1,519,386,624 |
| 46 个 shared experts | `46×3×2048×896` | 253,231,104 | 253,231,104 |
| 46 个 routers | `46×2048×96` | 9,043,968 | 9,043,968 |
| block RMSNorm scales | `48×2×2048` | 196,608 | 196,608 |
| final RMSNorm scale | `2048` | 2,048 | 2,048 |
| Q/K RMSNorm scales | `48×2×128` | 12,288 | 12,288 |
| **合计** |  | **25,857,439,744** | **3,066,640,384** |

### 5.3 12,288 参数差异的最终处理

先前报告中的 `25,857,427,456 / 3,066,628,096` 没有显式计入 Q/K RMSNorm 的 12,288 个 scale 参数。只要 R-Full 保留本文定义的 QK-Norm，生产账本就必须使用：

- **总参数：25,857,439,744**；
- **单 token active 参数：3,066,640,384**。这里 active 是单 token 触达的 unique parameter union：绑权重 embedding/head 只计一次，六个 routed experts 与一个 shared expert 均计入；不是 per-rank resident、batch-union 或 operation-weighted count。

差异恰为 `48×2×128=12,288`，不是随机误差。实现若得到旧值，说明 QK-Norm 没有真正注册参数；若得到更大的值，首先检查是否误用了 per-head scales、untied head 或 bias。

### 5.4 明确排除的参数

以下均不在上述合计中：

- MTP 模块；
- optimizer state、master weights、梯度与通信 buffer；
- RoPE cache、attention mask、KV cache；
- 视觉或音频 encoder；
- untied output embedding；
- 任何实验性 shared-expert gate；
- 评测/推理量化副本。

如果解除权重共享，需额外增加恰好 311,164,928 个参数，不能继续沿用 25.85744B 标签。

### 5.5 EP8 下每张 GPU 的逻辑参数

routed experts 被 EP8 均分，其余参数在普通模型副本中存在：

$$
\begin{aligned}
P_{\mathrm{nonrouted}} &= 25,857,439,744-24,310,185,984\
&=1,547,253,760,\
P_{\mathrm{local}} &=1,547,253,760+\frac{24,310,185,984}{8}\
&=4,586,027,008.
\end{aligned}
$$

这只是逻辑参数 ownership；distributed optimizer 会进一步切分 master weights 和 moments，但不会改变模型参数总数。

---

<a id="router-stability"></a>

## 6. 路由、负载均衡与数值稳定

### 6.1 冻结的 router 语义

对 MoE 层输入 `x_t`，先在 FP32 中计算：

$$
z_t=\mathrm{float32}(W_r x_t).
$$

然后：

$$
I_t=\mathrm{TopK}(z_t,6),
$$

并且只在选中的 6 个 logits 上归一化：

$$
g_{t,i}=\frac{\exp(z_{t,i}-m_t)}{\sum_{j\in I_t}\exp(z_{t,j}-m_t)},\quad i\in I_t,
$$

其中 `m_t=max_{j∈I_t}z_{t,j}`。

这对应 Megatron-Core 0.12 语义中的：

- `moe_router_score_function=softmax`；
- `moe_router_pre_softmax=false`；
- Top-k 后对 selected logits 做 softmax。

它**不是**以下任何一种：

- 对全部 96 个 logits 做 global softmax 后直接取未重归一化权重；
- sigmoid routing；
- `sqrt(softplus)` routing；
- Top-1 selected-softmax。

Top-6 selected-softmax 对主任务具有非零 router 梯度；教学教程中 Top-1 selected-softmax 的零主任务梯度角落情形不适用于本配置，但仍必须保留辅助 loss。

### 6.2 dropless 的精确定义

正常路径设置：

- `capacity_factor=null`；
- 不 padding 到固定 expert capacity；
- 不因专家过载丢弃 token；
- 不把溢出 token 送到默认专家；
- 不将 shared expert 当成 routed overflow handler。

生产保护不是“悄悄 drop”：若 dispatcher 预测本步 buffer 会超出已验证上限，应在执行 optimizer update 前使该步失败并保留上一个已提交 checkpoint。任何 emergency cap 被触发都视作训练正确性事件，不能把这批数据记作普通成功 step。

### 6.3 辅助负载均衡

前向 routing gate 与辅助统计分开定义。对完整 96-way logits 做 global softmax：

$$
p_{t,i}=\frac{\exp(z_{t,i})}{\sum_{j=1}^{N}\exp(z_{t,j})}.
$$

在一个 EP8 group 内，令 $T$ 为该组参与统计的 token 总数，$n_i$ 为 expert $i$ 被 Top-k 选中的总次数，$f_i=n_i/(kT)$，并令

$$
P_i=\frac{1}{T}\sum_t p_{t,i}.
$$

冻结目标为

$$
L_{\mathrm{aux}}=N\sum_{i=1}^{N}f_iP_i.
\qquad \alpha_{\mathrm{aux}}=10^{-3}\text{ 仅在总目标中乘一次。}
$$

这要求项目补丁在 EP group 内聚合完整 `n_i` 与 probability sums；只对各 rank 已算出的 scalar aux loss 做平均**不等价**。`n_i` 可 detach，`P_i` 的聚合必须保持 router 梯度，且必须处理后续 dense-DP 梯度平均的 EP 缩放。验收 oracle：固定同一组 logits/indices，EP1 reference 与 EP8 在 scalar loss、每个 expert 的 count/probability sum、router-weight gradient 上逐项一致。MCore 0.12 的 stock rank-local `switch_load_balancing_loss_func` 不满足此契约，因此 `AUX-001` 在补丁与梯度 parity 测试通过前是启动 blocker。

### 6.4 router z-loss

每层使用：

$$
L_z=\frac{1}{T}\sum_t\left(\log\sum_{i=1}^{N}\exp z_{t,i}\right)^2,
$$

系数 `λ_z=1.0e-4`。z-loss 在 FP32 计算，用于抑制 logits 无界漂移。必须分别记录未经系数缩放的 `L_z`、乘系数后的贡献以及 `max_abs_logit`。

### 6.5 总训练目标

$$
L_{\mathrm{train}}=L_{\mathrm{CE}}+
10^{-3}\sum_{l=2}^{47}L_{\mathrm{aux},l}+
10^{-4}\sum_{l=2}^{47}L_{z,l}.
$$

对外 perplexity 只由 `L_CE` 计算，不能把 auxiliary 或 z-loss 混入。训练日志同时报告三项和 total loss。

### 6.6 shared expert 语义

shared expert：

- 对每个 token 运行；
- 输入与 routed experts 相同，均为 FFN pre-norm 后的 2048 维向量；
- 输出不乘 router gate；
- 不经过 EP dispatch；
- 参数属于普通 dense-data-parallel 同步域，而不是 expert-data-parallel shard；
- 第一版不启用 shared-expert overlap，以便先验证数值等价；通过后才可作为纯系统优化打开。

应监控 `||y_shared||/||y_routed||` 的层分布。若比例异常，先排查初始化、clamp、归约和 router，而不是在正式训练中临时加入缩放。

### 6.7 专家负载指标

每层至少记录：

- 96 个专家 assignment counts；
- mean、std、CV、min/mean、max/mean；
- 零负载专家数；
- router probability entropy；
- selected gate entropy、最大 gate 和最小 gate；
- Top-1 占比与 Top-6 overlap 检查；
- 每 expert 输入 token 数和 grouped-GEMM padding 数；
- dispatch 与 combine 字节数、时间和重试数。

本计划使每张 GPU 每个 optimizer step 处理 32768 个本地 target tokens。一个 EP8 group 因此看到 262144 个 tokens、1,572,864 个 token-expert assignments；均匀时每个 expert 每步约 16,384 个 assignments。这一统计量在 4K/8K/16K/32K 阶段保持相同，有利于比较负载稳定性。

### 6.8 初始告警阈值

阈值在 EP8 与两节点 pilot 后只能整体版本化调整，不能在正式训练中逐步放宽：

| 指标 | warning | hard action |
|---|---:|---:|
| layer `max_load/mean_load`，100-step EMA | >1.35 | >1.75 持续 20 steps 停止 |
| layer `min_load/mean_load`，100-step EMA | <0.70 | <0.40 持续 20 steps 停止 |
| 零负载 expert | 任意层任意 step | 连续 3 steps 立即停止 |
| assignment CV，100-step EMA | >0.15 | >0.25 持续 20 steps 停止 |
| router `max_abs_logit` | >20 | >40 或非有限立即停止 |
| gate clamp 命中率 | >0.1% | >1% 持续 100 steps 停止 |
| up/linear clamp 命中率 | >0.1% | >1% 持续 100 steps 停止 |

阈值的目的不是要求每个 microbatch 完全均匀，而是及时发现失衡趋势、符号错误、重复 expert ID、统计域错误和 kernel corruption。


---

<a id="optimization"></a>

## 7. 精度、初始化、优化器、计数器与 LR 状态机

### 7.1 数值与 optimizer contract

| item | frozen value |
|---|---|
| model parameters | BF16 |
| router parameters | BF16；projection/logits/softmax/aux/z-loss 在 FP32 |
| gradient buffer / accumulation | FP32，`grad_reduce_in_fp32=true` |
| reduce-scatter wire dtype | FP32 |
| parameter all-gather wire dtype | BF16 |
| optimizer | AdamW，`beta1=0.9`、`beta2=0.95`、`eps=1e-8`、weight decay `0.1`、global grad clip `1.0` |
| loss scale | none；由自定义 finite consensus 负责 fail-closed |
| TF32 | disabled |

common 参数在 dense-DP120 分片 optimizer state；routed expert 参数在 EDP15 分片。任何框架 default 与本表不同都必须在 compiled argv 中显式覆盖。

### 7.2 初始化

- base linear/embedding：`Normal(0,0.02)`；
- residual output projection：`Normal(0,0.02/sqrt(2×48)) = Normal(0,0.0020412414523193153)`；
- router：`Normal(0,0.01)`、无 bias；MCore 0.12 没有独立 stock router-init knob，必须项目 patch 并逐 tensor 验证；
- RMSNorm/QK-RMSNorm scales：1；
- 不 zero-init residual，不使用 μP。

### 7.3 规范化计数器

本文只保留一套可执行语义：**立即停机、原批重放，不允许任何数值失败被提交或跳过**。

| 字段 | 精确定义 | 是否驱动数据/LR |
|---|---|---:|
| `successful_updates` | 已完成有限值检查、optimizer update 和逻辑提交的更新数；本文的 `step` 即它 | 是 |
| `update_tokens` | 已成功更新的 target tokens；baseline 恒等于 `successful_updates × 3,932,160` | 是 |
| `seen_tokens` | 已原子提交的数据 target tokens；本策略下恒等于 `update_tokens` | 是 |
| `attempted_batches_total` | append-only telemetry 中实际发起的 batch 尝试数，含失败与重放 | 否 |
| `failed_attempts_total` | 未提交的 data/numerical/runtime 尝试数 | 否 |

固定约束：

```text
seen_tokens == update_tokens == successful_updates * 3,932,160
0 <= successful_updates <= 254,313
```

阶段、LR、checkpoint、validation 与 eval cadence 均由 `successful_updates`/`update_tokens` 驱动。失败尝试不得推进这些字段，也不得推进 source cycle、sample key 或 stage。没有 `skipped_updates`、`skipped_tokens` 或“可接受 skip 比例”。

### 7.4 一次更新的事务边界

设当前已提交更新数为 `u0`，则下一批的规范 batch key 由 `u=u0+1` 唯一决定：

1. 根据 `u`、stage manifest 和本节冻结的 scheduler 生成 batch；读取数据本身不提交 cursor。
2. forward 中累积 loss/router/activation finite flags；所有 rank 先做一次 world consensus，避免某个 rank 提前抛异常造成 collective hang。
3. 若 forward 有 poison，立即写 append-only failure record，所有 rank 退出；不 backward、不更新、不提交数据。
4. backward 与梯度归约后，对 loss、router flags、所有 local gradients、global grad norm 做 finite scan 和 world consensus；失败则不调用 optimizer，立即退出。
5. 全局裁剪后执行 optimizer update；随后扫描更新后的 BF16 参数与本 rank 持有的 FP32 master/m/v shard。若发现 poison，禁止写 `COMMITTED`，终止进程；已触碰的内存状态随进程丢弃。
6. 只有所有检查通过，才将 `successful_updates` 设为 `u`，并原子提交 `seen_tokens/update_tokens`、scheduler state 与派生 data position。

任何停止后都从最后一个带 `COMMITTED` marker 的 checkpoint 恢复，因此同一 batch key 会被重放。若故障可复现，训练保持停止等待修复；不得通过丢弃该样本、移除 source 或重归一化权重来“解卡”。data/index/hash/short-read 错误也执行同一 fail-closed 规则。

这一区分两种原子性：进程内 optimizer kernel 不能廉价回滚，但**可接受 lineage 的持久状态**绝不包含失败更新；恢复会丢弃最后 durable checkpoint 之后的全部内存进度并确定性重放。

### 7.5 预更新 LR 索引

冻结 `successful_update` 为 1-based 预更新索引。尝试 update `u` 时，先用 `lr(u)` 完成该次 optimizer update；成功后 `successful_updates=u`。峰值 LR 为 `2.0e-4`，末值为 `2.0e-5`：

$$
\mathrm{lr}(u)=
\begin{cases}
2.0\times10^{-4}\cdot u/2543, & 1\le u\le2543,\
2.0\times10^{-4}, & 2544\le u\le228881,\
2.0\times10^{-5}+\frac{2.0\times10^{-4}-2.0\times10^{-5}}{2}
\left[1+\cos\left(\pi\frac{u-228881}{254313-228881}\right)\right],
& 228882\le u\le254313.
\end{cases}
$$

边界 oracle：

- `lr(1) = 2e-4 / 2543`；
- `lr(2543) = lr(2544) = lr(228881) = 2e-4`；
- `lr(254313) = 2e-5`。

scheduler checkpoint 必须保存 `schedule_version=rfull-token-wsd-v1`、`successful_updates` 与 `update_tokens`。stock MCore sample-based scheduler 只有在逐点向量测试与上式一致时才可复用，否则使用项目 token-scheduler 补丁。

### 7.6 cadence 与最终完成条件

- 每 2,000 个 `successful_updates` 保存 full recovery checkpoint；阶段边界和 final 强制额外保存。
- validation/eval milestone 同样按已提交 update 触发。
- 目标完成条件是 `successful_updates=254313` 且 `seen_tokens=update_tokens=999,999,406,080`。
- 任一未提交 attempt 只存在于原始 telemetry；它不会改变最终 token 预算。

---

<a id="data-reuse"></a>

## 8. 数据资产、tokenizer、packing 与 holdout

### 8.1 tokenizer 与 payload 冻结契约

必须同时区分两种 vocabulary：

| 项 | 冻结值 |
|---|---:|
| tokenizer native vocabulary | `151669` |
| model padded vocabulary / embedding rows | `151936` |
| EOT `<|endoftext|>` | `151643` |
| payload dtype | little-endian `uint32` |

硬约束：

- tokenizer identifier 固定为 `Qwen/Qwen3-8B`，但必须进一步写入 immutable revision/commit 与 tokenizer artifact SHA-256；未填即 `TOK-001` blocker；
- `add_special_tokens=false`、`add_bos=false`、不自动追加 EOS；文档分隔只使用 payload 中显式的 EOT `151643`；
- 禁止 EOS/zero fallback；`max(payload_id) < 151669`；
- 模型必须显式得到 `padded_vocab_size=151936`。因为 stock MCore 从 native `151669` 以默认 128 对齐只会得到 `151680`，不能靠默认 padding 推导本设计；需要受测的 explicit override/adapter；
- 输入 embedding 与 LM head 绑权重，行数必须为 `151936`，从而参数数为 `151936×2048=311,164,928`。

### 8.2 当前 indexed payload inventory

下表是当前仓库审计得到的 **indexed payload token count**，不是已具备密码学 provenance 的最终 manifest，也不再称作“unique document tokens”。在 `corpus_manifest_sha256` 生成前，它们是需要复算的设计输入。

| source | indexed payload tokens | target weight | exact planned target tokens | target / payload |
|---|---:|---:|---:|---:|
| DCLM | 320,239,478,022 | 35.40% | 353,999,781,888 | 1.1054× |
| FineWeb-Edu | 376,937,913,261 | 18.00% | 179,999,895,552 | 0.4775× |
| FinePDFs | 70,201,495,725 | 5.00% | 49,999,970,304 | 0.7122× |
| FinePhrase | 101,050,225,690 | 20.00% | 199,999,881,216 | 1.9792× |
| code | 39,527,688,034 | 15.00% | 149,999,910,912 | 3.7948× |
| FineMath | 19,734,484,408 | 3.12% | 31,199,969,280 | 1.5810× |
| InfiMath | 7,790,815,324 | 2.16% | 21,599,993,856 | 2.7725× |
| OWM | 2,526,172,306 | 1.32% | 13,200,003,072 | 5.2253× |
| **合计** | **938,008,272,770** | **100%** | **999,999,406,080** | **1.0661×** |

logical `uint32` payload alone is `3,752,033,091,080` bytes = `3.75203309108 TB` = `3.4124542172 TiB`, before index、document sidecar、filesystem/object overhead、cache 与 replica。target/payload 大于 1 只说明规划中的 window exposure/cycling，不证明逐文档无重复。

### 8.3 corpus manifest schema 与 canonical hash

每个 physical shard 的 manifest 至少包含：

```text
manifest_version, source_id, shard_id, uri/logical_path,
byte_size, dtype, endianness, token_count, max_token_id,
eot_id, eot_count, payload_sha256, index_sha256,
document_sidecar_sha256, tokenizer_artifact_sha256,
created_by_commit, source_order
```

canonical JSON 固定为 UTF-8、无 BOM、LF、key 按 Unicode code point 排序、无多余空白，且只允许 integer/string/bool/null；`manifest_sha256` 字段在哈希时省略。SHA-256 对 canonical bytes 计算。所有 rank 启动时必须得到相同 root hash，并逐 shard 校验 size/hash/count。任一 source 缺失、损坏、short read 或 hash 不匹配都全局停机；绝不删除 source 后重归一化其余权重。

当前 blocker：

```yaml
corpus_manifest_sha256: TBD-BLOCKER
tokenizer_artifact_sha256: TBD-BLOCKER
```

### 8.4 packing 与 physical-shard 语义

对长度为 `n` 的 physical shard、训练长度 `S`，候选窗口数严格为

$$
N_{\mathrm{win}}=\left\lfloor\frac{n-1}{S}\right\rfloor.
$$

第 `q` 个 aligned window 读取 `[qS,qS+S+1)`。禁止跨 shard；每个 shard 都重置 stride alignment。尾部不足 `S+1` 个 payload ID 的 transition 不进入 baseline。物理 shard 不是文档边界，历史约 1B-token shard 可能切断文档；因此 corpus manifest 必须提供 document sidecar 才能声称 document-level holdout。

EOT 是普通 CE target：不屏蔽 EOT，不屏蔽 EOT 后 token，不在 EOT 重置 attention/position。任何未来的 EOD-mask/reset 实验都必须是新 recipe，不能静默改变 baseline。

### 8.5 冻结 holdout 生成算法

每个 source 生成 **256 个 32K master windows**。算法版本为 `rfull-holdout-v1`：

1. 从 manifest 枚举所有满足 `[a,a+32769)` 的 aligned candidates，并用 document sidecar 找出其触达的完整 `doc_hash` 集合。
2. candidate key 为 `SHA256(domain || seed_u64_le || source_id || shard_id || a_u64_le)`；按 `(digest, shard_id, a)` 升序。
3. 依序扫描 candidate；只有其 token 区间与已选窗口不重叠，且其触达的 `doc_hash` 集合与当前 holdout set 不相交时才选中。每选一个，就把其触达的所有 `doc_hash` 加入 holdout set；直到得到 256 个 token-disjoint 且 document-disjoint 的 master windows。若候选耗尽仍不足 256 个，manifest 生成失败。
4. 训练 sample map 排除 holdout set 中文档的**全部 fragments**，包括跨 physical-shard 的 fragments；若 sidecar 无法证明这一点，manifest 生成失败。
5. 每个 master window 固定拆成 `8×4K`、`4×8K`、`2×16K`、`1×32K` 子窗口。所有坐标均为 0-based half-open payload intervals，并显式保存 input/label span。

由此每种长度都评估相同的 `67,108,864` 个有效 target tokens：

| suite | samples/source | global samples | global target tokens | pad-to-120 rows（loss mask=0） |
|---|---:|---:|---:|---:|
| 4K | 2,048 | 16,384 | 67,108,864 | 56 |
| 8K | 1,024 | 8,192 | 67,108,864 | 88 |
| 16K | 512 | 4,096 | 67,108,864 | 104 |
| 32K | 256 | 2,048 | 67,108,864 | 112 |

分布式 validation 按全局 sample index round-robin 到 batch-DP lanes；为 collective shape 对齐而补的 rows 必须使用全零 loss mask，不能复制并重复计数真实样本。

manifest 至少保存 selected doc hashes、master/subwindow coordinates、source ID、padding flags、tokenizer/corpus roots、selection seed/algorithm version 与 artifact hashes：

```yaml
holdout_manifest_sha256: TBD-BLOCKER
```

未生成并校验该文件前，只能跑诊断 validation，不能通过 launch gate。

---

<a id="data-scheduler"></a>

## 9. 无历史快进的数据调度器

### 9.1 canonical global sequence ordinal

数据位置不依赖 rank 或 world size。对 stage 内第 `v` 个 0-based successful update、该 update 内第 `q∈[0,G)` 个 global sequence slot：

$$
j=vG+q.
$$

runtime 再把 `q` 映射到当前 batch-DP lane、microbatch 与 accumulation slot。只要 `G` 不变，改变 world size/GA 不改变该 update 的 canonical sample set。baseline CP1 时，120 个 data lanes 与 GA8/4/2/1 分别实现 `G=960/480/240/120`。

下一位置只由 `successful_updates` 推导；fetch 不修改 cursor。checkpoint 仍保存 stage、stage-local update、`successful_updates`、`seen_tokens/update_tokens` 和 next-batch digest，用于恢复断言。

### 9.2 source schedule：10,000-entry exact cycle

source quota 固定为：

```text
[3540, 1800, 500, 2000, 1500, 312, 216, 132]
```

顺序对应 DCLM、FineWeb-Edu、FinePDFs、FinePhrase、code、FineMath、InfiMath、OWM，和为 10,000。每个完整 cycle 建立包含上述重复数的 canonical source-ID list；每个位置的排序 key 为

```text
SHA256("rfull-source-cycle-v1\0" || seed_u64_le ||
       stage_id_u32_le || cycle_id_u64_le || position_u32_le)
```

按 `(digest, original_position)` 排序。tail 先按 largest remainder 分配整数 quota，再用同一 hash-sort。禁止 Python `hash()`、未固定库 RNG 或 modulo-biased shuffle。

stage source quotas 已精确冻结：

| stage | DCLM | FineWeb-Edu | FinePDFs | FinePhrase | code | FineMath | InfiMath | OWM | total sequences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 69,140,788 | 35,156,333 | 9,765,648 | 39,062,592 | 29,296,944 | 6,093,764 | 4,218,760 | 2,578,131 | 195,312,960 |
| 8K | 6,481,938 | 3,295,901 | 915,528 | 3,662,112 | 2,746,584 | 571,290 | 395,508 | 241,699 | 18,310,560 |
| 16K | 864,298 | 439,474 | 122,076 | 488,304 | 366,228 | 76,175 | 52,737 | 32,228 | 2,441,520 |
| 32K | 107,984 | 54,907 | 15,252 | 61,008 | 45,756 | 9,517 | 6,589 | 4,027 | 305,040 |
| **all** | **76,595,008** | **38,946,615** | **10,818,504** | **43,274,016** | **32,455,512** | **6,750,746** | **4,673,594** | **2,856,085** | **216,370,080** |

对应 exact target-token matrix：

| stage | DCLM | FineWeb-Edu | FinePDFs | FinePhrase | code | FineMath | InfiMath | OWM | total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 283,200,667,648 | 144,000,339,968 | 40,000,094,208 | 160,000,376,832 | 120,000,282,624 | 24,960,057,344 | 17,280,040,960 | 10,560,024,576 | 800,001,884,160 |
| 8K | 53,100,036,096 | 27,000,020,992 | 7,500,005,376 | 30,000,021,504 | 22,500,016,128 | 4,680,007,680 | 3,240,001,536 | 1,979,998,208 | 150,000,107,520 |
| 16K | 14,160,658,432 | 7,200,342,016 | 2,000,093,184 | 8,000,372,736 | 6,000,279,552 | 1,248,051,200 | 864,043,008 | 528,023,552 | 40,001,863,680 |
| 32K | 3,538,419,712 | 1,799,192,576 | 499,777,536 | 1,999,110,144 | 1,499,332,608 | 311,853,056 | 215,908,352 | 131,956,736 | 9,995,550,720 |

> 上表 16K/32K 行由机器账本在最终校验时生成；任何人工抄写与 stage total 不一致都必须失败。规范真相源是 `stage_manifest.json` 与 accounting script，而不是 Markdown 手录值。

### 9.3 source-local window permutation

对每个 stage/source，manifest 固定 eligible shard/range 列表与每 shard 的 `N_win`。source occurrence ordinal `m` 先分解为 `pass_id` 与 pass 内 offset；每个 pass：

- shard order 使用 `SHA256("rfull-shard-order-v1\0" || seed || stage || source || pass || shard_id)` 排序；
- shard 内用 affine bijection `perm(i)=(a·i+b) mod N_win`；`a` 从 SHA-256 派生后递增到 `gcd(a,N_win)=1`，`b` 由独立 domain hash 派生；
- prefix counts 用二分查找定位 shard，复杂度为“重建一个 10K source cycle + `O(log n_shards)`”，不随历史 consumed sequences 增长；当前 cycle 可缓存。

stage manifest 必须显式记录 first-pass disjoint ranges、wrap boundary 与每次重复；不能用“epoch”一词掩盖 source exhaustion/cycling。

### 9.4 stage manifest 与恢复 oracle

```yaml
stage_manifest_sha256: TBD-BLOCKER
scheduler_algorithm: rfull-hash-affine-v1
```

manifest 包含 stage table、source quotas、eligible ranges、window counts、hash domains/seeds、largest-remainder tie-break、holdout exclusions 与 corpus/tokenizer roots。必须通过：

1. 同一 manifest 在不同 Python 进程/机器上产生相同前 10K、边界与随机抽样 digest；
2. 在 1M、100M 与最终 update 附近从 O(1) state 直接恢复，sample key 与连续运行一致；
3. 改 world size/GA 但保持 `G`，canonical batch sample multiset 不变；
4. 任意 corrupt/missing/short-read 注入触发全局停机且 next batch key 不变；
5. 4K→8K→16K→32K 边界的最后/第一批无 overlap，且所有 quota 与本节机器账本一致。

---

<a id="sequence-plan"></a>

## 10. 序列长度课程与精确 token 预算

### 10.1 frozen stage table

每个成功更新固定 `3,932,160` target tokens：

| stage | seq len | global sequences / update | GA（baseline CP1） | successful updates | samples | target tokens |
|---|---:|---:|---:|---:|---:|---:|
| A | 4,096 | 960 | 8 | 203,451 | 195,312,960 | 800,001,884,160 |
| B | 8,192 | 480 | 4 | 38,147 | 18,310,560 | 150,000,107,520 |
| C | 16,384 | 240 | 2 | 10,173 | 2,441,520 | 40,001,863,680 |
| D | 32,768 | 120 | 1 | 2,542 | 305,040 | 9,995,550,720 |
| **合计** | — | — | — | **254,313** | **216,370,080** | **999,999,406,080** |

离十进制 1T 只差 `593,920` tokens；禁止通过隐式 partial batch 补齐。

### 10.2 cumulative boundaries

| boundary | cumulative successful updates | cumulative samples | cumulative target tokens |
|---|---:|---:|---:|
| end 4K | 203,451 | 195,312,960 | 800,001,884,160 |
| end 8K | 241,598 | 213,623,520 | 950,001,991,680 |
| end 16K | 251,771 | 216,065,040 | 990,003,855,360 |
| final 32K | 254,313 | 216,370,080 | 999,999,406,080 |

边界 update 使用旧 stage 的长度完成并提交；其 post-update checkpoint 写入 completed stage。下一个 update 才切换长度。stage 切换不重置 optimizer、moments、weight decay 或 LR。

### 10.3 exactness and failure semantics

由于本设计采用 stop/replay 而不是 committed skip，accepted canonical lineage 的 `seen_tokens` 与本表**逐项相等**。失败尝试可增加 wall-clock 与 `attempted_batches_total`，但不会改变 token budget、source quota 或 stage boundary。任一最终 checkpoint 若计数不满足本表，应视为损坏而不是“近似完成”。

---

<a id="megatron"></a>

## 11. Megatron-Core/Megatron-LM 实现边界

### 11.1 选择理由

R-Full 的生产难点不是把 25.86B 参数“包一层分片”，而是同时正确处理：

- expert parallel process groups；
- token permute/dispatch/All-to-All/combine；
- grouped expert GEMM；
- expert-data-parallel gradient synchronization；
- distributed optimizer；
- expert-aware distributed checkpoint；
- 与 TP/PP/CP 可组合的 rank topology。

因此生产路径使用 Megatron-Core/Megatron-LM。现有 FSDP2 路径继续承担 Dense 模型、缩小版 MoE 和数值 reference 测试，但不承担正式 120-GPU R-Full。

### 11.2 已审计 API 与真正的生产锁

已对官方 Megatron-LM release 0.12 对应 commit `d580efc68a9f0dbf1945f834f6f6200cd01d3343` 做过 API 审计，确认它暴露本文需要的配置概念，包括：

- `expert_model_parallel_size`；
- 自定义 `moe_layer_freq`；
- `moe_ffn_hidden_size`；
- `moe_shared_expert_intermediate_size`；
- `moe_router_topk`、router dtype、aux/z-loss；
- `alltoall` dispatcher；
- grouped GEMM、permute fusion 和 MoE recompute；
- distributed optimizer/checkpoint。

这个 SHA 是**语义参考点，不是 MI300X 生产可执行环境已经锁定的声明**。正式环境必须是一个经过 ROCm pilot 的项目 fork commit，并在 `environment.lock.json` 中记录 exact SHA、patch set 和容器 digest。若最终基于更新版 MCore，必须重新跑全部语义/恢复测试。

### 11.3 需要项目实现或封装的部分

不假设上游开箱即用。项目侧至少需要以下明确边界：

1. **R-Full layer spec**：前两层 Dense、后 46 层 MoE；QK-RMSNorm 的 scale sharing 与本文一致。
2. **受限 SwiGLU**：三类 FFN 共享同一 clamp 语义和统计 hook。
3. **ROCm grouped GEMM backend**：通过 MCore 接口接入已验证的 TE/CK/AITER/Triton 路径之一。
4. **fused/chunked CE**：避免 16K/32K 的完整 FP32 vocab logits 常驻。
5. **全局 sample-ID dataset**：读取既有 raw `uint32` shards，不依赖旧 iterator fast-forward。
6. **checkpoint metadata extension**：保存数据/拓扑/环境 hashes 和 expert ownership。
7. **监控 hooks**：逐层 router、clamp、dispatcher 和 kernel-fallback 指标。
8. **配置验证器**：启动前重算参数量、process groups、batch arithmetic 和 source quotas。

这些 patch 应尽量局部化；不能复制一份难以追踪的 Megatron 核心代码后直接修改。

### 11.4 MCore 语义映射

下表是解析后配置语义，不保证未经项目 entrypoint 转换即可逐字作为命令行复制：

| R-Full 语义 | MCore/Megatron 配置意图 |
|---|---|
| 48 layers | `num_layers=48` |
| 2048 hidden | `hidden_size=2048` |
| 32Q/4KV×128 | `num_attention_heads=32`, `num_query_groups=4`, `kv_channels=128`, GQA on |
| Dense 5504 | `ffn_hidden_size=5504`, SwiGLU on |
| 96 experts | `num_experts=96` |
| Top-6 | `moe_router_topk=6` |
| expert width 896 | `moe_ffn_hidden_size=896` |
| one shared expert width 896 | `moe_shared_expert_intermediate_size=896`, shared gate off |
| first two Dense | `moe_layer_freq=[0,0]+[1]*46` |
| selected-logit softmax | `score_function=softmax`, `router_pre_softmax=false` |
| FP32 router compute | `moe_router_dtype=fp32`；router model parameter 仍为 BF16 |
| aux loss | EP-global project patch；stock rank-local `aux_loss` 仅作 API evidence；coefficient `1e-3` |
| z-loss | coefficient `1e-4` |
| dropless | no capacity factor; no pad-to-capacity |
| EP8 | `expert_model_parallel_size=8` |
| All-to-All | `moe_token_dispatcher_type=alltoall` |
| bias-free | disable linear bias |
| tied embedding | do not enable untied embeddings/output weights |
| RMSNorm | RMSNorm, `norm_epsilon=1e-6` |
| QK-Norm | QK layernorm feature with RMSNorm implementation |
| full RoPE | RoPE, rotary percent 1.0, base 1,000,000 |
| 32K ceiling | `max_position_embeddings=32768` |

配置解析后必须输出 JSON；验收脚本对 JSON 而不是 shell 字符串做断言。

### 11.5 第一版明确关闭的 MCore 优化

以下优化只有在 baseline 数值一致和 kernel trace 验收后才能逐个打开：

- shared-expert overlap；
- DeepEP/flex dispatcher；
- router fusion 改变精度的路径；
- async checkpoint；
- gradient/parameter communication overlap；
- FP8 expert GEMM；
- expert tensor parallel；
- pipeline interleaving；
- sequence parallel（TP1 下没有必要）。

`moe_grouped_gemm` 是生产性能所必需，但最初正确性测试必须同时保留一个仅供小规模对照的逐 expert reference path。正式训练发现退回 reference loop 时应 fail，而不是继续低效运行。

### 11.6 梯度同步规则

- attention、Dense FFN、shared experts、routers、norms、embedding/head：在 120-rank dense-DP 语义下同步。
- routed expert `e`：只在持有同一 global expert IDs 的 15 个 expert-DP replicas 之间同步。
- 一个 rank 不能把自己持有的 12 个不同 experts 的梯度混为一个“共享专家”。
- distributed optimizer 对 dense 与 expert 参数使用各自正确的 sharding group；
- explicit `padded_vocab_size=151936` override，而不是从 native vocab 默认推导；
- router `Normal(0,0.01)` 初始化；
- EP-global、autograd-aware auxiliary loss；
- exact limited-SwiGLU 覆盖 Dense、routed grouped-GEMM 与 shared-expert 三条路径；
- world-consensus finite scan + stop/replay transaction；
- token-indexed LR/data scheduler、immutable manifests 与 canonical telemetry lineage。
- 在两节点 pilot 中，必须逐参数比较 EP8×EDP2 与等价 reference 的 gradients。

---

<a id="parallel-layout"></a>

## 12. 120×MI300X 并行组

### 12.1 初始维度

正式 4K–32K 基线：

| 维度 | 大小 |
|---|---:|
| world size | 120 |
| nodes × GPUs/node | 15×8 |
| TP | 1 |
| PP | 1 |
| CP | 1 |
| EP | 8 |
| ordinary dense data lanes | 120 |
| expert-data-parallel replicas | 15 |
| experts/GPU/MoE layer | 12 |

EP 是从普通 data-parallel 维度中构造的专家通信组，而不是让 8 张 GPU 共同处理同一个 input batch。TP、PP、CP 都为 1 时，120 ranks 都读取不同 sequences。

### 12.2 rank 编号与 EP groups

冻结 global rank 映射：

$$
\mathrm{global\_rank}=8\times\mathrm{node\_id}+\mathrm{local\_gpu\_id}.
$$

每个节点是一个 EP group：

- node 0：ranks 0–7；
- node 1：ranks 8–15；
- …；
- node 14：ranks 112–119。

local GPU `r∈[0,7]` 在每个 MoE 层持有 global experts `[12r,12r+11]`。例如 local GPU 0 持有 0–11，local GPU 7 持有 84–95。

### 12.3 expert-data-parallel groups

相同 local GPU slot 跨 15 节点形成一个 EDP group：

$$
G_{\mathrm{EDP},r}=\{r,8+r,16+r,\ldots,112+r\}.
$$

这个组内的每个 rank 持有相同 12 个 global experts；它们的 expert gradients、optimizer shards 和初始化 checksum 必须一致。

### 12.4 dense-DP 与 EDP 不可混淆

“`EP8×DP15=120`”只是参数复制关系的一种简写，不是完整训练 batch 语义：

- Dense 模块看到 120 个不同 microbatch samples，形成 120-way 数据并行统计。
- 每个 global routed expert 只在 15 个节点上有 replica，因此 expert gradient 的复制域是 15。
- EP group 内 8 ranks 交换 token，但不因此减少输入 sample 数。

启动时打印每个 rank 的：dense-DP rank/group、EP rank/group、EDP rank/group、TP/PP/CP group、global expert IDs。至少抽查 ranks 0、1、7、8、119，并用机器脚本验证 120 个结果的集合关系。

### 12.5 batch arithmetic

| seq length | 本地 microbatch tokens | 每步 microsteps | 每 rank tokens/step | 全局 tokens/step |
|---:|---:|---:|---:|---:|
| 4096 | 4096 | 8 | 32768 | 3,932,160 |
| 8192 | 8192 | 4 | 32768 | 3,932,160 |
| 16384 | 16384 | 2 | 32768 | 3,932,160 |
| 32768 | 32768 | 1 | 32768 | 3,932,160 |

每次启动必须从 process-group 实际值反推这一表，而不是只相信 CLI。若框架报告 ordinary data lanes 为 15，则说明 batch 解释错误，禁止训练。

### 12.6 为什么 TP1/PP1

- EP8 后每卡逻辑参数约 4.586B，MI300X HBM 足以容纳 BF16 weights 和 optimizer 方案。
- TP 会让 attention/embedding 增加频繁 collective，而 residual width 2048 并不大。
- PP 会引入 pipeline bubble、层映射和 checkpoint 复杂性；48 层尚无必要。
- 节点内 8 GPU 优先全部用于 EP，可以固定 expert All-to-All 不跨节点。

若显存超预算，优先处理 fused CE、activation recompute 和 buffer 生命周期，而不是第一反应加入 TP/PP。

### 12.7 topology 断言

作业在首个数据读取前必须证明：

- 每个 EP group 的 8 ranks 位于同一物理节点；
- 任一 routed expert 恰有 15 个 replicas；
- 96 个 experts 在每个 EP group 中恰好覆盖一次；
- 没有 EP All-to-All 跨 NIC；
- EDP collectives 正确走跨节点 fabric；
- dense collectives 覆盖全部 120 ranks；
- GPU local rank 与物理设备 UUID 的映射稳定；
- 15 节点的系统时钟误差在日志关联容忍范围内。

---

<a id="rocm-stack"></a>

## 13. ROCm 内核与环境锁

### 13.1 环境 lock 必填字段

`environment.lock.json` 至少记录：

| 类别 | 必填内容 |
|---|---|
| 容器 | image repository、不可变 digest、构建 recipe commit |
| OS/driver | kernel、AMD driver、ROCm 完整版本 |
| Python | 版本、所有 wheel hashes、`pip freeze` |
| PyTorch | 版本、构建 commit、HIP arch、编译选项 |
| Megatron | 项目 fork URL、exact commit、dirty=false、patch list |
| Megatron-Core | exact package/source commit |
| communication | RCCL 版本/commit、UCX/MPI/PMI 版本与关键环境变量 |
| expert GEMM | TE/CK/AITER/Triton backend 名、commit、kernel ID |
| attention | fused attention backend、commit、支持的 shapes/dtypes |
| CE | fused/chunked CE backend 和版本 |
| checkpoint | distributed checkpoint backend/format version |
| hardware | 120 个 GPU UUID、节点拓扑、XGMI/NIC 映射 |
| configs | resolved model/data/train/topology JSON hashes |

旧项目笔记中的 ROCm 6.4.3 只是历史环境，不可自动继承为本次生产锁。

### 13.2 grouped expert GEMM 验收

96 experts/EP8 意味着每卡每层 12 个专家。生产 kernel 必须：

- 接受按 expert 排序的 ragged token counts；
- 正确执行 gate/up 两路和 down projection backward；
- 保持受限 SwiGLU 语义；
- 不为零 token expert 访问非法指针；
- 在负载不均时仍正确；
- 与逐 expert PyTorch reference 在 BF16 容差内一致；
- 输出 kernel trace 中实际使用的 grouped path；
- 不把全部 96 experts 的输出都计算后再 Top-k。

测试 shape 要覆盖每 expert 平均 token 数约 2K、4K、8K、16K，以及 0、1、极端过载和非 128 对齐的边界。

### 13.3 EP All-to-All 验收

节点内 RCCL 测试不能只跑默认小消息。应使用与生产接近的：

- dtype BF16；
- hidden width 2048；
- Top-k 6；
- 8 ranks；
- 每 rank sequence tokens 4096/8192/16384/32768；
- 均匀与人为倾斜两种 routing counts；
- forward dispatch、reverse combine 和 backward 全链路。

验证 token ID、expert ID、gate、排列逆变换和梯度；并用拓扑计数确认 payload 未经跨节点 NIC。

### 13.4 attention 与 CE

必须有 shape matrix：

| S | Q heads | KV heads | head dim | 要求 |
|---:|---:|---:|---:|---|
| 4096 | 32 | 4 | 128 | forward/backward、causal、RoPE、QK-Norm |
| 8192 | 32 | 4 | 128 | 同上 |
| 16384 | 32 | 4 | 128 | 不物化 S² attention |
| 32768 | 32 | 4 | 128 | 显存、数值和长时稳定性通过 |

CE 测试使用 `V=151936`、tied weights、S=4K–32K，和分块 FP32 reference 对比。kernel 若只支持 CUDA、特定 vocab 对齐或 untied weight，应在启动时明确失败。

### 13.5 第一版优化开启顺序

1. 正确但可能较慢的 fused attention + All-to-All + grouped GEMM；
2. permute/unpermute fusion；
3. distributed optimizer；
4. gradient reduce/parameter gather overlap；
5. shared-expert overlap；
6. async checkpoint。

每步都要用相同 sample IDs 做 loss/gradient parity 和 checkpoint-resume 测试。生产 baseline 至少需要前 3 项；后 3 项可保持关闭。

### 13.6 本地代码与共享数据

- 代码从每个节点本地 NVMe `$WORKDIR/code` 运行，所有节点是同一 commit。
- token payload 位于共享只读 `$DATA_ROOT`；小索引可复制到本地。
- checkpoint 写 `$CKPT_ROOT`，日志写结构化 `$LOG_ROOT`。
- 不从共享挂载目录 import Python，以免缓存和旧 `.pyc` 导致节点版本不一致。
- 不运行 keepalive；空闲 GPU 正常 idle。

---

<a id="budgets"></a>

## 14. 显存、计算与通信预算

### 14.1 每 rank 持有参数与持久训练状态

EP8 下每 rank 的逻辑参数持有量：

```text
common replicated parameters       1,547,253,760
local routed-expert parameters     3,038,773,248
local logical parameters           4,586,027,008
```

冻结 distributed-optimizer contract：

- model parameter storage：BF16；
- gradient accumulation buffer 与 reduce-scatter wire dtype：FP32；
- local optimizer shard：FP32 master weight + FP32 first moment + FP32 second moment；
- parameter all-gather wire dtype：BF16；
- common 参数的 optimizer shard group 为 dense-DP120；routed expert 参数的 shard group 为 EDP15。

无 allocator/alignment/padding 的持久下界：

| component | bytes/rank | GiB/rank |
|---|---:|---:|
| BF16 local model weights | 9,172,054,016 | 8.542141 |
| FP32 local grad buffer | 18,344,108,032 | 17.084282 |
| FP32 common master+m+v shard（DP120） | 154,725,376 | 0.144099 |
| FP32 routed master+m+v shard（EDP15） | 2,431,018,598.4 | 2.264063 |
| **persistent subtotal** | **30,101,906,022.4** | **28.034585** |

这是数学下界，不含 flat-buffer padding、bucket 对齐、allocator、FP32 router temporaries、activation、A2A staging、grouped-GEMM workspace、attention/CE workspace、RCCL buffers、checkpoint staging 与 profiler。上线表必须用实测峰值替换估计。

### 14.2 checkpoint logical payload

full recovery checkpoint 在不保存 gradients 的情况下，按 BF16 model + FP32 master/m/v 计 `14 bytes/parameter`：

```text
25,857,439,744 × 14 = 362,004,156,416 bytes
                         = 362.004156416 GB
                         = 337.142642975 GiB
```

`16 bytes/parameter = 413,719,035,904 bytes = 385.305877686 GiB` 只作为包含额外 tensor/布局开销的对照上界，不可冒充 DCP 实测。weights-only BF16 payload 为 `51,714,879,488 bytes = 48.163234711 GiB`。

### 14.3 activation 与 workspace 风险

必须分别量测：attention saved tensors、Dense FFN、routed/shared expert inputs、permutation metadata、dispatch/combine send/recv、grouped-GEMM workspace、FP32 router logits、CE workspace 与 backward overlap buckets。4K 可运行不能推出 32K 可运行。推荐起点是 full selective recompute；只有 profile 证明峰值安全后才逐项关闭。MoE recomputation 会增加一次 forward dispatch/combine，通信预算必须单列。

### 14.4 一阶训练 FLOP：两种 attention 约定

参数项使用

$$
F_{\mathrm{param}}\approx 6P_{\mathrm{active}}
=18,399,842,304\ \mathrm{FLOP/token}.
$$

attention 项有两种常见 convention：

- **full-square 近似**：$F_{\mathrm{attn,full}}\approx12L_{\mathrm{full}}Sd_q$；
- **causal-triangular 近似**：$F_{\mathrm{attn,causal}}\approx6L_{\mathrm{full}}Sd_q$，忽略 kernel tile 对被 mask 区域的额外执行。

这里 $L_{\mathrm{full}}=48,d_q=4096$。两者都不含 embedding gather、norm、softmax、router、top-k、permutation、all-to-all、optimizer、recompute 与 padding overhead。

| S | full-square GFLOP/token | causal-triangular GFLOP/token |
|---:|---:|---:|
| 4,096 | 28.063519 | 23.231681 |
| 8,192 | 37.727195 | 28.063519 |
| 16,384 | 57.054548 | 37.727195 |
| 32,768 | 95.709254 | 57.054548 |

本文历史 stage/总量采用 full-square convention：

| stage | scheduled tokens | full-square ZFLOP | causal-triangular ZFLOP |
|---|---:|---:|---:|
| 4K | 800,001,884,160 | 22.450867852 | 18.585388182 |
| 8K | 150,000,107,520 | 5.659083327 | 4.209530825 |
| 16K | 40,001,863,680 | 2.282288250 | 1.509158117 |
| 32K | 9,995,550,720 | 0.956666699 | 0.570291628 |
| **total** | **999,999,406,080** | **31.348906128** | **24.874368752** |

性能报告必须声明 convention，不能把 full-square 与 causal 数字混在同一趋势中。

### 14.5 idealized runtime scenarios

按 full-square `31.348906128 ZFLOP`，不计 checkpoint/eval/failure：

| sustained cluster PFLOP/s | idealized days |
|---:|---:|
| 20 | 18.14 |
| 25 | 14.51 |
| 30 | 12.09 |
| 35 | 10.37 |
| 40 | 9.07 |
| 45 | 8.06 |
| 50 | 7.26 |

这些是情景量，不是承诺 ETA。

### 14.6 EP All-to-All logical payload

routed payload width 固定为 `d_dispatch=d_model=2048`，不是 Dense FFN width `5504`。一个 token 在 46 个 MoE 层、Top-6、BF16 下 forward dispatch+combine 的 logical bytes：

$$
2\times46\times6\times2048\times2=2,260,992.
$$

EP8 均匀目标下 off-rank 比例 `7/8`：

```text
forward remote sent/token                    1,978,368 bytes
forward remote sent/rank/update         64,827,162,624 bytes = 60.375 GiB
forward+backward remote sent/rank/update 129,654,325,248 bytes = 120.75 GiB
NIC Tx+Rx/rank/update                    259,308,650,496 bytes = 241.5 GiB
```

4K/MBS1 单 rank microbatch 的 off-rank forward send 是 `8,103,395,328 bytes` across 46 layers；单层 forward 是 `176,160,768 bytes`，forward+backward four-leg 是 `352,321,536 bytes`。以上是无 metadata、alignment、padding、protocol 与 imbalance 的逻辑估算；shared expert 不进入 routed A2A。MoE recompute 另加一次 forward traversal。

### 14.7 frozen gradient collective convention

raw unpadded tensors：

| partition | FP32 local gradient bytes | BF16 local parameter bytes | group |
|---|---:|---:|---:|
| common/non-routed | 6,189,015,040 | 3,094,507,520 | dense-DP120 |
| local routed experts | 12,155,092,992 | 6,077,546,496 | EDP15 |

对 ring-equivalent distributed optimizer，每 rank **one-way sent**：

| partition | FP32 reduce-scatter sent | BF16 all-gather sent | total sent |
|---|---:|---:|---:|
| common, group 120 | 6,137,439,914.67 B | 3,068,719,957.33 B | 9,206,159,872.00 B |
| routed, group 15 | 11,344,753,459.20 B | 5,672,376,729.60 B | 17,017,130,188.80 B |
| **combined** | — | — | **26,223,290,060.80 B = 24.422 GiB** |

receive 与 send 对称时 endpoint Tx+Rx 约 `48.845 GiB/rank/update`。这些不是“线上字节”：runtime bucket padding、collective algorithm、chunking、overlap、hierarchical path 与协议开销只能由 RCCL/profiler trace 决定。验收必须同时报告 logical、profiler payload 与 NIC counters。

### 14.8 性能报告最少字段

每个 stage 至少报告：有效 target tokens/s、full-square 与 causal MFU、update latency p50/p95/p99、forward/backward/optimizer、attention、grouped GEMM、A2A、reduce-scatter、all-gather、data wait、checkpoint stall、HBM peak、workspace peak、logical/profiler/NIC bytes、padding ratio、expert imbalance、recompute 开关与所有 fallback。

---

<a id="checkpointing"></a>

## 15. Checkpoint、恢复、artifact lifetime 与容量

### 15.1 recovery checkpoint 必含状态

每个 full checkpoint 必须包含并 hash-bind：model、BF16 params、FP32 master/m/v、scheduler、所有 RNG、`successful_updates/update_tokens`、`attempted_batches_total_at_commit`、stage 与 stage-local update、next-batch digest、source/scheduler algorithm version、corpus/tokenizer/holdout/stage manifests、resolved model/stage config SHA、code/environment/backend lock、process-group/ownership/topology manifest、checkpoint schema version、lineage ID 与 parent checkpoint ID。stock `consumed_train_samples` 不能替代这些字段；提交点之后的 raw attempts 由外部 append-only telemetry 保留，恢复时与 checkpoint 快照对账，绝不把该原始计数误当成 canonical progress。

### 15.2 原子提交协议

1. 所有 rank 写入唯一 staging prefix；
2. 每个 shard 写 size/SHA-256/owner metadata；
3. coordinator 收齐并验证全体 shard，写 canonical root manifest；
4. `COMMITTED` marker 最后写入，为 `rfull-commit-v1` canonical-JSON envelope；它必须绑定 run/config-manifest、stage ID/artifact hash、`successful_updates/update_tokens`、root-manifest path/SHA、parent-commit SHA 与 lineage ID，而不是只放一个裸 hash；
5. marker 完成后再以原子替换更新 checkpoint-root 下的 `LATEST_COMMITTED`（`rfull-latest-v1`），其中只含 marker 相对路径与 marker SHA；恢复必须从该显式 canonical head 开始，禁止按目录名或最大 step 猜测；
6. 只有 pointer、marker、root manifest 与全部 hashes/计数器相互匹配的 checkpoint 可恢复；stage 内 checkpoint 必须恢复同一 stage，只有精确达到 stage endpoint 才可推进下一 stage；
7. interrupted staging 永不覆盖旧 checkpoint，后台 GC 只在 retention lease 外清理。

本地 rename 原子性不能无证明地外推到 blob/object storage；adapter 必须用该 backend 可证明的 conditional-create/marker 语义。

### 15.3 cadence 与 retention

- full recovery interval：每 2,000 successful updates；
- permanent full：`[2000,203451,241598,251771,254313]`；
- rolling full：额外保留最近 3 个 interval checkpoint；与 permanent 同 ID 时按内容 hash deduplicate；
- weights-only：每 10,000 updates 的 25 个里程碑，加 4 个 stage/final milestones，共 29 个逻辑 artifact；相同 ID/hash deduplicate；
- final：至少两份独立可恢复 DCP replica，加一个 release/export artifact；
- 所有删除必须在新 artifact `COMMITTED`、第二位置复制并恢复验证后执行。

### 15.4 artifact-lifetime peak simulation

用 14-byte full payload 和 BF16 weights payload 做**下界**：

| retained/in-flight item | count | bytes each | bytes |
|---|---:|---:|---:|
| full recovery（5 permanent + 3 rolling） | 8 | 362,004,156,416 | 2,896,033,251,328 |
| weights-only milestones | 29 | 51,714,879,488 | 1,499,731,505,152 |
| release export | 1 | 51,714,879,488 | 51,714,879,488 |
| second final DCP replica | 1 | 362,004,156,416 | 362,004,156,416 |
| one in-progress full staging | 1 | 362,004,156,416 | 362,004,156,416 |
| **checkpoint/export peak lower bound** | — | — | **5,171,487,948,800 B = 5.171 TB = 4.703 TiB** |
| corpus raw payload（if co-resident） | 1 | 3,752,033,091,080 | 3,752,033,091,080 |
| **co-resident subtotal** | — | — | **8,923,521,039,880 B = 8.924 TB = 8.116 TiB** |
| **+20% reserve lower bound** | — | — | **10.708 TB = 9.739 TiB** |

这仍未包含 DCP metadata/padding、indexes/sidecars、corpus replica、filesystem/object overhead、upload retry、evaluation cache、temporary export、logs 和 deleted-object retention。结论：**5 TB 不可签核**。若 corpus 在独立池，checkpoint pool 在 20% reserve 前就已超过十进制 5 TB；容量必须由生成的 lifetime timeline、真实 DCP size 和 storage-tier placement 重新签核。

### 15.5 恢复与 failure-injection 验收

必须覆盖：同 topology resume、进程 kill、节点 kill、save 中断、损坏 shard/marker、optimizer 后但 marker 前中断、stage 边界前后中断、数值 poison、world-size/GA 改变而 `G` 不变、DCP topology migration。每次恢复都验证 next-batch digest、LR、model/optimizer hashes（或冻结 tolerance）、source quotas 与 accepted-lineage metrics。

关键 boundary test：在 4K 最后一个 update 和 8K 第一个 update 分别注入 failure；恢复后必须重放同一 batch，且连续 control run 与 failure/resume run 在 stage、LR、sample IDs 和参数上符合确定性契约。

---

<a id="observability"></a>

## 16. 规范 telemetry、全局归约与 accepted lineage

### 16.1 三层真相源

1. **raw attempt telemetry**：per-rank append-only records，永不覆盖，含成功、失败、重放、rank、host、attempt/run UUID、parent checkpoint、monotonic/UTC timestamps；
2. **checkpoint ledger**：只记录带 `COMMITTED` marker 的 durable states；
3. **canonical view**：从当前被选择的 accepted durable checkpoint 沿 parent links 反向可达的 lineage，再加当前进程分支中以该 checkpoint 为直接 parent、已成功但尚未 checkpoint 的可证明尾段。发生 rollback 时，旧分支的 pre-rollback 尾段必须在 restore 前立即标为 noncanonical；重放尾段只有在同一 batch/update 再次成功提交后才进入新分支。abandoned attempts/tails 不进入 canonical curves；最终报告只由 final accepted checkpoint 可达的提交谱系重建。

任何 dashboard 都不得把重放的 sample/update 双计。

### 16.2 每 update 的必备字段

- run/attempt/update UUID 与 `successful_updates`；
- `seen_tokens/update_tokens`、stage、seq len、G、MBS、GA；
- next/processed batch digest 与 source counts；
- pre-update LR；
- global CE numerator、valid-target denominator、mean CE/PPL；
- aux/z loss numerator 与统计域；
- local/global grad norm、clip coefficient；
- finite consensus bitset；
- optimizer-step duration 与是否 commit；
- GPU kernel、data、A2A、reduce-scatter、all-gather、checkpoint timings；
- HBM、NIC 与 fallback flags。

### 16.3 全局 reduction 定义

训练/validation CE 统一上报

$$
\mathrm{CE}=\frac{\sum_r \mathrm{loss\_sum}_r}{\sum_r \mathrm{valid\_target\_count}_r},
\qquad \mathrm{PPL}=\exp(\mathrm{CE}),
$$

不得平均 rank mean。global clipping norm 必须与参数复制语义一致：

$$
\|g\|^2=
\frac{\sum_{r=1}^{120}\|g_{\mathrm{common},r}\|^2}{120}
+
\frac{\sum_{r=1}^{120}\|g_{\mathrm{routed-local},r}\|^2}{15}.
$$

router counts/probability sums 先在每个 EP8 group 聚合，再在 15 个 batch replicas 上按 token 数归约；layer 与 microstep ID 必须保留。只汇报 rank0 local 指标不合格。

### 16.4 finite consensus 与告警

world poison bit 是对 loss、router logits/softmax、activations sentinel、grads、global norm、post-update BF16 params、FP32 master/m/v 的 OR reduction。任一 bit 触发 stop/replay，不做 committed skip。大但有限的 norm（例如 `>max(10,100×rolling median)`）只作为 fail-closed diagnostic guard；它不会消费 batch。

### 16.5 系统指标与时间

每 rank 使用 monotonic clock 计算 duration，UTC 只用于跨系统关联；启动时记录 host clock offset。至少记录 GPU util/HBM、temperature/power、kernel name/fallback、RCCL collective traces、NIC Tx/Rx、data wait/page fault/cache、DCP bytes/stall。日志静止不等于 hang；诊断只用日志、GPU telemetry、`ps` 与 `/proc`，禁止 attach/stop 类侵入工具。

---

<a id="validation"></a>

## 17. 验证、评测与发布判据

### 17.1 unit/reference oracles

必须覆盖：精确参数 ledger、embedding shape `(151936,2048)` 与 tied storage、Q/K RMSNorm shapes、Dense/routed/shared limited-SwiGLU、Top-6 selected-softmax、96-way aux probabilities、EP1↔EP8 aux scalar/gradient parity、z-loss、router `Normal(0,0.01)`、dropless dispatch/combine、expert ownership、global clip norm、finite consensus、LR 边界、source-cycle/quota、holdout exclusion、checkpoint transaction 与 deterministic resume。

### 17.2 kernel parity

对 attention、CE、Dense FFN、grouped expert GEMM、permutation、A2A、shared expert、optimizer、DCP 分别保留 FP32/small-shape reference。报告 forward max/mean error、loss error、input/weight gradient error、NaN/Inf、determinism 与 fallback。只通过 forward 不够。

### 17.3 held-out validation

使用第 8.5 节 immutable holdout。每个 source/length 报告 valid targets、CE sum、mean CE、PPL；aggregate 同时给：

- token-weighted micro average：八 source 的 CE sums / target counts；
- source macro average：八个 source mean CE 的等权平均；
- 不允许对 PPL 直接求平均；先聚合 CE，再 exponentiate。

4K/8K/16K/32K 每套均有 `67,108,864` 个真实 targets；padding rows mask=0。训练中的 validation 只读取 holdout manifests，不访问 train ranges。

### 17.4 external benchmark matrix

工具链必须锁定 harness repo SHA、task versions、dataset revisions/hashes、few-shot examples、prompt template、tokenizer revision、generation/stop parameters、seed、batch size、dtype 与 per-item outputs。计划矩阵：

| cadence | likelihood/QA | generative/code | Chinese |
|---|---|---|---|
| `[2000,20000,50000,100000,150000,200000,203451,241598,251771,254313]` | MMLU、HellaSwag、ARC-C/E、PIQA、WinoGrande、OpenBookQA | — | C-Eval、CMMLU |
| every stage end + final | 上述全量 | GSM8K、HumanEval | 上述全量 |

具体 task/version/hash 未填前为 `EVAL-001` blocker：

```yaml
eval_harness_commit: TBD-BLOCKER
eval_dataset_manifest_sha256: TBD-BLOCKER
eval_prompt_manifest_sha256: TBD-BLOCKER
```

生成式评测远慢于 likelihood；提交前必须单独估 ETA，按 task/phase 隔离 `results.json`，单一任务失败不得使其他完成结果失效。

### 17.5 checkpoint 选择与发布

最终交付 checkpoint 按语义完整性选择：`successful_updates=254313`、所有 manifests/config/environment 匹配、两份 DCP 恢复通过、held-out validation 与 external matrix 完成。不得按单个 benchmark point-estimate 峰值挑 checkpoint，也不得在退火结束前把中段增长放缓称为“饱和”。

release package 至少包括 weights、config、tokenizer refs/hashes、model/parameter accounting、training token/stage/source ledger、environment/code hashes、eval artifacts、limitations、license/provenance 与 DCP→release logits parity。

---

<a id="rollout"></a>

## 18. 分阶段上线计划

### 18.1 Gate 0：静态设计与 CPU/reference

- 参数/shape/配置检查；
- 数据 manifest 与 global sample scheduler；
- 10,000-cycle 配额、holdout 排除和 O(1) seek；
- 缩小版模型 forward/backward；
- checkpoint metadata schema。

**通过条件**：所有账本精确相等，随机 sample IDs 可重复，跨 stage 不出现未声明 overlap。

### 18.2 Gate 1：单 GPU

使用缩小版 MoE 做至少 100 steps，并对 full geometry 做单层/推理 shape 测试：

- BF16 vs FP32 reference；
- QK-Norm/RoPE 顺序；
- limited SwiGLU 在 Dense、routed grouped-GEMM、shared expert 三条路径分别做 forward/backward parity；
- explicit `padded_vocab_size=151936` 与 tied embedding/head runtime assertion；
- router `Normal(0,0.01)` 初始化分布与 seed parity；
- fused attention、CE、grouped GEMM；
- optimizer 与 checkpoint round-trip。

**通过条件**：无 fallback、无非有限值、误差在冻结容差内。

### 18.3 Gate 2：单节点 EP8

先 100-step smoke，再至少 1000 steps full R-Full：

- 96 experts 每卡 12 个；
- 8 ranks 读取 8 个不同 samples；
- dispatch/combine forward/backward；
- EP-global aux 的 counts/probability sums/scalar/router-gradient 对 EP1 reference；
- dropless 极端倾斜；
- shared expert 与 routed overlap 关闭；
- 同步 full checkpoint/resume；
- 节点内 A2A trace。

**通过条件**：expert assignment/gradient parity、resume、HBM 与负载阈值全部通过。

### 18.4 Gate 3：两节点 EP8×EDP2

至少 1000 steps，覆盖：

- 同一 global experts 的两份 replica 初始化与更新一致；
- expert gradients 只在 EDP2 同步；
- dense gradients 跨 16 ranks 同步；
- distributed optimizer shard；
- DCP save/load；
- kill 一个 rank、留下 incomplete checkpoint、回到最近 committed checkpoint 的故障注入。

**通过条件**：无跨组污染；恢复后 sample IDs、LR、loss 与 control 对齐。

### 18.5 Gate 4：120-GPU smoke

固定正式 topology，运行 100 steps（393,216,000 tokens）：

- 打印和验证全部 process groups；
- 120 GPUs 均有真实 utilization/HBM；
- 验证 EP A2A 节点内、EDP/dense collectives 跨节点；
- 收集完整性能 profile，并对 FP32 reduce-scatter、BF16 all-gather、A2A 的 logical/profiler/NIC 三套字节闭环；
- 写并恢复一个 full checkpoint。

这 100 steps 只有在配置、数据、环境完全冻结且所有 gate 通过时，才可计入正式训练；否则作为 throwaway pilot。

### 18.6 Gate 5：正式 2K burn-in

在最终配置上跑到 step 2000，共 7,864,320,000 tokens：

- 前 200 steps 每 step 检查全量 router/clamp；
- 至少两次 validation；
- step 2000 保存永久 full checkpoint；
- 运行第一组 immutable-manifest external benchmark；
- 注入 forward/backward/post-update poison 与 checkpoint-save interruption，证明 stop/replay 不推进 sample/LR；
- 审核 loss、gradient、source quotas、专家负载、系统离群和 checkpoint overhead。

**通过后才解除长跑暂停点**。任何模型/optimizer/router/data 改动都会使这 2K steps 作废并重新从 step 0 开始；纯 telemetry bug 修复可由变更评审决定是否继续。

### 18.7 Gate 6：持续 1T

- 按第 10 节阶段计划运行；
- 每 2000 steps recovery checkpoint；
- 每个 stage end 人工签核并执行 checkpoint restore probe；
- 退火期不因中段指标“看似平台”提前结束；
- final step 后先完成 checkpoint 与验证，再终止资源。

---

<a id="failure-handling"></a>

## 19. 故障处理与运行纪律

### 19.1 故障分类

| 类别 | 证据 | 处理 |
|---|---|---|
| 应用错误 | Python traceback、HIP OOM、RCCL error、assert、NaN | 保留日志，修复/回滚，从健康 DCP 恢复 |
| 数据错误 | shard/hash/index/sample ID 断言 | fail closed；不跳 source、不改权重 |
| 数值错误 | 非有限、grad/router/clamp 越界 | 停止并重放诊断；不继续污染 moments |
| checkpoint 错误 | 缺 shard/marker/checksum | 忽略 incomplete，回退最近 committed |
| 基础设施错误 | 多节点同步重启、工作目录重建、无应用 traceback | 按平台中断处理，不在模型代码里臆测 bug |
| 性能退化 | kernel fallback、data wait、A2A/EDP 尾延迟 | 在健康边界停机分析，不改变数学语义绕过 |

### 19.2 启动成功的最低证据

以下全部出现后才可报告“训练已恢复/已启动”：

- 120 个 worker 和正确 topology；
- checkpoint model、optimizer、scheduler、data state 全部加载；
- 下一批 sample IDs 与 manifest 对齐；
- 首个新 metric 行和 optimizer update；
- 120 张 GPU 均有预期 HBM/utilization；
- router/expert counts 正常；
- checkpoint heartbeat 更新。

外层作业 `Running`、`pgrep` 有进程或 120 workers 刚拉起都不充分。

### 19.3 安全停止

1. 阻止 launcher 拉起下一阶段；
2. 请求训练在 optimizer-step 边界停止；
3. 若系统健康，写 emergency full checkpoint；
4. 等待 `COMMITTED`；
5. 再结束 worker/launcher；
6. 核对 GPU idle 与无残留 writer。

杀进程时先处理 wrapper/launcher，再处理 children；避免宽泛 `pkill -f` 匹配自身命令。

### 19.4 不允许的现场操作

- 对生产进程 attach `gdb`、`strace`、`ptrace`；
- 在 checkpoint 正写入时删除目录；
- 修改共享 Python 文件让不同节点热加载不同代码；
- 为通过 OOM 临时打开 token dropping；
- 发生 source 缺失后重新归一化 mix；
- 手工编辑 global step/token 计数；
- 只因日志暂时无新行就强杀进程；
- 未验证 GPU 活跃就宣布恢复成功。

### 19.5 恢复后审计

每次恢复自动生成一条 episode 记录：故障时间、最后健康 step、`rolled_back_uncheckpointed_tokens`（已回滚、需重放的计算量，绝非 accepted/lost tokens）、选择的 checkpoint、环境是否变化、恢复耗时、首 10 steps 对比、责任人和结论。相同错误重复两次必须升级为 blocker，不得无限自动重试。

---

<a id="long-context"></a>

## 20. 非规范 long-context continuation 迁移计划

本节**不属于 v0.1 1T baseline**，不得被 baseline launcher 读取。baseline checkpoint/config 固定 `max_position_embeddings=32768`；若继续到 128K/256K，必须创建新 project/recipe，显式受测地 override 到至少 `262144`。RoPE 无 learned table 不等于运行时、mask、DCP argument 或 kernel 自动兼容。

### 20.1 可执行候选（仍未验收）

| target S | CP | batch-DP=`120/CP` | EDP=`120/8` | global sequences | MBS/GA | target tokens/update |
|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 2 | 60 | 15 | 60 | 1/1 | 3,932,160 |
| 131,072 | 4 | 30 | 15 | 30 | 1/1 | 3,932,160 |
| 262,144 | 8 | 15 | 15 | 15 | 1/1 | 3,932,160 |

这些满足 stock MCore 的 `S % (2×CP) == 0` 与 `G % (MBS×batch-DP) == 0`。旧 CP3/CP5 提案被拒绝：`131072 % 6 = 2`、`262144 % 10 = 4`，且其 batch arithmetic 也不匹配。EDP 始终独立为 15，不能使用错误恒等式 `EP×CP×EDP=world`。

### 20.2 topology 与 communicator

rank order 仍为 `tp-cp-ep-dp-pp`。每个候选都必须打印 CP/EP/EDP/batch-DP group lists 和 rank-to-host；CP 与 EP 即使成员重合也使用独立 communicator。CP8 与 EP8 的同组竞争、CP2/4 的节点内布局、RCCL stream overlap 与 deadlock 必须 profile/pressure-test。

### 20.3 migration gate

需要单独通过：checkpoint-argument override、DCP topology migration、128K/256K attention forward/backward、memory、CP send/recv、EP A2A、combined CP×EP communicators、validation parity、resume 与至少 1K-step burn-in。通过前，64K–256K 只作为候选路线，不修改 v0.1 架构账本或已冻结 1T curriculum。

---

<a id="resolved-config"></a>

## 21. 已生成的机器可读配置与四阶段工件

旧版内嵌 YAML 仅是说明性骨架，不能区分 native/model vocabulary、不能展开 layer IDs，也不能表达 strict stop/replay；现已删除。仓库中的实际工件是：

- [`configs/rfull/rfull_v0_1.source.json`](../configs/rfull/rfull_v0_1.source.json)：唯一可编辑 source config；
- [`configs/rfull/rfull_v0_1.schema.json`](../configs/rfull/rfull_v0_1.schema.json)：JSON Schema Draft 2020-12，所有 object 均拒绝未知键；
- [`tools/compile_rfull_config.py`](../tools/compile_rfull_config.py)：schema 校验、跨字段算术验证、canonical hash 与 deterministic compile；
- [`configs/rfull/generated/manifest.json`](../configs/rfull/generated/manifest.json)：source/schema/compiler/stage hash 链；
- `stage_4k.json`、`stage_8k.json`、`stage_16k.json`、`stage_32k.json`：四个 standalone resolved stage config；
- [`tools/run_rfull_stages.py`](../tools/run_rfull_stages.py)：fail-closed verifier/orchestrator；
- [`tools/test_rfull_config.py`](../tools/test_rfull_config.py)：正向与负向 contract tests。

### 21.1 当前工件身份

| 工件 | SHA-256 语义 | 当前值 |
|---|---|---|
| source config | canonical JSON object | `a8139b28be8a8bb45c861d81907ad14780b1d09646d736fb4edd169849355356` |
| source schema | canonical JSON object | `c4e038e88ebf4fb3a338ec31e0b2e967f389694a5e11cca0192e96afa1a2f5f5` |
| compiler | 原始文件 bytes | `4d1b116ab308c65fa245a3b51b950327afb906b7385e47a78e67bc275d109d66` |
| generated manifest | canonical JSON，排除自身 `artifact_sha256` 字段 | `c2937941972e714390c6e9e88aff561a8d4fa28500b70897520fc4de652a950f` |
| 4K stage | 同上 | `18295177fe5e3c20622b0d51348f319f0d6b8268e342eca38d1a120c9d251949` |
| 8K stage | 同上 | `7916e562237c4f588870173e3db8b53002e51f460c69c5962e72c8942410f4a5` |
| 16K stage | 同上 | `b4269f3f0beec20d0a07ddd2446414d860a59de5893f7134f262dcc8f21a3c16` |
| 32K stage | 同上 | `880efd69fa4fe165c46dec31dad7a0af9c837884fb07c56dc8ca67b40805a898` |

canonical JSON 规则为 UTF-8、key 排序、无多余空白、integer/string/bool/null；配置中的十进制超参数使用 canonical decimal string，禁止 JSON float/NaN/Inf。生成工件的 `artifact_sha256` 不参与自身 hash 输入。

### 21.2 compiler 必须证明的约束

compiler 不是模板替换器；它在生成前必须 fail closed 地复算并断言：

- total/active parameters 分别为 `25,857,439,744` / `3,066,640,384`；
- native vocab=`151669`、model rows=`151936`，且显式证明默认 128 对齐会得到 `151680`；
- Dense layer IDs 为 `[0,1]`，MoE layer IDs 完整展开为 `[2,...,47]`；
- threshold=`10`、dispatch width=`2048`、Top-6 selected-softmax、EP-global autograd-aware aux contract；
- world/EP/EDP=`120/8/15`，EP groups node-local，FP32 reduce-scatter 与 BF16 parameter all-gather，并复算 A2A/raw collective byte ledger；
- 四阶段 batch arithmetic、每个 stage/source sequence quota、aggregate source tokens、`254313` successful updates 与 `999,999,406,080` target tokens；
- strict stop/replay 且 `max_committed_skips=0`；
- checkpoint retention 与 artifact-lifetime storage ledger（含 20% reserve）；
- baseline `max_position_embeddings=32768`，long-context continuation 未启用。

每个 stage artifact 都复制完整 model/router/optimizer/data/checkpoint/environment contract，而不是依赖运行时继承；同时给出 explicit topology groups、stage 起止 update/token、source sequence/token quota、输入 hash 与自身 hash。

### 21.3 可复现生成、验证与启动拒绝

```bash
python tools/compile_rfull_config.py
python tools/test_rfull_config.py
python tools/run_rfull_stages.py --plan
```

当前 manifest 明确记录 `launch_allowed=false` 和 **29 个** unresolved blocker，覆盖 tokenizer/corpus/holdout/stage manifest、eval provenance、11 项实现/系统 qualification evidence、ROCm/Megatron/backend lock 与 MCore argv adapter。因此 `--plan` 只能验证与打印计划；`--execute` 即使收到显式生产确认词也必须拒绝启动。

未来所有 blocker 关闭后，orchestrator 仍要求显式 confirmation、repository-local pinned argv adapter 和 checkpoint root；adapter 以 argv vector、`shell=False` 调用。每一阶段返回后必须找到对应 `step_<successful_update>/COMMITTED` 非空 marker，才能把该 checkpoint 作为下一阶段 parent。orchestrator state 采用临时文件加 `os.replace` 原子更新；失败阶段不得推进 canonical counters 或偷偷进入下一阶段。

这些工件把设计约束变成可执行检查，但**仍不等于 ROCm/Megatron 实现验收或 launch authorization**。

---

<a id="runbook"></a>

## 22. 启动与阶段切换 Runbook

### 22.1 正式启动前 24 小时

- [ ] 所有 blocker 有 owner、证据和关闭记录。
- [ ] 120 GPU UUID/topology inventory 完整。
- [ ] 容器、代码、数据和配置 hashes 固定。
- [ ] corpus/holdout/stage manifests 只读且有备份。
- [ ] artifact-only pool 按至少 `6.206 TB` usable 检查；若 corpus 与 artifact 同池，按至少 `10.709 TB` usable 检查；另计站点 replica、对象元数据与 tiering。
- [ ] EP8 A2A、dense/EDP collectives、attention、CE、grouped GEMM shape matrix 已通过。
- [ ] 2-node checkpoint/resume 与故障注入通过：单 rank NaN、梯度 Inf、坏样本、I/O 中断、单节点退出均触发全局停止、无 commit，并从最后 `COMMITTED` checkpoint replay 同一 canonical update/data position。
- [ ] dashboard、告警和 on-call 联系方式可用。
- [ ] 评测环境与训练环境解耦，不从训练 job 抢 GPU。
- [ ] 变更冻结；工作树 clean；每节点部署同一 SHA。

### 22.2 每节点预检

- [ ] 本地代码 commit 与 lock 一致。
- [ ] 8 张 GPU 可见、无残留进程、HBM idle。
- [ ] GPU/NIC/XGMI 健康，无新增 ECC/reset。
- [ ] `$DATA_ROOT` 只读可见，随机 shard hash 抽查通过。
- [ ] `$CKPT_ROOT` 可写，原子 commit 小测试通过。
- [ ] 时间同步、主机名/rank mapping 一致。
- [ ] ROCm/PyTorch/RCCL/backend 版本打印并匹配。

### 22.3 launcher 启动

launcher 必须：

1. 生成 resolved JSON 和 hash；
2. 运行静态参数/batch/process-group dry run；
3. 在所有节点核对代码与容器 digest；
4. 启动 120 ranks，失败时使外层 job 非零退出；
5. 打印 process groups 与 expert ownership；
6. 完成初始化 checksum；
7. 打开数据 manifest 并打印首批 sample IDs；
8. 执行首个 optimizer step；
9. 验证 120 GPU 活跃后才发出“RUNNING_HEALTHY”状态。

不要把具体内部集群名、账户或挂载路径写入公共配置；通过 `$WORKDIR`、`$DATA_ROOT`、`$CKPT_ROOT`、`$LOG_ROOT` 注入。

### 22.4 运行中每日检查

- [ ] 精确 step/seen/update tokens 与 ETA。
- [ ] 最近 24h failed attempts、stop/replay、NaN/Inf、clamp 与 router imbalance；canonical committed skip 必须为 `0`。
- [ ] source 实际配额与 pass counts。
- [ ] 120-GPU step-time/utilization 离群。
- [ ] EP/EDP/dense communication p50/p95/p99。
- [ ] 最近 3 个 recovery checkpoints 均有 COMMITTED 和校验结果。
- [ ] 异步评测状态；结果按 phase 独立。
- [ ] 存储增长与清理计划。
- [ ] 无环境/代码 drift。

### 22.5 阶段切换步骤

1. 到达精确 boundary step 后停止新 batch；
2. 写永久同步 full checkpoint；
3. 跑固定 4K 与当前长度 validation；
4. 生成 next-stage resolved config，并验证只有 allowlist 字段变化；
5. 预热新 shape kernels；
6. 从 boundary checkpoint 恢复；
7. 核对 LR、optimizer moments、sample-ID stage reset 与 global counters；
8. 运行 10-step enhanced monitoring；
9. 通过后恢复正常 checkpoint/eval cadence。

### 22.6 最终停止

1. 完成 step 254313，确认 exact token counters；
2. 写 final full DCP 并在另一存储位置复制/校验；
3. 写 final weights-only export；
4. 完成全长度 validation；
5. 启动最终外部评测；
6. 保存所有 manifests、logs 和 environment lock；
7. 生成全量重建的训练摘要；
8. 确认 writer 与 worker 退出、GPU idle；
9. 不因等待评测而保持训练 GPU keepalive。

---

<a id="acceptance-matrix"></a>

## 23. 规范配置编译、resolved artifacts 与 orchestrator

仓库中的规范入口不是手写 MCore YAML，而是：

```text
configs/rfull/rfull_v0_1.schema.json
configs/rfull/rfull_v0_1.source.json
configs/rfull/generated/stage_{4k,8k,16k,32k}.json
configs/rfull/generated/manifest.json
tools/compile_rfull_config.py
tools/run_rfull_stages.py
```

### 23.1 compiler contract

compiler 必须：

1. 用 JSON Schema 校验 source config，并拒绝 unknown keys；
2. 展开显式 `dense_layer_ids=[0,1]` 与 `moe_layer_ids=[2..47]`，禁止 `[2-47]` 这种字符串伪列表；
3. 验证参数、batch、token、source quota、topology、vocab、LR、retention 与 communication invariants；
4. 将四个 stage 编译为 standalone resolved JSON，每个只含标量/显式数组，不依赖运行时继承；
5. 使用第 8.3 节 canonical JSON 规则生成 SHA-256，写 generated manifest；
6. 发现任何 `TBD-BLOCKER`、环境漂移、hash 不匹配或 unqualified feature 时仍可生成审计 artifact，但 orchestrator 必须拒绝 launch。

### 23.2 orchestrator contract

当前 checked-in orchestrator 已实现 generated/source/schema/compiler hash-chain、四阶段顺序与最终 counter 验证；因为 blocker 未关闭，`--execute` 必须 fail closed。它只以 argv vector、`shell=False` 调用未来的 repository-local pinned adapter，并在阶段返回后验证对应非空 `COMMITTED` marker。**这还不是完整生产 adapter**：清除 `mcore_argv_adapter` blocker 之前，adapter 协议与 qualification evidence 必须另行实现并证明 code/container/backend/manifests、实际 rank topology、input checkpoint lineage、resolved-arguments hash、stage-end restore probe 与 validation；仅把某个路径填入配置不算关闭 blocker。

### 23.3 当前 blocker fields

```yaml
data.tokenizer_revision: TBD-BLOCKER
data.tokenizer_artifact_sha256: TBD-BLOCKER
data.corpus_manifest_sha256: TBD-BLOCKER
data.holdout_manifest_sha256: TBD-BLOCKER
data.stage_manifest_sha256: TBD-BLOCKER
evaluation.eval_harness_commit: TBD-BLOCKER
evaluation.eval_dataset_manifest_sha256: TBD-BLOCKER
evaluation.eval_prompt_manifest_sha256: TBD-BLOCKER
qualification.limited_swiglu_evidence_sha256: TBD-BLOCKER
qualification.router_initialization_evidence_sha256: TBD-BLOCKER
qualification.ep_global_aux_evidence_sha256: TBD-BLOCKER
qualification.finite_consensus_evidence_sha256: TBD-BLOCKER
qualification.strict_replay_checkpoint_evidence_sha256: TBD-BLOCKER
qualification.expert_optimizer_sharding_evidence_sha256: TBD-BLOCKER
qualification.topology_collective_evidence_sha256: TBD-BLOCKER
qualification.rocm_kernel_qualification_evidence_sha256: TBD-BLOCKER
qualification.checkpoint_throughput_restore_evidence_sha256: TBD-BLOCKER
qualification.storage_signoff_evidence_sha256: TBD-BLOCKER
qualification.burn_in_evidence_sha256: TBD-BLOCKER
environment.container_digest: TBD-BLOCKER
environment.rocm_version: TBD-BLOCKER
environment.pytorch_commit_or_build: TBD-BLOCKER
environment.megatron_fork_commit: TBD-BLOCKER
environment.megatron_core_commit: TBD-BLOCKER
environment.rccl_version: TBD-BLOCKER
environment.grouped_gemm_backend_commit: TBD-BLOCKER
environment.attention_backend_commit: TBD-BLOCKER
environment.fused_ce_backend_commit: TBD-BLOCKER
launch.mcore_argv_adapter: TBD-BLOCKER
```

生成配置只证明 schema/accounting 可执行，不证明 MI300X kernel 或 MCore fork 已验收。

---

<a id="blockers"></a>

## 24. Launch blocker register

| ID | blocker | close condition |
|---|---|---|
| TOK-001 | tokenizer immutable revision/artifact hash 未生成；explicit padded-vocab override 未实现 | artifact hashes + `(151936,2048)` runtime assertion + out-of-range payload tests |
| DATA-001 | corpus counts/sidecars/shard hashes 未统一 | canonical corpus manifest 全量校验 |
| DATA-002 | holdout doc-level exclusion 与 suite artifacts 未生成 | `rfull-holdout-v1` manifest/hash + leakage tests |
| DATA-003 | stage scheduler/manifests 未生成或跨机器未复现 | four stage manifests + O(1) seek/world-size oracle |
| CFG-001 | production MCore fork、argv adapter 与 environment locks 未填 | resolved config/argv/env hashes，无 `TBD-BLOCKER` |
| ROUTER-001 | router std 0.01 patch 未实现 | tensor-init distribution/seed tests |
| AUX-001 | stock aux rank-local，不满足 EP-global autograd objective | EP1↔EP8 scalar/stat/gradient parity |
| ACT-001 | limited-SwiGLU 未覆盖三条生产路径 | Dense/routed/shared forward+backward kernel parity |
| NUM-001 | all-rank finite consensus + stop/replay transaction 未实现 | poison/failure injection across forward/backward/post-update |
| OPT-001 | FP32 RS/BF16 AG 与 expert-aware sharding 未由 trace 证明 | state ownership + gradient parity + RCCL trace |
| KERN-001 | attention/grouped-GEMM/dispatcher/CE/DCP ROCm paths 未验收 | Gate 1–4 parity、fallback、throughput、memory |
| CKPT-001 | atomic backend commit、world-size migration 与 boundary resume 未验收 | DCP fault matrix + two-location final restore |
| STO-001 | capacity 与 artifact lifecycle 未签核 | generated peak timeline incl. corpus placement/overhead/reserve |
| EVAL-001 | eval harness/dataset/prompt versions 未锁 | immutable eval manifest + phase-isolated pilot |
| LC-001 | long-context override/CP 未验收 | separate continuation project; does not block ≤32K baseline unless invoked |

任何 active blocker（`LC-001` 除外，只要不启动 continuation）都禁止 1T production launch。Gate 0–5 的签核记录、owner、evidence URI、timestamp 与 config hash 必须进入 release ledger。

---

<a id="references"></a>

## 25. Provenance 与审计边界

### 25.1 本地代码证据

- data/trainer repository：`https://github.com/chicm/pretrain`，审计 commit `8546388a80d864d3c45a40b13129ecbc2417ba8a`，branch `dev-chicm`；
- production-design repository：`https://github.com/chicm/pretrain-moe`，本文修改前 HEAD `8ee72494d25efb831747c43688695194e657bb35`；
- MCore API evidence checkout：NVIDIA Megatron-LM commit `d580efc68a9f0dbf1945f834f6f6200cd01d3343`（0.12 release）；它只是审计证据，不是尚未填充的 production fork lock。

审计的 data files：

- `docs/data_scaling_1T_design.md`
- `docs/chimera_1t_technical_report.md`
- `src/data.py`
- `src/data_mix.py`
- `src/train.py`
- `src/training_progress.py`
- `recipes/chimera_8b_1t.sh`

审计的 MCore files/functions：

- `megatron/core/transformer/moe/moe_utils.py::{topk_softmax_with_capacity,switch_load_balancing_loss_func,z_loss_func}`
- `megatron/core/transformer/moe/router.py::TopKRouter`
- `megatron/core/transformer/moe/token_dispatcher.py`
- `megatron/core/parallel_state.py::RankGenerator`
- `megatron/core/distributed/{distributed_data_parallel_config.py,param_and_grad_buffer.py}`
- `megatron/training/{arguments.py,checkpointing.py}`

### 25.2 文档审计基线

修订前快照 SHA-256：

```text
e410a07b8916c58b6c0d981b834cb2bbef6365184aa99a2764b2a2fd82c6125a
```

该 hash 对应 `r_full_moe_production_training_design.md` 的 pre-correction 版本。本文后续 SHA 由最终 validation 输出记录；不能用短 hash 或误抄字符串替代。

### 25.3 外部 primary sources

架构研究的逐项出处保留在配套报告中；生产约束的核心 upstream source 包括 NVIDIA/Megatron-LM、Qwen/Hugging Face tokenizer/model artifacts、MoonshotAI/Kimi-K3、DeepSeek-V4、OLMoE 与 MegaBlocks 官方仓库/报告。Qwen3.8-Max 在审计截止日没有公开 official weights/config/model code，因此不从 proxy geometry 推断其未公开字段。

### 25.4 证据等级

1. local tensor/count/runtime artifact；
2. pinned source code/official config/weight index；
3. official report/blog；
4. third-party discovery only。

生产锁只接受 1–2 级证据；3 级可解释设计背景，4 级不得决定 geometry 或 launch semantics。

---

## 附录 A：参数审计最小断言

```python
V, d, L = 151_936, 2_048, 48
L_dense, L_moe = 2, 46
h_q, h_kv, d_h = 32, 4, 128
f_dense, n_experts, top_k, f_expert = 5_504, 96, 6, 896

embedding = V * d
attention = L * (
    d * (h_q * d_h)
    + d * (h_kv * d_h)
    + d * (h_kv * d_h)
    + (h_q * d_h) * d
)
dense = L_dense * 3 * d * f_dense
expert_one = 3 * d * f_expert
routed_total = L_moe * n_experts * expert_one
routed_active = L_moe * top_k * expert_one
shared = L_moe * expert_one
routers = L_moe * d * n_experts
norms = L * 2 * d + d + L * 2 * d_h

total = embedding + attention + dense + routed_total + shared + routers + norms
active = embedding + attention + dense + routed_active + shared + routers + norms

assert total == 25_857_439_744
assert active == 3_066_640_384
assert norms == 210_944
assert L_dense + L_moe == L
assert n_experts % 8 == 0
```

## 附录 B：每阶段计数断言

```python
TOKENS_PER_STEP = 3_932_160
stages = [
    # seq, global_seq_batch, grad_accum, steps
    (4_096, 960, 8, 203_451),
    (8_192, 480, 4, 38_147),
    (16_384, 240, 2, 10_173),
    (32_768, 120, 1, 2_542),
]

assert all(seq * gbs == TOKENS_PER_STEP for seq, gbs, _, _ in stages)
assert all(gbs == 120 * ga for _, gbs, ga, _ in stages)
assert sum(steps for _, _, _, steps in stages) == 254_313
assert sum(steps * TOKENS_PER_STEP for _, _, _, steps in stages) == 999_999_406_080
```

## 附录 C：生产启动的最短否决清单

出现任意一项，立即否决正式启动：

- 参数量不是精确的 25,857,439,744；
- QK-Norm 没有 12,288 个 scales 或 scale 形状错误；
- EP group 跨节点；
- EP8 ranks 共享同一输入 sample；
- 每卡不是每层 12 routed experts；
- router 不是 selected-logit Top-6 softmax；
- 正常路径有 token dropping/capacity factor；
- grouped expert GEMM 退化成 Python loop；
- 16K/32K attention 生成 S² tensor；
- 缺任一 source 后自动重归一化；
- resume 需要从 sample 0 快进；
- checkpoint 不含 optimizer/scheduler/data/topology；
- exact ROCm/PyTorch/Megatron/RCCL/kernel SHAs 未记录；
- 只看到进程而未看到 120 GPU 活跃；
- 任一生产调试步骤需要 attach/暂停训练进程。

---

## 附录 D：独立审计 finding disposition matrix

| finding | disposition | final contract / reason |
|---|---|---|
| 参数总量与 12,288 Q/K norm 差异 | **adopted** | total `25,857,439,744`；Q/K RMSNorm scales 全计入 |
| active parameter 术语含混 | **corrected** | 定义为单 token 触达的 unique union；不等于 resident/batch union/operation count |
| tokenizer vocab 被当成模型 rows | **corrected** | native `151669` 与 padded rows `151936` 分开；显式 override |
| 由 128 自动 padding 得到 151936 | **rejected** | 实际会得到 `151680`；必须 runtime assert |
| `step`/attempt/update/skip 语义冲突 | **corrected** | step=`successful_update`；失败 stop/replay；无 committed numerical skip |
| 允许 ≤25 skips 的替代方案 | **rejected by design choice** | 用户选定严格 stop/replay；最终 tokens 精确不变 |
| LR peak `2.4e-4` 的审计建议 | **rejected** | 冻结 peak `2.0e-4`、final `2.0e-5` 的 1-based 公式 |
| source mix 只有近似权重 | **corrected** | 10K cycle + largest remainder + exact stage/source matrix |
| Python RNG/hash 可复现性不足 | **corrected** | SHA-256 domain-separated ordering + affine bijection |
| resume 需要 O(history) fast-forward | **corrected** | batch key 由 update/global slot 直接寻址，O(1) state |
| indexed corpus count 被称为 unique tokens | **corrected** | 降级为 provisional indexed payload claim，等待 manifest |
| physical shard/document/EOT 语义不清 | **corrected** | `[a,a+S+1)`、stride-S、no cross-shard、EOT CE/no reset |
| holdout 缺算法/数量/聚合 | **corrected + blocked** | 冻结 256×32K/source master windows 与派生 suites；artifact 待生成 |
| selected-softmax 与 aux global-softmax 混淆 | **corrected** | forward gates 与 aux probabilities 明确分离 |
| stock MCore aux 是 rank-local | **adopted blocker** | 需要 EP-global autograd-aware patch 与 EP1/EP8 gradient parity |
| router FP32 被解释为 FP32 model storage | **rejected** | BF16 parameter + FP32 projection/logits/softmax/loss |
| router std 0.01 被假定 stock | **corrected blocker** | 项目 init patch + tensor test |
| limited-SwiGLU threshold 7 | **rejected** | 三路径统一 threshold 10 |
| shared+routed 输出方差是算术错误 | **rejected** | ungated addition 是冻结语义；监控而非静默 rescale |
| A2A dispatch width 5504 | **rejected** | token dispatcher 搬运 hidden rows，`d_dispatch=2048` |
| A2A 348.446 GB/rank/update | **rejected** | 正确 forward+backward remote sent 为 `120.75 GiB` |
| gradient comm 未冻结 dtype/算法 | **corrected** | FP32 RS + BF16 AG；common DP120、routed EDP15 |
| comm 未区分 logical/send/Tx+Rx | **corrected** | 表中分别给 raw、one-way sent、Tx+Rx，并要求 profiler/NIC |
| persistent state 16/18-byte 粗估 | **corrected** | distributed optimizer lower bound `30.101906 GB/rank` |
| 约 5 TB 足够 | **rejected** | checkpoint/export peak lower bound 5.171 TB；co-resident+20% 10.708 TB，仍缺 overhead |
| raw telemetry 覆盖与 replay 双计 | **corrected** | append-only raw + COMMITTED ledger + final reachable lineage |
| validation 仅写 PPL/benchmark 名 | **corrected + blocked** | exact CE reductions/suites/cadence；eval artifact hashes 待填 |
| CP3/CP5 long-context 计划 | **rejected** | slicing 与 batch arithmetic 均不合法 |
| baseline 强行改 max positions=262144 | **rejected** | baseline 32768；单独 continuation override/migration gate |
| CP2/CP4/CP8 候选 | **adopted as non-normative** | 64K/128K/256K 算术可行但未资格认证 |
| illustrative YAML 可直接 launch | **rejected** | schema source→4 resolved stages→hash manifest→orchestrator；blockers 时拒绝 |
| 上游 CUDA-oriented path 可直接用于 MI300X | **rejected** | ROCm kernel/backend/checkpoint gates 全是 hard blockers |
| full-square attention FLOP 未声明 | **corrected** | 同时报 full-square 与 causal-triangular convention |
| Qwen3.8-Max proxy geometry 当事实 | **rejected** | 未公开字段保持 unknown |

审计结论不是“可启动”：矩阵中标为 blocker 的项目必须用 artifact 或 runtime evidence 关闭。本文保留首页的 launch prohibition。
