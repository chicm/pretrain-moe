# 从 Dense Transformer 到现代 MoE：一份由浅入深的系统教程

> 面向已经理解 decoder-only Transformer、pre-norm、SwiGLU、GQA、AdamW 和数据并行，但此前没有系统学习过 Mixture-of-Experts（MoE）的读者。
>
> 本文是 [`moe_pretraining_architecture_research.md`](./moe_pretraining_architecture_research.md) 的教学配套文档。前者回答“这个 20–30B 项目应该如何选型”，本文回答“这些技术到底是什么、为什么有效、代码和系统里实际发生了什么”。资料观察截止到 **2026-08-05**。

---

<a id="reading-guide"></a>
## 0. 阅读指南

### 0.1 建议的阅读顺序

如果你只想先建立可工作的直觉：

1. 第 1 章：Dense FFN 如何变成 MoE；
2. 第 2 章：一个 token 穿过 MoE 层时发生了什么；
3. 第 3 章：total parameters、active parameters 和 FLOP 为什么是三件事；
4. 第 4 章：如何选择 expert 数量、宽度、Top-k 和 shared expert；
5. 第 5 章：router 如何打分、选择和获得梯度；
6. 第 6～9 章：负载均衡、dropless、grouped GEMM 和 Expert Parallel；
7. 第 15 章：把这些知识映射回本项目的 R-Full/R-Hybrid/X-K3。

如果你要负责训练实现，应继续读第 10～17 章。Kimi K3、DeepSeek V4 中那些看起来很“前沿”的模块，集中放在第 12～14 章，并明确区分了：

- **MoE 本身的技术**：expert、router、Top-k、负载均衡、dispatch、Expert Parallel；
- **经常与 MoE 一起出现但并不属于 MoE 的技术**：Gated DeltaNet、KDA、MLA、AttnRes、mHC、MTP、Muon、低精度 QAT。

### 0.2 全文目录

- [0. 阅读指南](#reading-guide)
- [1. 从 Dense FFN 到条件计算](#dense-to-moe)
- [2. 一个 token 穿过 MoE 层的完整过程](#token-flow)
- [3. 参数、激活参数与 FLOP](#parameter-accounting)
- [4. 专家数量、宽度、Top-k 与共享专家](#expert-design)
- [5. Router 的数学、梯度与路由变体](#routing)
- [6. 为什么需要负载均衡](#balancing)
- [7. Capacity、token dropping 与 dropless](#capacity)
- [8. Dispatch、排序与 grouped GEMM](#kernels)
- [9. Expert Parallel 与 all-to-all](#expert-parallel)
- [10. 显存、通信和吞吐的联合账本](#cost-model)
- [11. MoE 训练为何比 Dense 更容易失稳](#stability)
- [12. 从经典 MoE 到 DeepSeekMoE](#classic-to-deepseek)
- [13. Stable LatentMoE 与 Quantile Balancing](#latent-moe)
- [14. 与 MoE 相邻但不同的前沿模块](#adjacent-tech)
- [15. 本项目四个候选的逐项拆解](#candidate-walkthrough)
- [16. 从 Dense 代码库迁移到 MoE 的实施路线](#migration)
- [17. 监控与故障诊断手册](#debugging)
- [18. 常见误解](#misconceptions)
- [19. 术语表](#glossary)
- [20. 自测题与答案](#exercises)
- [21. 资料边界与延伸阅读](#sources)

### 0.3 统一记号

| 记号 | 含义 |
|---|---|
| `d` | residual stream / hidden size |
| `f` | Dense SwiGLU 的 intermediate size |
| `f_e` | 单个 routed expert 的 intermediate size |
| `f_s` | shared expert 的 intermediate size |
| `N` | routed expert 总数 |
| `k` | 每个 token 选择的 routed expert 数，即 Top-k |
| `T` | 当前路由统计域内的 token 数 |
| `L_MoE` | MoE 层数 |
| `EP` | Expert Parallel size |
| `DP/TP/PP/CP` | Data/Tensor/Pipeline/Context Parallel size |
| `d_l` | LatentMoE 的 latent hidden size |
| `S` | sequence length |

除非特别说明：

- FFN 使用无 bias 的 SwiGLU；
- 参数量按逻辑参数统计，不按量化权重文件字节数统计；
- “active”指一个 token 的前向路径会使用到的参数，不等于精确 FLOP；
- MiB/GiB 使用二进制单位，GB 使用十进制单位；
- 本文中的“专家专业化”通常是待观察的涌现行为，不是由配置自动保证的事实。

### 0.4 先记住一句话

> **MoE 不是让一个 token 计算整个更大的模型，而是把 Dense FFN 换成一个专家库，再让每个 token 只调用其中少数几个专家。**

它的核心交换是：

- 用更多**总参数和模型容量**，换取近似受控的**单 token 计算量**；
- 但同时引入路由不连续、负载不均、跨 GPU token 搬运和更复杂的 checkpoint/并行状态。

---

<a id="dense-to-moe"></a>
## 1. 从 Dense FFN 到条件计算

### 1.1 从你已经熟悉的 Dense SwiGLU 开始

一个 Dense Transformer block 中，忽略 norm 和 residual 后，SwiGLU FFN 可以写成：

$$
\operatorname{FFN}(x)
=W_{down}\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right)
$$

若 `x ∈ R^d`、intermediate size 为 `f`，三个核心矩阵为：

- `W_gate ∈ R^{f×d}`；
- `W_up ∈ R^{f×d}`；
- `W_down ∈ R^{d×f}`。

忽略 bias，参数量是：

$$
P_{\text{SwiGLU}}=3df
$$

Dense 的关键语义是：**所有 token 都经过同一组 FFN 参数。** 无论输入是中文、Python、数学公式还是空格，这个 token 都执行同样三块矩阵，只是激活值不同。

### 1.2 MoE 做的唯一根本替换

MoE 保留 attention、residual、norm 和自回归目标，只把某些 Dense FFN 替换为：

1. `N` 个独立的 FFN，即 routed experts；
2. 一个很小的 router，为当前 token 给所有专家打分；
3. Top-k 选择，只执行得分最高的 `k` 个专家；
4. 按 gate 权重组合这些专家输出。

形式上：

$$
z=W_rx
$$

$$
p=\operatorname{Softmax}(z)
$$

$$
\mathcal T(x)=\operatorname{TopK}(p,k)
$$

$$
y_{routed}
=\sum_{i\in\mathcal T(x)}\alpha_iE_i(x)
$$

其中 `E_i` 是第 `i` 个 SwiGLU expert，`W_r ∈ R^{N×d}` 是 router 矩阵，`α_i` 常常是选中概率重新归一化后的权重。

如果还有 shared expert：

$$
y=y_{routed}+E_{shared}(x)
$$

放回完整 pre-norm block 后，模块边界是：

$$
u_l=h_l+\operatorname{Attention}(\operatorname{Norm}_{attn}(h_l))
$$

$$
x_l=\operatorname{Norm}_{ffn}(u_l)
$$

$$
h_{l+1}=u_l+\operatorname{MoE}(x_l)
$$

前面的简化公式中，`x=x_l`、`y=MoE(x_l)`。MoE 替换的是第二条 residual branch 里的 FFN，不会跳过 attention residual。

### 1.3 “Mixture”不是把全部专家都算一遍

经典的 dense mixture 会计算所有分支，再按概率加权；稀疏 MoE 则先做 Top-k，只计算少数分支。

这一区别决定了 MoE 的价值：

- total parameters 近似随 `N` 增长；
- 单 token 的 expert matmul 近似随 `k` 增长；
- 当 `k ≪ N` 时，总容量可以远大于单 token 计算路径。

例如 `N=96, k=6` 时，一个 token 只执行 `6/96=1/16` 的 routed expert 库。

### 1.4 MoE 没有改变什么

第一次接触 MoE 时，最容易把它误解成另一种 Transformer。实际上它通常没有改变：

- causal mask；
- next-token prediction；
- embedding 与 LM head；
- attention 的 Q/K/V 语义；
- residual stream 的宽度 `d`；
- token 顺序；
- tokenizer。

因此，把 Dense 模型迁移到 MoE 时，最稳妥的模块边界通常是：

```text
Dense block:
    x -> attention -> residual -> dense_ffn -> residual

MoE block:
    x -> attention -> residual -> router/dispatch/experts/combine -> residual
```

### 1.5 为什么“容量更大”可能有帮助

粗略直觉是：Dense FFN 必须用同一组参数拟合所有 token 分布；MoE 允许不同 token 使用不同参数子集，理论上可以降低参数间的任务干扰，并增加条件容量。

但要特别注意：

- 专家不会因为编号不同就自动变成“代码专家”“中文专家”；
- router 可能按语法、频率、位置、token identity 或人类难以解释的特征分工；
- 多个专家也可能学到高度冗余的函数；
- 负载均衡只保证“被调用得差不多”，不保证“语义专业化得好”。

所以 MoE 提供的是**形成专业化的结构条件**，不是专业化本身的证明。

### 1.6 MoE 的四层问题

理解现代 MoE，必须同时看四层：

| 层次 | 核心问题 | 典型失败 |
|---|---|---|
| 模型层 | `N/k/f_e/shared` 如何选 | active 过大、容量不足 |
| 优化层 | router 如何学、如何平衡 | expert collapse、抖动、NaN |
| 内核层 | 不同 expert 的 ragged GEMM 怎么高效执行 | GPU 利用率低、padding 浪费 |
| 分布式层 | token 和 expert 如何跨卡映射 | all-to-all 成为瓶颈、checkpoint 不可恢复 |

只理解第一层，还不能训练出一个高效 MoE。

### 1.7 为什么通常把 FFN 做成 experts，而不是 attention

FFN 是最自然的条件计算位置：

- 参数量大，扩充它能显著增加 total capacity；
- 在 attention 完成 token mixing 后，FFN 对每个 token 独立，容易按 token dispatch；
- expert 输出仍是同一个 hidden size，可以直接回到 residual；
- 不需要改变 causal mask、KV cache 或 token 间依赖。

如果把 attention 本身专家化，选择一个 token 的 Q/K/V 路径往往会影响其他 token、cache layout 和因果通信，系统语义复杂得多。确实存在 sparse/expert attention 研究，但它不是本文所说的标准 FFN-MoE。

---

<a id="token-flow"></a>
## 2. 一个 token 穿过 MoE 层的完整过程

### 2.1 单 token 的概念路径

以标准 `N=4, k=2` 的 MoE 为例。输入 token hidden state 为 `x`。

router 产生四个 logits：

```text
expert:       E0      E1      E2      E3
logit:       1.2    -0.4     2.0     0.7
softmax:    0.25    0.05    0.55    0.15
```

Top-2 选中 `E2` 和 `E0`。如果对选中概率重新归一化：

$$
\alpha_2=\frac{0.55}{0.55+0.25}=0.6875
$$

$$
\alpha_0=\frac{0.25}{0.55+0.25}=0.3125
$$

最终输出：

$$
y=0.6875E_2(x)+0.3125E_0(x)
$$

如果有 shared expert，则再加 `E_shared(x)`。

### 2.2 真正的 GPU 执行不是逐 token 循环

实际 batch 中有很多 token。高效实现不会写成：

```text
for token:
    for selected_expert:
        run_one_expert(token)
```

那会产生大量微小 GEMM 和 kernel launch。真实流程更接近：

```text
hidden states [T, d]
        │
        ├─ router logits [T, N]
        ├─ top-k ids      [T, k]
        └─ top-k weights  [T, k]
                    │
          展开为 T×k 条 assignment
                    │
          按目标 GPU / expert 排序并打包
                    │
          all-to-all dispatch（若使用 EP）
                    │
          本地再按 expert 分组
                    │
          grouped GEMM / block-sparse GEMM
                    │
          inverse all-to-all
                    │
          按原 token 索引恢复并加权求和
                    ▼
               output [T, d]
```

这里真正被搬运和排序的基本单位，不再只是 token，而是 **token-expert assignment**。Top-6 意味着每个 token 产生六条 routed assignment。

#### 一个可逐行核对的 `T=4,N=4,k=2` batch

先假定 Top-k 已产生下表，暂不启用 capacity/drop：

| token | slot 0 `(expert,gate)` | slot 1 `(expert,gate)` |
|---:|---|---|
| 0 | `(E2,0.6875)` | `(E0,0.3125)` |
| 1 | `(E1,0.60)` | `(E3,0.40)` |
| 2 | `(E2,0.55)` | `(E1,0.45)` |
| 3 | `(E0,0.75)` | `(E3,0.25)` |

按 token-major 展开后，assignment ID `a=token*k+slot` 依次为 `a0…a7`。按 `(expert_id,a)` 做 stable sort：

| expert | sorted assignment IDs | packed token IDs | packed gates |
|---:|---|---|---|
| E0 | `[a1,a6]` | `[0,3]` | `[0.3125,0.75]` |
| E1 | `[a2,a5]` | `[1,2]` | `[0.60,0.45]` |
| E2 | `[a0,a4]` | `[0,2]` | `[0.6875,0.55]` |
| E3 | `[a3,a7]` | `[1,3]` | `[0.40,0.25]` |

于是：

```text
perm       = [1, 6, 2, 5, 0, 4, 3, 7]
counts     = [2, 2, 2, 2]
offsets    = [0, 2, 4, 6, 8]   # exclusive prefix sum
source_tok = [0, 3, 1, 2, 0, 2, 1, 3]
```

每个 expert 只处理 `packed_hidden[offsets[e]:offsets[e+1]]`。返回八条 expert 输出后，不能只做 inverse permutation 就结束；还要保留 `source_tok` 与 gate，用 scatter-add/index-add 语义组合：

$$
y_t=\sum_{a:\ source\_tok[a]=t}gate_a\,z_a
$$

例如 `y_0=0.3125z_{E0,0}+0.6875z_{E2,0}`。hidden-state 梯度沿 gather/scatter 的值路径回传到原 token；expert ID/排序索引本身不可导，router 通过 selected gates 的连续值路径以及可选 auxiliary loss 获得梯度。

若启用 capacity，必须明确是在 stable sort 前还是 owner 收齐后筛 assignment，并保留 `accepted` mask。只有规范明确要求时，才对每个 token 剩余 gates 重新归一化；drop 后 assignment、gate 和返回索引必须一起删除，不能只丢 hidden row。

### 2.3 为什么 Top-k 会复制 token

如果 token `t` 被分到六个专家，它的 hidden state 必须送到六个 expert computation。逻辑上会产生：

$$
T_{assign}=T\times k
$$

条 assignment。即使源 hidden 只存一份，dispatch buffer 也通常要包含 `k` 份待发送表示或等价的 gather 结果。

所以：

- `k` 不只增加 expert FLOP；
- 它也线性增加 dispatch/combine 数据量；
- 还会增加 router metadata、排序和反向传播流量。

### 2.4 Shared expert 为什么不需要 routed all-to-all

shared expert 对每个 token 都执行。最自然的实现是让每个 rank 都保存它的副本并本地计算：

```text
x ──> local shared expert ───────┐
 │                               ├─> add
 └─> routed experts via EP ──────┘
```

因此 shared expert：

- 增加每 token 固定计算；
- 增加 replicated parameters；
- 但通常不增加 routed token 的 all-to-all payload。

它的梯度仍必须在持有副本的 ranks 间同步。

### 2.5 反向传播经过哪里

一次普通 Top-k 前向后：

- 被选中的 experts 收到该 token 的梯度；
- 未被选中的 experts 不从该 token 获得主任务梯度；
- router 通过选中 gate 权重获得梯度；
- Top-k 的离散“选中哪一个”边界本身通常不可微，常见实现只对选中 score 的连续部分求导；
- auxiliary balance loss 或无辅助损失的 bias 控制器提供额外的负载调节信号。

这解释了“dead expert”为何可能自我强化：某专家一旦很少被选中，就更少得到训练信号，也更难重新变好。

### 2.6 训练和推理的路由差异

架构上二者相同，但系统行为不同：

| 项目 | 训练 | 自回归推理 |
|---|---|---|
| token 数 | 通常很大，可聚合成大 grouped GEMM | decode 时每步 token 少 |
| 负载统计 | 可跨 microbatch/DP group 统计 | 请求分布随时变化 |
| 通信重叠 | 比较容易用大 batch 隐藏 | 小 batch 时延敏感 |
| expert GEMM | 较容易做大 | 容易退化为许多小 GEMM |
| 容量溢出 | 影响训练语义 | 影响请求尾延迟或输出 |

因此“训练吞吐不错”不自动意味着“单请求推理高效”。

### 2.7 路由是“逐 token、逐层”发生的

标准 MoE 中，每个 MoE 层有自己的 router 和自己的 expert bank：

- 同一 token 在第 10 层和第 11 层可以选择完全不同的 experts；
- 同一句话中的不同 token 也可以走不同路径；
- 第 10 层的 expert 7 与第 11 层的 expert 7 没有共享身份，只是各层局部编号；
- router 通常不把整条 sequence 固定交给某个“总专家”。

因此不要把一次路由想象成“先给文档分领域，再用一个子模型处理到底”。真实路径是每层重新做条件计算，组合空间远大得多。

---

<a id="parameter-accounting"></a>
## 3. 参数、激活参数与 FLOP

### 3.1 一个 routed expert 有多少参数

标准 SwiGLU expert 仍是三个矩阵：

$$
P_{expert}=3df_e
$$

`N` 个 routed experts：

$$
P_{routed,total}=N\cdot3df_e
$$

每个 token 只执行 `k` 个：

$$
P_{routed,active}=k\cdot3df_e
$$

再加一个 shared expert 和 router：

$$
P_{\text{MoE,total}}
=N\cdot3df_e+3df_s+dN
$$

$$
P_{\text{MoE,active}}
=k\cdot3df_e+3df_s+dN
$$

router 必须先给全部 `N` 个专家打分，因此 `dN` 是固定激活的。

### 3.2 “3B active”不是“模型只有 3B”

一个 26B total / 3.7B active 的模型意味着：

- checkpoint 和 optimizer state 仍要容纳约 26B 逻辑参数；
- 单 token 只走其中约 3.7B 参数对应的路径；
- 不同 token 走的 3.7B 子集不同；
- 整个数据集长期训练后，几乎所有专家都会被使用。

因此：

- **total** 更接近存储、模型容量和全量 checkpoint 问题；
- **active** 更接近每 token 线性层计算问题；
- **local parameters/GPU** 取决于 EP、TP 和 optimizer sharding。

三者不能互换；实际通信 bytes 还要另列第四本账。

### 3.3 Active parameters 也不是精确 FLOP

常用训练规划近似：

$$
F_{token}\approx6P_{active}+12L_{full}Sd_q
$$

这里 `L_full` 是 full-attention 层数，`S` 是该训练样本的 context length，`d_q=n_q d_h` 是总 query width（标准设计里常等于 hidden size `d`）。第一项把参与 dense matmul 的参数按前向和反向粗略换算；第二项近似 full attention 的 QK/AV 二次项。该式是训练规划近似，不是 profiler 定义；对纯推理 forward，系数和计数口径都不同。

它没有精确包含：

- embedding lookup；
- router softmax/Top-k；
- token 排序与复制；
- padding；
- norm 和激活函数；
- grouped GEMM 的利用率损失；
- all-to-all 等待；
- Gated DeltaNet/KDA 的 recurrent update；
- activation checkpoint 的重计算。

所以 active parameters 是架构比较指标，不是 profiler 的替代品。

### 3.4 用 R-Full 做一次完整计算

R-Full 的核心 MoE 几何：

- `d=2048`；
- 共 48 层，其中前两层 dense；
- 46 个 MoE 层；
- `N=96, k=6`；
- `f_e=f_s=896`；
- 一个 shared expert。

46 层 routed expert 总参数：

$$
46\times96\times3\times2048\times896
=24,310,185,984
$$

每 token 激活的 routed expert 参数：

$$
46\times6\times3\times2048\times896
=1,519,386,624
$$

46 个 shared experts：

$$
46\times3\times2048\times896
=253,231,104
$$

46 个 router：

$$
46\times2048\times96=9,043,968
$$

一层 32Q/4KV、head dim 128 的 GQA **projection** 参数为：

$$
P_{GQA\ proj/layer}
=d(n_qd_h)+2d(n_{kv}d_h)+(n_qd_h)d
=18,874,368
$$

下面把所有分项都列出来。这里“active”采用本文统一的结构口径；只有 routed expert 项按 `k/N` 稀疏，其余参数视为固定路径：

| 分项 | total | active |
|---|---:|---:|
| tied embedding/LM head | 311,164,928 | 311,164,928 |
| 两层 Dense SwiGLU-5504 | 67,633,152 | 67,633,152 |
| 46 层 routed experts | 24,310,185,984 | 1,519,386,624 |
| 46 层 shared experts | 253,231,104 | 253,231,104 |
| 46 个 routers | 9,043,968 | 9,043,968 |
| 48 层 GQA projections | 905,969,664 | 905,969,664 |
| 48 层 learned Q/K RMSNorm scales | 12,288 | 12,288 |
| 主 RMSNorms | 198,656 | 198,656 |
| **合计** | **25,857,439,744** | **3,066,640,384** |

这也揭示了一个重要事实：R-Full 的 total 几乎完全由 routed experts 主导，但 active 中 attention、embedding、shared expert 和 dense layers 占比明显上升。这里按项目现有 `QKNorm` 实现计入每层两个长度为 128、在同类 heads 间共享的可学习 scales（归一化仍逐 head 执行）；若未来增加额外 projection 或 MTP tensor，必须把它们作为新分项重新发布账本。

### 3.5 为什么 tied embedding 很重要

本项目词表 `V=151936`、`d=2048`：

$$
Vd=311,164,928
$$

如果 input embedding 和 output head 不共享权重，就额外增加同样的 311,164,928 参数。实际算子语义上，input embedding 是按 token ID 查一行，output head 才对整个词表做 logits matmul；二者并不是都在输出阶段执行。本文为了与公开模型命名和候选账本可比，把整块 embedding/head tensor 计入结构 active count，因此 untie 也会增加这一口径的 active 数。

这不是 MoE 特有问题，但在严格 20–30B total 预算中，0.311B 已经足以改变候选边界。

### 3.6 MTP 为什么必须单独计数

Multi-Token Prediction 可能增加额外 projection、norm、embedding/head 复用或独立 block。不同实现的 tensor 几何不同。

因此不能只说“加一层 MTP”就继续沿用原来的 total/active：

1. 先列出每个新增 tensor；
2. 判断训练时是否始终执行；
3. 判断推理部署时是否保留；
4. 分别发布 backbone-only 和 backbone+MTP 的 total/active。

本文候选数字均不包含可选 MTP。

### 3.7 “本 step 没激活”不等于“不需要保存状态”

某个 routed expert 在当前 microbatch 可能没有收到 token，因而主任务 gradient 为零；但它仍然是模型参数的一部分：

- checkpoint 必须保存它；
- AdamW moments/master weights 仍占显存；
- 后续 batch 仍可能激活它；
- expert replica group 必须对零 token/零 gradient 情况保持一致语义。

所以稀疏激活主要节省每 token matmul，不会按 `k/N` 自动缩小 optimizer state。

---

<a id="expert-design"></a>
## 4. 专家数量、宽度、Top-k 与共享专家

### 4.1 四个最重要的旋钮

MoE 层最核心的四个设计量是：

| 旋钮 | 增大后首先增加什么 | 常见代价 |
|---|---|---|
| `N` 专家总数 | total capacity | checkpoint、router、并行复杂度 |
| `k` Top-k | active capacity | FLOP、A2A、combine 开销 |
| `f_e` 单专家宽度 | 单专家表达力 | total 和 active 都增大 |
| `f_s` shared 宽度 | 每 token 固定通路 | replicated state 和固定 FLOP |

在标准 MoE 中：

- total routed width 与 `N f_e` 成正比；
- active routed width 与 `k f_e` 成正比；
- 稀疏比率可以粗看作 `k/N`，但 shared expert 和非 MoE 参数不会随它稀疏。

### 4.2 粗粒度专家与细粒度专家

**粗粒度**：专家较少、每个很宽，例如 Mixtral 8×7B 的 8 experts / Top-2。

**细粒度**：专家更多、每个较窄，例如 Qwen3、DeepSeekMoE，以及本项目 R 系列。

在固定 active expert width 下，可以有：

```text
粗粒度：2 个 × 3072
细粒度：6 个 × 1024
更细粒度：8 个 × 768
```

三者 active intermediate width 都约 6144，但系统行为不同。

细粒度的潜在优点：

- 一个 token 可组合更多子功能；
- 专家组合数量更大；
- 固定 total budget 下可以提供更细的条件参数分配。

潜在代价：

- assignment 数随 `k` 增加；
- 单 expert token batch 更小，GEMM 更难跑满；
- router 元数据和排序开销更高；
- 专家数过大时 EP 布局和负载统计更复杂。

所以“专家越多越好”不成立。专家粒度必须与 grouped GEMM、EP 拓扑和 batch token 数共同设计。

### 4.3 为什么 R 系列选择 96 / Top-6

R 系列的 routed 激活比例：

$$
\frac{k}{N}=\frac{6}{96}=\frac1{16}
$$

Qwen3-30B-A3B 的比例：

$$
\frac8{128}=\frac1{16}
$$

R 系列加一个 shared expert 后，每 token 的 FFN intermediate width 为：

$$
6\times896+1\times896=6272
$$

Qwen3-30B-A3B routed 路径为：

$$
8\times768=6144
$$

这是一种有意义的控制：保留接近的 active FFN width 与稀疏比例，同时减少专家数和 Top-k，降低 grouped GEMM 与路由复杂度。

### 4.4 Shared expert 是什么

shared expert 是不经 Top-k、所有 token 都执行的 FFN。设计动机通常是：

- 给所有 token 一条稳定的通用通路；
- 让 routed experts 更有机会学习差异化残差；
- 在 router 暂时不稳定时保留一定计算能力。

但这些是设计动机，不是自动发生的因果事实。shared expert 也可能：

- 吸收过多能力，让 routed experts 利用不足；
- 增加固定 active FLOP；
- 因为在 EP ranks 上复制而增加每卡状态；
- 需要完整 replica-group 梯度同步。

因此要监控 shared/routed 输出 RMS、梯度范数和消融结果，而不是只凭直觉判断。

### 4.5 为什么前 1～2 层常保留 Dense

一些公开 MoE 在最前面保留 Dense FFN。常见假设是：

- 浅层 token 表示尚不稳定，立即稀疏路由可能增加噪声；
- 先做通用特征变换，再让深层条件分工；
- 减少最浅层高 token-rate 路由的工程复杂度。

本项目保留前两层 dense，属于保守工程选择。但必须注意：仅由 config 看到“首层 dense”，不能证明它一定提升稳定性；这仍应通过消融验证。

### 4.6 专家是否真的会按领域专业化

可以观察，但不要先验命名。合理分析包括：

- 不同数据域的 expert load 分布；
- token ID、词类、语言、代码符号与专家的互信息；
- 同一个 token 在不同上下文中的路由变化；
- 专家输出相似度和权重相似度；
- 层间专业化是否不同；
- dead/redundant experts 数量。

一个专家经常接收 Python token，不代表它就是“Python 专家”：可能只是频率、位置或格式特征造成的相关性。

### 4.7 设计时最实用的等式

先冻结 active budget，再搜索 `N/k/f_e/f_s`：

$$
W_{active}=kf_e+f_s
$$

再检查 total expert budget：

$$
W_{total}=Nf_e+f_s
$$

最后检查系统可执行性：

- `N` 能否均匀分到 EP ranks；
- `f_e` 是否适合 GPU tile；
- 每个 local expert 每 microbatch 能收到多少 token；
- `k` 带来的 A2A 是否可接受；
- 最坏负载下是否 OOM。

模型公式只完成了设计的一半。

### 4.8 每个 expert 实际看到多少训练 token

若训练语料有 `D` 个有效 token、路由近似均匀，则每层每个 routed expert 获得的 assignment 数约为：

$$
D_{per\ expert}\approx D\frac{k}{N}
$$

R 系列 `k/N=1/16`。若训练 1T token，每个 routed expert 在每个 MoE 层约看到 62.5B token assignments；shared expert 和 dense layers 则看到全部 1T token。

这带来两个设计约束：

- 固定数据量下无限增加 `N`，会降低单 expert 的训练样本；
- 专家变窄可能降低单 expert 所需容量，但不能假设极少数据也能学好；
- 真实不均衡会让冷门 experts 看到的 token 更少；
- 数据域变化还会改变各 expert 的有效训练分布。

因此 total parameters 很大不代表每个参数都获得了与 Dense 模型相同数量的训练信号。

---

<a id="routing"></a>
## 5. Router 的数学、梯度与路由变体

### 5.1 最常见的 learned token-choice routing

“token choice”表示每个 token 选择专家。最常见形式：

$$
z_t=W_rh_t
$$

$$
p_{t,i}=\frac{e^{z_{t,i}}}{\sum_j e^{z_{t,j}}}
$$

然后对每个 token 取 Top-k：

$$
\mathcal T_t=\operatorname{TopK}_i(p_{t,i},k)
$$

选中后常做重新归一化：

$$
\alpha_{t,i}
=\frac{p_{t,i}}{\sum_{j\in\mathcal T_t}p_{t,j}},
\quad i\in\mathcal T_t
$$

重新归一化让 routed mixture 的权重和为 1；不重新归一化则保留“router 总置信度”尺度。两种语义不能在不验证的情况下互换。

### 5.2 Router 很小，为什么却很重要

router 只有 `dN` 参数，通常远小于 experts，但它控制：

- 哪些专家得到主任务梯度；
- 每卡收到多少 token；
- grouped GEMM 的 batch shape；
- all-to-all 的 send/recv count；
- 训练是否 OOM；
- 推理尾延迟。

因此 router 参数小，不代表它对系统影响小。

### 5.3 Top-k 的不可微边界

在普通实现里，Top-k 选中的 score 可继续参与连续求导，但“第 k 名和第 k+1 名谁被选中”是离散变化。

当两者 margin 很小时，极小数值扰动就可能改变路径：

$$
\Delta_t=z_{t,(k)}-z_{t,(k+1)}
$$

`Δ_t` 很小意味着路由脆弱。应监控：

- Top-k margin 分布；
- 固定 probe batch、关闭随机 jitter 后跨 checkpoints 的 route flip rate；
- router entropy；
- 每 expert 的 token count。

普通训练中相邻 steps 的 batch 不同，不能把两批不同 token 的 expert IDs 直接比较成“flip”。对固定 probe 中同一 layer、同一 token position，若两次 Top-k 集合分别为 `S_a,S_b`，可定义：

$$
\operatorname{flip}=1-\frac{|S_a\cap S_b|}{k}
$$

还应同时看 score margin；大 flip 可能来自参数快速变化，也可能只是大量 token 长期位于几乎并列的决策边界。

### 5.4 为什么第一版把 FP32 router 当作保守基线

BF16/FP16 的 score 舍入可能改变接近边界的 Top-k 次序。为了建立容易审计的 reference，可先采用：

1. hidden state 和 router weights 仍可按配置存为 BF16；
2. projection 使用 FP32 accumulation，logits/score normalization 在 FP32 中处理；
3. Top-k ID 在 FP32 scores 上决定；
4. combine weights 再转换到 expert/combine 所需 dtype。

这不是所有前沿模型的统一要求：成熟实现可能在 BF16/FP8 或 fused kernel 中完成部分 router 计算。若优化 dtype，必须用 score、Top-k IDs、load、loss 和长时稳定性对齐来证明，而不是默认与 FP32 reference 等价。

### 5.5 Router jitter/noise 的作用

训练早期可向 logits 或输入加入小噪声，目的不是提高随机性本身，而是：

- 打破专家完全对称；
- 避免少数专家因微小初始化优势永久垄断；
- 让更多专家获得早期梯度。

噪声过大则会：

- 增加 route churn；
- 破坏专家形成稳定分工；
- 放大分布式非确定性。

所以它更像有限时长的探索机制，应有显式 schedule 和监控。

### 5.6 Hash routing

Hash routing 不学习“哪个专家最好”，而是依据 token ID 或其他离散键进行确定性映射。

优点：

- 给定同一 hash 函数、token ID、层号和实现版本时，路由是确定性的；
- 期望负载可通过映射设计得较均匀；
- 不存在 learned router 的早期 collapse。

但确定性不等于任何 batch 都均衡：token 频率偏斜、hash 冲突或 tokenizer 分布都可能造成实际负载不均。

缺点：

- 同一 token 在不同上下文中可能被强制送到同一集合；
- 路由不能根据语义自适应；
- 高频 token 的负载仍需特殊处理；
- hash 规则本身成为模型归纳偏置。

DeepSeek V4 在前若干层使用 token-ID hash routing，之后再使用 score routing。这是特定大规模训练方案，不意味着中型 MoE 默认也应采用。

### 5.7 Expert Choice

普通 token-choice 是“每个 token 选 k 个专家”。Expert Choice 反过来：

- 每个 expert 在一个 token 集合中选择自己最想处理的固定数量 token；
- expert batch 天然有上限，负载更容易均衡；
- 但每个 token 获得的专家数可能不同，甚至可能没有被任何 expert 选中；
- 自回归、因果分布式和在线推理中的语义与实现更复杂。

它适合用来理解“负载均衡也可以改变选择方向”，但不是本项目第一版路线。

### 5.8 `sqrt(softplus)` score

DeepSeek V4 的 score routing 不直接用普通 softmax score，而使用正值变换：

$$
s_i=\sqrt{\operatorname{softplus}(z_i)}
$$

直觉上：

- `softplus` 保证正值且比指数增长温和；
- 平方根进一步压缩大 score 的动态范围；
- Top-k 仍可依据修正后的 score/bias 选择；
- 选中专家的权重可再归一化。

这是一种具体系统中的设计，不能只替换一行激活就假设会复制其稳定性；它与 correction bias、sequence-wise balance、初始化和整体训练 recipe 共同工作。

### 5.9 Group-limited routing

如果 EP 跨多个节点，可限制 token 只在少数 expert groups 内选择，减少跨节点目的地和 all-to-all fan-out。

代价是：

- token 不再能从全部专家中自由选择；
- group 本身可能失衡；
- 需要分层打分和组级容量控制。

本项目 EP8 完全放在单节点 xGMI 域内，因此第一版没有必要增加这层限制。

### 5.10 路由算法的选择原则

对第一版中型 MoE，优先级通常是：

1. 行为可解释；
2. 单卡 reference 可验证；
3. 多卡统计可复现；
4. 与 grouped GEMM/EP kernel 匹配；
5. 再考虑是否比 softmax Top-k 有质量收益。

因此本项目默认从 **FP32 softmax Top-k + 小权重辅助均衡** 开始，而不是一次引入 hash、无辅助 bias、Quantile Balancing 和 Anticipatory Routing。

<a id="balancing"></a>
## 6. 为什么需要负载均衡

### 6.1 Expert collapse 是怎样形成的

假设初始化时 expert 3 因随机波动比其他专家多收到一些 token：

1. expert 3 获得更多主任务梯度；
2. 它更快变得“有用”；
3. router 更倾向继续选择它；
4. 其他专家获得的训练信号更少；
5. 正反馈最终让少数 expert 过载，其余 expert 接近死亡。

这会同时损害：

- **质量**：大部分总参数没有被有效训练；
- **吞吐**：过载 expert 决定整层尾延迟；
- **显存**：单 expert 的动态 token buffer 可能溢出；
- **容错**：负载尖峰使训练偶发 OOM。

因此 MoE 的“均衡”不是为了图表好看，而是模型语义和系统可执行性的共同约束。

### 6.2 Load 与 importance 是两个量

对 `T` 个 token、Top-k=`k`：

**离散负载**：expert `i` 实际收到多少 assignment：

$$
n_i=\sum_{t=1}^{T}\mathbf1[i\in\mathcal T_t]
$$

目标平均负载：

$$
q=\frac{Tk}{N}
$$

归一化 load fraction：

$$
f_i=\frac{n_i}{Tk}
$$

**连续 importance**：先定义 Top-k 之前、跨全部 `N` 个 experts 的 router 概率 `p_{t,i}`，再取平均概率质量：

$$
\bar p_i=\frac1T\sum_{t=1}^{T}p_{t,i},
\qquad \sum_i\bar p_i=1
$$

它不是最终 combine gate。若选中集合内重新归一化，则真正参与混合的是：

$$
g_{t,i}=\mathbf1[i\in\mathcal T_t]
\frac{p_{t,i}}{\sum_{j\in\mathcal T_t}p_{t,j}},
\qquad \sum_{i\in\mathcal T_t}g_{t,i}=1
$$

二者不能混用：某专家可能经常排第 `k+1`，`\bar p_i` 不低，却从未真正被选中；也可能大量低权重地进入 Top-k，load 很高但 importance 一般。

### 6.3 经典 auxiliary balance loss

Switch Transformer 原始的 Top-1 形式可写成：

$$
L_{aux}=N\sum_{i=1}^{N}f_i\bar p_i,
\qquad f_i=\frac{n_i}{T}
$$

本文对 Top-k 明确采用 assignment-normalized 扩展，即沿用 §6.2 的 `f_i=n_i/(Tk)`：

$$
L_{aux}^{(top-k)}=N\sum_{i=1}^{N}f_i\bar p_i
$$

在理想均匀状态 `f_i=\bar p_i=1/N` 时，两式的基准值都是 1。训练实际加入：

$$
L=L_{LM}+\lambda_{aux}L_{aux}^{(top-k)}
$$

直觉是：已经获得较多离散 assignment 的 expert，不应继续积累过高的全专家概率质量。

这只是一个**已声明的 Top-k 约定**，不是所有代码库都相同。有的实现把选中指示量除以 `T`（此时 `\sum_i f_i=k`），有的平衡选中后的 gate mass，有的按 sequence 或 layer 求和；这些差异会改变基准值与有效系数，`λ_aux` 不能脱离公式、统计域和 reduction 方式直接搬运。

实现细节：

- `f_i` 来自离散 Top-k，通常停止梯度；
- 梯度主要通过全专家分布 `\bar p_i` 回到 router；
- `p_{t,i}`、选中后 `g_{t,i}` 与离散 `f_i` 必须使用不同变量名；
- Top-k>1 时必须把 assignment 归一化和 gate 归一化分别写清；
- padding token、被 mask token 不应混入统计；
- 跨 EP ranks 的统计域与 layer/sequence reduction 必须明确。

### 6.4 为什么 auxiliary loss 不能太大

`λ_aux` 太小：

- 来不及纠正 collapse；
- expert max/mean load 迅速增大；
- dropless buffer 压力上升。

`λ_aux` 太大：

- router 优先满足“平均分配”，而不是“把 token 交给最合适专家”；
- 专家选择趋近人为均匀；
- 主语言建模 loss 可能受损；
- 路由专业化被削弱。

所以 auxiliary loss 是约束，不是第二个主任务。第一版应从已验证的小系数开始，并让监控决定是否调整。

### 6.5 CV、max/mean 和熵分别告诉你什么

常用负载指标：

$$
\operatorname{CV}(n)
=\frac{\operatorname{Std}(n_i)}{\operatorname{Mean}(n_i)}
$$

$$
R_{max}=\frac{\max_i n_i}{\operatorname{Mean}(n_i)}
$$

router entropy：

$$
H_t=-\sum_i p_{t,i}\log p_{t,i}
$$

解释：

- CV 高：整体离散不均衡；
- max/mean 高：存在决定尾延迟的热点；
- entropy 极低：router 很确定，可能过早塌缩；
- entropy 极高：router 近似随机或没有学到区分；
- entropy 正常也不保证 load 正常，因为 Top-k 边界可能集中。

还应监控 min load、零负载 expert 数、Top-k margin 和 route flip rate。

### 6.6 统计域：token、sequence、microbatch、EP group

“负载均衡”必须先回答“在哪个集合上均衡”。

- **单 sequence**：防止一条长序列的 token 全部挤到少数专家；
- **microbatch**：最易实现，但统计噪声可能大；
- **EP group**：与真正共享专家集合的 dispatch 域一致；
- **跨 DP 全局**：统计更稳，但同步开销更大，且可能掩盖某个本地 EP group 的热点。

[DeepSeek V4 技术报告](https://arxiv.org/abs/2606.19348)保留轻量 sequence-wise balance，正是为了避免只看大 batch 平均后，单条序列内部仍严重集中。

### 6.7 Router z-loss

z-loss 约束 log-sum-exp 的尺度：

$$
L_z=\frac1T\sum_t
\left(\log\sum_i e^{z_{t,i}}\right)^2
$$

作用是避免 logits 整体绝对值无限漂大。包含 z-loss 时，本文采用的目标写成：

$$
L=L_{LM}+\lambda_{aux}L_{aux}^{(top-k)}+\lambda_zL_z
$$

这里 `L_z` 对当前统计域内的有效 token 取 mean。`λ_z`、是否逐层再求和/平均、padding mask 和 accumulation reduction 都必须进入版本化配置与 checkpoint manifest；若另一实现对 token 或 layer 采用 sum，数值相同的 `λ_z` 并不等价。

它和负载均衡不同：

- z-loss 管数值尺度；
- auxiliary balance 管专家使用分布；
- 一个稳定不代表另一个稳定。

### 6.8 无辅助损失 correction bias

Auxiliary-loss-free routing 的核心思想是把“选择均衡”从主模型梯度中拆出来。

设原始 affinity 为 `s_{t,i}`，每个 expert 有一个不参与反向传播的 correction bias `b_i`：

$$
\mathcal T_t=\operatorname{TopK}_i(s_{t,i}+b_i,k)
$$

但 combine weight 仍由**未加 bias 的原始 affinity**计算。训练过程中：

- expert 过载：降低它的 `b_i`；
- expert 欠载：提高它的 `b_i`；
- `b_i` 是控制器状态，不是靠 LM loss 学出的参数。

优点：不把均衡梯度直接注入语言建模目标。风险：

- bias 更新步长过大时路由振荡；
- 更新太慢时来不及处理热点；
- 需要正确聚合统计并随 checkpoint 保存；
- 控制器稳定不等于专家质量更好。

DeepSeek 系列采用了这类思路，但第一版中型 MoE 用小权重 auxiliary loss 更容易调试。

训练结束后，auxiliary loss 本身不参与推理；而最终 correction bias 若属于路由定义，通常要随 checkpoint 保留并在推理时冻结使用。丢掉它会改变 Top-k 路径。具体是否保留必须由模型实现/config 明确，不能把“无梯度”误解成“可以不保存”。

### 6.9 Quantile Balancing 在哪里不同

普通 correction bias 常直接根据“本 step 多了还是少了多少 token”更新。Quantile Balancing 则观察每个专家 margin/score 分布的目标分位点，估计使该专家获得目标容量 `q=T_ctrl k/N` 所需的 bias；这里 `T_ctrl` 是控制器定义的全局统计域，而不是默认等于单 rank microbatch。机制事实以 [Kimi K3 官方技术报告](https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf) 为准。

它仍遵守两个重要原则：

1. bias 只影响 Top-k 选择；
2. combine weight 使用原始 affinity，不让 bias 伪装成模型置信度。

第 13 章会详细解释。现在只需知道：它是更复杂的负载控制器，不是另一种 expert FFN。

### 6.10 一个可执行的均衡监控表

每层至少记录：

| 指标 | 目的 |
|---|---|
| min/mean/max tokens per expert | 看死专家和热点 |
| load CV、max/mean | 看离散不均衡和尾部 |
| probability importance CV | 看 router 连续偏好 |
| router entropy | 看过度确定或近似随机 |
| Top-k margin p1/pk/p(k+1) | 看选择稳定性 |
| fixed-probe route flip rate | 看同一批 token 跨 checkpoints 的路由抖动 |
| dropped assignments | 保证 dropless 语义 |
| per-expert input/output RMS | 看某些专家数值爆炸 |
| per-expert grad norm | 看训练信号是否缺失 |
| correction bias 范围 | 看控制器是否发散 |

只记录全模型平均值是不够的；MoE 问题往往只发生在少数层和少数专家。

---

<a id="capacity"></a>
## 7. Capacity、token dropping 与 dropless

### 7.1 为什么要有 capacity

先定义容量统计域。令 `T_cap` 为**进入同一个容量竞争域的有效 token 数**（不含 padding），`N` 为该域中的 global experts 数，每 token Top-k。若 capacity 在完整 EP routing group 的 expert owner 侧、all-to-all 后统一应用，则 `T_cap` 应是该 EP group 汇总的有效 token；若实现选择源 rank/microbatch 局部 capacity，那是另一套语义，必须单独记录，不能混用。

平均每 expert assignment 为 `q=T_cap k/N`，可以定义容量：

$$
C=\left\lceil c\frac{T_{cap}k}{N}\right\rceil
$$

其中 `c` 是 capacity factor，例如 1.0、1.1 或 1.25。还必须冻结 capacity 是按单个 microbatch、整个 gradient-accumulation window，还是其他 routing group 计算；这些选择会改变溢出率和可复现性。

一个 R-Full 数字例子：EP8 中每个 source rank 有 1,024 个有效 token，若 capacity 在 owner 收齐整个 EP group 的 assignments 后执行，则 `T_cap=8×1,024=8,192`、`N=96`、`k=6`。于是：

$$
q=\frac{8,192\times6}{96}=512,
\qquad C=\lceil1.25\times512\rceil=640
$$

每个 rank 拥有 12 个 experts，所以本 rank 的静态 capacity rows 为 `12×640=7,680`；均匀路由下预期收到 `8,192×6/8=6,144` 条 assignments。若误把 source-local 的 `T_cap=1,024` 代入同一 global-expert 公式，会得到 `C=80`，恰好少 8 倍并造成非预期 dropping。实际 incoming rows 仍会随路由偏斜波动，因此 `C=640` 不是显存峰值的替代品。

固定 capacity 的好处：

- buffer shape 可静态预分配；
- 每个 expert 的最大 GEMM 大小已知；
- 更容易做静态图和内存规划。

但真实负载不会刚好均匀。超过 `C` 的 assignment 必须被处理。

### 7.2 Token dropping 到底丢掉什么

通常丢的是某个 token 到某个 expert 的 **assignment**，不一定把整个 token 从 Transformer 删除。

例如 Top-2 token 的第二条 assignment 溢出：

- 可能只保留第一位 expert；
- 可能把溢出分支输出设为零；
- residual stream 仍然存在；
- combine 权重可能需要重新归一化。

这改变了模型函数，而且拥塞最严重的 token 受到的改变最多。训练和推理若采用不同 drop 语义，还会产生分布偏移。

### 7.3 Capacity factor 的两难

`c` 较小：

- padding/空闲容量少；
- 但 drop 多，语义受损。

`c` 较大：

- drop 少；
- 但静态 buffer 和 padding 浪费增大；
- 最大 expert 决定的 GEMM 可能拖慢全层。

负载 loss 能缓解，但无法保证每个 microbatch 完全均匀。

### 7.4 Dropless 的准确含义

Dropless MoE 的目标是：**所有有效 Top-k assignment 都被执行，不因固定 expert capacity 而静默丢弃。**

这不意味着：

- 每个 expert 负载相同；
- 没有 padding；
- 没有 OOM；
- 没有尾延迟；
- router 可以不做均衡。

Dropless 把问题从“超过容量就删”变成“如何高效执行 ragged workload”。

### 7.5 MegaBlocks 的思路

MegaBlocks 把不同 expert 的不规则 token 块映射成 block-sparse 或 grouped matrix operations：

- 不必把每个 expert padding 到统一最大容量；
- 不必为静态容量而丢 assignment；
- 让大矩阵内核处理由多个小 expert batch 组成的块结构。

它解决的是执行效率，不替 router 做负载均衡。若一个 expert 收到十倍 token，它仍会占用更多计算和内存。

### 7.6 为什么 dropless 仍要 OOM safety cap

生产实现应有明确的最坏负载保护，例如：

- 最大 local assignment 数；
- 最大单 expert token 数；
- dispatch buffer 上限；
- 超限时 fail-fast、重试或缩小 microbatch。

这不是“恢复 token dropping”。关键区别是：

- 不应在主训练中静默改变模型语义；
- 应让异常显式暴露并被监控；
- safety threshold 是故障保护，不是常态容量机制。

### 7.7 Padding token 不能进入路由账本

packing 或固定长度 batch 中的 padding 若被送入 router，会：

- 污染 load statistics；
- 浪费 expert FLOP；
- 造成某些 padding pattern 的伪专业化；
- 让不同 sequence length 的实验不可比较。

有效 token mask 必须同时作用于主 loss、router、负载统计和 dispatch。

---

<a id="kernels"></a>
## 8. Dispatch、排序与 grouped GEMM

### 8.1 为什么不能给每个 expert 单独调用一次 MLP

单个 expert 的计算形状是：

$$
[M_i,d]\times[d,f_e]
$$

其中 `M_i` 是该 expert 当前收到的 token 数。若 `N=96`，逐 expert 启动三个 SwiGLU GEMM，会产生：

- 大量 kernel launch；
- 很多小 `M_i` GEMM；
- GPU occupancy 低；
- CPU/driver launch 开销显著；
- 不均衡导致部分 kernel 极小、部分很大。

因此 MoE 高效与否，往往不由理论 FLOP 决定，而由“能否把 ragged expert workload 合成高效 GPU 工作”决定。

### 8.2 Dispatch 前的元数据

Top-k 后至少有：

- `expert_id[T,k]`；
- `gate_weight[T,k]`；
- `source_token_id[T,k]`；
- `destination_rank[T,k]`；
- 每个 destination/expert 的 count 与 prefix offset。

常见步骤：

1. flatten 成 `T×k` assignments；
2. 按 destination rank 计数；
3. prefix sum 得到 send offsets；
4. gather hidden states 到连续 send buffer；
5. all-to-all；
6. 按 local expert 再排序或直接按预计算 offset 布局。

反向和 combine 需要保存足够的逆映射。

### 8.3 Grouped GEMM

Grouped GEMM 接收一组不同 `M_i`、但共享或相近 `K/N` 维度的矩阵乘：

```text
expert 0: [M0, d] × [d, f_e]
expert 1: [M1, d] × [d, f_e]
...
expert r: [Mr, d] × [d, f_e]
```

由一个调度器/内核批量执行，减少 launch 开销并改善资源利用率。

SwiGLU 至少涉及：

1. gate/up projections；
2. activation 与 elementwise multiply；
3. down projection；
4. backward 的 dgrad 和 wgrad。

只验证 forward grouped GEMM 远远不够；训练吞吐常受 backward、wgrad accumulation 和 workspace 支配。

### 8.4 Block-sparse 与 grouped GEMM 的关系

两者都在解决 ragged experts：

- grouped GEMM：把多个独立 dense GEMM 作为一个 group 调度；
- block-sparse：把 token-expert 关系组织成稀疏块矩阵，再调用稀疏矩阵内核。

哪个更好取决于：

- expert 数；
- 每 expert token 数分布；
- `d` 与 `f_e` tile；
- dtype；
- 硬件和 kernel 库；
- backward 支持。

不能因为某篇论文用 block-sparse，就假设 ROCm 上同样最优。

### 8.5 Combine 阶段

expert 输出返回源 rank 后，要按 `source_token_id` scatter/reduce：

$$
y_t=\sum_{j=1}^{k}\alpha_{t,j}y_{t,j}
$$

关键细节：

- 是否在发送前乘 gate weight，还是返回后乘；
- accumulation 使用 BF16 还是 FP32；
- Top-k 权重是否重新归一化；
- 多 assignment 写同一 token 时是否确定性归约；
- shared expert 在何处相加。

这些细节会让“看似相同”的两个实现产生数值差异。

### 8.6 内存布局影响很大

专家权重可按：

```text
[N_local, 3, f_e, d]
```

或拆成 gate/up/down 张量。高效布局需要考虑：

- contiguous tile；
- transpose 是否在 load 时完成；
- optimizer state 的同构分片；
- checkpoint tensor 命名；
- quantization block；
- kernel 期望的 expert-major 或 matrix-major 顺序。

一旦 checkpoint 已大规模产生，改变 expert layout 的迁移成本很高，所以必须在试训阶段冻结。

### 8.7 ROCm/MI300X 上必须单独验收什么

至少要测：

- BF16 forward/dgrad/wgrad 正确性；
- grouped GEMM 对 `f_e=896` 的 tile 利用率；
- 不同 load CV 下吞吐曲线；
- all-to-all 与计算 overlap；
- non-contiguous/packing 输入；
- activation checkpoint 重算；
- 多轮 checkpoint/resume 后数值一致；
- 8 卡 xGMI 域内 collectives；
- 24～72 小时 soak test。

HF modeling code 能说明 tensor 几何，不等于提供这些训练内核。

---

<a id="expert-parallel"></a>
## 9. Expert Parallel 与 all-to-all

### 9.1 EP 的基本思想

若一个 MoE 层有 96 个 routed experts、EP=8，则每个 EP rank 保存：

$$
96/8=12\text{ experts}
$$

输入 token 初始分布在 8 个 ranks 上，但它选择的 expert 可能位于任意 rank。因此每层需要：

1. dispatch all-to-all：把 hidden 送到 expert 所在 rank；
2. 本地 expert 计算；
3. combine all-to-all：把输出送回 token 的源 rank。

EP 分的是 expert weights，不是把一个 expert 的矩阵切成八份；后者是 TP。

### 9.2 All-to-all 与 all-reduce 不同

- **all-reduce**：每个 rank 对同形张量求和并得到相同结果；
- **all-to-all**：每个 rank 给每个其他 rank 发送不同数据片段；
- **all-to-all-v**：各目的地长度可不同，更符合不均衡路由，但实现与性能更复杂。

MoE 每层的 token dispatch 是 all-to-all；optimizer step 的 replica gradient 同步通常是 all-reduce 或 reduce-scatter/all-gather。

一个最小 EP=2 例子：令 rank 0 拥有 `E0,E1`，rank 1 拥有 `E2,E3`。若源 rank 0 的四条 assignments 有两条去各目的地，源 rank 1 的四条中一条去 rank 0、三条去 rank 1，则 dispatch count matrix（行是 source，列是 destination）为：

$$
\operatorname{send\_counts}=
\begin{bmatrix}2&2\\1&3\end{bmatrix}
$$

所以 rank 0 的 `send_counts=[2,2]`、`recv_counts=[2,1]`；rank 1 的 `send_counts=[1,3]`、`recv_counts=[2,3]`。除了 hidden row，dispatch metadata 至少要让 owner 知道或可恢复 `(source_rank, source_token, topk_slot, gate)`。combine 返回时按相反方向发送 expert outputs，再由 source 用 `(source_token,topk_slot)` scatter-add；仅保存 expert ID 不足以把输出送回原 token。

### 9.3 逻辑 A2A payload 公式

忽略 metadata/padding，标准 MoE 每 token 的 dispatch+combine 逻辑流量：

$$
B_{A2A/token}
=2L_{MoE}k d_{dispatch}\times b
$$

其中：

- 第一个 `2`：去 expert 一次、回来一次；
- `b`：每元素字节数，BF16 为 2；
- 标准 MoE 的 `d_dispatch=d`；
- Stable LatentMoE 若在本地先压缩，则 `d_dispatch=d_l`。

R-Full：

$$
2\times46\times6\times2048\times2
=2.15625\text{ MiB/token}
$$

在 EP8 均匀路由下，约 `7/8=87.5%` assignment 去远端 GPU，远端部分约 1.89 MiB/token。

### 9.4 为什么这不是网卡监控读数

上式是逻辑 payload，不是实际 wire bytes。实际值取决于：

- 本地 expert assignment 不需要远程发送；
- collective 算法和分块；
- metadata、alignment、padding；
- 是否量化通信；
- 是否与 expert compute 重叠；
- 网络重传和协议；
- 每层 load imbalance。

它适合比较架构，不替代 profiler。

### 9.5 为什么优先把 EP8 放在单节点

本项目每节点有 8 张 MI300X。让 EP8 完全对应单节点：

- routed token A2A 走节点内 xGMI；
- 避免每个 MoE 层都做跨节点 RDMA all-to-all；
- 96 experts 正好每卡 12 个；
- `d=2048, f_e=896` 不再被 TP 切碎；
- 跨节点网络主要处理梯度同步而不是逐层 token 搬运。

这是拓扑感知设计：先看硬件通信域，再选 EP。

### 9.6 EP、DP、TP、PP、CP 分别切什么

| 并行轴 | 切分对象 | MoE 中的主要代价 |
|---|---|---|
| EP | experts | token all-to-all |
| DP | batch | replica gradient sync |
| TP | 单个矩阵维度 | 每层 tensor collective，小 expert GEMM 变碎 |
| PP | layers | pipeline bubble、跨 stage activation |
| CP | sequence | attention/context communication，recurrent prefix scan |

这些轴可以正交组合，但 process groups、rank mapping 和 checkpoint metadata 必须一致。

在 `EP8×TP1×PP1×DP15` baseline 中，120 个 global ranks 的坐标可画成：

```text
node 0:  r0  r1  r2  r3  r4  r5  r6  r7    <- EP group 0
node 1:  r8  r9 r10 r11 r12 r13 r14 r15    <- EP group 1
...
node 14: r112 ...                     r119  <- EP group 14

expert-DP group for local expert slot e:
{r_e, r_(8+e), r_(16+e), ..., r_(112+e)}   # 15 replicas
```

同一节点的 8 ranks 共同拥有 96 个 global experts，每 rank 物理持有 12 个；相同 local-EP 坐标在 15 个节点上构成 expert-DP replica group。attention、dense/shared FFN、router 等 non-routed 参数在 baseline 中复制于全部 120 ranks；若以后引入 FSDP/ZeRO 或 CP，归约组必须重新显式定义，不能继续照抄这张图。

一个训练 stage 的 collective 顺序是：

1. 各 source rank 本地算 router scores、Top-k 和 `send_counts`；
2. 在节点内 EP group 交换 counts/metadata，并 A2A dispatch hidden rows；
3. owner rank 对本地 12 个 experts 做 grouped GEMM；
4. 反向 A2A 把 expert outputs 送回 source，按 `(source_token, topk_slot)` combine；
5. backward 逆向经过 combine、第二次 A2A 与 experts；
6. routed-expert gradients 只在对应的 15-rank expert-DP group 归约；non-routed gradients 则在其真实 replica group 归约。

这解释了为什么“EP8×DP15”不是只有一个二维标签：token collectives、expert-gradient collectives 和 non-routed-gradient collectives 是三组不同通信。

### 9.7 为什么本项目不默认使用大 TP

对 `f_e=896`：

- TP2 后局部 intermediate 448；
- TP4 后 224；
- TP8 后 112。

这些小 GEMM 很难充分利用 GPU。只要 EP8 后状态能放进 HBM，就优先 TP1。

### 9.8 Expert Data Parallel

EP8 只保存一份完整 expert set；15 个节点各有一个 EP8 group，相当于 expert set 有 15 个数据并行副本。

某个 expert 的梯度应在 15 个节点上持有该 expert 的对应 local EP rank 间同步。也就是说：

- routed expert replica group 大小约为 15；
- attention、dense、router、shared expert 等非 routed 参数在更多 ranks 上有副本；
- 两类参数不能盲目使用同一个 optimizer sharding group。

### 9.9 Non-routed parameters 的分层归约

EP ranks 都需要 attention 和 shared expert 的副本。一个高效方案可以是：

1. 节点内先对 dense/non-routed 梯度聚合；
2. 对应代表 rank 跨节点同步；
3. 节点内广播或 all-gather 结果。

具体算法取决于框架，但原则是：**所有持有同一逻辑参数副本的 ranks 最终必须得到一致更新。**

### 9.10 长上下文下的 CP

基础映射：

```text
EP8 × CP1 × DP15 = 120 GPUs
```

若 128K/256K activation 过大，可保持每节点 EP8，再让多个节点组成 CP：

```text
128K pilot: EP8 × CP3 × DP5
256K pilot: EP8 × CP5 × DP3
```

此时：

- full attention context communication 跨节点；
- Gated DeltaNet 需要正确的 chunk boundary/prefix-state scan；
- DP 降低，要用 gradient accumulation 恢复 global tokens/step；
- CP 与 EP 必须是经过验证的正交 process groups。

### 9.11 Checkpoint 为什么更难

Dense checkpoint 通常只需知道 DP/TP/FSDP shard。MoE 还必须保存：

- global expert ID 到 rank 的映射；
- expert tensor layout；
- routed 与 non-routed optimizer group；
- correction bias/Quantile controller 状态；
- EP/DP/TP/PP/CP topology metadata；
- RNG 和 dataloader 状态；
- 是否允许在不同 EP size 下 reshard。

一个可执行的 manifest 至少应有如下 schema；字段名可以不同，语义不能缺失：

| 类别 | 关键字段 |
|---|---|
| identity | `format_version`、代码 SHA、model/config/tokenizer hash、committed global step、consumed tokens |
| tensor index | canonical tensor name、global shape、dtype、shard axes/ranges、文件 URI、byte length、checksum |
| topology | world size、rank coordinates、EP/DP/TP/PP/CP groups、global expert ID→tensor/rank mapping |
| optimizer/scheduler | stable param-group ID、hyperparameters、per-tensor moments/master weights、scheduler/scaler state |
| router/controller | correction bias、histogram/quantile config、update counter、统计域、gradient-accumulation phase |
| data/RNG | data shard/cursor、sampler epoch、packing buffer、Python/NumPy/framework RNG 及各 rank RNG |
| transaction | generation ID、expected shard list、每 shard checksum、`complete=true` commit record |

保存协议应先把某一 generation 的所有 shards 写到临时位置并校验，再最后原子发布小型 committed manifest；loader 只接受 manifest 中列全且 checksum 匹配的 shards，不能把两个 steps 的文件拼成“完整 checkpoint”。对象存储不保证目录 rename 原子时，也要依靠 generation ID 与最后写入的 commit record，而不是依赖目录观感。

same-topology exact resume 与 elastic reshard 是两种不同保证：前者要求下一批数据、路由、loss 和 optimizer update 在规定容差内重现；后者还要从 canonical global tensor ranges 重建新的 expert ownership，并验证 optimizer/controller 状态没有被遗漏。二者都要覆盖“保存后杀进程重启”，还要专门测试在 gradient accumulation 中途、写 shard 中途以及 controller 更新边界发生故障。

### 9.12 零 token expert 与 collective 对齐

某个 local expert 在某一步可能收到零 token。实现仍必须：

- 支持零行 grouped GEMM 或安全跳过；
- 为其产生定义明确的零 gradient/无 gradient 语义；
- 让所有 ranks 以相同顺序参加需要的 collectives；
- 保证 AdamW weight decay 和 optimizer state 更新规则一致；
- 不因动态跳过而造成 graph recompilation 或 collective deadlock。

普通 Dense DDP 的“每个参数每步都有梯度”假设在这里不再可靠。

### 9.13 Decode 时 EP 的特殊压力

自回归 decode 每个请求每步只产生一个新 token。若 batch 很小：

- 每 expert 的 `M_i` 极小；
- all-to-all latency 难以被 GEMM 隐藏；
- 所有 EP ranks 仍要协同完成一次 token；
- 热门 expert 会影响 tail latency。

因此服务系统常需要连续 batching、请求级调度、expert-aware placement 或专用 inference kernel。预训练 EP 拓扑可以作为起点，但不是自动最优的推理拓扑。

---

<a id="cost-model"></a>
## 10. 显存、通信和吞吐的联合账本

### 10.1 每参数 16 bytes 的训练状态近似

对 AdamW、未做 optimizer sharding 的常见粗估：

| 状态 | bytes/parameter |
|---|---:|
| BF16 weight | 2 |
| BF16 gradient | 2 |
| FP32 master weight | 4 |
| FP32 first moment | 4 |
| FP32 second moment | 4 |
| 合计 | 16 |

真实实现可能使用 FP32 gradient、无 master weight、分片 moments 或压缩 optimizer，因此必须以实际 state dict 为准。

### 10.2 R-Full 的每卡状态

R-Full routed expert 总参数 24.310B。EP8 后每卡 local routed：

$$
24.310/8=3.039\text{B}
$$

non-routed 参数：

$$
25.857-24.310=1.547\text{B}
$$

本地总逻辑参数约 4.586B，粗略状态：

$$
4.586\times16\approx73.4\text{GB/GPU}
$$

R-Hybrid 因 attention/recurrent 模块更多，本地约 5.221B，对应约 83.5GB/GPU。

### 10.3 73GB 并不代表 192GB HBM 很宽松

训练还要容纳：

- layer activations；
- attention scores 或 recurrent temporary state；
- dispatch/send/recv/combine buffers；
- grouped GEMM workspace；
- activation checkpoint metadata；
- communication buckets；
- framework allocator fragmentation；
- 临时 FP32 router/normalization buffers。

必须用目标 sequence length 和 microbatch 实测峰值。

### 10.4 一次完整的 per-rank HBM 峰值算例

峰值不是把各模块各自的历史最大值机械相加，而是某一时间点所有 live buffers 的并集。一个更接近实现的写法是：

$$
H_{peak}=H_{state}+H_{saved\_act}
+\max_t\bigl(H_{layer\_temp}(t)+H_{async}(t)\bigr)
+H_{comm\_extra}+H_{allocator\_reserve}
$$

其中 `H_state` 已包含持久权重、梯度、master weight 和 Adam moments；若通信 bucket 直接 alias gradient，就不能再重复计数。下面给出一个**教学用预算例子，不是 MI300X 实测值**：R-Full、EP8、AdamW 不分片，单卡按 192 GB 十进制容量做验收。

| 同时存活的分项 | 教学假设（GB/GPU） | 如何在实现中替换 |
|---|---:|---|
| BF16 weights + BF16 grads + FP32 master + Adam moments | 73.4 | 枚举本 rank state tensors；此行已含 9.17GB 逻辑 gradient |
| checkpoint 保留的 activations | 28.0 | 目标 `S×microbatch` 下实测 saved tensors |
| attention/recurrent 临时量 | 10.0 | 在目标 kernel 与 checkpoint policy 下 profile |
| packed send/recv/return/combine + permutation metadata | 8.0 | 按每层 owner 收到的 `A_max`，不是平均 load |
| grouped-GEMM workspace | 6.0 | 取真实 kernel workspace 峰值 |
| 非 alias 的 communication buckets | 6.0 | 核对 bucket 是否复用 gradient storage |
| allocator/graph capture/fragmentation reserve | 15.0 | 同时报 allocated、reserved 与 driver 可用量 |
| **保守峰值合计** | **146.4** | 假定这些 buffer 因异步 overlap 或 capture 同时保留 |
| **对 192GB 的余量** | **45.6（23.75%）** | 仍须通过长时间 soak 的 p99/max 峰值验收 |

如果 allocator 能证明 attention temporary 与 MoE workspace 生命周期互斥，可以按时间线复用，而不是把 10GB 与 6GB 永久相加；反之，异步 A2A overlap 可能让上一层 receive buffer 与下一计算阶段同时存活。验收时至少扫均匀、p95 偏斜、p99 偏斜和 safety-cap 邻界四种 routing 分布，并分别记录 steady state、单步 peak、长时间 `max_reserved`。

这个算例也说明为什么平均每 expert 512 rows 不足以做 HBM 预算：动态 buffer 应从“单层单 rank 最大 incoming assignments `A_max`”反推，并把 hidden rows、expert outputs、inverse permutation、gate、counts/offsets 与通信对齐 padding 一起计入。

### 10.5 Gradient buffer 也是通信账本

EP8 下，R-Full 每卡 local routed + replicated non-routed 参数约 4.586B。BF16 gradient buffer 逻辑大小：

$$
4.586\text{B}\times2\approx9.17\text{GB}
$$

它不是单次实际 wire bytes：

- routed 和 non-routed 使用不同 replica groups；
- reduce-scatter 与 all-reduce 流量不同；
- gradient accumulation 可降低同步频率；
- bucket overlap 可隐藏一部分时间；
- FP32 gradient 会翻倍。

### 10.6 每 expert 能拿到多少 token

若每个 EP rank 在当前 microbatch 有 `T_local` 个有效 token，整个 EP8 group 有约 `8T_local`。均匀路由时，一个 expert 的平均 assignment：

$$
M_{expert}
=\frac{EP\cdot T_{local}\cdot k}{N}
$$

对 EP8、`N=96,k=6`：

$$
M_{expert}=0.5T_{local}
$$

如果每 rank 只有 256 token，则平均每 expert 仅 128 rows；再考虑负载波动，一些 expert 更小。这个 `M` 直接决定 grouped GEMM 是否高效。

### 10.7 KV cache 不是 MoE cache

MoE 替换的是 FFN，不会自动缩小 attention KV cache。标准 GQA 的 batch-1 BF16 cache：

$$
B_{KV}=2LSn_{kv}d_h\times2\text{ bytes}
$$

第一个 `2` 是 K 和 V。R-Full 在 32K/128K 约为 3/12 GiB。

只有同时改变 attention，例如 Gated DeltaNet、MLA、CSA/HCA，才会改变 KV/recurrent-state 账本。不能把这些收益归因于 MoE。

### 10.8 吞吐不是只看 TFLOP

MoE step time 可以粗分为：

$$
t_{step}\approx
\max(t_{dispatch},t_{expert})
+t_{combine}
+t_{nonMoE}
+t_{grad-sync}
+t_{unhidden}
$$

如果通信与 expert compute 完全重叠，`max` 近似合理；若没有重叠，则更接近相加。

需要同时报告：

- tokens/s/GPU；
- effective TFLOP/s；
- dispatch/combine 时间；
- grouped GEMM 时间与 tile 利用率；
- load CV 与最大 expert tokens；
- 网络/xGMI 带宽；
- hidden-overlap 百分比。

### 10.9 架构选择应通过三个 roofline

一个候选至少要过：

1. **HBM roofline**：模型状态、activation 和 buffers 能否容纳；
2. **GEMM roofline**：expert batch shape 能否跑满矩阵核心；
3. **communication roofline**：A2A 和梯度同步能否被带宽/计算重叠承受。

只通过参数公式，不能说明它可训练。

<a id="stability"></a>
## 11. MoE 训练为何比 Dense 更容易失稳

### 11.1 Dense 的数值问题还在，MoE 又增加了离散反馈环

Dense 模型的 loss spike 可能来自激活爆炸、梯度爆炸、数据异常或低精度溢出。MoE 在此之上多了：

```text
参数变化
  -> router score 变化
  -> Top-k 离散路径变化
  -> 每个 expert 收到的数据分布变化
  -> expert 参数和负载变化
  -> 下一步 router score 再变化
```

这个闭环可能把一个普通数值扰动放大成路由重排和系统负载尖峰。

### 11.2 常见失败模式

| 现象 | 模型侧原因 | 系统侧后果 |
|---|---|---|
| 少数 expert 垄断 | router collapse | OOM、尾延迟 |
| dead experts | 得不到主任务梯度 | total capacity 浪费 |
| route churn | Top-k margin 太小、数值噪声 | cache/排序不稳定、质量波动 |
| logits 爆大 | router scale 漂移 | softmax 溢出、选择过硬 |
| 单 expert activation 爆炸 | 数据与参数正反馈 | NaN 扩散到 residual |
| load 平均但质量下降 | 均衡约束太强 | 专家无法按任务选择 |
| resume 后路由突变 | controller/RNG 未恢复 | loss 不连续 |
| 吞吐随 step 抖动 | token 分布改变负载 | grouped GEMM/A2A 尾部变化 |

### 11.3 初始化：既要打破对称，又不能一开始就塌缩

如果所有 experts 完全相同且 router 完全相同，理论上存在对称；如果差异太大，又可能让某专家抢跑。

常见原则：

- expert 权重按与 Dense FFN 一致的 scaled initialization 独立初始化；
- router 权重使用受控的小尺度；
- router bias 通常从零开始，若有 correction bias 也从中性状态开始；
- 可在训练早期使用小 jitter 打破 ties；
- depth scaling 保持 residual branch 输出尺度；
- 第一批就检查每层 load histogram，而不是等 loss spike。

### 11.4 Router 精度与 norm

推荐的低风险组合：

- pre-RMSNorm；
- router 输入尺度受控；
- router logits/softmax/Top-k 使用 FP32；
- Q/K 使用 QK-Norm，防止 attention score 爆炸；
- router z-loss 约束 logit 绝对尺度；
- expert 输出 RMS 分层监控。

QK-Norm 不是 MoE 技术，但 attention 爆炸会改变进入 router 的 residual 分布，所以会间接影响路由稳定性。

### 11.5 SwiGLU limiting

SwiGLU 的两个乘法分支都可能产生大值：

$$
\operatorname{SiLU}(W_gx)\odot W_ux
$$

DeepSeek V4 使用低成本限制：

- linear/up branch clamp 到 `[-10,10]`；
- gate branch 上限 clamp 到 `10`。

其价值是直接限制乘法中的极端坐标。实施时必须记录：

- clamp 命中率；
- 命中值占激活比例；
- 每层/每 expert 的命中分布；
- clamp 前后 RMS 和最大值。

如果大量值长期命中上限，clamp 只是在掩盖更深层问题。

### 11.6 SiTU-GLU

[Kimi K3 官方技术报告](https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf)使用更平滑的 bounded GLU。定义 soft cap：

$$
\operatorname{softcap}(x,\beta)=\beta\tanh(x/\beta)
$$

SiTU-GLU 可写成：

$$
\left[
\beta_1\tanh\left(\frac{W_gx}{\beta_1}\right)
\odot\operatorname{Sigmoid}(W_gx)
\right]
\odot
\left[
\beta_2\tanh\left(\frac{W_ux}{\beta_2}\right)
\right]
$$

K3 报告使用 `β1=4, β2=25`。它在原点附近近似 SwiGLU，同时让两分支大值平滑饱和。

与 hard clamp 相比：

- 函数和梯度更平滑；
- 但引入新的激活和超参数；
- 需要 fused forward/backward；
- 在多个模型尺度上的公开复现少于 SwiGLU clamp。

因此它属于研究项，而不是第一版默认项。

### 11.7 Anticipatory Routing 不是常规路由器

[DeepSeek V4 技术报告](https://arxiv.org/abs/2606.19348)中的 Anticipatory Routing 是 loss spike 恢复机制。其核心逻辑：

1. 在 step `t` 检测到 spike；
2. 丢弃该 step 的异常 checkpoint/更新；
3. 回到 spike 前状态；
4. 重放最初一批时冻结/复用前一时刻路由决策，使 expert 输入分布不立即重排；
5. 随后恢复正常动态路由。

它在诊断上区分两类问题：

- 固定路由后稳定：router/expert 反馈环可能是放大器；
- 固定路由仍异常：问题更可能来自数据、attention 或普通数值路径。

这要求训练系统能精确回滚数据、模型、optimizer、RNG 和 router 状态，工程复杂度很高。

### 11.8 AdamW 与 Muon

AdamW 对每参数维护一、二阶矩，成熟而稳健。Muon 面向二维矩阵，对更新做正交化/谱形状处理，常借助 Newton–Schulz 迭代。

Kimi/DeepSeek 的混合思路大致是：

- 大型二维矩阵使用 Muon；
- embedding、norm、bias、标量/向量和不适合正交化的张量使用 AdamW；
- Kimi 的 Per-Head Muon 进一步按 attention head 组织更新。

对 MoE 来说还多一层困难：

- 每个 expert 矩阵都要维护 optimizer 状态；
- expert replica group 与 dense replica group 不同；
- optimizer state reshard/checkpoint 更复杂；
- Muon 本身的计算、通信和数值行为必须单独验证。

所以不能把“优化器更先进”和“MoE 更稳定”直接画等号。本项目应先用 AdamW 建立基线，再做等 token/等 FLOP A/B。

### 11.9 FP8、FP4 与 QAT

低精度对 MoE 很有吸引力，因为大多数 total parameters 位于 experts：

- 降低权重和 optimizer/通信成本；
- expert GEMM 可能更快；
- checkpoint 更小。

但低精度同时放大：

- per-expert scale 差异；
- 小 expert batch 的统计噪声；
- activation outlier；
- router 边界的精度敏感性。

MXFP4/MXFP8 QAT 不是“把 BF16 checkpoint 转一下”这么简单，它需要：

- block scale 和量化布局；
- fake/real quantization 训练路径；
- forward/backward kernel；
- optimizer/master-weight 策略；
- checkpoint 与部署格式；
- 精度回归。

第一版应先完成 BF16 稳定训练。

### 11.10 可重复性比 Dense 更难

即使 seed 相同，以下因素也可能改变 Top-k：

- all-to-all 到达顺序；
- 并行归约顺序；
- BF16 score 舍入；
- 非确定性 sort/top-k；
- microbatch 划分；
- EP topology；
- correction bias 同步时机。

可重复性验收应分级：

1. 单卡 FP32 reference：严格比对；
2. 单节点 BF16：允许小数值误差，但 route IDs 应高度一致；
3. 多节点：比较 loss、load histogram 和统计轨迹，不一定逐 bit 相同；
4. resume：恢复后的第一步与不中断对照必须在预设容差内。

### 11.11 最稳妥的功能引入顺序

```text
Dense baseline
  -> 单卡 softmax Top-k MoE reference
  -> shared expert + 前两层 dense
  -> dropless grouped GEMM
  -> 单节点 EP8
  -> 多节点 expert-DP / dense-DP
  -> 长时间 BF16 soak
  -> 长上下文 CP
  -> aux-loss-free / Quantile / Muon / MTP / 低精度等消融
```

每一步只增加一个主要自由度，失败时才有可定位性。

---

<a id="classic-to-deepseek"></a>
## 12. 从经典 MoE 到 DeepSeekMoE

### 12.1 历史主线不是“专家越来越多”这么简单

现代稀疏 MoE 的演进可以粗略看成：

| 路线 | 主要贡献 | 主要遗留问题 |
|---|---|---|
| Sparsely-Gated MoE | learned sparse routing | 分布式和负载复杂 |
| GShard | Transformer 中的大规模 Top-2 MoE | capacity/drop、系统复杂 |
| Switch Transformer | Top-1 简化路由和通信 | 单 token 组合能力更少 |
| ST-MoE | 稳定性、router regularization | 仍依赖成熟训练 recipe |
| Mixtral | 成熟的粗粒度 Top-2 开放权重 | active experts 很宽 |
| DeepSeekMoE | 细粒度 experts + shared expert isolation | 专家/路由/EP 更复杂 |
| OLMoE | 训练代码、数据、checkpoint 与分析开放 | 模型规模较小 |
| MegaBlocks | dropless block-sparse 执行 | 不替代模型和路由设计 |

### 12.2 Top-1 与 Top-2/Top-k

Top-1：

- 每 token 只执行一个 routed expert；
- active compute/A2A 最低；
- combine 简单；
- 单个路由错误没有第二专家补充。

Top-2 或更大 `k`：

- token 可组合多个 experts；
- 在 expert 宽度不变时，单 token 的 active FFN width 更大且有多条可组合路径；是否训练更平滑或质量更好是经验问题，total expert capacity 并未因此改变；
- assignment、通信和计算近似线性增加；
- 负载控制更复杂。

不存在脱离硬件和质量目标的最佳 `k`。

### 12.3 Mixtral：粗粒度专家的直观基线

Mixtral 8×7B 每层 8 experts、Top-2，单 expert FFN 很宽。优点是：

- expert 数少；
- 路由和部署直观；
- 单 expert GEMM 大，硬件利用率较好。

但 Top-2 激活两个大 FFN，single-token active parameters 较高。它适合用来理解“总参数稀疏”，不应直接作为 25B total / 3–4B active 的等比例模板。

### 12.4 DeepSeekMoE 的细粒度 expert segmentation

[DeepSeekMoE 论文](https://arxiv.org/abs/2401.06066)的核心动机是把一个大 expert 拆成多个较小专家，然后为每 token 激活更多小专家，在相近 active width 下获得更多组合。

假设一个粗 expert width 为 `F`，拆成 `m` 个 width `F/m` 的小专家。为了保持 active width，Top-k 也相应增大。理论参数/FLOP可保持接近，但：

- 专家组合数增加；
- router 可以更细地分配子能力；
- assignment 和 metadata 增加；
- 每 expert GEMM 的 `M`/`f_e` 变小。

所以 fine-grained 是模型容量与内核效率的交换。

### 12.5 Shared expert isolation

DeepSeekMoE 把一部分专家设为始终激活的 shared experts，意图承载通用知识，让 routed experts 更专注差异化部分。

“isolation”可理解为结构上分开：

- shared 路径不参加 routed Top-k 竞争；
- routed experts 竞争剩余条件计算；
- 最终输出相加。

但 shared/routed 实际学到什么必须通过输出、梯度和路由分析验证，不能由名称直接推断。

### 12.6 DeepSeekMoE-16B 给本项目的启示

公开配置展示了同一模型中可以同时存在：

- hidden 2048；
- 首层 dense；
- 64 routed / Top-6；
- 2 shared experts；
- fine-grained expert FFN。

这说明这些模块在中型模型上有公开先例。它不能证明 R 系列的 `96/Top-6/1 shared/f=896` 是最优，只能降低“这个组合完全没有先例”的风险。

### 12.7 OLMoE 的特殊价值

[OLMoE 官方仓库](https://github.com/allenai/OLMoE)所对应的 OLMoE-1B-7B 使用 64 experts、Top-8、dropless token-choice routing。它的重要性在于开放了较完整的：

- 训练代码；
- 数据；
- 中间 checkpoint；
- 评测；
- 路由分析。

学习如何真正训练 MoE 时，可复现的小模型链路往往比只有 HF inference modeling code 的超大模型更有工程价值。

### 12.8 不要从专家数推断模型先进程度

`N=896` 不自动优于 `N=96`：

- 大模型可能有不同 hidden、latent dimension、batch 和 EP 网络；
- 896 experts 需要极高效 grouped GEMM 和负载控制；
- 中型模型中每 expert token 数可能太少；
- 路由收益可能低于系统损失。

应比较等 total、等 active FLOP、等数据和等训练预算的对照，而不是比较专家数字大小。

---

<a id="latent-moe"></a>
## 13. Stable LatentMoE 与 Quantile Balancing

### 13.1 标准 MoE 的参数瓶颈

标准 expert 在 full hidden `d` 上工作：

$$
E_i:\mathbb R^d\rightarrow\mathbb R^d
$$

每 expert 参数 `3df_e`。当 `d` 很大、又想把 `N` 提到数百时，total parameters 快速增长。

### 13.2 LatentMoE 的基本想法

先用共享 down projection 把所有 token 从 `d` 压到 `d_l`：

$$
h_l=W_{down}^{shared}h,
\quad d_l<d
$$

routed experts 在 latent space 工作：

$$
E_i:\mathbb R^{d_l}\rightarrow\mathbb R^{d_l}
$$

组合后再用共享 up projection 回到 full hidden：

$$
y_{routed}=W_{up}^{shared}
\left(\sum_{i\in\mathcal T}\alpha_iE_i(h_l)\right)
$$

单 expert 从 `3df_e` 降为 `3d_lf_e`。当 `d_l=d/2` 时，专家核心参数约减半，可以在固定 total 下增加 `N` 或 `f_e`。

### 13.3 为什么朴素 LatentMoE 容易不稳定

共享 down projection 输出尺度一旦漂移，会同时影响所有 routed experts；共享 up projection 又把所有 expert 输出耦合回同一空间。可能出现：

- latent activation RMS 波动；
- bottleneck 丢失通用信息；
- 少数 expert 的异常经共享 up projection扩散；
- shared projections 成为固定计算和梯度热点。

### 13.4 Stable LatentMoE 的两个稳定化要点

[Kimi K3 官方技术报告](https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf)与[公开模型定义](https://huggingface.co/moonshotai/Kimi-K3)所示的 Stable LatentMoE 设计包含：

1. routed latent 路径在 shared down projection 后做 RMSNorm；
2. shared expert 保持在 full hidden space，而不是也被迫穿过 latent bottleneck。

概念公式：

$$
h_l=\operatorname{RMSNorm}(W_dh)
$$

$$
y_{routed}=W_u
\left(\sum_{i\in\mathcal T}\alpha_iE_i(h_l)\right)
$$

$$
y=y_{routed}+E_{shared}(h)
$$

这样：

- routed experts 获得尺度受控的 latent 输入；
- full-hidden shared expert 为通用信息提供旁路；
- routed expert 仍有各自独立参数，不是一个共享 MLP。

### 13.5 参数公式

忽略小 norm/bias：

$$
P_{LatentMoE,total}
=N\cdot3d_lf_e+2dd_l+3df_s+dN
$$

active 把 `N` 替换成 `k`：

$$
P_{LatentMoE,active}
=k\cdot3d_lf_e+2dd_l+3df_s+dN
$$

注意 `2dd_l` 的 down/up projections 对每个 token 都执行。

### 13.6 LatentMoE 的通信收益

如果 down projection 在源 rank 本地执行，dispatch 的是 `[d_l]` 而不是 `[d]`：

$$
d_{dispatch}=d_l
$$

X-K3 envelope 中：

- `L_MoE=46`；
- `k=8`；
- `d_l=1024`；
- BF16。

所以：

$$
2\times46\times8\times1024\times2
=1.4375\text{ MiB/token}
$$

不是按 full hidden 2048 计算的 2.875 MiB/token。这个收益只有在实现真的“先压缩、后 dispatch，latent combine、再上投影”时成立。

### 13.7 LatentMoE 的代价

- 每层固定执行 down/up projections；
- latent bottleneck 可能限制表示；
- shared projection 可能成为计算热点；
- 更多专家/更大 Top-k 仍增加 assignments；
- checkpoint layout 与标准 MoE 不同；
- 高效 kernel 和反向路径公开成熟度有限。

因此 X-K3 在调研中只是约 24.79B/3.08B 的**预算包络**，不是可以直接编码的冻结配置。

### 13.8 Quantile Balancing 的算法直觉

Kimi K3 的 router 先得到正 affinity，例如：

$$
s_i=\operatorname{Sigmoid}(W_r x_i)
$$

选择使用带 bias 的 score：

$$
\mathcal T_i=\operatorname{TopK}(s_i+b,k)
$$

但 mixture weight 使用原始 score：

$$
p_{i,j}
=\frac{s_{i,j}}{\sum_{r\in\mathcal T_i}s_{i,r}}
$$

训练 step `u` 对 `s_i+b^(u)` 做 Top-(k+1)：前 `k` 名是实际 routes，第 `k+1` 名的 biased score 记为行级 cutoff `α_i^(u)`。固定这些 cutoffs 后，候选下一步 bias `b_hat_j^(u+1)` 会让 expert `j` 接收满足下式的 tokens：

$$
s_{i,j}+\hat b_j^{(u+1)}>\alpha_i^{(u)}
$$

注意这里用于构造分布的是**原始 score 相对 biased cutoff 的 margin**：

$$
m_{i,j}=s_{i,j}-\alpha_i^{(u)}
$$

旧 bias 只通过 `α_i^(u)` 进入该更新。若本 step 有 `T` 个 tokens，目标负载是 `q=Tk/N`，则可设：

$$
\hat b_j^{(u+1)}
=-\operatorname{Quantile}_{1-k/N}
\left(\{s_{i,j}-\alpha_i^{(u)}\}_{i=1}^{T}\right)
$$

再减去所有 experts 的 bias 均值；共同平移不改变 Top-k。新 bias 只在下一 step 生效，避免用当前 batch 推导出的 bias 反过来路由同一 batch。

### 13.9 为什么用分位点而不是固定步长

固定 sign update 只知道“过载/欠载”，不知道需要移动多少 score 才会改变恰当数量的 Top-k 边界。

Quantile update 利用当前 margin 分布估计：

- bias 移动到哪里会让目标数量 token 改变 assignment；
- 每个 expert 的 score 尺度不同，也能自适应；
- 大专家数下比统一固定步长更有机会稳定。

K3 用每 expert 的直方图估计全局分位点，最后对 biases 做 mean-centering；报告还把跨 step 的 quantile EMA 作为进一步降低 batch 噪声的改进。最终 bias 在推理时冻结。

### 13.10 Quantile Balancing 的系统难点

- 分位点统计本身比 count 更昂贵；
- 统计域必须覆盖 controller 所定义的完整 global batch，包括相应 token shards 和 gradient-accumulation microbatches；
- 不能把各 rank 的 local quantile 取平均来冒充 pooled global quantile；
- distributed approximate quantile 受 histogram bin width 约束，仍可能带误差；
- 小 batch 时分位点离散且噪声大；
- bias/controller 必须保存进 checkpoint；
- 更新时机要与 gradient accumulation 对齐；
- 中型模型上缺少充分公开复现。

还必须冻结 exact quantile 还是 histogram 近似、histogram 范围/bin/overflow、deterministic merge、controller 更新周期、mean-centering 与推理冻结规则。一个不隐藏状态边界的 controller 伪流程是：

```text
for each microbatch in the controller window:
    score = router_affinity(hidden)                  # raw semantic score
    alpha = kth_plus_one(score + bias)              # biased cutoff
    route = topk(score + bias)                       # selection only
    mixture_weight = normalize(score[route])         # do not inject bias
    histogram += hist(score[:, expert] - alpha)      # every expert

all_reduce(histogram, controller_group)              # spans intended DP/EP domain
bias_next = -quantile(histogram, 1 - k / N)
bias_next = bias_next - mean(bias_next)
commit bias_next only at the documented global-step boundary
checkpoint(histogram/window_step/bias/controller_version)
```

若 histogram 覆盖多个 gradient-accumulation microbatches，不能在窗口中途用半成品 bias 改变后续 microbatch 的控制语义；恢复时也必须回到相同 window phase。exact-quantile 与 histogram-quantile 的误差、overflow 比例和多 rank merge 顺序都要进入 parity test。

因此它适合在 softmax+aux 基线稳定后做研究消融。

### 13.11 Kimi K3 的 896 experts 为什么不能机械缩放

K3 同时具备：

- 2.78T total / 104.2B active；
- 896 routed / Top-16；
- latent dimension 3584；
- 专用 LatentMoE、Quantile Balancing、SiTU-GLU；
- 大规模 EP 和定制 kernel；
- 与 KDA/Gated MLA、Muon、量化训练共同设计。

把 896 直接按参数比例缩到 25B，会改变：

- 每 expert token batch；
- EP mapping；
- A2A fan-out；
- router statistical regime；
- grouped GEMM shape；
- latent bottleneck 比例。

可迁移的是机制和公式，不是专家数本身。

---

<a id="adjacent-tech"></a>
## 14. 与 MoE 相邻但不同的前沿模块

### 14.1 为什么必须单独分类

MoE 解决的是 **FFN 条件参数容量**。下面这些模块解决的是其他问题：

| 模块 | 主要目标 |
|---|---|
| GQA/MLA/CSA/HCA | attention 与 KV cache |
| Gated DeltaNet/KDA | 长上下文的线性或 recurrent mixing |
| AttnRes/mHC | residual 信息流 |
| MTP | 训练目标/多 token 预测 |
| Muon | 优化器更新几何 |
| SwiGLU clamp/SiTU | 激活数值稳定 |
| FP4/FP8 QAT | 精度、存储和算力 |

它们常与 MoE 一起出现，是因为前沿模型会同时优化多个瓶颈，不是因为 MoE 必须依赖它们。

### 14.2 GQA：低风险 full-attention 基线

Grouped-Query Attention 让多个 query heads 共享较少 K/V heads，降低 KV cache，同时保留标准 softmax attention。

它的优势：

- kernel 成熟；
- 训练/推理语义清晰；
- 与 RoPE、FlashAttention 和 checkpoint 兼容；
- 不引入 recurrent state。

因此 R-Full 用完整 GQA，是第一代 MoE 最稳妥的 attention 搭档。

### 14.3 Gated DeltaNet

DeltaNet/linear attention 用固定大小 recurrent state 近似长期序列记忆。概念上每步会：

1. 从 token 产生 key/query/value；
2. 用 decay/gate 更新 per-head state；
3. query 从状态读取输出；
4. 配合短 depthwise convolution 捕获局部模式。

训练可用 chunk-wise parallel scan，推理只维护固定状态，避免 full KV cache 随 `S` 线性增长。

Qwen3.5/Next 的公开几何还包含：

- 不同的 key/value head widths；
- output gate；
- decay/update gates；
- conv kernel 4；
- 按 3 个 Gated DeltaNet 层 + 1 个 full-attention 层组成 3:1 pattern。

风险在于高性能 backward、FP32 recurrent state、chunk boundary、CP prefix scan 和 ROCm fused kernel。

### 14.4 KDA：Kimi Delta Attention

KDA 也是 delta-rule/linear-attention 家族，但 Kimi K3 对 decay 做了有界参数化，并设计适合 Tensor Core 的 tiled 算法。

调研报告只使用一个说明“有界 log-decay”思想的简化式：

$$
g=g_{min}\cdot\operatorname{Sigmoid}(\exp(A_h)z)
$$

其中 [Kimi K3 官方技术报告](https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf)给出 `g_min=-5`、`A_h` 初始为 0。真实 KDA 还包含 low-rank decay projection、tile 变换、chunk 算法和定制 backward；不能靠这个简式复刻。

从实现审计角度，至少要把下列**抽象 shape 契约**写清；它们不是 K3 exact tensor config：

| 状态/张量 | 抽象 shape | 必须冻结的语义 |
|---|---|---|
| 输入 | `[B,S,d]` | padding、document boundary、position reset |
| `q,k` | `[B,S,H_k,d_k]` | head grouping、normalization、layout |
| `v` | `[B,S,H_v,d_v]` | `H_v` 与 `H_k` 的映射、output packing |
| decay/update gate | `[B,S,H_state,...]` | 低秩投影维度、值域、应用在 update 前还是后 |
| recurrent state | `[B,H_state,d_k,d_v]`（概念形） | FP32/BF16、reset、detach、checkpoint/recompute |
| convolution cache | `[B,C,kernel-1]`（概念形） | causal padding、chunk 首尾、streaming 更新 |

并行训练通常把长度 `S` 切为 chunks，在 chunk 内做 tiled recurrence/scan，并把末状态传给下一 chunk；逐 token 解码则每步原地更新 recurrent state 和 conv cache。两条路径必须在相同初始状态与 mask 下对齐。尤其要测试非整 chunk 长度、packed documents、sequence reset、prefill→decode 切换，以及不同归约顺序导致的有限浮点误差。

KDA 的价值主张是长上下文复杂度与硬件效率，但迁移到 ROCm 的风险高于已开放并广泛集成的 full attention。

### 14.5 MLA 与 Gated MLA

Multi-head Latent Attention 用低维 latent 表示压缩 K/V 相关状态。若投影能在推理时“吸收”进 query/output 路径，KV cache 可显著降低。

关键区别：

- **MLA latent** 压缩 attention K/V；
- **LatentMoE latent** 压缩 routed expert hidden；
- 二者名字相似，但作用位置完全不同。

Gated MLA 还在 attention output 上加入 elementwise gate。要获得 cache 收益，推理 kernel 必须直接维护和消费 latent cache；若 HF reference 先展开为普通 K/V，收益会消失。

### 14.6 Attention output gate

Qwen3.5 风格的 gated full attention 从 projection 中同时产生 query-related gate：

$$
y=\operatorname{Gate}(x)\odot\operatorname{Attention}(x)
$$

它让模型按通道调节 attention 输出强度，但：

- 增加 projection 参数；
- 改变初始化和 residual 尺度；
- 需要 fused kernel 才能避免额外带宽开销。

它不是 expert gate；一个控制 attention 输出通道，一个选择 FFN experts。

### 14.7 Block AttnRes

普通 residual 主要累加相邻层输出。Block AttnRes 每隔若干层收集多个历史 residual candidates：

1. 各候选做 RMS normalize；
2. 学习标量/一维投影 score；
3. softmax 得到权重；
4. 混合为新的 block residual。

它让深层显式选择不同深度的信息，但增加：

- 多 residual activation 的保存；
- norm 和融合计算；
- checkpoint/recompute 与 PP 复杂度。

K3 每 12 层使用一次；对 48 层中型模型应先作为消融。

### 14.8 CSA/HCA

DeepSeek V4 的长上下文 attention 组合包括：

- 初始滑动窗口层；
- Compressed Sparse Attention：压缩 K/V 后用索引选 Top-512，并保留局部窗口；
- Hierarchical Compressed Attention：更高压缩率的粗粒度全局信息。

其 1M context cache 收益依赖专用压缩、索引、sparse gather 和 cache layout。不能把论文中的 cache 比例直接套到普通 HF attention 实现。

### 14.9 mHC residual

mHC 把 residual stream 扩展成多个通道，概念式：

$$
X_{l+1}=B_lX_l+C_lF_l(A_lX_l)
$$

DeepSeek V4 使用 multiplier 4，并通过流形约束和 Sinkhorn 迭代稳定映射。

代价是 residual activation、通信和 kernel 复杂度大幅上升。它与 MoE、长上下文和 activation checkpoint 同时启用时，调试空间会呈乘法增长。

### 14.10 MTP

Multi-Token Prediction 在主 next-token loss 外，增加预测更远未来 token 的训练分支。潜在作用：

- 给 hidden state 更密集的未来监督；
- 改善表示学习；
- 某些实现可为 speculative decoding 提供 draft proposals。

训练时的 MTP branch 和生产中的“每步接受多个 token”不是同一件事。上线收益还取决于 acceptance rule、验证成本、拒绝后的回退、KV-cache 回滚/提交语义，以及 serving engine 是否实现相应 speculative path；没有这些系统支持，保留 MTP 权重也不会自动提高解码吞吐。

它不改变 routed expert 定义，但会改变：

- 每 step 训练 FLOP；
- total/active parameters；
- loss 权重；
- checkpoint；
- 训练与部署保留哪些模块。

所以本项目在 exact tensor geometry 冻结前不把 MTP 计入 headline。

### 14.11 QK-Norm、partial RoPE 与长上下文

- QK-Norm：分别归一化 query/key，控制 attention logits；
- partial RoPE：只给部分 head dimensions 施加旋转位置编码；
- 大 `rope_theta`：改变位置频率尺度。

R-Hybrid 的 full-attention 层采用 25% partial RoPE 和 `rope_theta=10,000,000` 的代理几何。是否原生训练 128K 或 256K仍必须二选一，不能只改 max position 字段就宣称支持。

### 14.12 为什么不能一次堆满所有前沿功能

若同时启用：

```text
MoE + aux-free + Quantile + GDN/KDA + MLA
+ AttnRes/mHC + Muon + MTP + FP4 QAT
```

任何 loss spike 都可能有十余种交互原因，且缺少成熟对照。第一代生产模型应只引入“提供主要收益、已有成熟内核、可独立验收”的最小集合。

---

<a id="candidate-walkthrough"></a>
## 15. 本项目四个候选的逐项拆解

### 15.1 统一对照

| 候选 | total | active | 核心定位 |
|---|---:|---:|---|
| C0† | 30.532122624B | 3.353032704B | Qwen3 控制组，超过 30B 上界 |
| R-Full | 25.857439744B | 3.066640384B | 第一代生产默认 |
| R-Hybrid | 26.492484352B | 3.701684992B | 128K/256K 条件候选 |
| X-K3 | ≈24.786217536B | ≈3.080694336B | 研究预算包络 |

所有数字：

- 都是 text-only；
- R-Full、R-Hybrid 和 X-K3 使用 tied embedding；C0 按公开 Qwen3-30B-A3B geometry 使用 untied embedding/head；
- 都不含可选 MTP；
- X-K3 attention 数字是预算，不是冻结 tensor config。

C0 与 R-Full 均计入 48 层 learned Q/K RMSNorm scales，共 12,288 参数；归一化逐 head 执行，但每层 Q 与 K 各自只存一个跨同类 heads 共享的 `d_h` 长度 scale，因此公式是 `2L_full d_h`，不是再乘 head 数。R-Hybrid 还计入 36 个 GDN 层的 `dt_bias`、`A_log` 和 gated RMSNorm scale（共 6,912），以及 12 个 full-attention 层的 Q/K RMSNorm scales（共 6,144）；这 13,056 个参数均在固定 active 路径上。X-K3 仍是 `≈` 预算 envelope，不能把同样的精确位数含义套到它上面。

### 15.2 C0：控制组的价值

C0 复现 Qwen3-30B-A3B 公开几何：

- 48 layers；
- `d=2048`；
- 128 routed / Top-8；
- `f_e=768`；
- 无 shared expert；
- full GQA；
- untied embedding/head。

独立核算得到 30.532B total，严格超过 30B，所以不能作为生产候选。它仍很重要，因为可以：

- 校验参数计数代码；
- 对照已有 HF modeling geometry；
- 比较 `128/8/no-shared` 与 `96/6/shared`；
- 为 kernel benchmark 提供外部基线。

### 15.3 R-Full：一个 token 如何前向

每个深层 block：

```text
hidden d=2048
  -> RMSNorm
  -> full GQA: 32 Q heads / 4 KV heads / head_dim 128
  -> residual
  -> RMSNorm
  -> FP32 router scores over 96 experts
  -> select Top-6
  -> EP8 dispatch full hidden d=2048
  -> 6 routed SwiGLU experts, f=896
  -> EP8 combine
  + 1 local shared SwiGLU expert, f=896
  -> residual
```

优点：

- attention 和 MoE 两个风险域相对独立；
- full attention kernel 成熟；
- 参数、KV、A2A 和 checkpoint 容易建模；
- 适合作为 ≤32K 第一版。

### 15.4 R-Hybrid：MoE 主体相同，attention 更激进

MoE body 与 R-Full 相同；attention pattern 为：

```text
Gated DeltaNet
Gated DeltaNet
Gated DeltaNet
Gated full attention
... 重复 12 组
```

总计 36 GDN + 12 full-attention layers。它降低长 context 的 full-attention 二次项和 KV cache，但引入 recurrent state、conv、scan 和 output gating。

只有在以下条件都通过后才应晋升：

- fused GDN forward/backward；
- FP32 recurrent state 正确性；
- chunk 与 full recurrent 对照；
- CP prefix scan；
- checkpoint/recompute；
- 128K 或 256K 目标冻结；
- 长时 soak 无漂移。

### 15.5 X-K3：研究包络如何工作

概念配置：

- 36 KDA + 12 gated MLA；
- 128 routed / Top-8；
- `d=2048, d_l=1024`；
- latent expert `f_e=1280`；
- full-space shared expert `f_s=896`；
- AttnRes interval 12。

单 token 的 MoE 路径：

```text
h[2048]
  -> shared down projection -> latent[1024]
  -> latent RMSNorm
  -> router selects 8 of 128 experts
  -> dispatch latent[1024]
  -> latent expert SwiGLU f=1280
  -> latent combine
  -> shared up projection -> routed output[2048]

h[2048] -> local full-space shared expert f=896

routed output + shared output
```

其主要吸引力是低 A2A 和更多专家容量；主要问题是 KDA、MLA、LatentMoE、AttnRes 与 Quantile/kernel 同时不成熟。故不能直接作为实现规格。

### 15.6 为什么默认选 R-Full

R-Full 不是“理论最先进”，而是风险可分解：

- MoE 本身已经是首次引入的主要复杂度；
- attention 沿用成熟 full GQA；
- EP8 与单节点拓扑自然匹配；
- 25.857B 落在总参数预算中；
- 3.067B active 保持合理计算；
- 后续可以逐个替换 attention、router 或 optimizer，而不是一次重写全部。

### 15.7 何时选择 R-Hybrid

必须先冻结业务目标：

- 如果主要训练/部署 ≤32K：R-Full；
- 如果必须原生 128K 或 256K，且 full attention 成本不可接受：做 R-Hybrid pilot；
- 不能同时把 128K 和 256K 都当作“自然支持”，因为数据 curriculum、position scaling、batch 和 CP 规划不同。

### 15.8 第一版明确不做什么

- 不按比例复制 896 experts；
- 不默认启用 mHC/AttnRes；
- 不在 MTP 未定义时把它塞入 headline；
- 不以 FP4 QAT 作为首次稳定性基线；
- 不因某个 HF modeling file 可前向，就宣称训练栈可复现；
- 不跨节点做逐层 expert A2A，除非单节点确实放不下。

<a id="migration"></a>
## 16. 从 Dense 代码库迁移到 MoE 的实施路线

### 16.1 第一步不是优化，而是定义模块语义

先写清楚一个 MoE layer 的契约：

**输入**：

- hidden states `[batch, seq, d]` 或展平后的 `[T,d]`；
- valid-token mask；
- optional process-group/topology metadata。

**输出**：

- 与输入同形 hidden；
- router logits 或可选采样统计；
- auxiliary/z losses；
- per-expert counts；
- dropped/overflow 状态；
- 可供诊断的 gate weights/margins。

**语义必须冻结**：

- Top-k 前还是后加 correction bias；
- combine weights 是否重归一化；
- shared expert 与 routed mixture 如何相加；
- padding 如何处理；
- auxiliary loss 的统计域；
- router dtype；
- dropless 异常如何报错。

如果这些没有先写进测试，后续 fused kernel 很容易“性能正确、模型语义错误”。

### 16.2 建立慢但可信的单卡 reference

reference 可以使用普通张量操作和逐 expert 循环，目标不是快，而是：

- 公式一眼可读；
- Top-k IDs 和 weights 可导出；
- 每个 expert 输入输出可检查；
- backward 可与数值梯度/独立实现比较；
- 作为 grouped GEMM 和 EP 的 oracle。

不要一开始就只保留 fused implementation，否则出错时没有真相源。

#### 如果要从 Dense checkpoint warm-start

本文目标是 from-scratch 预训练，因此 Dense checkpoint 不是必需输入；但做迁移实验时，必须把 tensor mapping 写成显式转换清单：

| Dense tensor | 目标 | 处理 |
|---|---|---|
| embedding/head、attention、主 norms | 同 shape tensor | 逐名复制并校验 checksum |
| Dense SwiGLU `W_g,W_u,W_d` | shape 相同的 cloned experts | 可复制到每个 expert；若 combine weights 和为 1，初始函数可保持一致 |
| Dense SwiGLU | `f_dense != f_e` 的小 experts/shared expert | 不能声称直接等价；需 neuron partition、factorization 或蒸馏，并单独验输出误差 |
| 不存在的 router | 新 router | 新建；固定 seed，避免全零 logits 加固定 tie-break 让少数 experts 独占 |
| optimizer moments | 兼容旧 tensor/新增 tensor | 兼容项复制；新增项显式清零或重建，不伪造 shape mapping |

若把一个 Dense FFN 的中间神经元切成 `k` 组，每组形成一个小 expert，则 Dense 输出本质上是这些组输出之和；而标准 MoE combine weights 往往归一化为和 1。要做函数保持测试，就必须相应缩放每组的 down projection 或临时使用求和 combine，并固定路由到完整的 `k` 组。R-Full 的 `5504→896` 不是整除关系，且有 96 个 routed experts，因此这只能作为受控初始化实验，不能写成无损通用转换。

另一种受控实验是把同一个完整 dense FFN 复制给每个 routed expert。复制瞬间所有 `E_i(x)` 相同；只要选中集合上的 combine gates 满足 `Σ_i g_i=1`，routed 分支就仍等于该 dense FFN，未归一化 gate 则会按 `Σ_i g_i` 缩放输出。若同时复制一个 shared expert，必须显式设定 routed/shared 两支的系数，否则很容易把 dense 输出加两遍。固定 probe 的初始化验收应同时比较 dense/MoE 输出、gate-sum、Top-k IDs 和 routed/shared 分支值。

转换产物应记录 source checkpoint hash、每个 source→target tensor rule、随机 seed、未映射 tensor 列表和输出误差报告。先在固定 probe 上比较 hidden states/logits/loss，再决定是否导入旧 optimizer；仅仅 `load_state_dict(strict=False)` 成功不构成迁移正确性。

### 16.3 最小单元测试集合

#### 结构退化测试

1. `N=1,k=1,no shared` 应退化为 Dense FFN；
2. 所有 experts 权重相同且 combine weights 和为 1 时，routed 输出应等于该公共 FFN；
3. router logits 全相等时，tie-breaking 行为应确定且有文档；
4. shared-only/routed-only 路径分别可测试。

#### Dispatch/combine 测试

1. permutation 后 inverse permutation 恢复原 token 次序；
2. 一个 token 的 `k` 条 assignment 不丢失、不重复；
3. padding 产生零 assignment；
4. combine 权重和、dtype、累加误差符合约定；
5. 极端负载下仍 dropless 或显式 fail-fast。

#### 梯度测试

1. expert weight gradient；
2. router selected-score gradient；
3. shared expert gradient；
4. auxiliary/z-loss gradient；
5. EP1 与 reference 的 input/weight gradient 对齐；
6. activation checkpoint 开/关对齐。

### 16.4 第二阶段：单 GPU grouped GEMM

保持 router 和语义不变，只替换 expert execution：

```text
reference expert loop
        ↓
local permutation + grouped GEMM
        ↓
compare output / loss / gradients
```

验收维度：

- 多种 `T`；
- 均匀与高度偏斜 load；
- BF16/FP32；
- `f_e=896` 真实 shape；
- forward、dgrad、wgrad；
- compile/eager；
- checkpoint recompute。

### 16.5 第三阶段：单节点 EP8

先保持：

- 一个节点；
- `TP=PP=CP=1`；
- 固定数据；
- 相同初始化；
- 小模型或少层。

对比 EP1 与 EP8：

- forward output；
- Top-k ID；
- auxiliary loss；
- 每 expert 梯度；
- optimizer 更新后参数；
- dispatch/combine token 数守恒。

**守恒检查**：

$$
\sum_i n_i=T_{valid}k
$$

只要 dropless，这个等式每层每步都应成立。

### 16.6 第四阶段：多节点 DP15

加入 15 个 EP8 groups 后，重点不再只是 token A2A，而是两类 gradient group：

- routed expert replicas：相同 global expert ID 跨 15 节点同步；
- non-routed replicas：attention、dense、shared、router、norm 的全部副本同步。

验收方法：

1. 所有副本初始 checksum 相同；
2. 一步 optimizer 后 checksum 仍一致；
3. 故意注入某 rank gradient，验证只在正确 group 传播；
4. 重启后 global expert ID 不错位。

### 16.7 Checkpoint/resume 是功能，不是收尾项

必须尽早做：

```text
连续跑 20 steps 作为对照

另一条：
跑 10 steps -> save -> 杀进程 -> restart -> 再跑 10 steps

比较：
loss、参数、optimizer、router load、数据位置、RNG
```

checkpoint 至少覆盖：

- expert weights 和 optimizer；
- non-routed weights 和 optimizer；
- global step、scheduler；
- correction bias/QB 状态；
- RNG；
- dataloader/corpus cursor；
- topology 和 tensor layout 版本。

### 16.8 第一版推荐配置

从调研结论出发，第一版保守组合是：

- R-Full；
- BF16；
- FP32 router；
- softmax Top-6；
- 小权重 auxiliary balance；
- router z-loss；
- dropless + 显式 OOM safety cap；
- 一个 shared expert；
- 前两层 Dense；
- QK-Norm；
- monitored SwiGLU limiting；
- AdamW；
- EP8/TP1/PP1/CP1/DP15；
- tied embedding；
- 不启用 MTP、Muon、Quantile、KDA、mHC、FP4。

### 16.9 训练 curriculum

建议按风险分阶段：

1. **短程正确性**：小 token、短 context，验证 loss 能下降；
2. **单节点性能**：找到 expert token batch 与 microbatch sweet spot；
3. **多节点稳定性**：验证 replica groups、A2A、checkpoint；
4. **中程统计**：观察专家是否死亡、路由是否稳定；
5. **长时 soak**：覆盖数据域变化和 checkpoint 轮转；
6. **上下文扩展**：再从短 context 进入 32K；
7. **高级消融**：一次只加一个 router/optimizer/attention 特性。

### 16.10 配置和 checkpoint 的版本化

MoE 特别需要把以下字段写入 checkpoint manifest：

```text
num_experts
num_experts_per_tok
expert_intermediate_size
shared_expert_intermediate_size
num_dense_layers
router_score_function
router_dtype
normalize_topk_prob
aux_loss_coefficient
z_loss_coefficient
dropless_semantics
expert_parallel_size
expert_id_mapping
expert_weight_layout
```

配置字段相同也不一定语义相同，所以还应保存实现版本/SHA 和 kernel 版本。

### 16.11 性能验收不能只跑均匀 synthetic routing

至少构造：

- 完全均匀；
- 轻度偏斜；
- 单 expert 热点；
- 长尾 Zipf load；
- 不同 `T_local`；
- 多种 `k`；
- shared expert 开/关。

真实路由很少完全均匀，kernel 只在均匀 benchmark 上快没有生产意义。

### 16.12 何时才可以添加 R-Hybrid

R-Full 主干达到以下门槛后再分叉：

- 多节点 BF16 长时稳定；
- checkpoint/resume 多次通过；
- MoE 监控和报警完整；
- 训练吞吐达到预期；
- downstream 基线可信。

然后保持 MoE body 不变，只替换 attention，才能把差异归因于 hybrid attention。

---

<a id="debugging"></a>
## 17. 监控与故障诊断手册

### 17.1 最小 dashboard

#### 主训练

- LM loss、perplexity；
- learning rate、grad norm；
- tokens/s、step time；
- MFU/有效 TFLOP/s；
- HBM peak、allocator reserved。

#### Router

- auxiliary/z loss；
- entropy；
- Top-1/Top-k score；
- `p_k-p_{k+1}` margin；
- fixed-probe route flip rate（相同 token、jitter off、跨 checkpoints）；
- correction bias min/max/std。

#### Experts

- 每层 min/mean/max/CV load；
- dead experts；
- input/output RMS；
- weight/grad norm；
- clamp hit rate；
- grouped GEMM rows/expert。

#### 通信

- dispatch/combine bytes；
- A2A time、wait time；
- local vs remote assignment 比例；
- gradient collective time；
- overlap ratio。

### 17.2 Loss spike 的排查顺序

1. 定位首次异常 step，而不是只看最终 NaN；
2. 检查数据 batch 和 token IDs；
3. 找出首先出现非有限值的层/张量；
4. 对照 router load 是否在 spike 前先变化；
5. 检查 expert activation/clamp；
6. 检查 attention logits/QK-Norm；
7. 检查 optimizer update norm；
8. 用 spike 前 checkpoint 固定路由重放；
9. 只在能复现后修改机制。

不要一看到 MoE 就默认问题一定是 router。

### 17.3 Load imbalance 高

| 检查项 | 可能结论 |
|---|---|
| importance 也偏 | router 本身偏好少数 experts |
| importance 均匀但 load 偏 | Top-k 边界/tie 或统计实现问题 |
| 只有某些 sequence 偏 | 需要 sequence-wise 诊断 |
| 只有一个 EP group 偏 | process-group/statistics bug 或数据分片差异 |
| bias 振荡 | correction update 过快 |
| aux loss 很大但不改善 | 系数/梯度路径/统计域错误 |

先验证计数和 process group，再调 `λ_aux`。

### 17.4 Dead expert

定义不能只看单 step。可用滑动窗口：

- 连续若干百 step load 接近零；
- grad norm 接近零；
- router probability 也持续很低。

可能处理：

- 检查初始化和 tie-breaking；
- 临时增加早期 jitter；
- 调整辅助均衡；
- 检查该 expert 是否实际映射到错误 rank；
- 检查 grouped GEMM offset 是否漏掉它；
- 谨慎考虑重初始化，避免破坏 optimizer/checkpoint 语义。

### 17.5 偶发 OOM

MoE OOM 可能只在某个特殊 batch 发生。记录：

- 当步 `T_valid`；
- max tokens/expert；
- 每 destination rank send counts；
- dispatch/workspace 大小；
- activation checkpoint 状态；
- allocator fragmentation；
- 数据域和 sequence length。

如果平均显存很低但偶发 OOM，优先怀疑负载尾部和临时 buffer，而不是模型静态参数。

### 17.6 理论 FLOP 低但吞吐差

按顺序检查：

1. 每 expert `M_i` 是否太小；
2. `f_e` 是否与 tile 不匹配；
3. 是否逐 expert launch；
4. sort/permutation 是否占比过高；
5. A2A 是否未 overlap；
6. 某个热点 expert 是否形成 straggler；
7. shared expert 是否串行执行；
8. backward/wgrad 是否没有 fused；
9. graph compile 是否频繁重编译动态 shape。

MoE 常见现象是“数学 FLOP 更少，但墙钟更慢”。

### 17.7 A2A 很慢

- 确认 EP group 是否真的位于单节点；
- 检查 rank placement 与 xGMI topology；
- 区分 payload 大和 collective latency 大；
- 检查 send counts 是否极不均匀；
- 检查是否多次小 all-to-all，而非合并后的较大调用；
- 检查 compute overlap 时间线；
- 检查 metadata/padding 是否远大于公式估算；
- 避免把 CP/DP group 错当 EP group。

### 17.8 Resume 后 loss 不连续

逐项核对：

- expert ID 映射；
- optimizer moment 是否随 expert 一起恢复；
- router correction bias；
- scheduler step；
- RNG；
- dataloader cursor；
- gradient accumulation phase；
- dynamic loss scale；
- EP/DP topology；
- tied weight alias 是否恢复。

只比较 state dict key 数量不够；要比较逻辑 global tensor checksum。

### 17.9 评测时 expert load 与训练不同是否正常

可能正常，因为：

- 数据域不同；
- batch/sequence packing 不同；
- 推理 decode token 数少；
- router jitter 在 eval 被关闭；
- capacity/drop 语义可能不同。

但必须保证：

- Top-k 和 combine 语义相同；
- 不发生静默 token drop；
- shared expert 保持一致；
- router dtype 不因推理框架改变；
- 量化没有改变大量边界路由。

### 17.10 一个故障表

| 症状 | 首先检查 | 不要先做 |
|---|---|---|
| NaN | 首个非有限张量、数据、激活 | 盲目增大 aux loss |
| 热点 expert | 计数/stat group、importance | 立即扩大 capacity 隐藏问题 |
| dead experts | 路由与 offset 正确性 | 直接删除专家 |
| 偶发 OOM | max load、动态 buffer | 只看平均 HBM |
| 吞吐低 | `M_i`、kernel timeline、A2A | 只算 active params |
| resume 漂移 | topology/controller/RNG | 重新训练掩盖 checkpoint bug |
| eval 变差 | route 语义和数据域 | 按单个 expert 名称讲故事 |

---

<a id="misconceptions"></a>
## 18. 常见误解

### 误解 1：MoE 26B/3B 就是一个 3B 模型

错。它存储和训练约 26B 参数，只是每 token 使用约 3B 路径。

### 误解 2：Active parameters 就是精确计算量

错。它忽略 attention 二次项、路由、排序、padding、通信和 kernel 利用率。

### 误解 3：专家越多，专业化越强

错。更多专家也可能更冗余、更难训练、每 expert batch 更小。

### 误解 4：负载均衡意味着语义分工良好

错。均衡只说明调用量接近，不说明专家学到了互补函数。

### 误解 5：Shared expert 一定存通用知识

错。这是设计动机，需要输出和消融证据验证。

### 误解 6：Dropless 意味着没有容量问题

错。它只是不静默丢 assignment，仍有热点、OOM 和尾延迟。

### 误解 7：EP 只是另一种 TP

错。EP 分专家并搬 token；TP 分单个矩阵并做 tensor collective。

### 误解 8：MoE 自然减少 KV cache

错。KV cache 由 attention 决定。R-Full 的 MoE 不改变 GQA cache。

### 误解 9：HF modeling code 等于训练代码

错。它通常不包含生产级 EP、grouped GEMM backward、optimizer sharding、checkpoint 和稳定性 recipe。

### 误解 10：总权重文件字节数就是参数量

错。量化、scale、metadata、打包格式都会改变 bytes/parameter。

### 误解 11：Top-k 越大越稳

不一定。更大 `k` 提供更多路径，但增加 FLOP、A2A 和负载控制难度。

### 误解 12：Auxiliary loss 越强越均衡，因此越好

错。太强会让 router 为均衡牺牲主任务选择。

### 误解 13：把 EP8 改成 EP16 就能放更多 expert

可能，但会跨节点做每层 token A2A，通信代价可能远高于节省的状态。

### 误解 14：Kimi/DeepSeek 的一个机制可以单独复制

不一定。score、bias、norm、optimizer、kernel 和恢复协议经常共同构成系统。

### 误解 15：只要平均 load 正常就安全

错。单层、单 expert、单 sequence 或单 EP group 的尾部可能仍异常。

### 误解 16：一次加入所有先进模块能节省实验时间

通常相反。交互项让任何回归都无法归因，最终更慢。

### 误解 17：推理与训练使用同一组权重，性能就会相同

错。decode 的 token batch 很小，expert GEMM 和 A2A 的效率 regime 完全不同。

### 误解 18：模型名中的 A3B 是严格可复现的精确 active count

不一定。命名通常是近似口径；embedding、router、attention、MTP 是否计入可能不同，应逐 tensor 核算。

---

<a id="glossary"></a>
## 19. 术语表

| 术语 | 教学定义 |
|---|---|
| Expert | 一个可独立执行的 FFN 子网络，通常是 SwiGLU |
| Routed expert | 由 router/Top-k 条件选择的 expert |
| Shared expert | 所有 token 固定执行、不参与 Top-k 竞争的 expert |
| Router/Gate | 根据 token hidden 给 experts 打分的模块 |
| Router logit | score 激活/归一化前的原始打分 |
| Affinity/score | 用于排序或 mixture weight 的专家相关性 |
| Top-k | 每 token 选择得分最高的 k 个 routed experts |
| Token-choice | 每个 token 主动选择 experts |
| Expert Choice | 每个 expert 在 token 集合中选择固定容量 token |
| Assignment | 一条 token→expert 的执行关系 |
| Dispatch | 把 assignment 的 hidden state 送往 expert |
| Combine | 把多个 expert 输出送回并按 gate 权重合并 |
| Load | expert 实际收到的 assignment 数 |
| Importance | router 给 expert 的连续概率质量 |
| Expert collapse | 少数 expert 垄断负载的正反馈状态 |
| Dead expert | 长期几乎没有 token/梯度的 expert |
| Route churn | token 的 expert 选择频繁变化 |
| Top-k margin | 第 k 名与第 k+1 名 score 的间隔 |
| Auxiliary balance loss | 通过可微损失推动专家使用均衡 |
| Router z-loss | 约束 router log-sum-exp 尺度的正则 |
| Correction bias | 只影响选择、由负载控制器更新的 expert bias |
| Quantile Balancing | 用 score margin 分位点更新 correction bias 的方法 |
| Capacity | 固定实现允许单 expert 处理的最大 assignment 数 |
| Capacity factor | 相对于平均负载的容量倍率 |
| Token dropping | 超容量时不执行某些 assignment |
| Dropless | 不因固定容量静默丢弃有效 assignment |
| Grouped GEMM | 一次调度多个不同 `M_i` 的 expert dense GEMM |
| Block-sparse MoE | 用稀疏块矩阵表达 ragged expert workload |
| EP | 按 experts 切分权重的并行轴 |
| Expert-DP | 同一 global expert 在不同 EP groups 间的数据并行副本 |
| All-to-all | ranks 互相发送不同数据片段的 collective |
| All-reduce | ranks 归约同形张量并获得相同结果的 collective |
| TP | 把单个矩阵维度切到多个 ranks |
| PP | 把层切成流水线 stages |
| CP | 把 sequence/context 切到多个 ranks |
| Fine-grained experts | 数量较多、单个较窄的专家设计 |
| Expert segmentation | 把粗 expert 拆成多个小 expert |
| Shared expert isolation | 将固定共享通路与 routed competition 分开 |
| LatentMoE | 在低维 latent space 中执行 routed experts |
| Stable LatentMoE | 加 latent norm 与 full-hidden shared path 的 LatentMoE |
| Active parameters | 一个 token 的执行路径使用到的参数口径 |
| Total parameters | 模型包含的全部逻辑参数 |
| Local parameters | 某个 rank 实际持有的参数，取决于并行布局 |
| Router jitter | 训练早期为探索/打破对称加入的小噪声 |
| Hash routing | 按 token ID 等离散键确定 expert 的固定路由 |
| Group-limited routing | 先限制 expert groups，再在组内 Top-k |
| Anticipatory Routing | loss spike 回放时暂时冻结先前路由的恢复机制 |
| GQA | 多个 Q heads 共享较少 KV heads 的 full attention |
| Gated DeltaNet | 带 decay/update/output gates 和 recurrent state 的线性注意力 |
| KDA | Kimi 的 bounded-decay delta attention |
| MLA | 通过 latent representation 压缩 attention KV 的机制 |
| AttnRes | 在多个历史 residual candidates 间学习混合 |
| mHC | 扩展多通道 residual stream 的机制 |
| MTP | 同时预测多个未来 token 的辅助训练目标/模块 |
| Muon | 对二维矩阵更新做正交化/谱整形的优化器家族 |
| QAT | 训练中显式模拟/使用低精度量化的训练方法 |
| Soak test | 长时间持续运行以暴露稀有稳定性和资源问题 |

---

<a id="exercises"></a>
## 20. 自测题与答案

### 20.1 基础题

**题 1：** `d=2048, f_e=896` 的单个 SwiGLU expert 有多少参数？

**答案：**

$$
3\times2048\times896=5,505,024
$$

---

**题 2：** `N=96,k=6` 的 routed expert active ratio 是多少？

**答案：** `6/96=1/16=6.25%`。但整模 active/total 比例不会等于 6.25%，因为 attention、embedding、router、shared 和 dense layers 固定激活。

---

**题 3：** Top-6 为什么让一个 token 产生六条 assignment？

**答案：** 因为 token hidden 要分别作为六个 routed experts 的输入，六个输出再按 gate weight 组合。

---

**题 4：** shared expert 是否计入 routed A2A？

**答案：** 通常不计。它在每个 rank 本地复制并执行，但其参数梯度需要副本同步。

### 20.2 计算题

**题 5：** R-Full 单个 MoE 层的 routed total 和 active 参数是多少？

**答案：**

$$
P_{routed,total}=96\times3\times2048\times896
=528,482,304
$$

$$
P_{routed,active}=6\times3\times2048\times896
=33,030,144
$$

---

**题 6：** 标准 MoE 有 46 层、Top-6、`d=2048`，BF16 dispatch+combine 的逻辑 payload 是多少？

**答案：**

$$
2\times46\times6\times2048\times2
=2,260,992\text{ bytes}
=2.15625\text{ MiB/token}
$$

---

**题 7：** 同样配置若 `d_l=1024, k=8`，为什么 payload 不是按 `d=2048` 算？

**答案：** 只有当实现先在源 rank 下投影到 latent、dispatch/combine 都在 latent space，再本地上投影时，发送维度才是 `d_l=1024`。这正是 Stable LatentMoE 通信收益成立的前提。

---

**题 8：** EP8、每 rank `T_local=1024`、`N=96,k=6` 时，均匀情况下每 expert 平均收到多少 assignment？

**答案：**

$$
\frac{8\times1024\times6}{96}=512
$$

---

**题 9：** 延续题 8。capacity 在 owner 收齐 EP-group assignments 后执行，`capacity_factor=1.25`。求 `T_cap`、每 expert 的 `C` 和每 owner rank 的总 capacity rows；若误用 source-local domain，会得到什么？

**答案：** `T_cap=8×1,024=8,192`，所以：

$$
C=\left\lceil1.25\times\frac{8,192\times6}{96}\right\rceil=640
$$

每 rank 拥有 12 个 experts，总 capacity 是 `12×640=7,680` rows。误用 `T_cap=1,024` 会得到 `C=80`，少一个 EP8 因子。若发生 dropping，还要验证 accepted+dropped=`T_cap×k`，并只对 accepted gates 按已声明规则重归一化。

---

**题 10：** EP=2 时 dispatch count matrix（行是 source，列是 destination）为 `[[2,2],[1,3]]`。两个 rank 的 `send_counts/recv_counts` 分别是什么？combine 至少还需保留什么 metadata？

**答案：** rank 0 为 `send=[2,2]、recv=[2,1]`；rank 1 为 `send=[1,3]、recv=[2,3]`。除了 destination expert，还要能恢复 `(source_rank, source_token, topk_slot, gate)` 或等价信息；返回时按相反 splits 发送，再用 `(source_token,topk_slot)` scatter-add。只存 expert ID 无法可靠逆置换。

---

**题 11：** 按 §10.4 的教学 HBM 表，求峰值、余量和余量比例。若要求至少保留 15%（28.8GB）余量，初始预算是否通过？若 saved activations 再增加 20GB 呢？

**答案：** 初始峰值为 `73.4+28+10+8+6+6+15=146.4GB`，余量 `192-146.4=45.6GB`，即 23.75%，通过。增加 20GB 后峰值 166.4GB、余量 25.6GB（13.33%），不通过 15% 门槛。73.4GB 状态行已含 gradient，不能再把 §10.5 的 9.17GB 重复相加。

### 20.3 理解题

**题 12：** 为什么 load balance loss 不能证明 expert specialization？

**答案：** 它只约束调用量/概率质量，不约束不同 experts 学到互补语义。所有专家学同一个函数也可以完美均衡。

---

**题 13：** 为什么 dropless 仍可能 OOM？

**答案：** dropless 必须处理所有 assignment；极端热点会扩大某 rank 的动态 buffer、workspace 和 GEMM rows，反而更需要 safety cap。

---

**题 14：** 为什么 EP8 比 EP15 更自然地匹配本项目硬件？

**答案：** 每节点正好 8 GPU，EP8 可把逐层 token A2A 留在 xGMI 域；EP15 必然跨节点，并且 96 experts 也不能均匀除以 15。

---

**题 15：** R-Hybrid 的 KV cache 更小，是否说明其 MoE 更高效？

**答案：** 不能。KV 收益来自 Gated DeltaNet/full-attention 混合，不来自 MoE body；两者应分别归因和验收。

---

**题 16：** correction bias 为什么不应直接进入 combine weight？

**答案：** bias 是负载控制信号，不代表模型对 expert 的语义置信度。若进入 mixture weight，会把系统均衡偏置注入模型输出。

---

**题 17：** 为什么在 benchmark 中要测试偏斜 routing？

**答案：** 真实 load 不均；均匀 synthetic routing 会高估 grouped GEMM 利用率、低估最大 buffer 和 straggler 时间。

---

**题 18：** 什么时候才应尝试 Quantile Balancing？

**答案：** 在 softmax Top-k + 小权重 auxiliary loss 的单卡、多卡、checkpoint 和长时训练基线都稳定后，以单变量消融方式尝试。

### 20.4 设计题

**题 19：** 若保持 active routed width `k f_e≈5376`，列出两种候选。

**答案示例：**

- `k=6,f_e=896`；
- `k=7,f_e=768`（5376）；
- `k=8,f_e=672`（5376）。

参数相近不代表系统等价：`k` 越大，assignment/A2A 越多；`f_e` 越小，GEMM tile 可能越差。

---

**题 20：** 如何证明 EP8 实现没有漏 token？

**答案：** 对每层每步验证 `Σ_i n_i=T_valid k`，并将 EP8 output/grad 与同权重 EP1/reference 比较；同时检查 inverse permutation 和源 token IDs。

---

**题 21：** 若 resume 后 loss 略高但所有 weight key 都加载成功，下一步查什么？

**答案：** expert ID/rank mapping、optimizer moments、router correction state、RNG、dataloader cursor、scheduler/accumulation phase，而不是只看 key 数量。

---

<a id="sources"></a>
## 21. 资料边界与延伸阅读

### 21.1 本文与调研报告的分工

本文解释概念和工程因果链；精确的模型事实、参数表、许可证和来源审计以配套报告为准：

- [`moe_pretraining_architecture_research.md`](./moe_pretraining_architecture_research.md)

特别是：

- 20–30B 指 **total parameters**；
- 项目集群、既有 dense run、tokenizer/EOT 和 1T tokenized corpus 是项目输入；
- Qwen3.8-Max 截止 2026-08-05 没有公开可核验的 exact config/weights；
- Qwen3.5/Qwen3-Next 只能作为明确标注的代理；
- X-K3 是预算包络，不是冻结实现；
- Kimi K3 为自定义/source-available license，不应笼统称作 OSI 开源；
- HF 模型定义不等于完整训练栈。

### 21.2 建议按这个顺序读原始资料

1. **DeepSeekMoE**：先理解 fine-grained experts 与 shared expert isolation；
2. **OLMoE**：看训练代码、数据、checkpoint 和路由分析如何连成完整链路；
3. **MegaBlocks**：理解 dropless 与 block-sparse/grouped execution；
4. **Mixtral**：对照少专家、粗粒度 Top-2；
5. **Qwen3/Qwen3-Next/Qwen3.5 config**：学习公开 tensor geometry；
6. **Kimi K3 report**：再看 LatentMoE、Quantile、KDA、AttnRes、SiTU；
7. **DeepSeek V4 report**：再看 hash/score routing、aux-free、Anticipatory、CSA/HCA、mHC 和 Muon。

### 21.3 主要一手资料

- Kimi K3 官方仓库：<https://github.com/MoonshotAI/Kimi-K3>
- Kimi K3 官方技术报告 PDF：<https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf>
- Kimi K3 Hugging Face：<https://huggingface.co/moonshotai/Kimi-K3>
- DeepSeek-V4-Flash-0731 Hugging Face：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- DeepSeek V4 技术报告：<https://arxiv.org/abs/2606.19348>
- DeepSeekMoE：<https://arxiv.org/abs/2401.06066>
- OLMoE 官方仓库：<https://github.com/allenai/OLMoE>
- MegaBlocks 官方仓库：<https://github.com/databricks/megablocks>
- Qwen3-30B-A3B-Base config：<https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/raw/main/config.json>
- Qwen3.5-35B-A3B-Base config：<https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base/raw/main/config.json>

### 21.4 最后的工程原则

> 先用最简单、可验证的 MoE 建立正确性和稳定性；再逐个证明高级路由、长上下文 attention、优化器和低精度技术确实提供独立收益。

对第一次做 MoE 的团队，最重要的能力不是一次性复制某个前沿模型，而是建立：

- 可核算的参数与通信账本；
- reference 与 fused kernel 双实现；
- topology-aware process groups；
- 每层每 expert 的可观测性；
- 可演练的 checkpoint/resume；
- 一次只改变一个变量的实验纪律。
