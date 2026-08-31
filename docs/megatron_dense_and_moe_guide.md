# 在 Megatron-LM 上写 Dense 与 MoE 模型并跑起来

一份原理文档，以及 FSDP2 与 Megatron 的实战对比

---

**背景** 我们用两套栈训练过两个模型：

| | 8B Dense | 25.86B MoE |
|---|---|---|
| 框架 | **FSDP2**（纯 PyTorch） | **Megatron-LM** + MCore 0.12.4 |
| 模型代码 | 自己写，264 行 | **一行没写** |
| 硬件 | MI300X | 15 节点 × 8 MI300X = 120 GPU |
| 启动耗时 | **很快** | **很久**（本文的主题） |

这个反差不是偶然，也不是"Megatron 更难用"这么简单。本文拆解两者的抽象层次差异，
说明为什么同样是"把模型跑起来"，代价可以差一个数量级。

> **阅读提示**：如果你没用过 Megatron，请先读第零章。
> 后文大量出现 TP / PP / EP / DP / CP 这些缩写，它们是理解一切的前提。

---

## 零、Megatron 原理：五种并行

### 0.1 一个根本矛盾

训练大模型时，单张 GPU 装不下下面这些东西：

| 占显存的东西 | 8B 模型（bf16）举例 |
|---|---|
| 参数 | 16 GB |
| 梯度 | 16 GB |
| 优化器状态（Adam 的 m 和 v，fp32） | 64 GB |
| 激活值（随 batch × seq 增长） | 几十 GB |
| **合计** | **>100 GB** |

MI300X 有 192 GB，勉强能放 8B。但 25.86B 就是 3 倍以上，**必须切开**。

问题是：**沿哪个方向切？** 一个 Transformer 训练任务有多个可切分的维度，
Megatron 的核心就是把这些维度组织成一个正交的并行体系。

先看一个 Transformer 的计算长什么样：

```
输入 [batch, seq, hidden]
   │
   ├─ Layer 1 ─┐
   │   Attention:  Q,K,V = x @ Wq, x @ Wk, x @ Wv    ← 矩阵可以切
   │   FFN:        h = x @ W1;  y = h @ W2           ← 矩阵可以切
   ├─ Layer 2 ─┘                                     ← 层可以切
   ├─ ...
   ├─ Layer 48
   │
输出 [batch, seq, vocab]
```

可切分的维度有四个：**batch（样本）**、**layer（层）**、
**矩阵内部（hidden 维）**、**seq（序列长度）**。
MoE 还多一个：**expert**。

五种并行正好对应这五个维度。

---

### 0.2 DP —— Data Parallel（数据并行）

**切什么**：切 batch。

**怎么工作**：每张 GPU 保留**完整的模型副本**，各自处理不同的样本。
反向传播后，把各自的梯度 **all-reduce** 求平均，保证所有副本参数一致。

```
GPU 0: 样本 0-7   ─┐
GPU 1: 样本 8-15  ─┼─ all-reduce 梯度 ─→ 所有 GPU 参数保持一致
GPU 2: 样本 16-23 ─┘
       ↑ 每张卡都有完整模型
```

**优点**：最简单，通信只在反向末尾发生一次，容易和计算重叠。
**缺点**：**不省显存** —— 每张卡还是要放下整个模型。

**这就是 FSDP 要解决的问题**：FSDP（包括我们用的 FSDP2）是 DP 的增强版，
它把参数、梯度、优化器状态也沿 DP 维**切开**，用时才 all-gather 回来：

```
FSDP:  GPU 0 只存参数的 1/N，forward 到某层时才 all-gather 出完整权重，
       用完立刻丢弃（reshard）
```

所以 FSDP ≈ "DP + 参数分片"。它解决了显存问题，但**只有这一个并行维度**。
8B 够用，25.86B MoE 不够。

---

### 0.3 TP —— Tensor Parallel（张量并行）

**切什么**：切**单个矩阵乘法**的内部。

**怎么工作**：把权重矩阵按列或按行切到多张卡上，每张卡算一部分，再拼起来。

以 FFN 的 `y = (x @ W1) @ W2` 为例，TP=2：

```
       W1 按列切              W2 按行切
GPU 0: h0 = x @ W1[:, :半]    y0 = h0 @ W2[:半, :]  ─┐
                                                      ├─ all-reduce → y
GPU 1: h1 = x @ W1[:, 半:]    y1 = h1 @ W2[半:, :]  ─┘
```

巧妙之处：W1 按列切之后，中间结果 `h` 天然是切开的，
不需要通信；只在最后 `y0 + y1` 时做一次 all-reduce。

**优点**：能把单层放不下的矩阵切开，显存直接减半/减 N。
**缺点**：**通信在每一层内部**，且是同步的 all-reduce，
对带宽极其敏感 —— 所以 **TP 组必须放在同一节点内**（走 NVLink/xGMI，不走网络）。

**我们的选择**：TP=1。25.86B 在 192 GB 上放得下，
加 TP 只会引入不必要的 all-reduce。

---

### 0.4 PP —— Pipeline Parallel（流水线并行）

**切什么**：切**层**。

**怎么工作**：把 48 层分成几段，每张卡负责一段，像流水线一样传递激活值。

```
GPU 0: 层 1-12  ─→ GPU 1: 层 13-24 ─→ GPU 2: 层 25-36 ─→ GPU 3: 层 37-48
       forward 时激活值往右传，backward 时梯度往左传
```

**问题：流水线气泡（bubble）**。GPU 0 算完第一段后要等整条流水线走完
才能开始下一个 micro-batch，中间是空闲的：

```
时间 →
GPU 0: [F1][F2][F3]              [B3][B2][B1]
GPU 1:     [F1][F2][F3]      [B3][B2][B1]
GPU 2:         [F1][F2][F3][B3][B2][B1]
              ↑ 斜边就是气泡，GPU 在空转
```

缓解办法是把 batch 切成很多 micro-batch 填满流水线，
但气泡无法完全消除，比例约为 `(PP-1) / micro_batch_数`。

**优点**：跨节点友好 —— 只在段与段之间传激活值，通信量小。
**缺点**：有气泡；实现复杂（1F1B、interleaved 等调度策略）。

**我们的选择**：PP=1。48 层能放下，不引入气泡。

---

### 0.5 EP —— Expert Parallel（专家并行，MoE 专属）

**切什么**：切 **expert**。

**背景**：MoE 模型把一个大 FFN 换成 N 个小 FFN（expert），
每个 token 只激活其中 k 个。我们的配置是 96 个 expert，每 token 激活 6 个。

96 个 expert 的参数量很大，但每个 token 只用 6 个 —— 这正是 MoE 的价值：
**参数量大而计算量小**。

**怎么工作**：把 96 个 expert 分到 8 张卡上，每卡 12 个。
但问题来了：token 在 GPU 0 上，它要用的 expert 可能在 GPU 5 上。

解决办法是 **all-to-all 通信**：

```
1. 路由：router 决定每个 token 去哪些 expert
2. dispatch（all-to-all）：把 token 发送到 expert 所在的卡
        GPU 0 的 token ──┬─→ GPU 0 的 expert
                         ├─→ GPU 3 的 expert
                         └─→ GPU 5 的 expert
3. 计算：每张卡用本地 expert 处理收到的 token
4. combine（all-to-all）：把结果发回原来的卡
```

**这是 MoE 最麻烦的地方**，原因有三：

1. **通信量大且形状可变** —— 每个 expert 收到多少 token 取决于数据，
   不是固定形状。这比 all-reduce 更容易触发框架的边界情况。
2. **每层两次 all-to-all** —— 46 个 MoE 层就是 92 次。
3. **负载不均** —— 如果 router 偏好某几个 expert，那些卡就成为瓶颈，
   其余卡空等。这就是为什么需要 `--moe-aux-loss-coeff` 这样的
   负载均衡损失。

**我们的选择**：EP=8，**恰好等于单节点的 GPU 数**。
这样 all-to-all 完全走节点内的 xGMI 互联，不碰 InfiniBand。
这个决策后来在排障时意外救了场（见 §4.1）。

---

### 0.6 CP —— Context Parallel（上下文并行）

**切什么**：切 **序列长度**。

**为什么需要**：Attention 的显存和计算量随序列长度 **平方增长**。
序列 128K 时，单卡放不下 attention 的中间结果。

**怎么工作**：把序列切成几段，每卡算一段。但 attention 需要每个 token
看到所有其他 token，所以要用 ring attention 之类的算法，
让各卡轮流交换 K/V 分块。

```
GPU 0: token 0-32K   ─┐
GPU 1: token 32-64K  ─┼─ 环形传递 K/V，每卡逐步算完对全序列的 attention
GPU 2: token 64-96K  ─┘
```

**我们的选择**：CP=1。序列长度 4096 完全放得下，不需要。

CP 通常只在超长上下文（32K 以上）训练时才用。

---

### 0.7 五种并行的对比

| | 切什么 | 省显存 | 通信位置 | 通信量 | 适合放在 |
|---|---|---|---|---|---|
| **DP** | batch | ❌（FSDP 才省） | 反向末尾 | 中 | 跨节点 |
| **TP** | 矩阵内部 | ✅ 强 | **每层内部** | **大** | **节点内** |
| **PP** | 层 | ✅ 强 | 段之间 | 小 | 跨节点 |
| **EP** | expert | ✅ 强 | 每 MoE 层两次 | 大且形状可变 | **节点内**（如果能） |
| **CP** | 序列 | ✅（激活） | attention 内部 | 大 | 节点内 |

**组合规则**：这五个维度是**正交**的，可以同时使用，
总 GPU 数 = TP × PP × EP × DP × CP。

一个典型的大规模配置可能是：
`TP=8, PP=4, DP=16` → 512 GPU。

**我们的配置**：

```
120 GPU = EP(8) × DP(15)
TP = 1, PP = 1, CP = 1
```

即：96 个 expert 切到 8 张卡（节点内），这套"8 卡的模型"复制 15 份做数据并行。

---

### 0.8 Megatron 还负责什么

除了并行，Megatron 还内置了大规模训练需要的一整套设施：

| 功能 | 说明 |
|---|---|
| **distributed optimizer** | Adam 状态沿 DP 维分片（类似 ZeRO-1），省大量显存 |
| **激活重计算** | 用计算换显存，可选择性重算某些算子 |
| **通信/计算重叠** | 梯度 all-reduce 与反向传播重叠 |
| **融合算子** | fused softmax、fused RoPE、fused Adam 等 |
| **分布式 checkpoint** | 每个 rank 存自己那份，避免汇聚到单卡 |
| **数据管线** | 预 tokenize 成 bin/idx 格式，mmap 读取 |

这些也是它值得用的原因 —— **自己实现一遍的成本远高于学习配置的成本**。
前提是：你确实需要 TP/PP/EP 中的至少一个。

---

### 0.9 该用哪个？

```
模型能放进单卡？
  ├─ 能 → 普通 DDP
  └─ 不能 → 需要 TP / PP / EP 吗？
        ├─ 不需要（≤10B，纯 dense）→ FSDP / FSDP2   ← 我们的 8B
        └─ 需要 → Megatron                          ← 我们的 25.86B MoE
```

**判断标准很简单**：如果 FSDP 的"DP + 参数分片"这一个维度就够用，
就别引入 Megatron 的复杂度。一旦需要 MoE（EP）或模型大到必须切层/切矩阵，
Megatron 几乎是唯一成熟选择。

---

## 一、两种范式

### 1.1 FSDP2：你拥有模型，框架只管切分

我们的 8B dense 是 `src/model.py`，264 行，结构一目了然：

```python
class ModelArgs   # 超参 dataclass
class RMSNorm
class Attention   # GQA + RoPE + SDPA
class SwiGLU
class Block       # Attention + SwiGLU + 两个 norm
class Chimera     # embedding + N×Block + lm_head
```

分布式化只有两行：

```python
for layer in model.layers:
    fully_shard(layer, mp_policy=mp, **fsdp_kw)   # 每层一个通信组
fully_shard(model, mp_policy=mp, **fsdp_kw)       # 根模块兜底
```

**心智模型**：模型是普通的 `nn.Module`，`fully_shard` 把参数沿 DP 维切开，
forward 前 all-gather、backward 后 reduce-scatter。你能 `print(model)`、
能单步调试、能在 CPU 上跑一个 tiny 配置。

**代价**：只有一个并行维度（数据并行 + 参数分片）。没有张量并行、没有流水线并行、
没有专家并行。8B 模型这够用；25.86B MoE 不够。

### 1.2 Megatron：框架拥有模型，你只填参数

Megatron 里我们**没有写任何模型代码**。整个入口是这样的：

```python
# tools/pretrain_entry.py —— 全文不到 30 行
from moe_rebuild import rocm_shim
rocm_shim.apply()                                  # ROCm 适配
sys.argv[0] = f"{MEGATRON_DIR}/pretrain_gpt.py"
runpy.run_path(sys.argv[0], run_name="__main__")   # 交给上游
```

模型完全由**命令行参数**决定。`moe_rebuild/config.py`（437 行）的唯一职责是
把 dataclass 翻译成 argv：

```python
a += ["--num-layers", "48", "--hidden-size", "2048"]
a += ["--num-experts", "96", "--moe-router-topk", "6"]
a += ["--expert-model-parallel-size", "8"]
a += ["--moe-layer-freq", "[0]*2+[1]*46"]
...
```

**心智模型**：Megatron 内部有一套 spec 系统（`TransformerLayerSpec` 等），
根据参数组装出并行化的层。你看不到一个完整的 `nn.Module` 定义，
调试时面对的是 `megatron/core/transformer/moe/token_dispatcher.py:499` 这种位置。

**收益**：TP / PP / EP / CP 全都有，且经过大规模验证。25.86B 在 120 GPU 上
跑到 82.8 TFLOP/s，这是 FSDP2 给不了的。

### 1.3 核心差异

| | FSDP2 | Megatron |
|---|---|---|
| 模型来源 | 你写 | 框架按参数生成 |
| 并行维度 | DP + 参数分片 | TP × PP × EP × DP × CP |
| 出错时你面对的 | **你的代码** | **框架内部 + 版本组合** |
| 调试回路 | 秒级（本地 tiny 配置） | 分钟到小时（要占集群） |
| 适用规模 | ≤10B 量级 | 10B–1T |

**一句话**：FSDP2 的复杂度在你的代码里，Megatron 的复杂度在配置和版本兼容性里。
前者可以本地复现，后者往往只在 120 GPU 上才暴露。

这就是为什么 8B dense "很快就跑起来"，而 MoE 花了很久 —— 不是模型更难，
是**故障面从"我的代码"转移到了"框架 × ROCm × torch 版本"的笛卡尔积**。

---

## 二、在 Megatron 上写一个 Dense 模型

先从简单的开始。Dense 模型只需要正确设置这几组参数。

### 2.1 最小可用参数集

```bash
# 几何
--num-layers 24 --hidden-size 2048 --ffn-hidden-size 5504
--num-attention-heads 32
--seq-length 4096 --max-position-embeddings 32768

# 现代 LLM 标配
--swiglu                          # SwiGLU 激活（ffn-hidden 要按 2/3 折算）
--normalization RMSNorm
--position-embedding-type rope
--group-query-attention --num-query-groups 4    # GQA
--untie-embeddings-and-output-weights
--disable-bias-linear

# 精度与优化器
--bf16
--use-distributed-optimizer
--lr 2e-4 --min-lr 2e-5 --lr-decay-style cosine
--clip-grad 1.0 --weight-decay 0.1

# 数据
--tokenizer-type HuggingFaceTokenizer --tokenizer-model <path>
--data-path <bin/idx prefix>
--data-cache-path <本地盘路径>       # ← 见下文陷阱
```

### 2.2 Dense 阶段的三个坑

**坑 1：`--ffn-hidden-size` 与 SwiGLU 的关系**

SwiGLU 有三个矩阵（gate / up / down）而不是两个，所以同参数量下 hidden 要乘 2/3。
写 `--swiglu` 时若沿用 dense FFN 的 4×hidden，参数量会多 50%。
我们的 dense 层用 5504 ≈ 2048 × 2.7，就是折算后的值。

**坑 2：`--data-cache-path` 必须在本地盘**

Megatron 会用 `numpy.load(..., mmap_mode='r')` 读三个索引文件。
若 cache 在 blobfuse/NFS 上，mmap 缺页失败时内核直接发 **SIGBUS/SIGSEGV**，
没有 errno、没有异常、无法重试，而且崩溃发生在 `ThreadPoolExecutor` worker 里。

这是我们在 1T dense 训练时踩过的坑，症状是**罕见随机崩溃**（约 13 次里 2 次），
且规模越大越容易命中（每个 rank 都 mmap）。

> 规则：任何 `mmap` / `np.load(mmap_mode)` 的路径必须落在本地盘。
> 大文件顺序读没事，**mmap 随机缺页才是杀手**。

**坑 3：gbs 变化会改变 cache key**

`--global-batch-size` 参与数据索引的 cache key。改了 gbs 就要重新预热 cache，
否则每个 rank 会在启动时各自重建，非常慢。

### 2.3 Dense 验证结果

我们把 dense 当作**基线探针**而非目标：

| 规模 | 结果 |
|---|---|
| 1B / 2 节点 | 30/30 迭代，loss 12.33→7.67，**209 TFLOP/s** |
| 1B / 15 节点 120 GPU | 40/40 迭代，**206 TFLOP/s**，rc=0，0 NaN |

**这一步的真正价值**：它证明了硬件、RCCL、InfiniBand、数据管线全部正常。
后来 MoE 出问题时，这个 206 TFLOP/s 的参照系直接把嫌疑范围压缩到
"MoE 特有的代码路径"，省掉了大量方向性的错误。

> **建议**：上 MoE 之前一定先跑通同规模 dense。它是最便宜的"排除硬件"实验。

---

## 三、在 Megatron 上写一个 MoE 模型

### 3.1 MoE 参数集

在 dense 基础上增加：

```bash
--num-experts 96                    # expert 总数
--moe-router-topk 6                 # 每 token 激活几个
--moe-ffn-hidden-size 896           # 单个 expert 的 FFN 宽度
--moe-shared-expert-intermediate-size 896   # 共享 expert（可选但推荐）
--moe-layer-freq "[0]*2+[1]*46"     # 哪些层是 MoE
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 0.001
--moe-z-loss-coeff 0.0001
--moe-router-dtype fp32             # 路由用 fp32
--moe-token-dispatcher-type alltoall
--expert-model-parallel-size 8      # EP
```

### 3.2 几个设计决策及理由

**`--moe-layer-freq "[0]*2+[1]*46"`：前 2 层保持 dense**

底层特征通用性强，路由收益低；而训练早期 router 尚未学好，
让最靠近 embedding 的层走稠密路径可以稳定梯度。这是社区常见做法。

注意这个字符串是**被 `eval()` 的 Python 表达式**，`[0]*2+[1]*46` 展开成
48 个元素的列表，0=dense、1=MoE。长度必须等于 `--num-layers`。

**`--expert-model-parallel-size 8`：EP 组锁在节点内**

96 experts / EP=8 = 每 rank 12 个 local expert。选 8 而不是更大，
是因为**一个节点恰好 8 张 GPU** —— token dispatch 的 alltoall 完全走节点内
xGMI 互联，不触碰 InfiniBand。

这个决策后来意外地救了场：它使"单节点复现"成为可能，
而单节点复现正是我们最终定位根因的决定性一步（见 §4.1）。

**dropless（不设 capacity factor）**

我们实测对比过：

| 配置 | it1 |
|---|---|
| dropless | 233 s |
| `--moe-expert-capacity-factor 1.25` + pad | **更慢**，9 分钟 0 迭代 |

固定容量需要 padding 到容量上限，在我们的 shape 下反而更贵。
**dropless 是实测更优，不是省事。**

**`--moe-router-dtype fp32`**

路由是 argmax + softmax，bf16 下容易出现 tie-breaking 抖动，
导致同一 token 在不同 step 路由到不同 expert。fp32 路由几乎不增加成本
（router 只是 `hidden × num_experts` 的小矩阵）。

### 3.3 并行策略怎么选

```
world = 120 = EP(8) × DP(15)
TP = 1, PP = 1
```

| 维度 | 值 | 理由 |
|---|---|---|
| EP | 8 | 锁在节点内，alltoall 走 xGMI |
| DP | 15 | 每个 expert 副本被 15 路数据并行共享 |
| TP | 1 | 25.86B 在 192 GiB 上放得下，避免额外 all-reduce |
| PP | 1 | 无流水线气泡 |

**MoE 的显存特点**：参数量大（25.85B）但激活量小（每 token 只激活 6/96）。
所以 TP=1 是可行的，这与同等参数量的 dense 模型很不一样。

实测每 rank 4,585,502,720 参数，58.7 GiB reserved / 192 GiB —— 相当宽裕。

### 3.4 micro batch 的取舍

| mbs | TFLOP/s | 结果 |
|---|---|---|
| 8 | 139 | **OOM**（需约 160 GiB 激活） |
| 1 | 48 → 82.8 | 可行，配 8 步梯度累积 |

MoE 的激活显存随 mbs 线性增长，且因为 dropless 无上限。mbs=1 是唯一选择。

---

## 四、我们实际遇到的问题

按严重程度排序。这一节是本文的核心 —— **这些问题没有一个能靠读文档避免**。

### 4.1 `--moe-grouped-gemm` 死锁 MoE backward（最严重）

**现象**：48 层 MoE 在 120 GPU 上**从未完成一次迭代**。2 小时 20 分 0 iteration，
且 120 分钟的分布式超时**从未触发**。

`--moe-grouped-gemm` 的本意是把每个 rank 上 12 个 expert 的 GEMM 融合成一次
grouped 调用，理论上应该更快。在此 ROCm 构建上它会死锁。

**单变量对照**（48 层单节点，只改这一个 flag）：

| grouped-gemm | it1 | 之后 | TFLOP/s |
|---|---|---|---|
| **开** | 从未完成 | 8/8 rank 卡在 `backward_step` @250 W | 2.4 |
| **关** | 52 s | **22/25 迭代，loss 12.338→7.797** | **82.5** |

**34 倍差距，且是"能跑 / 不能跑"的区别。**

**为什么花了这么久才找到**，两个方法论错误：

**错误一：混淆变量。** 我扫了 DP 宽度 8 / 16 / 120，得到"停顿随宽度加剧"的
漂亮单调结论。但**唯一的 48 层配置也是唯一的 120 宽配置** —— 所有 bisect 臂
都是 12 层，深度和宽度在数据里完全共线。效应被记在了正在扫的那个变量头上。

固定 DP=16 只把深度 12→48，**立刻复现**：

| 深度 | DP | it1 | 之后 |
|---|---|---|---|
| 12 | 16 | ~6 s | 22/25 收敛 |
| **48** | **16** | **310 s @ 2.4 TFLOP/s** | it2 十二分钟未完成 |

再进一步，**48 层单节点也复现** —— 8 个 rank 全卡在 backward，
功耗 248–254 W（sd < 3 W = RCCL 自旋等待），**零跨节点流量**。
这一刀砍掉了整条网络排查线。

**错误二：栈帧误导。** 停顿的栈一直在移动：
`token_permutation` → `custom_backward` → `get_grad_norm_fp32`。
三次以为找到热点，三次都错 —— 它们全是**下游受害者**。
一个 rank 卡在融合 GEMM 里，其余 119 个堆在"下一个 collective"上，
具体是哪个取决于各自到达时刻。

沿途被单变量证伪的假设（每一条都花了至少一轮实验）：

capacity factor（两次）· DDP bucket size · overlap-grad-reduce ·
distributed optimizer · global batch size · gradient clipping ·
token dispatcher（alltoall→allgather，**也卡**）· IB fabric（错误计数全 0）·
显存（65/192 GB，零 allocator retry）· 单节点硬件 · 120-rank 裸集合通信（全正常）

### 4.2 checkpoint 保存：两个 torch/MCore 版本错配

第一次生产跑到 **iteration 2000**（loss 已降到 8.66）后死于 checkpoint 保存，
挂满 120 分钟超时，**2000 个干净迭代全部损失**。

**缺陷 A：`_write_item` 参数个数不匹配**

```
TypeError: _write_item() missing 1 required positional argument:
           'serialization_format'
```

torch 2.10-dev 给 `filesystem._write_item` 加了第 6 个参数；
MCore 0.12.4 的 `filesystem_async.py:189` 只传 5 个。

异常发生在 checkpoint **worker 进程**里，于是 rank 1 退出码 1，
其余 119 个永远阻塞在收集写入结果的 `gather_object`。
又一次"报错的是受害者"。

修复放在 `rocm_shim.py`，把新参数绑定到默认值。**关键细节**：
MCore 用的是 `from ... import _write_item`（按值导入），
所以必须**同时 patch torch 的模块属性和 MCore 已导入的引用**。

**缺陷 B：async worker 永不归队**

修好 A 后仍挂，但形态变了：rank 0 卡在 `process.join()` 而
**checkpoint worker 进程根本不存在**，零字节写出。

`torch_dist` 的整个 async 子系统在此构建上不可用。
解决：`--ckpt-format torch` 走非 async 路径。

**验证**（新增第 20 步保存的测试臂，把验证成本从 5.5 小时降到 10 分钟）：
302 秒写完 357.31 GB，15 个 `mp_rank_*` 目录，训练继续跑到 40/40，
保存前后步时 10.06 / 10.00 秒无影响。

### 4.3 blobfuse 缓存不跨节点失效

失败保存留下的 `latest_checkpointed_iteration.txt` 删不掉：

1. node-14 上 `rm -rf` 整个目录 → 其他节点仍看得见
2. 逐个节点删 15 次 → 仍有 14 个节点看得见
3. 最后读到损坏字节：`FileNotFoundError: .../iter_4474717655932076052`

解决：不跟缓存较劲，checkpoint 目录名做成常量，污染时直接 bump 版本号。

### 4.4 五个 ROCm 适配 shim

Megatron 假设 NVIDIA 环境，在 ROCm 上需要适配。我们的纪律是
**Megatron checkout 保持零改动**（`git status --porcelain` 输出 0 行），
所有适配走 `rocm_shim.py`：

| shim | 问题 |
|---|---|
| flash-attn 版本门 | TE 硬编码 ≤2.8.1，把 2.8.3 报告为 2.8.1 才能解锁 FlashAttention 后端（5.4 ms/0.89 GiB vs unfused 14.0 ms/6.76 GiB） |
| `fused_kernels.load` | 上游需要 nvcc；该函数在任何平台上都不构建东西，置空 |
| EP group timeout | 上游给这个组漏传 `timeout=`（`parallel_state.py:1133`） |
| `_write_item` arity | 见 §4.2 |
| SIGUSR1 | 改为非致命的全线程栈转储，用于在线诊断 |

### 4.5 自伤：`pkill -f` 匹配到 trainer 的 argv

修复生效后生产已健康跑了 37 个迭代 @ 75.5 TFLOP/s。
我为重启 TensorBoard 执行了：

```bash
pkill -f '[t]ensorboard'      # ← 120 个 rank 全灭
```

trainer 的命令行含 `--tensorboard-dir /scratch/.../tensorboard`，
`-f` 匹配整条 argv。

> **规则**：按绝对可执行路径杀进程，且任何 `pkill` 前先 `pgrep -af` 确认命中范围。
> `--tensorboard-dir`、`--save`、`--load`、`--data-path` 会把目录名带进 argv，
> 用工具名做模式尤其危险。

---

## 五、为什么 FSDP2 快而 Megatron 慢

这不是框架优劣，是**抽象层次决定的故障模式差异**。

### 5.1 故障面对比

| | FSDP2（8B dense） | Megatron（25.86B MoE） |
|---|---|---|
| 模型正确性 | 自己写的，可本地验证 | 由参数生成，只能在集群验证 |
| 出错位置 | 你的 264 行代码 | 框架 × ROCm × torch 版本组合 |
| 最小复现 | 本地 tiny 配置，秒级 | 至少 1 节点 8 卡，分钟级 |
| 典型 bug | 形状不匹配、loss 不降 | **死锁**、版本 API 错配、缓存不一致 |
| 报错质量 | Python traceback，指向你的行号 | 120 个 rank 在不同 collective 处堆积 |

**最关键的一行**：FSDP2 的 bug 会**抛异常**，Megatron 的 bug 会**挂住**。
异常有栈、有行号、有类型；挂住只有"120 个进程都在等"。

### 5.2 为什么 MoE 特别难

Dense 的 bug 通常是确定性的：形状错了就是错了，第一次 forward 就崩。

MoE 引入了三个新的复杂度来源：

1. **数据依赖的控制流** —— 每个 token 路由到哪个 expert 取决于数据，
   所以各 rank 的工作量不同、通信量不同、到达 collective 的时刻不同。
   一个 rank 慢，全体等待，而**报错的是等待者**。
2. **额外的通信模式** —— alltoall 的形状是可变的（variable-split），
   比 all-reduce 更容易触发框架/库的边界情况。
3. **深度放大** —— 46 个 MoE 层意味着 46 次 dispatch/combine。
   单层看不出的问题，48 层会被放大到死锁。
   （我们的单层 profile 预测 48 层约 1 秒，实测 310 秒。）

### 5.3 如果重来一次

**建议的顺序**（我们的实际顺序恰好把最贵的放在了最前）：

1. **先跑通同规模 dense** —— 最便宜的"排除硬件"实验
2. **MoE 单节点最小配置** —— 少层数、少 expert，先验证模型能构建
3. **单节点扫深度** 12 → 24 → 48 —— 深度是**便宜维度**，
   改一个数字，1 节点可测
4. **再扩节点数** —— 节点数是**昂贵维度**，要占满集群
5. **最后才是生产规模**

我们把 3 和 4 做反了：先在 15 节点上扫 DP 宽度，代价是 120 GPU × 2.3 小时，
而且因为深度/宽度共线，得出了错误结论。

**核心教训**：**代理配置只能证伪，不能证实。**
小配置跑不通 → 生产一定跑不通（有用）；
小配置跑通了 → 生产未必跑通（无信息量）。
我们拿 12 层的成功给 48 层背书，这是最贵的一次错误。

---

## 六、Megatron 上的诊断手法

从这轮排障沉淀出的、可复用的方法。

### 6.1 挂死判定的两个必要条件

**必须同时成立**：
1. 进度指标不再变化（要有可观测推进证据，不是"日志没新行"）
2. rank 间进度不对齐

`timeout` 返回 124 只意味着"没在 N 秒内跑完"。
我们曾把 6 次 `rc=124` 全部误判为死锁，实际是首步 autotune 177 秒
撑爆了自己设的 420 秒超时。

### 6.2 探针清单（MI300X 实测）

| 探针 | 判读 |
|---|---|
| **GPU 功耗** `rocm-smi -P` | idle≈140 W / **RCCL 自旋 245–260 W，节点内 sd<8 W** / 真 GEMM 400–700 W，sd>50 |
| **进程 io** `/proc/<pid>/io` | delta=0 → 完全静止；**不同 rank delta 不同 = 不对齐** |
| **CPU jiffies** `/proc/<pid>/stat` 14/15 列 | 20 s 只走 1–2 → 静止，且排除"在编译 kernel" |
| **TensorBoard 字节数** | 冻结时刻 = 停止推进时刻，比日志行可靠 |
| **日志尺寸横比** | 12/15 节点字节级相同 → 全体卡同一处，无异类 |
| **watchdog 计数** | 数量 ≠ 总 rank 数 → 找差额；**全 0 也是信号** |

`GPU use% = 100` **无意义** —— 自旋和真算都是 100。

### 6.3 沉默 rank 规则

**在集合通信超时里，报错的 rank 是受害者，沉默的才是真凶。**

先数 watchdog 数量是否等于总 rank 数，不等就去找差额那部分。
`grep -rl` 按文件定位哪些节点**没有**报错，比分析报错内容更快。

本轮验证：checkpoint 失败时 node-0 日志 97 KB 而其余 1.2 MB，
exitcode 1 而其余 -15（我的 SIGTERM）—— 异类立刻显形。

### 6.4 在线栈转储

`ptrace_scope=1` 时 gdb 用不了、py-spy 装不上，只剩 Python 层手段：

```python
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
```

发信号前必须**排除 torchrun agent**（它的 argv 也含入口脚本名但不导入 shim，
信号会杀死它）。判据是 `/proc/<pid>/status` 的 `SigCgt` 第 9 位。

### 6.5 超时本身要能触发

`distributed_timeout_minutes=120` 在 T+129 分钟**零 watchdog** ——
卡点不在被监控的集合通信里。后果：崩溃退化成 120 个 GPU 的无限期占用。

对策：
- 超时降到 **20 分钟**（稳态步时 9 秒，超过 20 分钟必死）
- 加**外部看门狗**：独立于框架，按日志字节停滞 30 分钟判杀，
  用 double-fork + `nohup` 使 `ppid=1`，不受 ssh 会话影响

---

## 七、结论

**什么时候用 FSDP2**
- 模型 ≤10B，单一数据并行够用
- 需要快速迭代模型结构
- 团队要能读懂、能调试模型代码

**什么时候用 Megatron**
- 模型 >10B，或需要 MoE / TP / PP
- 结构稳定，不打算频繁改模型
- 有预算承受"框架内部问题"的排查成本

**用 Megatron 时的准备**
1. 先跑通同规模 dense 作为基线
2. 把所有框架适配集中到一个 shim 文件，保持上游 checkout pristine
3. 建立便宜的复现路径（单节点、少层数），**别在生产规模上调试**
4. 装外部看门狗，别信框架自带的超时
5. 每个实验**只改一个变量**，并检查它是否与别的变量共线

**最终结果**：25.86B MoE 在 120 GPU 上稳定运行，
**82.8 TFLOP/s/GPU**，步时中位 9.17 秒，**0/776 迭代超过 30 秒**，
loss 12.3330 → 4.1954。

代价是几天的排障。回头看，其中大部分本可以用
"先在 1 节点上扫深度"这一个决定省掉。

---

## 附录：完整生产配置

```bash
# 几何：48 层 = 2 dense + 46 MoE
--num-layers 48 --hidden-size 2048
--ffn-hidden-size 5504                    # dense 层
--moe-ffn-hidden-size 896                 # 每个 expert
--num-attention-heads 32
--group-query-attention --num-query-groups 4
--seq-length 4096 --max-position-embeddings 32768
--swiglu --normalization RMSNorm --position-embedding-type rope
--qk-layernorm --untie-embeddings-and-output-weights --disable-bias-linear

# MoE
--num-experts 96 --moe-router-topk 6
--moe-shared-expert-intermediate-size 896
--moe-layer-freq "[0]*2+[1]*46"
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 0.001 --moe-z-loss-coeff 0.0001
--moe-router-dtype fp32
--moe-token-dispatcher-type alltoall
# 注意：不加 --moe-grouped-gemm（会死锁）
# 注意：不设 capacity factor（dropless 更快）

# 并行：120 = EP(8) × DP(15)
--expert-model-parallel-size 8
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1

# 训练
--micro-batch-size 1 --global-batch-size 960
--train-iters 203451
--lr 2e-4 --min-lr 2e-5 --lr-decay-style cosine --lr-warmup-iters 2543
--clip-grad 1.0 --weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.95
--bf16 --use-distributed-optimizer

# 稳定性
--ckpt-format torch                       # torch_dist 的 async 子系统损坏
--save-interval 2000
--distributed-timeout-minutes 20          # 不是 120
--timing-log-level 0                      # timers 在热路径上做 barrier
```
