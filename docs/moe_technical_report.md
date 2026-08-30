# R-Full 25.86B MoE 预训练技术报告

从模型设计到 120 GPU 生产训练启动的完整记录

---

**状态** 生产训练运行中 · `rfull_moe_prod_0830_215058`
**硬件** 15 节点 × 8 × AMD MI300X = 120 GPU（集群 `chec-mi300-7`）
**软件** Megatron-LM `5cb6dbb` / MCore 0.12.4 · ROCm 7.1.0 · HIP 7.1.25424 · RCCL 2.27.7
**代码** `github.com/chicm/pretrain-moe`，分支 `main` = `megatron-clean` = `8aea377`

---

## 1. 摘要

本项目在 stock Megatron-LM 上重建了 R-Full 25.86B MoE 模型的预训练流程，目标是
0.80T token 的完整训练。整个过程分三阶段验证：dense 1B 多节点 → MoE 单节点 →
120 GPU 生产。

**最终结果**

| 指标 | 数值 |
|---|---|
| 总参数 | **25.85B**（transformer 25.54B + embedding 0.31B） |
| 每 rank 参数 | 4,585,502,720 |
| 吞吐 | **82.7 TFLOP/s/GPU**（中位） |
| 步时 | 中位 **9.18 s**，P90 9.49 s，最大 16.56 s |
| 停顿率 | **0 / 686** 迭代超过 30 s |
| 收敛 | loss **12.3330 → 4.3900**（691 迭代） |
| 显存 | 58.7 GiB reserved / 192 GiB |
| 预计耗时 | **21.6 天** |

**关键结论一句话**：阻塞整个项目最久的不是分布式问题，而是一个单节点就能复现的
配置项 —— `--moe-grouped-gemm` 在此 ROCm 构建上会死锁 MoE backward。

---

## 2. 模型设计

### 2.1 架构总览

| 维度 | 取值 | 说明 |
|---|---|---|
| 层数 | 48 | **前 2 层 dense + 后 46 层 MoE**（`--moe-layer-freq [0]*2+[1]*46`） |
| hidden | 2048 | |
| 注意力头 | 32 | |
| KV 组 | **4** | GQA，KV cache 降至 1/8 |
| 序列长度 | 4096 | max_position 32768（预留长上下文扩展） |
| 位置编码 | RoPE | |
| 归一化 | RMSNorm | 外加 `--qk-layernorm` |
| 激活 | SwiGLU | |
| 精度 | bf16 | |
| 词表 | 151,643 | |

### 2.2 MoE 配置

| 维度 | 取值 |
|---|---|
| expert 数 | **96** |
| top-k | **6** |
| MoE 层数 | 46（第 3–48 层；前 2 层保持 dense，FFN=5504） |
| expert FFN hidden | 896 |
| 共享 expert | 896（每 token 恒定激活） |
| 路由负载均衡 | `aux_loss`，系数 0.001 |
| z-loss | 0.0001 |
| router 精度 | **fp32**（路由决策对数值敏感） |
| token dispatcher | `alltoall` |
| 容量 | **dropless**（不设 capacity factor） |

**稀疏度**：每 token 激活 6/96 = 6.2% 的 expert。每 MoE 层 expert 参数 0.53B，
实际激活 38.5M（experts 33.0M + shared 5.5M），**稀疏比 7.2%**。

共享 expert 的作用是给所有 token 一条恒定路径，降低路由抖动对早期训练的影响。

**前 2 层保持 dense** 是常见做法：底层特征通用性强，路由收益低，而早期训练时
router 尚未学好，让最靠近 embedding 的层走稠密路径可以稳定梯度。

### 2.3 并行策略

```
world = 120 = EP(8) × DP(15)
TP = 1, PP = 1
```

| 并行维度 | 大小 | 理由 |
|---|---|---|
| Expert Parallel | **8** | 96 experts / 8 = 每 rank 12 个 local expert；EP 组限制在**节点内**，alltoall 走 xGMI 而非 IB |
| Data Parallel | 15 | 每个 expert 副本被 15 路 DP 共享 |
| Tensor Parallel | 1 | 25.86B 在 192 GiB 上无需 TP，避免额外 all-reduce |
| Pipeline Parallel | 1 | 无 PP 气泡 |

**EP=8 而非更大**是一个刻意选择：EP 组恰好落在单节点 8 张 GPU 内，token
dispatch 的 alltoall 完全走节点内 xGMI 互联，不触碰 InfiniBand。这个设计后来在
排障时意外地帮了大忙 —— 它让"单节点复现"成为可能。

### 2.4 训练超参

| 参数 | 取值 |
|---|---|
| global batch | 960 |
| micro batch | **1** |
| 梯度累积 | 8 步 |
| 总迭代 | 203,451 |
| 总 token | **0.80T** |
| 学习率 | 2e-4 → 2e-5，cosine |
| warmup | 2,543 步（1.25%） |
| Adam | β=(0.9, 0.95)，wd=0.1 |
| grad clip | 1.0 |
| 优化器 | distributed optimizer |
| checkpoint | 每 2000 步 |

**`micro_batch_size = 1` 是实测结论**：mbs=8 能达到 139 TFLOP/s（对比 48），
但需要约 160 GiB 激活显存而 OOM。mbs=1 + 8 步梯度累积是唯一能跑的配置。

---

## 3. 工程结构

刻意与之前的 JSON 配置树决裂，改为代码化配置：

```
moe_rebuild/
  config.py       构建 argv，所有超参的单一真相源
  specs.py        RunSpec 注册表：生产臂 + 各实验臂
  rocm_shim.py    全部 ROCm 适配（Megatron 保持 pristine）
tools/
  launch.py       多节点启动
  monitor.py      进度解析
  pack.py         打包 + 陈旧性防护
  warm_cache.py   数据索引预热
  pretrain_entry.py
tests/            41 个离线测试
docs/rebuild_status.md   878 行诊断记录
```

**核心纪律：Megatron-LM checkout 保持零改动。**
`git status --porcelain` 输出 0 行。所有 ROCm 适配走 `rocm_shim.py`，共 4 个 shim：

| shim | 作用 |
|---|---|
| `_enable_flash_attention` | TE 硬编码 flash-attn ≤2.8.1，把 2.8.3 报告为 2.8.1 以解锁 FlashAttention 后端 |
| `_noop_load` | `megatron.legacy.fused_kernels.load` 置空（ROCm 上无需编译 CUDA kernel） |
| `_install_ep_group_timeout_fix` | 把 collective timeout 传给 MCore 的 EP 子组 |
| `_install_write_item_arity_fix` | 修 torch/MCore 的 checkpoint API 版本错配（见 §5.2） |

配套的陈旧性防护：`pack.py` 生成 `_deploy.b64`，`_deploy.py` 拒绝过期 tarball。
这避免了"改了代码但集群跑的是旧版"这类最难查的问题。

---

## 4. 三阶段验证

| 阶段 | 内容 | 结果 |
|---|---|---|
| Phase 1 | dense 1B，2 节点 | 30/30 迭代，loss 12.33→7.67，209 TFLOP/s |
| Phase 1b | dense 1B，15 节点 120 GPU | 40/40 迭代，206 TFLOP/s，rc=0，0 NaN |
| Phase 2 | MoE 48L 几何构建 | 25.85B，每 rank 4.59B，82 GiB / 192 GiB |
| Phase 3 | 120 GPU 生产 | **见下文，一路踩坑** |

Phase 1 的 206 TFLOP/s 后来成为关键参照系 —— 它证明了硬件、RCCL、IB 都是好的，
把问题范围压缩到 MoE 特有的代码路径。

---

## 5. 问题与解决

按发现顺序，共 4 个真实缺陷 + 1 个自伤事故。

### 5.1 核心缺陷：`--moe-grouped-gemm` 死锁 MoE backward

**现象**
48 层 MoE 在 120 GPU 上**从未完成一次迭代**。2 小时 20 分钟 0 iteration，
120 分钟的分布式超时**从未触发**，最后靠 NCCL watchdog 才中止。

**排查过程中被逐一证伪的假设**（每个都是单变量实验）

| 假设 | 结果 |
|---|---|
| capacity factor 不足 | 证伪两次；dropless 反而**更快** |
| DDP bucket size / overlap-grad-reduce | 证伪，且引入了回归 |
| distributed optimizer | 关掉，仍卡 |
| global batch 960→120 | 仍卡 |
| gradient clipping（grad-norm 的 DP 规约） | `--clip-grad 0` 后仍卡 |
| token dispatcher alltoall→allgather | 仍卡 |
| IB fabric | 8×400 Gb/s NDR，错误计数全 0，RCCL 均匀使用 |
| 显存 | 65 GB / 192 GB，零 allocator retry |
| node-5 硬件 | 排除 |
| 120-rank 裸集合通信 | all_reduce 0.3 ms，reduce_scatter 4.57 ms，barrier 0.33 ms，全正常 |

**为什么绕了这么久**

两个方法论错误：

1. **混淆变量**。我扫了 DP 宽度 8/16/120，得到"停顿随宽度加剧"的漂亮结论。
   但**唯一的 48 层配置也是唯一的 120 宽配置** —— 深度和宽度完全共线，
   所有 bisect 臂都是 12 层。效应被记在了正在扫的那个变量头上。

2. **栈帧误导**。停顿的栈一直在移动：`permute` → `custom_backward` →
   `get_grad_norm_fp32`。我三次以为找到了热点，三次都是错的 ——
   它们全是**下游受害者**：一个 rank 卡在融合 GEMM 里，其余 119 个堆在
   "下一个 collective"上，具体是哪个取决于各自到达时刻。

**决定性实验**

固定 DP=16，只把深度从 12 换成 48：

| 深度 | DP | it1 | 之后 |
|---|---|---|---|
| 12 | 16 | ~6 s | 22/25，loss 12.345→7.900 |
| **48** | **16** | **310 s @ 2.4 TFLOP/s** | it2 十二分钟未完成 |

复现了！而且只用 2 个节点、9 分钟，替代了 15 节点 140 分钟。

再进一步 —— **48L 单节点也复现**。这一刀直接砍掉了整条网络排查线：
8 个 rank 全部卡在 `backward_step`，功耗 248–254 W（sd < 3 W = RCCL 自旋等待），
**零跨节点流量**。

**根因确认**（48L 单节点，只改一个 flag）

| `--moe-grouped-gemm` | it1 | 之后 | TFLOP/s | >30s |
|---|---|---|---|---|
| **开** | 从未完成 | 0 iter，8/8 rank 卡 backward @250W | 2.4 | 100% |
| **关** | 52 s | **22/25，loss 12.338→7.797** | **82.5** | **0/21** |

**34 倍差距，且是"能跑 / 不能跑"的区别。**

**解决**：`moe_grouped_gemm = False`，保留为可切换项以便未来 ROCm 回归测试。

### 5.2 checkpoint 保存的两个版本错配

第一次生产跑到 **iteration 2000**（loss 已降到 8.66）后死于 checkpoint 保存。
保存挂满 120 分钟超时，NCCL watchdog 中止全部 120 rank，**2000 个干净迭代全部损失**。

**缺陷 A：`_write_item` 参数个数不匹配**

```
TypeError: _write_item() missing 1 required positional argument:
           'serialization_format'
```

torch 2.10-dev 给 `filesystem._write_item` 加了第 6 个参数；MCore 0.12.4 的
`filesystem_async.py:189` 只传 5 个。异常发生在 checkpoint **worker 进程**里，
于是 rank 1 退出码 1，其余 119 个永远阻塞在收集写入结果的 `gather_object` ——
又一次"报错的是受害者"。

修复在 `rocm_shim.py`：把新参数绑定到 `SerializationFormat.TORCH_SAVE`。
关键细节是 MCore 用的是 `from ... import _write_item`（按值导入），
所以**必须同时 patch torch 的模块属性和 MCore 已导入的引用**。5 个单测覆盖。

**缺陷 B：async worker 永不归队**

修好 A 后仍挂，但形态变了：
- rank 0 卡在 `async_utils.py:248 process.join()`，而 **checkpoint worker 进程根本不存在**
- 其余 rank 卡在 `gather_object`
- **零字节写出**

整个 `torch_dist` async 子系统在此构建上不可用。

**解决**：`--ckpt-format torch`，走非 async 路径（`checkpointing.py:398`），
通过独立的 `distrib_optim.pt` 支持 distributed optimizer。

**验证**（新增 `moe_prod_15n_ckpttest` 臂，第 20 步保存而非第 2000 步，
把验证成本从 5.5 小时降到 10 分钟）

| 检查项 | 结果 |
|---|---|
| 保存完成 | **是**，302 秒 |
| 大小 | **357.31 GB**，15 个 `mp_rank_*` 目录 |
| `latest_checkpointed_iteration.txt` | 20 |
| 保存后继续训练 | 是，跑到 40/40 |
| 前后步时 | 10.06 s / 10.00 s，无影响 |

### 5.3 blobfuse 缓存不跨节点失效

失败保存留下的 `latest_checkpointed_iteration.txt` 删不掉：

1. node-14 上 `rm -rf` 整个目录 → node-0/7 仍看得见 `latest=20`
2. 逐个节点删 15 次 → 仍有 14 个节点看得见
3. 最后读到**损坏字节**：`FileNotFoundError: .../iter_4474717655932076052`

**解决**：不跟缓存较劲。checkpoint 目录名做成常量 `PROD_CKPT_DIR`，
污染时直接 bump（`rfull_moe_prod` → `rfull_moe_prod_v2`）。
测试臂更进一步，每次启动生成带时间戳的唯一目录。

顺带发现一个隐患：`_ckpt_test` 修改器改了 `run_id`，但 `save`/`load` 是在构造
RunSpec 时由**原始** run_id 派生的，不会回溯更新 —— 测试臂差点读写了生产的
checkpoint 目录。**修改器改了字段，必须检查所有派生字段。**

### 5.4 自伤事故：`pkill -f` 匹配到 trainer 的 argv

修复生效后，生产已健康跑了 **37 个 iteration @ 75.5 TFLOP/s**。
我为重启 TensorBoard 服务执行了：

```bash
pkill -f '[t]ensorboard'      # ← 全灭
```

trainer 的命令行含 `--tensorboard-dir /scratch/.../tensorboard`，
`-f` 匹配整条 argv，**120 个 rank 全部收到 SIGTERM**。

这是同一个坑的第三次变体（前两次：停止脚本匹配到自身；SIGUSR1 探针命中
torchrun agent）。

**规则**：按**绝对可执行路径**杀进程，且任何 `pkill` 前先 `pgrep -af` 确认命中范围。
`--tensorboard-dir`、`--save`、`--load`、`--data-path` 这类参数会把目录名带进 argv，
所以用工具名做模式尤其危险。

```bash
pkill -f '/opt/venv/bin/tensorboard'   # 安全，已验证 trainer 存活
```

---

## 6. 诊断方法论

这轮排障沉淀出的可复用手法。

### 6.1 挂死判定的两个必要条件

**必须同时成立**才能叫挂死：
1. 进度指标不再变化（不是"日志没新行"，要有可观测推进证据）
2. rank 间进度不对齐

`timeout` 返回 124 只意味着"没在 N 秒内跑完"。曾把 6 次 `rc=124` 全部误判为死锁，
实际是首步 autotune 177 s 撑爆了我自己设的 420 s 超时。

### 6.2 探针清单（MI300X 实测）

| 探针 | 判读 |
|---|---|
| **GPU 功耗** `rocm-smi -P` | idle≈140 W / **RCCL 自旋 245–260 W，节点内 sd<8 W** / 真 GEMM 400–700 W，sd>50 |
| **进程 io** `/proc/<pid>/io` | delta=0 → 完全静止；**不同 rank delta 不同 = 不对齐** |
| **CPU jiffies** `/proc/<pid>/stat` 14/15 列 | 20 s 只走 1–2 → 静止，且排除"在编译 kernel" |
| **TensorBoard 字节数** | 冻结时刻 = 停止推进时刻，比日志行可靠 |
| **日志尺寸横比** | 12/15 节点字节级相同 → 全体卡同一处，无异类 |
| **watchdog 计数** | 数量 ≠ 总 rank 数 → 去找差额；**全 0 也是信号** |

`GPU use% = 100` **无意义** —— 自旋和真算都是 100。

### 6.3 沉默 rank 规则

**在集合通信超时里，报错的 rank 是受害者，沉默的才是真凶。**

做法：先数 watchdog 数量是否等于总 rank 数，不等就去找差额那部分。
`grep -rl` 按文件定位哪些节点**没有**报错，比分析报错内容更快。

本轮验证：checkpoint 失败时 node-0 日志 97 KB 而其余 1.2 MB，
exitcode 1 而其余 -15 —— 异类立刻显形。

### 6.4 代理配置只能证伪，不能证实

用 12 层的成功给 48 层生产背书，代价是 120 GPU × 2.3 小时。

- 小配置**跑不通** → 生产一定跑不通（有用）
- 小配置**跑通了** → 生产未必跑通（无信息量）

而且当时手里就有警告：12L/DP=120 只跑 1 个 iteration 就停。我把它归进了
"停顿会自愈"的叙事。

### 6.5 先扫便宜维度

深度是便宜维度（改一个数字，2 节点可测），节点数是昂贵维度（要占满集群）。
我把顺序做反了。

### 6.6 超时本身要能触发

`distributed_timeout_minutes=120` 在 T+129 分钟**零 watchdog** ——
卡点不在被监控的集合通信里。后果：崩溃退化成 120 个 GPU 的无限期占用。

**对策**：
- 超时降到 **20 分钟**（稳态步时 10 s，超过 20 分钟必死）
- 加**外部看门狗**：独立于框架，按日志字节数停滞 30 分钟判杀，
  用 double-fork + `nohup` 使 `ppid=1`，不受 ssh 会话影响

---

## 7. 当前状态

**生产运行** `rfull_moe_prod_0830_215058`

| 项 | 值 |
|---|---|
| 进度 | 691 / 203,451 迭代 |
| loss | 12.3330 → **4.3900** |
| 吞吐 | 82.7 TFLOP/s/GPU（峰值 84.0） |
| 步时 | 中位 9.18 s，P90 9.49 s |
| 停顿 | **0 / 686** 超 30 s |
| 节点健康 | 15/15，每节点 proc=41 |
| ETA | 21.6 天 |

**监控**
- TensorBoard：node-14:6006，本地隧道 `localhost:6006 → node-0:16006 → node-14:6006`
- 外部看门狗：node-14，`ppid=1`
- 定时巡检：45 分钟一次

**生产配置的三个不可改动项**
```
--moe-grouped-gemm            关闭（开启 → MoE backward 死锁）
--ckpt-format torch           （torch_dist async 子系统损坏）
--distributed-timeout-minutes 20
```

**仓库** `main` = `megatron-clean` = `8aea377`，51 个 commit，41 个测试通过，
Megatron-LM 零改动。

---

## 8. 待验证

1. **iteration 2000 的生产 checkpoint** —— 已在 ckpttest 臂验证（302 s / 357 GB），
   但生产路径尚未实际走到
2. **长期稳定性** —— 目前最长连续运行约 2000 迭代
3. **grouped GEMM 回归** —— 未来 ROCm 升级后应重测，它本应带来可观加速

---

## 附录：术语

- **EP (Expert Parallel)** —— expert 按 rank 切分，token 经 alltoall 路由到对应 rank
- **dropless** —— 不设 expert 容量上限，不丢弃 token（对比固定 capacity factor）
- **victim/cause** —— 集合通信中报错的 rank 通常在等待，真正的故障 rank 往往沉默
- **straggler** —— 单个 rank 变慢拖累全局，表现为其余 rank 在各种 collective 处堆积
