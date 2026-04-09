---
tags: [HCCL_BUFFSIZE, FUSED_MC2, layer_sharding, MoE, memory, DeepSeek-V3, W8A8, FlashComm2, tiling, o_proj, q_b_proj]
---

# vLLM-Ascend 特性开关与显存优化总览

## 结论速览

- 本文聚焦三个最容易被一起问到的 Ascend 主题：`HCCL_BUFFSIZE`、`VLLM_ASCEND_ENABLE_FUSED_MC2`、`layer_sharding`。
- `HCCL_BUFFSIZE` 不是泛化的“调大 HCCL 性能”开关，而是 A3 MoE 通信算子 tiling 阶段使用的窗口容量上限，默认 200 MB。
- `VLLM_ASCEND_ENABLE_FUSED_MC2` 不是“开启 MC2”的总开关，而是“允许用 fused MoE 通信算子替换默认 MC2 / ALLTOALL 路径”的开关。
- `layer_sharding` 的核心作用是按 layer 维度分散大线性层权重驻留，减少每张 NPU 常驻的权重体积；计算语义不应改变，代价是额外的广播与预取。
- 在 `DeepSeek-V3.2` 这类 A3 + W8A8 + MoE 场景下，这三类配置常常一起出现，但它们分别作用于：
  - MoE 通信窗口容量
  - MoE 通信算子选择
  - 大权重层的驻留与加载方式

## `HCCL_BUFFSIZE`

### 1. 真实语义

`HCCL_BUFFSIZE` 在 `vllm_ascend/csrc/moe_dispatch_normal/op_host/moe_dispatch_normal_tiling.cpp` 和 `vllm_ascend/csrc/moe_combine_normal/op_host/moe_combine_normal_tiling.cpp` 中被读取。

实现要点：

- 环境变量名固定为 `HCCL_BUFFSIZE`
- 默认值是 `200`
- 单位是 MB
- 代码里会换算成字节：`HCCL_BUFFSIZE * 1024 * 1024`

因此：

- `export HCCL_BUFFSIZE=200` 的直接含义就是把这类 MoE 通信窗口上限设为 200 MB

### 2. 它影响什么

这里影响的不是“所有 HCCL 通信”，而是 A3 MoE 自定义算子在 tiling 阶段可用的 window 大小判断。

两个典型检查点：

- `moe_dispatch_normal` 会校验 dispatch 所需窗口是否超过上限
- `moe_combine_normal` 会校验 combine 所需窗口是否超过上限

如果窗口不够，tiling 阶段就会直接失败，而不是运行时默默退化。

### 3. 应该怎么理解

- 值太小：大 batch、大 `k`、大 hidden size 场景下容易直接报错
- 值调大：本质上是在为更重的 MoE 通信预留更大的窗口空间

所以它更像“容量阈值”，不是笼统的性能调优旋钮。

## `VLLM_ASCEND_ENABLE_FUSED_MC2`

### 1. 取值定义

环境变量定义在 `vllm_ascend/envs.py`：

- `0`：默认路径，不启用 fused 替换
- `1`：允许使用 `dispatch_ffn_combine`
- `2`：允许使用 `dispatch_gmm_combine_decode`

代码判断主要发生在：

- `vllm_ascend/ascend_forward_context.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`

### 2. 它控制的不是“MC2 开或关”

真正的语义是：

- 先根据机型、token 数、量化类型、EP/DP 规模等因素决定当前更适合 `ALLGATHER`、`ALLTOALL`、`MC2`
- 在满足条件时，再允许用 fused 算子替换默认 MoE 通信实现

所以：

- `VLLM_ASCEND_ENABLE_FUSED_MC2=1/2` 是“替换策略开关”
- 不是“默认 MoE 通信一定走 MC2”的开关

### 3. `=1` 和 `=2` 的差别

#### `=1`

`dispatch_ffn_combine` 路径的约束更偏向：

- A3
- `w8a8_dynamic`
- EP 不超过 32
- 非 draft model
- 非 MTP
- 非 dynamic EPLB

这条路径在 prefill 和 decode 都可能生效。

#### `=2`

`dispatch_gmm_combine_decode` 更偏向：

- decode 侧
- `w8a8_dynamic`
- speculative 路径满足额外条件

代码里对 prefill 会直接关闭这条 fused 选项，因此它本质上是 decode 定向优化。

### 4. 和 MTP 的关系

不应把它理解成“只有开 MTP 才有效”。

更准确地说：

- `=1` 明确是非 MTP 路径
- `=2` 也不是以 MTP 为前提，只是在某些 speculative method 是 `mtp` 时会再多一层量化约束

因此 MTP 不是必要条件，只是部分分支下的附加限制条件。

## `layer_sharding`

### 1. 它解决什么问题

`layer_sharding` 的配置入口在：

- `vllm_ascend/ascend_config.py`

官方说明在：

- `docs/source/user_guide/feature_guide/layer_sharding.md`

核心实现位于：

- `vllm_ascend/ops/layer_shard_linear.py`

它面向的是“跨很多层重复出现、结构相同但权重不同的大线性层”，目标是减少每张 NPU 上常驻的总权重体积。

### 2. 真实机制

核心机制不是切分单层矩阵，而是“按 layer 维度分片存放”：

- 同一类层按 layer index 组成一个 series
- 第 `i` 层权重只常驻在通信组内的 `i % K` 号设备
- 其余设备只保留 dummy tensor
- 运行时通过异步 broadcast 预取下一层真实权重

`vllm_ascend/ops/layer_shard_linear.py` 中可以直接看到：

- 非 source 设备会释放真实权重
- 权重加载后会对同一 series 做统一 post-process
- 每到一层前会等待对应权重 ready
- 执行当前层时会预取后续层权重

因此它是“layer 维度的驻留与预取优化”，不是近似计算。

### 3. 为什么常与 `q_b_proj` / `o_proj` 一起出现

在 `DeepSeek-V3.2` 这类 DSA-CP 场景里：

- `q_b_proj`
- `o_proj`

都属于非常吃显存的大权重层。

所以 `docs/source/tutorials/models/DeepSeek-V3.2.md` 会使用：

```bash
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'
```

这意味着：

- 每张卡不再常驻所有 hidden layers 的这两类权重
- 只常驻自己负责的那一部分 layer 权重
- 其余 layer 在运行时按需广播过来

### 4. 与 FLASHCOMM2 / DSA-CP 的关系

官方 feature guide 里明确写到：

- FLASHCOMM2 场景下，推荐只对 `o_proj` 做 layer sharding
- DSA-CP 场景下，`q_b_proj` 与 `o_proj` 都可能是高价值对象

同时 `vllm_ascend/utils.py` 也明确限制：

- 若开启 FLASHCOMM2，只接受 `layer_sharding == ["o_proj"]`
- 其他组合会直接报错

所以：

- `["q_b_proj", "o_proj"]` 更适合 DSA-CP 一类场景
- `["o_proj"]` 更适合 FLASHCOMM2 一类场景

### 5. 代价与收益

收益：

- 降低每张卡常驻权重规模
- 帮助更深模型或更激进并行配置装入显存

代价：

- 增加运行时广播
- 若当前层计算无法完全掩盖预取通信，吞吐或时延可能变差

因此它是显存换通信的优化，不是无代价提升。

## 面向 `DeepSeek-V3.2` 的理解方式

`docs/source/tutorials/models/DeepSeek-V3.2.md` 里的典型 A3/W8A8/MoE 配置同时出现：

- `HCCL_BUFFSIZE=200`
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
- `--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'`

更合理的拆解方式是：

- `HCCL_BUFFSIZE`：解决 MoE 通信窗口容量问题
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`：影响 runtime shape 与通信优化路径
- `VLLM_ASCEND_ENABLE_FUSED_MC2`：进一步改变 MoE 通信算子选择
- `layer_sharding`：减少大线性层权重常驻显存

这些项可以同时出现，但并不是同一个维度的开关。

## 易混淆点

- `HCCL_BUFFSIZE` 不是“越大越快”，它首先是容量约束参数。
- `VLLM_ASCEND_ENABLE_FUSED_MC2` 不是“开启 MC2 的总开关”，而是 fused 替换策略开关。
- `layer_sharding` 是按 layer 分散权重驻留，不是把单层矩阵做数值近似或改变计算语义。
- `["q_b_proj", "o_proj"]` 与 `["o_proj"]` 的适用场景不同，不能在 FLASHCOMM2 场景里随意混用。
