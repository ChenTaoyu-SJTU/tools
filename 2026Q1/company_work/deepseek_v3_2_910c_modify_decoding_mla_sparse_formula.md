# DeepSeek-V3.2-910C modify：Decoding 分段成本 MLA 公式整理

## 范围

- 工作表：`DeepSeek-V3.2-910C modify`
- 区域标题：`Decoding 分段成本 MLA`
- 本文只分析 sparse 路径
- 已忽略的非 sparse 项只有 3 个：
  - `attn非sparse计算耗时`
  - `attn非sparse访存耗时`
  - `非sparse per_layer耗时`

## 先把表里的参数翻译成人话

下面不再使用单元格编号，统一直接写参数含义。

### 并行与 workload 参数

- `spec_step = 2`
- `accept_rate = 0.8`
- `accept_len = 1.8`
- `decode_world_size = 16`
- `attention_tp_world_size = 16`
- `attention_dp_world_size = 1`
- `ffn_world_size = 16`
- `top_k = 8`

### 模型结构参数

- `hidden_size = 7168`
- `q_head_num = 128`
- `q_lora_rank = 1536`
- `index_head_dim = 128`
- `index_n_heads = 64`
- `k_head_dim = 576`
- `v_head_dim = 512`
- `ffn_dim = 2048`
- `num_experts = 256`

### 解码阶段硬件与经验系数

- `ffn_compute_tflops = 376`
- `memory_bandwidth_TBps = 1.6`
- `tp_collective_bandwidth_GBps = 900`
- `attention_compute_tflops = 376`
- `all2all_bandwidth_GBps = 392`
- `bf16_compute_tflops = 148.4`
- `qkvgemm_sol = 0.4`
- `outproj_gemm_sol = 0.4943323544838017`
- `group_ffn0_sol = 0.3`
- `nccl_tp_sol = 0.4`
- `nccl_a2a_sol = 0.4`
- `memory_sol = 0.7`
- `dsa_sol = 0.7`

## 这一行数据本身的 4 个输入量

对于 `Decoding 分段成本 MLA` 里的任意一行，真正随行变化的输入量可以直接理解成：

- `context_len`
- `spec_global_batch_size`
- `local_batch_size`
- `global_batch_size`

它们之间的关系是：

```text
local_batch_size = spec_global_batch_size / attention_dp_world_size / spec_step
global_batch_size = local_batch_size * attention_dp_world_size
local_spec_dp_batch_size = local_batch_size * spec_step
```

在当前这张表里：

```text
attention_dp_world_size = 1
spec_step = 2
```

所以当前数值上等价于：

```text
local_batch_size = spec_global_batch_size / 2
global_batch_size = local_batch_size
local_spec_dp_batch_size = 2 * local_batch_size
```

## 每一项公式都改写成含义

### 1. 固定写死的经验项

```text
q_a_proj计算耗时 = 10
q_index_b_proj计算耗时 = 15
kv_index_proj计算耗时 = 15
index_weight_proj计算耗时 = 5
kv_a_proj计算耗时 = 10
```

这 5 项不是推出来的，是表里直接给的经验常数。

### 2. 投影和索引相关耗时

#### q_a_proj 访存耗时

```text
q_a_proj访存耗时 =
(
  local_spec_dp_batch_size * hidden_size
  + 2 * local_spec_dp_batch_size * q_lora_rank
  + q_lora_rank * hidden_size
)
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### q_b_proj 计算耗时

```text
q_b_proj计算耗时 =
2 * local_spec_dp_batch_size
* (q_head_num * (128 + 64) / attention_tp_world_size)
* q_lora_rank
* 1000 * 1000
/ ffn_compute_tflops
/ 1024^4
/ qkvgemm_sol
```

这里的 `128 + 64` 是表里直接写死的输出维度拆分，原表没有再用独立命名单元格抽出来。

#### q_b_proj 访存耗时

```text
q_b_proj访存耗时 =
(
  local_spec_dp_batch_size * q_lora_rank
  + 2 * local_spec_dp_batch_size * (q_head_num * (128 + 64) / attention_tp_world_size)
  + (q_head_num * (128 + 64) / attention_tp_world_size) * q_lora_rank
)
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### q_index_b_proj 访存耗时

```text
q_index_b_proj访存耗时 =
(
  local_spec_dp_batch_size * q_lora_rank
  + 2 * local_spec_dp_batch_size * (index_head_dim * index_n_heads)
  + (index_head_dim * index_n_heads) * q_lora_rank
)
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### kv_index_proj 访存耗时

```text
kv_index_proj访存耗时 =
(
  local_spec_dp_batch_size * hidden_size
  + 2 * local_spec_dp_batch_size * index_head_dim
  + index_head_dim * hidden_size
)
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### index_weight_proj 访存耗时

```text
index_weight_proj访存耗时 =
(
  2 * local_spec_dp_batch_size * index_n_heads
  + 2 * local_spec_dp_batch_size * hidden_size
  + 2 * index_n_heads * hidden_size
)
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### index 计算耗时

```text
index计算耗时 =
context_len * spec_step * local_batch_size * 2
* (
    index_n_heads * index_head_dim / attention_compute_tflops
    + index_n_heads / bf16_compute_tflops
  )
* 1000 * 1000
/ 1024^4
/ dsa_sol
```

这项的结构说明它把两部分代价合在一起算：

- 一部分是 `index_n_heads * index_head_dim`
- 一部分是 `index_n_heads`

前者走 attention 算力口径，后者走 bf16 算力口径。

#### index 访存耗时

```text
index访存耗时 =
(
  spec_step * local_batch_size * index_n_heads * index_head_dim
  + context_len * local_batch_size * index_head_dim
  + spec_step * local_batch_size * index_n_heads * 4
)
* 1000 * 1000
/ 1024^4
/ memory_bandwidth_TBps
/ memory_sol
```

### 3. K / V / O 路径

#### k_b_proj 计算耗时

```text
k_b_proj计算耗时 =
2 * q_head_num * local_spec_dp_batch_size * 128 * v_head_dim
* 1000 * 1000
/ bf16_compute_tflops
/ 1024^4
/ qkvgemm_sol
```

这里的 `128` 是表中直接写死的每头维度常数。

#### k_b_proj 访存耗时

```text
k_b_proj访存耗时 =
q_head_num
* (
    2 * local_spec_dp_batch_size * v_head_dim
    + 2 * local_spec_dp_batch_size * 128
    + 2 * 128 * v_head_dim
  )
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### sparse attn 计算耗时

```text
sparse_attn计算耗时 =
2 * min(context_len, 2048)
* spec_step * local_batch_size
* 2 * q_head_num / attention_tp_world_size
* (k_head_dim + v_head_dim)
* 1000 * 1000
/ 1024^4
/ attention_compute_tflops
/ dsa_sol
```

这里最关键的 2 个特征是：

- 它只按 `min(context_len, 2048)` 算，说明 sparse attention 的主计算窗口被截断在 2048
- 它按 `k_head_dim + v_head_dim` 算，说明这里把 K/V 两部分一起计入了 attention 主计算量

#### sparse attn 访存耗时

```text
sparse_attn访存耗时 =
min(context_len, 2048)
* local_batch_size
* (k_head_dim + v_head_dim)
* 1000 * 1000
/ 1024^4
/ memory_bandwidth_TBps
/ memory_sol
```

#### v_b_proj 计算耗时

```text
v_b_proj计算耗时 =
2 * q_head_num * local_spec_dp_batch_size * 128 * v_head_dim
* 1000 * 1000
/ bf16_compute_tflops
/ 1024^4
/ qkvgemm_sol
```

#### v_b_proj 访存耗时

```text
v_b_proj访存耗时 =
q_head_num
* (
    2 * local_spec_dp_batch_size * v_head_dim
    + 2 * local_spec_dp_batch_size * 128
    + 2 * 128 * v_head_dim
  )
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

#### o_proj 计算耗时

```text
o_proj计算耗时 =
2 * local_spec_dp_batch_size
* (q_head_num * 128 / attention_tp_world_size)
* hidden_size
* 1000 * 1000
/ ffn_compute_tflops
/ 1024^4
/ outproj_gemm_sol
```

#### o_proj 访存耗时

```text
o_proj访存耗时 =
(
  local_spec_dp_batch_size * hidden_size
  + 2 * local_spec_dp_batch_size * (q_head_num * 128 / attention_tp_world_size)
  + (q_head_num * 128 / attention_tp_world_size) * hidden_size
)
* 1000 * 1000
/ memory_bandwidth_TBps
/ 1024^4
/ memory_sol
```

### 4. Elementwise 与通信项

#### element_wise、topk、share 合并开销

```text
element_wise_topk_share开销 =
150 + local_spec_dp_batch_size * 2.3
```

这是纯经验公式：

- `150` 是固定底噪
- `2.3 * local_spec_dp_batch_size` 是随 token 数线性增加的经验项

#### all_reduce

```text
all_reduce耗时 =
2 * local_spec_dp_batch_size
* (attention_tp_world_size - 1) / attention_tp_world_size
* hidden_size
* 1000 * 1000
/ tp_collective_bandwidth_GBps
/ 1024^3
/ nccl_tp_sol
```

#### all2all_0

```text
all2all_0耗时 =
50
+ 4
* (local_batch_size * accept_len / attention_tp_world_size)
* min(decode_world_size, top_k)
* hidden_size
* 1000 * 1000
/ all2all_bandwidth_GBps
/ 1024^3
/ nccl_a2a_sol
```

这里的 `accept_len = 1.8` 来自表里的：

```text
accept_len = (1 - accept_rate^spec_step) / (1 - accept_rate)
```

#### all2all_1

```text
all2all_1耗时 = all2all_0耗时
```

#### all_gather

```text
all_gather耗时 =
local_spec_dp_batch_size
* (attention_tp_world_size - 1)
* hidden_size
* 1000 * 1000
/ tp_collective_bandwidth_GBps
/ 1024^3
/ nccl_tp_sol
```

### 5. MoE FFN 两段

#### ffn0 计算耗时

```text
ffn0计算耗时 =
2 * (spec_global_batch_size * top_k / ffn_world_size)
* (2 * ffn_dim)
* hidden_size
* 1000 * 1000
/ ffn_compute_tflops
/ 1024^4
/ group_ffn0_sol
```
比如：某rank上有8个experts，然后发了10个token到该卡上。其中：expert0 (3个token), expert1~7 (每个有1个token).那么ffn0的耗时就是：2 * (3 + 7) * 2 * ffn_dim * hidden_size / 1000 / 1000 / 1024 / 4 = 1000


#### ffn0 访存耗时

```text
ffn0访存耗时 =
0.5 * (num_experts / ffn_world_size)
* hidden_size
* (2 * ffn_dim)
* 1000 * 1000
/ memory_bandwidth_TBps
/ memory_sol
/ 1024^4
```

#### ffn1 计算耗时

```text
ffn1计算耗时 =
2 * (spec_global_batch_size * top_k / ffn_world_size)
* hidden_size
* ffn_dim
* 1000 * 1000
/ ffn_compute_tflops
/ 1024^4
/ group_ffn0_sol
```

#### ffn1 访存耗时

```text
ffn1访存耗时 =
0.5 * (num_experts / ffn_world_size)
* ffn_dim
* hidden_size
* 1000 * 1000
/ memory_bandwidth_TBps
/ memory_sol
/ 1024^4
```

## 最核心的总公式：per_layer耗时

最终 sparse 路径的单层耗时是：

```text
per_layer耗时 =
max(q_a_proj计算耗时, q_a_proj访存耗时)
+ max(q_b_proj计算耗时, q_b_proj访存耗时)
+ max(q_index_b_proj计算耗时, q_index_b_proj访存耗时)
+ max(kv_index_proj计算耗时, kv_index_proj访存耗时)
+ max(index_weight_proj计算耗时, index_weight_proj访存耗时)
+ max(index计算耗时, index访存耗时)
+ max(k_b_proj计算耗时, k_b_proj访存耗时)
+ max(kv_a_proj计算耗时, sparse_attn计算耗时)
+ max(sparse_attn计算耗时, sparse_attn访存耗时)
+ max(v_b_proj计算耗时, v_b_proj访存耗时)
+ max(o_proj计算耗时, o_proj访存耗时)
+ max(ffn0计算耗时, ffn0访存耗时)
+ max(ffn1计算耗时, ffn1访存耗时)
+ element_wise_topk_share开销
+ all_reduce耗时
+ all2all_0耗时
+ all2all_1耗时
+ all_gather耗时
```

这里要特别注意一件事：

```text
max(kv_a_proj计算耗时, sparse_attn计算耗时)
+ max(sparse_attn计算耗时, sparse_attn访存耗时)
```

也就是说，`sparse_attn计算耗时` 在聚合时被用了两次。原表不是把 `kv_a_proj / sparse_attn计算 / sparse_attn访存` 三者一次性并成一个 `max`，而是拆成了两个连续阶段。因此它会重复参与汇总。这是原表的建模方式，不是本文改写导致的。

## 这套建模在表达什么

- 它不是纯理论推导，而是“矩阵规模公式 + 带宽公式 + 经验常数”混合模型
- attention 的 sparse 收益主要体现在 `sparse_attn计算耗时` 和 `sparse_attn访存耗时` 都只看 `min(context_len, 2048)`
- 单层总耗时的核心思想是：每一段都取 `max(计算, 访存)`，再把通信类和共享类开销直接累加
- 因为当前表里的 `attention_dp_world_size = 1`、`spec_step = 2`，所以很多式子在本表里会自然退化成更简单的数值关系

## 最简结论

- 这块的最终目标量就是 `per_layer耗时`
- `per_layer耗时` 本质上是 decoding MLA sparse 路径的单层执行预算
- 公式里最值得关注的结构是 `min(context_len, 2048)`，这就是 sparse MLA 的主要收益来源
- 如果后续你要继续找瓶颈，优先看：
  - `sparse_attn计算耗时`
  - `sparse_attn访存耗时`
  - `all_reduce耗时`
  - `all2all_0耗时`
  - `all_gather耗时`
