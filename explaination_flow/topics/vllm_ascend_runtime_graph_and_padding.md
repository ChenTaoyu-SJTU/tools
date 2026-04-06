# vLLM-Ascend runtime、ACLGraph 与 DP padding 总览

## 结论速览

- `profile_run`、`dummy run`、`graph capture` 是三件不同的事：profile run 主要用于初始化阶段的显存/shape 预热，graph capture 则是后续固定 shape 执行优化。
- GPU 路径里，`vllm/v1/worker/gpu_worker.py` 会在显存 profiling 阶段调用 `model_runner.profile_run()`；GPU dummy batch 的总 token 数基线来自 `scheduler_config.max_num_batched_tokens`。
- Ascend 的 `NPUModelRunner.profile_run()` 不是完全独立实现，而是在 GPU V1 runner 基础上增加了 EPLB warmup、MC2 容量预热以及 PCP 下的 token 调整。
- Ascend 侧 ACLGraph 要捕获哪些 shape，最终以 `compilation_config.cudagraph_capture_sizes` 为准；runner 内部会把它整理成 `self.cudagraph_batch_sizes`。
- DP 场景下的 shape 决策顺序是：本 rank 先本地 dispatch -> 再跨 DP 同步/对齐 -> 再按对齐后的 token 数重新 dispatch。
- `num_tokens_across_dp` 是“每个 DP rank 各自 token 数组成的向量”，不是一个总和。
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 首先推动的是 token 维度 padding；`num_reqs_padded` 变大通常是 padding 后的 token shape 被投影成 request 数，不代表真实请求数真的变多。
- ROCm 没有单独的 `rocm_graph.py`；它复用 `vllm/compilation/cuda_graph.py`，差别主要体现在平台配置与可用模式上。

## profile run、dummy run、graph capture 的关系

### 1. GPU 通用逻辑

GPU worker 在 `vllm/v1/worker/gpu_worker.py` 中，初始化 KV cache 前会调用：

- `self.model_runner.profile_run()`

这里的目的是做显存 profiling 和必要的编译/预热，不等于“马上开始 cudagraph capture”。

在 `vllm/v1/worker/gpu_model_runner.py` 中：

- `profile_run()` 的 dummy batch 总 token 数基线是 `scheduler_config.max_num_batched_tokens`
- 不是 `max_model_len`
- 也不是 `max_num_seqs * max_model_len`

### 2. Ascend V1 runner 的特化

`vllm_ascend/worker/model_runner_v1.py` 里，`NPUModelRunner.profile_run()` 会先做几件 Ascend 特有的事：

- `eplb_warmup()`
- 如果当前 MoE 通信方法需要 MC2 / FUSED_MC2，且 `max_num_tokens` 超过容量，会先用 `mc2_tokens_capacity` 做一次 `_dummy_run(..., is_profile=True)`
- PCP 场景下会临时把 `max_num_tokens` 调整为分片后的值，再调用父类 `profile_run()`

所以 Ascend 的 profile run 仍然建立在 GPU V1 runner 逻辑之上，但运行前多了一层 Ascend 侧 warmup 和 shape 修正。

### 3. graph capture 什么时候发生

是否启用 ACLGraph，先看 `NPUModelRunner._use_aclgraph()`，它要求至少满足：

- `compilation_config.cudagraph_mode != NONE`
- `compilation_config.mode == VLLM_COMPILE`

另外，`load_model()` 里只有当：

- `compilation_config.cudagraph_mode.has_full_cudagraphs()`

为真时，模型才会被 `ACLGraphWrapper` 包起来。

因此：

- profile run 可以发生在不开图的场景
- 开图也不代表所有 shape 都会被立即 capture
- 真正的 ACLGraph capture 仍然由后续固定 shape 执行路径触发

## `cudagraph_capture_sizes` 与 ACLGraph sizes 的来源

### 1. runner 内部如何保存 sizes

GPU 和 Ascend V1 runner 都会把：

- `compilation_config.cudagraph_capture_sizes`

整理成：

- `self.cudagraph_batch_sizes = sorted(...)`

Ascend 的对应逻辑在 `vllm_ascend/worker/model_runner_v1.py`，GPU 的对应逻辑在 `vllm/v1/worker/gpu_model_runner.py`。

因此“ACLGraph 要捕获哪些 batch size”本质上仍然来自 compilation config，而不是 runner 自己重新发明了一套配置。

### 2. Ascend 如何把 sizes 注册给 ACLGraph

在 `NPUModelRunner._check_and_update_cudagraph_mode()` 末尾：

- `set_graph_params(self.cudagraph_batch_sizes)`
- speculative decode 场景下还会调用 `set_draft_graph_params(self.cudagraph_batch_sizes)`

`vllm_ascend/compilation/acl_graph.py` 里的 `set_graph_params()` / `set_draft_graph_params()` 会把这些 size 变成内部缓存字典的 key。

所以：

- `self.cudagraph_batch_sizes` 是 runner 视角的 size 列表
- `_graph_params` / `_draft_graph_params` 是 ACLGraph 运行时缓存视角的 size 索引

### 3. 为什么实际 sizes 可能比配置少

Ascend 侧会在 `vllm_ascend/utils.py::update_aclgraph_sizes()` 中，根据：

- 层数
- DP/TP/EP 等并行因子
- 通信方式
- `HCCL_OP_EXPANSION_MODE`

估算“当前硬件/模型最多能 capture 多少个 shape”。

如果原始 `cudagraph_capture_sizes` 太多，代码会均匀采样一个代表子集，并通过 `update_cudagraph_capture_sizes()` 写回配置。

因此最终真正参与 ACLGraph 的 shape 集合，以运行期更新后的 `compilation_config.cudagraph_capture_sizes` 为准。

## DP 场景下 token、padding 与 re-dispatch 的顺序

### 1. 先按本 rank 的 `num_tokens` dispatch

`vllm_ascend/worker/model_runner_v1.py` 中，runner 会先根据本 rank 当前看到的 `num_tokens` 走一轮 cudagraph dispatch，得到初始的：

- `cudagraph_mode`
- `batch_descriptor`

### 2. 再跨 DP 同步 shape

若 `data_parallel_size > 1`，runner 会调用 `_sync_batch_across_dp(...)`，拿到：

- `num_tokens_across_dp`
- `synced_cudagraph_mode`

其中 `num_tokens_across_dp` 是长度为 `dp_size` 的向量，每个位置对应一个 DP rank 的 token 数。

它不是总 token 数；如果需要总量，需要自己再对这个向量求和。

### 3. 发生 DP padding 后，再次 dispatch

如果同步结果要求对齐，各 rank 会把自己本轮执行 shape pad 到统一 token 数。随后代码会：

- 取出本 rank 对应的 `num_tokens_padded`
- 再用这个对齐后的 token 数重新走一轮 dispatch

所以 DP 场景里的 runtime shape 不是“一次决定到底”，而是“本地初判 -> DP 协调 -> 重新判定”。

## FlashComm1 对 `num_reqs_padded` 的影响

`VLLM_ASCEND_ENABLE_FLASHCOMM1` 的读取入口在 `vllm_ascend/envs.py`。

从执行顺序看，FlashComm1 直接影响的是：

- token 维度对齐

常见链路是：

1. 先按 TP 粒度做 token padding
2. 再叠加 cudagraph shape 对齐
3. DP 场景还可能继续 pad 到所有 rank 的最大 token 数

若此时刚好是：

- uniform decode
- `query_len == 1`

那么 padding 后的 token 数又会投影成 `batch_desc.num_reqs`，这时你会看到：

- 真实 `num_reqs` 很小
- 但 `num_reqs_padded` 很大

这更接近“执行形状需要这么大的 padded slots”，不代表真实请求条数真的变成了这个值。

## GPU CUDAGraph、Ascend ACLGraph、ROCm 的边界

### 1. GPU

GPU 走 `vllm/compilation/cuda_graph.py` 这套通用 CUDAGraph 逻辑。

### 2. Ascend

Ascend V1 runner 走 `vllm_ascend/compilation/acl_graph.py` 的 `ACLGraphWrapper`，并结合 Ascend 自己的：

- shape 裁剪
- DP shape 协调
- MoE / MC2 / FlashComm1

形成一条 NPU 特化路径。

### 3. ROCm

ROCm 在 `vllm/platforms/rocm.py` 里仍然返回 `vllm.compilation.cuda_graph.CUDAGraphWrapper`。

因此：

- ROCm 没有单独的 `rocm_graph.py`
- 仍复用 `torch.cuda.CUDAGraph()` 这套接口
- 是否退化到 `PIECEWISE` 或不用图，取决于平台检查和具体配置，而不是切换到另一套捕获框架

## 通用行为与 Ascend 特化行为的分界

### 通用 vLLM 行为

- GPU worker 初始化时会做 `profile_run`
- `cudagraph_capture_sizes` 来自 compilation config
- cudagraph batch sizes 会在 runner 内部排序保存
- 多种平台都会先决定 runtime mode，再进入具体 graph wrapper

### Ascend 特化行为

- `NPUModelRunner.profile_run()` 会做 EPLB / MC2 / PCP 相关额外处理
- `ACLGraphWrapper` 与 `_graph_params` / `_draft_graph_params` 是 Ascend 独有的 ACLGraph 运行时机制
- DP padding 与 FlashComm1、MoE 通信方式、ACLGraph shape 裁剪之间耦合更紧

## 易混淆点

- `profile_run` 不等于“已经 capture 完所有图”；很多时候它只是为了初始化、预热和显存估算。
- `num_tokens_across_dp` 不是总 token 数，而是每个 DP rank 的 token 分布。
- `num_reqs_padded` 可以远大于真实请求数，尤其在 uniform decode + `query_len=1` 场景。
- Ascend 的 graph size 列表可能被运行期裁剪；最终 shape 集合要以更新后的 compilation config 为准。
