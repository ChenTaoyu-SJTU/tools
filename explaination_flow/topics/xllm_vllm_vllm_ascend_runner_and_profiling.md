---
tags: [xLLM, vLLMModelRunner, profiling, Ascend_Profiling, launcher, DP, enable_attention_dp, NPUModelRunner, inputs_sharing, vanilla_vllm]
---

# xLLM、vLLM 与 vllm-ascend 的 runner 接线与 profiling 总览

## 结论速览

- `Ascend_Profiling_new.py` 不是直接调用 vLLM 官方 `serve` 路径，而是用 xLLM 自己的 built-in launcher 构造与 vLLM 兼容的分布式环境，再进入 xLLM 的 `vLLMModelRunner`。
- xLLM 的 `ParallelConfig` 同时支持两种 DP 语义：
  - `enable_attention_dp=True`：DP 被乘进 world，走“Attention DP + EP”语义
  - `enable_attention_dp=False`：pure DP，多副本 driver；inferencer 层看到的 `dp_size` 会被重置为 1
- xLLM 的 vanilla_vllm runner 会把 `data_parallel_size`、`data_parallel_rank`、`nnodes` 等信息重新组装成 vLLM 的 `EngineArgs`，并显式打开 `enable_expert_parallel=True`。
- 在 NPU 路径下，xLLM 会先做一批 vllm-ascend patch / 注册动作，再调用 vLLM 的 `init_distributed_environment()` 和 `ensure_model_parallel_initialized()`，最后落到 `vllm_ascend/worker/model_runner_v1.py::NPUModelRunner`。
- `inputs_sharing_groups` 当前应该绑定 TP group，不应该绑定 DP group 或 WORLD group。
- 在线 profiling 的 `/start_profile`、`/stop_profile` 是 vLLM OpenAI server 路径；`Ascend_Profiling_new.py` 更像离线 runner 级别的 profile / prewarm / forward 调试入口。

## `Ascend_Profiling_new.py` 在做什么

入口是：

- `examples/model_runner/vllm/Ascend_Profiling_new.py`

这份脚本的核心不是“自己实现一套分布式框架”，而是：

1. 解析一组尽量贴近 `vllm serve` 语义的参数
2. 推导 launch topology
3. 为每个本地进程写好与 vLLM/xLLM 兼容的环境变量
4. 最终调用原有 `Ascend_Profiling.py` 里的 `run_standalone(...)`

### 1. built-in launcher 如何推导拓扑

`resolve_launch_topology()` 会计算：

- `dp_size`
- `dp_size_local`
- `nnodes`
- `node_rank`
- `dp_group_world_size`
- `local_dp_group_world_size`
- `local_start_dp_rank`
- `world_size`

`build_rank_info()` 再把单个本地进程映射成：

- `global_rank`
- `dp_rank`
- `local_dp_rank`
- `rank_in_dp_group`
- `visible_devices`

这份映射和 vLLM 常见的“单个 DP 副本内部 world + 跨 DP 全局 world”语义是一致的。

### 2. 脚本如何把环境变量补齐

`worker_main()` 会写入：

- `MASTER_ADDR` / `MASTER_PORT`
- `RANK` / `WORLD_SIZE`
- `LOCAL_RANK` / `LOCAL_WORLD_SIZE`
- `ASCEND_RT_VISIBLE_DEVICES`
- `XLLM_DP_RANK`
- `XLLM_LOCAL_DP_RANK`
- `XLLM_DP_GROUP_RANK`
- `XLLM_DP_GROUP_WORLD_SIZE`
- `XLLM_DP_INIT_PORTS`

因此这份脚本虽然不是 `vllm serve`，但会主动把与 vLLM / xLLM / vllm-ascend 对接所需的关键信息都补齐。

## xLLM 如何表达 DP 语义

### 1. xLLM 自己的 `ParallelConfig`

定义在：

- `xllm/backend/distributed.py`

最关键的分叉是：

- `enable_attention_dp=True`
- `enable_attention_dp=False`

#### `enable_attention_dp=True`

这是 xLLM 想让底层 runner 真正感知 DP 的模式。

`xllm/backend/engine/xengine.py` 中会：

- 先按 `CP x SP x TP x PP` 算基础 world
- 再把 `dp_size` 乘进去

也就是说，DP 会进入底层 world 语义。

#### `enable_attention_dp=False`

这是 pure DP 模式。

`xllm/service/launcher/launcher.py` 会把：

- `global_dp_size = dp_size`
- `dp_size = 1`

这样 inferencer / model runner 看到的底层 world 不再包含 DP，而 DP 只体现在“上层有多少个独立 driver 副本”。

## xLLM 如何把配置接到 vLLM

### 1. `vLLMModelRunner` 如何组装 `VllmConfig`

主要逻辑在：

- `xllm/backend/v2/model_runner/backend/vanilla_vllm/llm_runner.py`

`_build_vllm_config()` 会把 xLLM 侧信息改写成 vLLM 的 `EngineArgs`，包括：

- `tensor_parallel_size`
- `pipeline_parallel_size`
- `data_parallel_size`
- `data_parallel_rank`
- `data_parallel_start_rank`
- `data_parallel_size_local`
- `nnodes`
- `node_rank`
- `data_parallel_address`
- `data_parallel_rpc_port`

同时它会显式写入：

- `enable_expert_parallel=True`

所以 vanilla_vllm backend 当前默认就是按“vLLM 侧允许 EP”来构造配置的。

### 2. 为什么它还会回写 `parallel_config`

`EngineArgs.create_engine_config()` 内部会做一轮拓扑合法性检查和参数推导，因此 xLLM 在拿到 `vllm_config` 后，如果 `dp_size > 1`，还会重新把这些字段回写到：

- `vllm_config.parallel_config.data_parallel_rank`
- `vllm_config.parallel_config.data_parallel_index`
- `vllm_config.parallel_config.data_parallel_rank_local`
- 多机场景下的 `data_parallel_size_local`

这样做的原因是：

- profiling 脚本中的每个 xLLM worker 已经绑定到一个具体 DP rank
- xLLM 需要保留这个“当前进程就是哪个 DP rank”的精确视图

## xLLM 如何接到 vllm-ascend

### 1. 初始化顺序

`vLLMModelRunner.__init__()` 的 NPU 路径会先做一批 Ascend 相关准备：

- `init_batch_invariance()`
- `adapt_patch()`
- `ops.register_dummy_fusion_op()`
- `_register_atb_extensions()`
- `register_ascend_customop(...)`
- `init_ascend_config(...)`
- `check_ascend_device_type()`

然后才调用：

- `init_distributed_environment(...)`
- `ensure_model_parallel_initialized(...)`
- `init_ascend_model_parallel(...)`

这意味着 xLLM 不是简单地“把 vllm-ascend 当黑盒 import 进来”，而是先完成了 Ascend 运行时所需的 patch 和注册。

### 2. 真正选中哪种 forward engine

选择逻辑在：

- `xllm/backend/v2/model_runner/backend/vanilla_vllm/forward_engine/factory.py`

当 `forward_engine_type == "npu.v0.15.0"` 时，会创建：

- `forward_engine/npu_forward_engine/npu_forward_engine_main.py::LLMAscendForwardEngine`

这条 engine 在 `_initialize_backend_engine()` 中真正实例化：

- `vllm_ascend/worker/model_runner_v1.py::NPUModelRunner`

所以最终执行模型前向的是 vllm-ascend 的 NPUModelRunner，而不是 xLLM 自己重新实现的一套 NPU runner。

## 为什么 `inputs_sharing_groups` 应该绑 TP group

逻辑在：

- `xllm/backend/v2/model_runner/backend/vanilla_vllm/llm_runner.py::_resolve_inputs_sharing_ranks()`

当前实现直接返回：

- `get_tp_group()`

这个选择是合理的，因为 `InputsShareCommunicator` 的语义是：

- 一组 rank 共享同一份 forward inputs

TP shard 正好符合这个前提：

- 同一 TP group 内本来就应该消费同一份 `input_ids`、`positions`、`slot_mappings`、`query_start_loc`

反过来说：

- DP group 内通常对应不同副本、不同 batch，不应强行共享一份输入
- WORLD group 的范围更大，语义也更错

因此当前代码把 inputs sharing 绑定到 TP group 是对的。

## profiling 有两条不同语义的路径

### 1. vLLM OpenAI server 的 profiling

在线服务路径在：

- `vllm/entrypoints/serve/profile/api_router.py`

接口是：

- `POST /start_profile`
- `POST /stop_profile`

调用链最终会到：

- `vllm/v1/engine/llm_engine.py::start_profile()`
- `vllm/v1/engine/llm_engine.py::stop_profile()`

这条路径的含义是：

- 服务已经跑起来了
- 通过 API 控制 profiler 的启停

### 2. xLLM `Ascend_Profiling_new.py` 的 profiling

这条路径更接近：

- 启动一个 xLLM runner
- 构造 NPU forward engine
- 走 vllm-ascend 的 `profile_run()`、dummy run、forward、ACLGraph 预热

因此它不是在线服务层的 profile 控制接口，而是 runner / forward engine 级别的调试和预热入口。

## 这条接线链路的边界

### xLLM 负责的部分

- built-in launcher 与进程拓扑推导
- 自己的 `ParallelConfig` 语义
- 把配置重写为 vLLM `EngineArgs`
- 选择 GPU / NPU forward engine
- 输入共享与 KV cache 适配

### vLLM 负责的部分

- `VllmConfig` / `ParallelConfig` 的正式构造与校验
- `init_distributed_environment()` / `initialize_model_parallel()`
- v1 worker / engine / profiling 框架

### vllm-ascend 负责的部分

- NPU worker / model runner
- Ascend graph、MoE 通信、layer sharding、FlashComm1 等 NPU 特化实现

## 易混淆点

- `Ascend_Profiling_new.py` 不是 `vllm serve`，但它会主动模拟一套与 vLLM 兼容的分布式环境。
- xLLM 的 pure DP 和 Attention DP 不是一回事；是否把 DP 乘进底层 world，取决于 `enable_attention_dp`。
- xLLM vanilla_vllm backend 当前默认会把 `enable_expert_parallel=True` 传给 vLLM，这意味着很多问题不能只从 xLLM 自己的配置表面理解。
- `inputs_sharing_groups` 绑定 TP group 是语义驱动的设计，不是随手选的默认值。
