# vLLM 分布式 DP/TP/EP 拓扑、rank 与初始化总览

## 结论速览

- `ParallelConfig.world_size` 表示单个 DP 副本内部的并行规模，计算方式是 `PP x TP x PCP`；它默认**不包含** DP。
- `ParallelConfig.world_size_across_dp` 才是把 DP 乘进去后的全局 WORLD 大小，即 `world_size x data_parallel_size`。
- `data_parallel_size_local`、`data_parallel_start_rank`、`nnodes`、`node_rank` 共同决定当前节点到底承载几个 DP 副本，以及这些副本在全局 DP 拓扑中的位置。
- `run_server()` / `run_headless()` 负责启动 engine 进程；真正创建 worker 的主入口在 executor 层，不在 `parallel_state.py`。
- `init_distributed_environment()` 接收到的 `world_size` / `rank` 往往还是“单个 DP 副本视角”，进入函数后才会按 `data_parallel_rank` 扩成全局 WORLD。
- 在在线服务路径里，MoE 模型会保留 vLLM 内部 DP 语义；dense 模型的多个 DP engine 更像独立副本，`EngineCore` 会把 `data_parallel_size` 重置成 1，但保留 `data_parallel_index` 记录“这是第几个 DP engine”。
- 开启 `enable_expert_parallel` 后，EP 组不是只看 TP；在 `PP=1`、`PCP=1` 时，EP 组大小等于 `DP x TP`。

## 代码主链路

### 1. 参数入口与 `ParallelConfig` 字段来源

CLI 参数首先进入 `vllm/engine/arg_utils.py` 的 `AsyncEngineArgs` / `EngineArgs`，随后在 `create_engine_config()` 中生成 `ParallelConfig`。

和本主题最相关的字段是：

- `data_parallel_size`
- `data_parallel_rank`
- `data_parallel_size_local`
- `data_parallel_start_rank`
- `data_parallel_address`
- `data_parallel_rpc_port`
- `nnodes`
- `node_rank`

这些字段最终会落到 `vllm/config/parallel.py`。

`ParallelConfig` 中几个最容易混淆的量：

- `world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size`
- `world_size_across_dp = world_size * data_parallel_size`
- `node_rank_within_dp = node_rank % nnodes_within_dp`
- `local_world_size = world_size // nnodes_within_dp`

因此：

- `world_size` 是“单个 DP 副本内部”的 TP/PP/PCP worker 数。
- `local_world_size` 是“当前节点上，单个 DP 副本实际落到本机的 worker 数”。

### 2. `nnodes` / `data_parallel_size_local` / `data_parallel_start_rank` 的推导

`vllm/engine/arg_utils.py` 会按两类多机路径推导：

#### 路径 A：`nnodes > 1`

这表示单个 TP/PP world 本身跨机器。代码会：

- 先算 `world_size = DP x PP x TP`
- 再算 `local_world_size = world_size // nnodes`
- 用 `node_rank` 推导 `inferred_data_parallel_rank`
- 若未显式传 `data_parallel_size_local`，则按当前节点能容纳的 DP 副本数自动推导

这条路径更像“一个更大的 distributed world 被切到多台机器上”。

#### 路径 B：不传 `nnodes`，但传 DP 相关参数

这对应很多 `vllm serve` 多机 DP 教程，尤其是 `docs/source/tutorials/models/DeepSeek-V3.2.md` 里的形式：

- node0: `--data-parallel-size 2 --data-parallel-size-local 1`
- node1: `--headless --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-start-rank 1`

这时：

- `nnodes` 仍然保持默认值 1
- 但 `data_parallel_size > 1`
- 机器之间的协作不是靠 `nnodes` 路径，而是靠 DP 地址、RPC 端口和后续 WORLD PG 初始化拼起来

这也是为什么“多机 DP 但不传 `--nnodes`”依然能建立跨机 WORLD。

## 启动链路

### 1. `run_server()` / `run_headless()`

入口在 `vllm/entrypoints/cli/serve.py`。

- 非 `--headless` 节点走 `run_server()` 或 `run_multi_api_server()`
- `--headless` 节点走 `run_headless()`

`run_headless()` 做两件关键事情：

1. 如果 `node_rank_within_dp > 0`，说明当前节点只是同一个 DP 副本里的非 leader 节点，此时直接拉起 `MultiprocExecutor`
2. 否则说明当前节点承载的是本地 DP engine，需要通过 `CoreEngineProcManager` 启动本地 `EngineCore` 进程

### 2. `EngineCore` 进程如何注入 DP 信息

在 `vllm/v1/engine/core.py` 的 `EngineCoreProc.run_engine_core()` 中：

- `data_parallel_rank_local = local_dp_rank`
- `data_parallel_index = dp_rank`

随后分两种情况：

#### MoE 路径

若当前模型是 MoE 且处于 DP 场景：

- `data_parallel_rank = dp_rank`
- 使用 `DPEngineCoreProc`

这表示 vLLM 内部真的保留了 DP 组语义，后续 WORLD/DP/EP 组都会按全局拓扑构建。

#### Dense 路径

若模型不是 MoE：

- `data_parallel_size = 1`
- `data_parallel_size_local = 1`
- `data_parallel_rank = 0`

这说明多个 DP engine 只是多个相互独立的副本；但 `data_parallel_index` 仍然保留原始 DP 编号，方便上层负载均衡与状态区分。

### 3. Executor 如何选择，worker 又在哪里创建

executor 选择逻辑在 `vllm/v1/executor/abstract.py::Executor.get_class()`：

- `mp` -> `MultiprocExecutor`
- `ray` -> `RayDistributedExecutor`
- `uni` -> `UniProcExecutor`
- `external_launcher` -> `ExecutorWithExternalLauncher`

默认多卡本地服务最常见的是 `mp`。

`MultiprocExecutor` 在 `vllm/v1/executor/multiproc_executor.py` 中真正创建 worker：

- 先根据 `node_rank_within_dp` 计算 `global_start_rank`
- 再循环 `local_rank in range(local_world_size)`
- 为每个 local rank 启动一个 `WorkerProc`

因此“创建多个 worker”的位置是 executor，不是 `vllm/distributed/parallel_state.py`。

## rank / local_rank / dp_rank 的统一语义

建议把几个 rank 分成三层理解：

### 1. DP engine 级别

- `data_parallel_rank`: 第几个 DP 副本
- `data_parallel_rank_local`: 当前节点上的第几个本地 DP 副本
- `data_parallel_index`: engine 视角保留的 DP 编号，dense 路径尤其要靠它区分副本

### 2. 单个 DP 副本内部的 worker 级别

- `worker.rank`: 当前 worker 在单个 DP 副本内部的 rank
- `worker.local_rank`: 当前 worker 绑定的本地设备序号

在 `MultiprocExecutor` 创建 worker 时，传进去的 `rank` / `local_rank` 还是“单个 DP 副本视角”。

### 3. 真正的 torch WORLD 级别

进入 `vllm/distributed/parallel_state.py::init_distributed_environment()` 后，如果发现：

- `nnodes > 1`
- 或 `data_parallel_size > 1`

就会做两步改写：

- `rank = data_parallel_rank * world_size + rank`
- `world_size = world_size_across_dp`

因此：

- 传入函数前，`rank` 常常还是 `0..TPxPP-1`
- 进入 torch WORLD PG 时，它才会变成全局 rank

## `init_distributed_environment()` 之后会建哪些组

`vllm/distributed/parallel_state.py` 里先初始化 WORLD，再调用 `initialize_model_parallel()` 构建 TP / DCP / PCP / PP / DP / EP 组。

几个关键点：

- 组布局按 WORLD reshape 后再转置切分，本质上是“先排好全局 rank，再按维度切片”
- `DP` 组来自对 DP 维转置后的 reshape
- `EP` 组在 MoE 场景下创建，大小与 `data_parallel_size x prefill_context_parallel_size x tensor_parallel_size` 相关
- 若 `nnodes_within_dp > 1`，还会额外构建 `_INNER_DP_WORLD`，用于“单个 DP 副本横跨多机”时的内部协作

## 典型拓扑

### 1. 单机 `DP=2, TP=8, PP=1`

这是最典型的 A3 单机在线服务例子。

初始配置：

- `world_size = 8`
- `world_size_across_dp = 16`

运行期：

- 启动 2 个 DP engine 进程
- 每个 engine 再启动 8 个 TP worker
- 每个 engine 内部最初的 `worker.rank` 都是 `0..7`
- 进入 `init_distributed_environment()` 后：
  - DP0 变成 WORLD rank `0..7`
  - DP1 变成 WORLD rank `8..15`

如果是 Ascend，本地设备通常还会借助 `ASCEND_RT_VISIBLE_DEVICES` 在 engine 进程级别先切卡，因此两个 engine 里的 `npu:0..7` 可以映射到不同物理卡组。

### 2. 双机 DP，但不传 `nnodes`

这正是 `DeepSeek-V3.2.md` 多机教程的常见写法。

特点：

- `nnodes` 仍然是 1
- node1 通过 `--headless` + `--data-parallel-start-rank 1` 告诉 vLLM：本机承载从哪个 DP rank 开始的 engine
- DP 的 WORLD 初始化仍然会走跨机地址和端口，只是不依赖 `nnodes` 的推导分支

这条路径适合“按 DP 副本切机器”的部署方式。

### 3. `nnodes > 1`

这条路径适合“单个 DP 副本本身跨机器”的场景。

关键量：

- `nnodes_within_dp`: 一个 DP 副本横跨多少台机器
- `node_rank_within_dp`: 当前节点在所属 DP 副本中的相对节点号
- `local_world_size = world_size // nnodes_within_dp`

`MultiprocExecutor` 会用 `node_rank_within_dp` 计算当前节点这一段 worker 的全局起始 rank。

### 4. `DP + TP + EP`

在 `PP=1`、`PCP=1` 时可以直接记：

- `ep_size = dp_size x tp_size`

所以双机 32 卡常见几种形态：

- `dp=2, tp=16, ep` -> `ep_size=32`
- `dp=4, tp=8, ep` -> `ep_size=32`
- `dp=32, tp=1, ep` -> `ep_size=32`

是否真正进入这套 EP 逻辑，还取决于：

- 模型是不是 MoE
- `enable_expert_parallel` 是否开启

## `external_launcher` 需要单独区分

`external_launcher` 对应 `ExecutorWithExternalLauncher`，主要面向 torchrun 一类外部 launcher。

这条路径的特点是：

- 使用 `env://`
- `RANK` / `LOCAL_RANK` / `MASTER_ADDR` / `MASTER_PORT` 都由外部 launcher 提供
- `ParallelConfig.__post_init__()` 会先把 `world_size` 乘上 `data_parallel_size`

所以它与常规 `mp` 路径的差别是：

- 常规 `mp` 是先按单个 DP 副本建 world，再在 `init_distributed_environment()` 里扩到全局
- `external_launcher` 更接近“外部已经把全局 world 摆好了”

## 易混淆点

- `worker.rank` 不是最终 torch WORLD rank；DP 场景下它只是进入 `init_distributed_environment()` 前的局部 rank。
- `world_size` 默认不含 DP，别把它直接当成全局 WORLD 大小。
- 多机部署不一定需要 `--nnodes`；很多在线服务教程走的是“多机 DP + headless”的另一条路径。
- dense 模型的多 DP 在线服务，更接近多个独立副本；MoE 才会真正把 DP 纳入 vLLM 内部并行语义。
- “创建 worker”与“初始化并行组”是两层逻辑：前者发生在 executor，后者发生在 `parallel_state.py`。
