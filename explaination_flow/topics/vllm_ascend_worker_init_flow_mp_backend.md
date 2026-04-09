---
tags: [worker, init, mp-backend, NPUWorker, distributed, kv-cache, Phase-A/B, ACLGraph, init_device, load_model]
---

# vllm-ascend Worker 初始化流程（mp backend）

## 结论速览

- 使用 mp backend 时，worker 初始化分两个阶段：**Phase A** 在子进程 `WorkerProc.__init__` 中同步完成（`init_device` + `load_model`），**Phase B** 由 engine 进程通过 RPC 驱动（KV cache 相关）。
- Phase A 顺序：`NPUWorker()` → `init_device()`（含分布式初始化 + 构造 `NPUModelRunner`）→ `load_model()`（加载权重）→ 发送 READY。
- Phase B 顺序：`get_kv_cache_spec()` → `determine_available_memory()`（profile_run）→ `initialize_from_config()`（分配 KV cache）→ `compile_or_warm_up_model()`（warmup + graph capture）→ `initialize_cache()`（同步 block 数）。
- 分布式初始化链：`init_batch_invariance()` → `init_distributed_environment(..., "hccl")` → `ensure_model_parallel_initialized(tp, pp, pcp, dcp)` → `init_ascend_model_parallel()`（MC2, FineTP, FlashComm2, shard weight 等 Ascend 专有组）。
- KV cache 分配是 Ascend 特有的：K/V 分离分配 + 2MB 对齐（支持 PD disaggregation），MLA 按 `kv_lora_rank : qk_rope_head_dim` 拆分，sparse MLA 三路拆分。
- graph capture 在 Ascend 上使用 `ACLGraphWrapper`，通过 `_torch_cuda_wrapper()` 和 `_replace_gpu_model_runner_function_wrapper()` 把上游 CUDA graph 逻辑适配为 ACL Graph。

## 代码链路

### 总调用链

```
EngineCore.__init__()  (vllm/v1/engine/core.py)
│
├── MultiprocExecutor.__init__() → _init_executor()  (vllm/v1/executor/multiproc_executor.py)
│       └── spawn → WorkerProc.worker_main() → WorkerProc.__init__()
│           ├── WorkerWrapperBase.init_worker() → NPUWorker(**kwargs)
│           │     ├── adapt_patch()
│           │     ├── register ops (dummy_fusion, atb_extensions, ascend_customop)
│           │     ├── init_ascend_config(vllm_config) → singleton AscendConfig
│           │     └── super().__init__() → WorkerBase.__init__()
│           ├── NPUWorker.init_device()
│           │     ├── _init_device()
│           │     │   ├── torch.npu.set_device(npu:local_rank)
│           │     │   ├── MemorySnapshot()
│           │     │   ├── _init_worker_distributed_environment()
│           │     │   │   ├── init_batch_invariance()
│           │     │   │   ├── init_distributed_environment(..., "hccl")
│           │     │   │   ├── ensure_model_parallel_initialized(tp, pp, pcp, dcp)
│           │     │   │   ├── init_ascend_model_parallel(parallel_config)
│           │     │   │   └── ensure_ec_transfer_initialized()
│           │     │   ├── set_random_seed()
│           │     │   ├── init_device_properties_triton()
│           │     │   └── bind_cpus() (optional)
│           │     ├── init_workspace_manager(device, 1)
│           │     └── NPUModelRunner(vllm_config, device)
│           ├── _init_message_queues()
│           ├── NPUWorker.load_model()
│           │     └── NPUModelRunner.load_model()
│           │         ├── get_model(vllm_config) → 加载权重
│           │         ├── drafter.load_model() (optional)
│           │         └── ACLGraphWrapper() (if full cudagraph mode)
│           └── send READY
│
├── EngineCore._initialize_kv_caches()  (vllm/v1/engine/core.py)
│       ├── RPC: get_kv_cache_spec()
│       │     → NPUModelRunner.get_kv_cache_spec()
│       ├── RPC: determine_available_memory()
│       │     → NPUWorker.determine_available_memory()
│       │       └── NPUModelRunner.profile_run()
│       │           ├── eplb_warmup()
│       │           ├── _dummy_run(mc2_cap) (if MC2)
│       │           └── super().profile_run() → _dummy_run(max_num_tokens)
│       ├── get_kv_cache_configs() (engine 侧)
│       ├── RPC: initialize_from_config(kv_cache_configs)
│       │     → NPUWorker.initialize_from_config()
│       │       └── NPUModelRunner.initialize_kv_cache()
│       │           ├── initialize_attn_backend()
│       │           ├── may_reinitialize_input_batch()
│       │           └── initialize_kv_cache_tensors()
│       │               ├── _allocate_kv_cache_tensors() (K/V 分离 + 2MB 对齐)
│       │               ├── _reshape_kv_cache_tensors()
│       │               └── bind_kv_cache()
│       └── RPC: compile_or_warm_up_model()
│             ├── _dummy_run(warmup_sizes)
│             ├── capture_model() → ACL Graph
│             └── _warm_up_atb()
│
└── RPC: initialize_cache(num_gpu_blocks, num_cpu_blocks)
```

### 关键文件

| 组件 | 文件路径 |
|------|---------|
| Engine Core | `vllm/vllm/v1/engine/core.py` |
| Mp Executor | `vllm/vllm/v1/executor/multiproc_executor.py` |
| Abstract Executor | `vllm/vllm/v1/executor/abstract.py` |
| Worker Base | `vllm/vllm/v1/worker/worker_base.py` |
| NPU Worker | `vllm-ascend/vllm_ascend/worker/worker.py` |
| NPU Model Runner | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` |
| GPU Model Runner (base) | `vllm/vllm/v1/worker/gpu_model_runner.py` |
| Distributed State | `vllm/vllm/distributed/parallel_state.py` |
| Ascend Parallel State | `vllm-ascend/vllm_ascend/distributed/parallel_state.py` |
| Ascend Config | `vllm-ascend/vllm_ascend/ascend_config.py` |
| Batch Invariance | `vllm-ascend/vllm_ascend/batch_invariant.py` |
| KV Cache Interface | `vllm/vllm/v1/kv_cache_interface.py` |

## 各步骤详述

### NPUWorker.__init__

- `adapt_patch()`：注册 vllm-ascend 对上游 vLLM 的 monkey-patch。
- 注册 ops：`register_dummy_fusion_op()`、`_register_atb_extensions()`（非 A5）、`register_ascend_customop()`。
- `init_ascend_config(vllm_config)`：单例解析 `additional_config` 中的 Ascend 配置（编译配置、fusion、fine-grained TP、EPLB、PD 参数等）。
- `WorkerBase.__init__`：绑定 `vllm_config`、rank 信息、各子 config。
- 可选 sleep mode 准备、signal handler（NPU Graph + Static Kernel）。

### _init_device

- `torch.npu.set_device()` + 内存快照 + 空闲内存校验。
- `_init_worker_distributed_environment()`：
  1. `init_batch_invariance()`：确定性推理时注册 NPU 确定性算子。
  2. `init_distributed_environment(..., "hccl")`：初始化 HCCL 进程组，DP 场景重计算 rank/world_size。
  3. `ensure_model_parallel_initialized(tp, pp, pcp, dcp)`：创建 TP/PP/PCP/DCP/DP/EP/EPLB group。
  4. `init_ascend_model_parallel()`：Ascend 专有组（`_MC2`、`_P_TP`、`_OTP`/`_LMTP`/`_EMBED_TP`/`_MLP_TP`、`_FLASHCOMM2_*`、`_SHARD_WEIGHT`、`_DYNAMIC_EPLB`）。
  5. `ensure_ec_transfer_initialized()`。
- `set_random_seed()`、`init_device_properties_triton()`、可选 `bind_cpus()`。

### NPUModelRunner.__init__

- 在 `_torch_cuda_wrapper()` 下调用 `GPUModelRunner.__init__`（cuda→npu 重定向）。
- PCP 场景扩展 buffer；设置 `AscendSampler`、Ascend attention backend。
- 获取 DCP/PCP 组信息，创建 `PCPManager`。
- 配置 drafter、KV transfer 角色、cos/sin 预计算、MC2 容量/mask。
- 动态 EPLB 启动独立进程。
- 构造 `NPUInputBatch`。

### NPUModelRunner.load_model

- `get_model(vllm_config)` 加载权重到 NPU。
- 可选 `model_register()`（动态 EPLB）。
- 可选 drafter/LoRA 加载。
- 如果 full cudagraph：`ACLGraphWrapper` 包裹模型。

### get_kv_cache_spec

- 遍历 `AttentionLayerBase` 层：
  - `Attention`：KV sharing → skip；否则 `get_kv_cache_spec()`。
  - `MLAAttention`：sparse → 自定义 `MLAAttentionSpec`（三路 head_size）；否则默认。
  - `MambaBase`：获取 spec，对齐 attention page size。

### determine_available_memory / profile_run

- `memory_profiling` context 记录内存变化。
- `eplb_warmup()` → 可选 MC2 预 profile → PCP 调整后 `super().profile_run()`。
- `_dummy_run(max_num_tokens)` 触发所有运行时内存分配。
- 返回 `requested_memory - non_kv_cache_memory`。

### initialize_from_config → initialize_kv_cache

- `ensure_kv_transfer_initialized()`。
- `initialize_attn_backend()`：初始化 attention group 后端。
- `_allocate_kv_cache_tensors()`：K/V 分离 + 2MB 对齐；MLA 按 `kv_lora_rank:qk_rope_head_dim` 拆分；sparse MLA 三路拆分。
- `_reshape_kv_cache_tensors()`：一维 → 多维。
- `bind_kv_cache()`：绑定到 `static_forward_context`。
- 注册到 KV transfer group。

### compile_or_warm_up_model

- warmup `_dummy_run` 按从大到小执行。
- `capture_model()`：在 `_torch_cuda_wrapper()` + `_replace_gpu_model_runner_function_wrapper()` 下调用 `GPUModelRunner.capture_model()` → ACL Graph 录制。
- `_warm_up_atb()`：预热 ATB matmul（非 A5）。
- 重置随机种子。

## 易混淆点

- `init_device` 和 `load_model` 是在子进程的 `WorkerProc.__init__` 中**同步**执行的，**不是**通过 RPC。只有 Phase B 的步骤才是 RPC。
- `_torch_cuda_wrapper()` 不是真的用 CUDA，而是把 `torch.cuda.*` 临时重定向到 `torch.npu.*`，让上游代码可以在 NPU 上跑。
- `ACLGraphWrapper` 是 Ascend 版的 `CUDAGraphWrapper`，但底层使用 ACL Graph 而非 CUDA Graph。
- KV cache 分配中 K/V 分离 + 2MB 对齐是为了支持 PD disaggregation（prefill disaggregation 要求 K cache 的首地址对齐）。
- `init_ascend_model_parallel()` 创建的通信组是在 vLLM 标准组（TP/PP/DP/EP 等）之上的**额外**组，不是替代。
- `profile_run` 中如果 MC2 容量不够，会先用小的 capacity 做一次 dummy run 再做正式的 max tokens profile，以避免 MC2 通信 OOM。
