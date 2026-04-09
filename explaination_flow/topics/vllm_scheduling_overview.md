---
tags: [scheduling, async, scheduler, AsyncScheduler, RecomputeScheduler, executor, async_scheduling, mp, uni]
---

# vllm + vllm-ascend Scheduling 总览

## 结论速览

- vllm 和 vllm-ascend 在常规配置下**默认开启 async scheduling**。配置字段默认值为 `None`，由 `VllmConfig.__post_init__` 自动推导，仅在命中特定不兼容条件时降级为 `False`。
- vllm-ascend 的 `AscendPlatform._fix_incompatible_config` **不覆盖** `async_scheduling`，完全继承上游逻辑。
- `async_scheduling=True` 时使用 `AsyncScheduler`，`False` 时使用 `Scheduler`。

---

## 子主题：Async Scheduling 默认行为

### 配置字段

`SchedulerConfig.async_scheduling`，定义于 `vllm/vllm/config/scheduler.py`：

```python
async_scheduling: bool = Field(default=None)
```

默认值是 `None`，不是 `True`。实际生效值由 `VllmConfig.__post_init__` 自动推导。

用户可通过 CLI `--async-scheduling` 显式指定 `true` 或 `false`。

### 自动推导逻辑

位于 `vllm/vllm/config/vllm.py` 的 `VllmConfig.__post_init__` 中：

1. **`async_scheduling` 为 `True`（用户显式开启）**：校验兼容性，不兼容则 `raise ValueError`。
2. **`async_scheduling` 为 `None`（默认）**：依次检查四个降级条件，全部不命中则设为 `True`。
3. **`async_scheduling` 为 `False`（用户显式关闭）**：直接跳过，不做任何推导。

四个降级条件（命中任一即设为 `False`）：

| 条件 | 说明 |
|------|------|
| 非 Eagle/MTP 投机解码 | `speculative_config.method not in get_args(EagleModelTypes)` |
| `disable_padded_drafter_batch=True` | 投机解码的 padded batch 被禁用 |
| executor backend 不支持 | 仅 `mp`、`uni`、`external_launcher` 支持 async scheduling |
| Mamba prefix cache | `cache_config.mamba_cache_mode != "none"` |

全部不命中时：

```python
self.scheduler_config.async_scheduling = True
```

### Scheduler 类选择

位于 `vllm/vllm/config/scheduler.py` 的 `SchedulerConfig.get_scheduler_cls()`：

- `async_scheduling=True` → `vllm.v1.core.sched.async_scheduler.AsyncScheduler`
- `async_scheduling=False` → `vllm.v1.core.sched.scheduler.Scheduler`

vllm-ascend 的 `RecomputeScheduler` 也遵循同样分支：

```
async_scheduling=True  → AsyncRecomputeScheduler
async_scheduling=False → RecomputeScheduler
```

逻辑位于 `vllm-ascend/vllm_ascend/core/recompute_scheduler.py`。

### vllm-ascend 行为

`AscendPlatform.check_and_update_config()` 调用 `_fix_incompatible_config()`，但其中**没有对 `async_scheduling` 做任何覆盖**（位于 `vllm-ascend/vllm_ascend/platform.py`）。

因此 vllm-ascend 完全继承上游 `VllmConfig.__post_init__` 的自动推导逻辑，在常规配置下同样默认开启。

### 生效点

async scheduling 开启后影响的关键路径：

| 组件 | 文件 | 影响 |
|------|------|------|
| Executor | `vllm/vllm/v1/executor/multiproc_executor.py` | `max_concurrent_batches` 返回 2；`WorkerProc` 启动 `async_output_busy_loop` 线程 |
| Executor | `vllm/vllm/v1/executor/uniproc_executor.py` | `max_concurrent_batches` 返回 2 |
| Model Runner (GPU) | `vllm/vllm/v1/worker/gpu/model_runner.py` | `use_async_scheduling` 控制返回 `AsyncGPUModelRunnerOutput` |
| Model Runner (NPU) | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | 同上，额外控制 PCP 和 spec decode 下的 token 处理路径 |

### 代码链路

```
SchedulerConfig.async_scheduling = None  (默认)
    │
    ▼
VllmConfig.__post_init__()  (vllm/vllm/config/vllm.py)
    ├── 用户显式 True  → 校验兼容性，不兼容 raise
    ├── None (默认)    → 四个降级条件检查 → 全不命中 → 设为 True
    └── 用户显式 False → 直接跳过
    │
    ▼
SchedulerConfig.get_scheduler_cls()  (vllm/vllm/config/scheduler.py)
    ├── True  → AsyncScheduler
    └── False → Scheduler
    │
    ▼
MultiprocExecutor / UniprocExecutor
    └── max_concurrent_batches = 2 if async_scheduling else 1
    │
    ▼
GPUModelRunner / NPUModelRunner
    └── use_async_scheduling → 控制 output 类型和 token 处理路径
```

---

## 易混淆点

- **`None` 不等于 `True`**：字段默认值是 `None`，需要走自动推导才变成 `True`。直接检查 `if async_scheduling:` 在推导前会得到 `False`（因为 `None` 是 falsy）。
- **CPU 平台强制关闭**：`vllm/vllm/platforms/cpu.py` 中 `CpuPlatform.check_and_update_config()` 直接将 `async_scheduling` 设为 `False`，不走推导。Ascend 平台**没有**类似的强制关闭。
- **executor backend 限制**：Ray executor 不在默认支持列表中（`mp`/`uni`/`external_launcher`），使用 Ray 时 async scheduling 会被自动关闭。但 Ray executor 的 `max_concurrent_batches` 有独立的 PP 并发逻辑。

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `vllm/vllm/config/scheduler.py` | `SchedulerConfig` 定义，`get_scheduler_cls()` |
| `vllm/vllm/config/vllm.py` | `VllmConfig.__post_init__` 自动推导逻辑 |
| `vllm/vllm/engine/arg_utils.py` | CLI 参数 `--async-scheduling` 定义 |
| `vllm/vllm/v1/core/sched/scheduler.py` | 同步 `Scheduler` |
| `vllm/vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler` |
| `vllm/vllm/v1/executor/multiproc_executor.py` | mp executor 中 async output 线程 |
| `vllm/vllm/v1/worker/gpu/model_runner.py` | GPU model runner 中 `use_async_scheduling` |
| `vllm/vllm/platforms/cpu.py` | CPU 平台强制关闭 async scheduling |
| `vllm-ascend/vllm_ascend/platform.py` | Ascend 平台 `_fix_incompatible_config`（无覆盖） |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | NPU model runner 中 async scheduling 相关路径 |
| `vllm-ascend/vllm_ascend/core/recompute_scheduler.py` | Ascend recompute scheduler 分支 |
