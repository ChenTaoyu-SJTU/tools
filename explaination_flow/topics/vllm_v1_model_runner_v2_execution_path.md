---
tags: [v2-runner, execute_model, prepare_inputs, sample, async-scheduling, CUDA-graph, InputBatch, chunked-prefill, spec-decode, RequestState]
---

# vLLM V1 GPU ModelRunner V2 Execution Path

## 结论速览
- 在 `VLLM_USE_V2_MODEL_RUNNER=1`、`async_scheduling=True` 的纯文本生成路径里，一次 step 的主链是：
  `EngineCore.preprocess_add_request()` -> `Scheduler.schedule()` -> `Worker.execute_model()` -> `GPUModelRunner.execute_model()` -> `GPUModelRunner.sample_tokens()` -> `Scheduler.update_from_output()`
- v2 scheduler 不显式维护 “prefill phase / decode phase” 状态机。真正区分 prefill 和 decode 的，是请求当前的 `num_computed_tokens` 相对 runner 侧 `prefill_len` 的关系。
- v2 的关键特征是：`SchedulerOutput.scheduled_new_reqs` 会额外携带 `NewRequestData.prefill_token_ids`。对新请求，它通常等于完整 prompt；对被抢占后恢复的请求，v2 也会把它重新放进 `scheduled_new_reqs`，并把当前 `req._all_token_ids` 当作新的 `prefill_token_ids` 发送给 runner。
- runner 侧真正决定输入 token 的地方不是 scheduler，而是 `RequestState + InputBatch`：
  - `RequestState.prefill_token_ids/prefill_len/num_computed_tokens/last_sampled_tokens/draft_tokens`
  - `prepare_prefill_inputs()` 负责取 prefill token
  - `combine_sampled_and_draft_tokens()` 负责在 decode/spec 场景把 `last_sampled_tokens` 和 draft token 填回 `input_ids`
- chunked prefill 中间轮不会返回 `new_token_ids`。只有当本轮把 prompt 走到尾部，或者本轮本身就是 decode/spec 轮次，`sample_tokens()` 返回的结果才会在 `Scheduler.update_from_output()` 中真正追加到 `Request._all_token_ids`。
- **Async scheduling 的 overlap 机制**：同一 step 内 `prepare_inputs` 和 `execute_model`（forward）在 GPU 上是串行的（共享 default CUDA stream）。真正的 overlap 发生在跨 step：(1) D2H output copy 在单独的 `output_copy_stream` 上与下一步 CPU schedule 并行；(2) batch queue（`max_concurrent_batches=2`）实现两步流水线；(3) `synchronize_input_prep` 保护跨步 CPU buffer 安全复用。
- **CUDA Graph 只捕获 `model(...)` 前向，不包含 `prepare_inputs`**。`prepare_inputs` 有动态控制流和可变 tensor 形状，无法被固化到 CUDA graph 中。`InputBuffers` 充当两者之间的桥梁：固定 GPU 地址的预分配 buffer，`prepare_inputs` 每步写入新数据，CUDA graph replay 从同一地址读取。

## 代码链路

### 1. 请求进入 EngineCore
- `vllm/vllm/v1/engine/core.py`
  - `EngineCore.preprocess_add_request()` 会把 `EngineCoreRequest` 转成内部 `Request`。
  - 对 structured output，请求在这里触发 grammar 初始化。
- `vllm/vllm/v1/request.py`
  - `Request.from_engine_core_request()` 生成内部请求对象。
  - 请求的核心状态是：
    - `prompt_token_ids`
    - `_all_token_ids`
    - `_output_token_ids`
    - `num_computed_tokens`
    - `spec_token_ids`
    - `num_output_placeholders`

### 2. Engine 主循环驱动一次 step
- `vllm/vllm/v1/engine/core.py`
  - `EngineCore.step()` 的顺序很固定：
    1. `scheduler_output = self.scheduler.schedule()`
    2. `future = self.model_executor.execute_model(scheduler_output, non_block=True)`
    3. `grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)`
    4. 如果 `execute_model()` 返回 `None`，立刻调用 `sample_tokens(grammar_output)`
    5. 最后调用 `self.scheduler.update_from_output(scheduler_output, model_output)`

### 3. Scheduler 只追“还差多少 token 没算”
- `vllm/vllm/v1/core/sched/scheduler.py`
  - `schedule()` 开头就写明：scheduler 不区分 prefill/decode，只看 `num_computed_tokens` 是否追上 `num_tokens_with_spec`。
  - 对 RUNNING 请求，本轮预算是：
    `num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens`
  - 对 WAITING 请求，本轮预算通常是：
    `request.num_tokens - num_computed_tokens`
  - 如果开启 chunked prefill，本轮只会取 token budget 允许的前一段 prompt。
  - v2 下构造 `SchedulerOutput` 时：
    - `scheduled_new_reqs` 中的 `prefill_token_ids` 直接来自 `req._all_token_ids`
    - `scheduled_cached_reqs` 只传增量 block/table 等缓存状态
  - `_update_after_schedule()` 会先把 scheduler 侧 `request.num_computed_tokens` 前推，然后计算：
    `request.is_prefill_chunk = request.num_computed_tokens < (request.num_tokens + request.num_output_placeholders)`

### 4. Async scheduler 的额外动作
- `vllm/vllm/v1/core/sched/async_scheduler.py`
  - `AsyncScheduler._update_after_schedule()` 在普通 `_update_after_schedule()` 之后继续做两件事：
    - 对非 `is_prefill_chunk` 请求，给本轮预留 `1 + cur_num_spec_tokens` 个 `num_output_placeholders`
    - 把 `request.spec_token_ids` 暂时改成占位列表，等待真实 draft token 在后续路径中被消费或校验
  - `_update_request_with_output()` 在 token 回来后再把 `num_output_placeholders` 扣回去。

### 5. Worker 选择 v2 runner 并下发 `SchedulerOutput`
- `vllm/vllm/v1/worker/gpu_worker.py`
  - `init_device()` 在 `VLLM_USE_V2_MODEL_RUNNER=1` 时实例化 `vllm/vllm/v1/worker/gpu/model_runner.py::GPUModelRunner`
  - `execute_model()` 基本就是把 `SchedulerOutput` 原样传给 model runner

### 6. GPUModelRunnerV2 先同步 request state，再准备输入
- `vllm/vllm/v1/worker/gpu/model_runner.py`
  - `execute_model()` 开头先做：
    - `finish_requests()`
    - `free_states()`
    - `add_requests()`
    - `update_requests()`
    - `self.block_tables.apply_staged_writes()`
  - `add_requests()` 把 `scheduled_new_reqs` 写进 runner 侧 `RequestState`
  - `update_requests()` 给已有请求追加新 block ids
- `vllm/vllm/v1/worker/gpu/states.py`
  - `RequestState` 保存 v2 runner 真正依赖的长期状态：
    - `prefill_token_ids`
    - `prefill_len`
    - `num_computed_prefill_tokens`
    - `num_computed_tokens`
    - `last_sampled_tokens`
    - `draft_tokens`
    - `next_prefill_tokens`
- `vllm/vllm/v1/worker/gpu/model_runner.py`
  - `prepare_inputs()` 会把本轮 batch 拆成 `InputBatch`
  - 这里的顺序是 “decode first, then prefill”，即按每个请求本轮 `num_scheduled_tokens` 排序，小 query 往前，大 query 往后
  - `block_tables` 在这里有两个作用：
    - `gather_block_tables()` 取出本轮请求对应的 block table
    - `compute_slot_mappings()` 把当前 token 的逻辑位置映射到 KV cache 物理 slot

### 7. InputBatch 是 prefill/decode 真正分流的地方
- `vllm/vllm/v1/worker/gpu/input_batch.py`
  - `prepare_prefill_inputs()`：
    - 仅当 `num_computed_tokens < prefill_len` 时，从 `prefill_token_ids` 拷 token 到 `input_ids`
    - 如果已经不在 prefill，直接跳过
  - `prepare_pos_seq_lens()`：
    - `positions = num_computed_tokens + local_offset`
    - `seq_lens = num_computed_tokens + query_len`
  - `combine_sampled_and_draft_tokens()`：
    - prefill 请求：不写 `last_sampled_tokens`
    - decode 请求：把 `last_sampled_tokens` 写到本轮输入开头
    - spec 请求：继续把 draft token 接在后面
  - `get_num_sampled_and_rejected()`：
    - 如果 `seq_len < prefill_len`，认定当前还是 chunked prefill，本轮 `num_sampled=0`、`num_rejected=0`
  - `post_update()`：
    - 把 `last_sampled_tokens` 更新为本轮最后一个真正采样出的 token
    - 把 `num_computed_tokens += query_len - num_rejected`

### 8. 前向、采样、异步拷回
- `vllm/vllm/v1/worker/gpu/model_runner.py`
  - `execute_model()` 只负责前向，把 `(hidden_states, input_batch, kv_connector_output)` 暂存在 `self.execute_model_state`
  - `sample_tokens()` 才真正：
    - `sample()`
    - `compute_prompt_logprobs()`
    - 构造 `ModelRunnerOutput`
    - 构造 `AsyncOutput`
    - `postprocess()`
    - 若开 spec decode，调用 `propose_draft()`
- `vllm/vllm/v1/worker/gpu/async_utils.py`
  - `AsyncOutput` 负责把 GPU 上的 sampled tokens、logprobs、prompt logprobs 异步拷回 CPU
  - `get_output()` 最终把它们整理成 `ModelRunnerOutput`

### 9. Scheduler 根据 `ModelRunnerOutput` 回写请求
- `vllm/vllm/v1/core/sched/scheduler.py`
  - `update_from_output()` 遍历本轮 `num_scheduled_tokens`
  - 对 spec decode：
    - 根据 `scheduled_spec_decode_tokens` 和 `generated_token_ids` 计算 accepted/rejected
    - 如果有 rejection，会把 `request.num_computed_tokens` 和 `request.num_output_placeholders` 回退
  - `_update_request_with_output()` 真正把新 token append 到 `Request._all_token_ids`
  - 如果本轮没有 `new_token_ids`，则不会产生 `EngineCoreOutput`
  - 因此 “中途 chunked prefill 不向前端流式吐 token” 是 scheduler 显式保证的行为

## 场景说明

### 场景 1：新请求首轮 prefill，且首轮就走完整个 prompt
假设：
- 请求 `r1`
- prompt token ids = `[11, 12, 13, 14]`
- `max_num_batched_tokens >= 4`
- 无 spec

过程：
1. `Request` 初始状态：
   - `_all_token_ids = [11, 12, 13, 14]`
   - `num_computed_tokens = 0`
2. `Scheduler.schedule()` 产出：
   - `scheduled_new_reqs = [r1]`
   - `NewRequestData.prefill_token_ids = [11, 12, 13, 14]`
   - `num_scheduled_tokens = {"r1": 4}`
3. runner `add_requests()` 后，`RequestState` 中：
   - `prefill_len = 4`
   - `num_computed_tokens = 0`
4. `prepare_prefill_inputs()` 生成：
   - `input_ids = [11, 12, 13, 14]`
   - `query_start_loc = [0, 4]`
   - `positions = [0, 1, 2, 3]`
   - `seq_lens = [4]`
5. 因为这轮 `seq_len == prefill_len`，它不是 chunked prefill，中间最后一个 hidden state 会被拿去 sample，第一个输出 token 会在本轮产生。

结论：
- “完整 prompt prefill” 与 “首个 decode token” 可以发生在同一个 step。

### 场景 2：chunked prefill 到首个 decode 的切换
假设：
- 请求 `r2`
- prompt token ids = `[21, 22, 23, 24, 25]`
- token budget 只能先给 3 个 token
- 无 spec

第 1 轮：
- scheduler 给 `r2` 分配 `num_scheduled_tokens = 3`
- runner 侧：
  - `prefill_len = 5`
  - `input_ids = [21, 22, 23]`
  - `positions = [0, 1, 2]`
  - `seq_lens = [3]`
- `get_num_sampled_and_rejected()` 看到 `seq_len < prefill_len`，把 `num_sampled` 改成 0
- 所以：
  - 本轮没有 `new_token_ids`
  - 但 `post_update()` 会把 runner 侧 `num_computed_tokens` 推到 3

第 2 轮：
- scheduler 再给 `r2` 分配剩余 `2` 个 prompt token
- runner 侧：
  - `input_ids = [24, 25]`
  - `positions = [3, 4]`
  - `seq_lens = [5]`
- 这时 `seq_len == prefill_len`，说明 prompt 已走完
- 本轮会产出第一个输出 token，例如 `90`

结论：
- chunked prefill 的中间轮只推进 `num_computed_tokens`
- 真正第一次向前端返回 token，是“最后一段 prompt 结束”的那一轮

### 场景 3：decode + async + spec
假设某一轮 scheduler 已经给请求 `r3` 产出了这样的 batch 语义：
- 本轮对 `r3` 总共调度 3 个 token
- 其中 1 个是 decode 主 token，2 个是 `scheduled_spec_decode_tokens = [91, 92]`
- runner 侧已有：
  - `last_sampled_tokens = [90]`
  - `prefill_len = 4`
  - `num_computed_tokens >= 4`

那么在 runner 里：
1. `prepare_prefill_inputs()` 不再写 prompt，因为已经 `num_computed_tokens >= prefill_len`
2. `combine_sampled_and_draft_tokens()` 会把：
   - `last_sampled_tokens` 写入本轮输入开头
   - draft token `[91, 92]` 接在后面
   - 所以本轮 `input_ids` 逻辑上等价于 `[90, 91, 92]`
3. `sample()` 后如果 rejection sampling 的结果是：
   - 接受第 1 个 draft
   - 拒绝第 2 个 draft
   - 并重新采样出一个新 token `95`
   - 那么 `generated_token_ids` 会是 `[91, 95]`
4. `Scheduler.update_from_output()` 会据此计算：
   - `num_draft_tokens = 2`
   - `num_accepted = len(generated_token_ids) - 1 = 1`
   - `num_rejected = 1`
   - 然后把 `request.num_computed_tokens` 回退 1
   - async scheduler 还会同步把 `request.num_output_placeholders` 回退 1

结论：
- spec decode 的本质是：runner 先把 “上一步真实 token + 本步 draft token” 一起送进模型验证，再由 scheduler 用 accepted/rejected 结果修正 request 状态。

## Async Scheduling 的 Overlap 机制

### 核心结论

同一个 step 内，`prepare_inputs` 和 `execute_model`（forward）在 GPU 上是**串行**的。它们共享同一条 CUDA default stream，GPU kernel 按提交顺序执行。

Async scheduling 真正 overlap 的是**跨 step 的 CPU/GPU 流水线**：step N 的 GPU forward/sample 与 step N+1 的 CPU schedule/grammar_bitmask 并行，以及 step N 的 D2H output copy 与后续 CPU 工作并行。

### 为什么 prepare_inputs 和 forward 不能 GPU 并行

- `prepare_inputs` 中的 GPU 操作（`async_copy_to_gpu` 的 H2D 拷贝、`prepare_prefill_inputs`/`prepare_pos_seq_lens` 等 Triton kernel、`build_attn_metadata`）都在 default CUDA stream 上
- `model.forward()` 也在 default CUDA stream 上
- 同一个 stream 上的 kernel 天然按提交顺序串行执行
- `async_copy_to_gpu`（`vllm/vllm/v1/worker/gpu/buffer_utils.py`）使用 `non_blocking=True` 只意味着 CPU 不等待 H2D 完成就继续执行后续 CPU 代码，但在 GPU stream 层面这些 H2D copy 仍排在 default stream 上
- `forward` 依赖 `prepare_inputs` 的输出（`input_ids`、`positions`、`attn_metadata` 等），有数据依赖

### 跨 step 的真正 overlap

#### (a) D2H output copy 与 CPU 调度并行

`AsyncOutput`（V2 runner：`vllm/vllm/v1/worker/gpu/async_utils.py`）和 `AsyncGPUModelRunnerOutput`（主 runner：`vllm/vllm/v1/worker/gpu_model_runner.py`）在单独的 `output_copy_stream` / `async_output_copy_stream` 上做 D2H 拷贝（sampled_token_ids、logprobs 等），不阻塞 default stream。

工作流程：
1. `copy_stream.wait_stream(default_stream)` —— 等 default stream 上的 sample 完成
2. 在 `copy_stream` 上发起 D2H non-blocking 拷贝
3. `copy_event.record(copy_stream)` —— 记录完成事件
4. Engine 侧通过 `get_output()` -> `copy_event.synchronize()` 获取最终结果

这意味着 GPU worker 返回 `AsyncOutput` 后，CPU 可以立即开始下一步的 `schedule()`，D2H 拷贝在后台完成。

#### (b) Batch Queue 实现两步流水线

当 `async_scheduling=True` 时，`max_concurrent_batches=2`，启用 `step_with_batch_queue()`：
- `vllm/vllm/v1/executor/uniproc_executor.py`：`max_concurrent_batches` 返回 2
- `vllm/vllm/v1/executor/multiproc_executor.py`：同理
- `vllm/vllm/v1/engine/core.py` 的 `step_with_batch_queue()` 利用 batch queue 实现：
  1. schedule + `execute_model(non_block=True)` 提交 step N+1
  2. 如果 batch queue 未满且前一个 batch 还没完成，直接 return，不阻塞
  3. 否则 pop 前一个 batch 的 future，阻塞等待结果

#### (c) `synchronize_input_prep` 保护跨步 CPU buffer

主 runner 的 `synchronize_input_prep()`（`vllm/vllm/v1/worker/gpu_model_runner.py`）不是为了让 `prepare_inputs` 和 `forward` 并行，而是保证跨步 CPU buffer 安全：

当 step N 的 `prepare_inputs` 用 `non_blocking=True` 把 CPU tensor 异步拷到 GPU 后，CPU 端的 tensor 可能还没被 GPU 消费完。如果 step N+1 立即覆写这些 CPU buffer 会导致数据竞争。`prepare_inputs_event` 确保上一步的异步拷贝在 GPU 侧完成后，才开始下一步的 `prepare_inputs`。

#### (d) Multiproc Worker 的 async output 线程

当 `async_scheduling=True` 且使用 multiproc executor 时，每个 `WorkerProc` 会启动一个 `async_output_busy_loop` 后台线程（`vllm/vllm/v1/executor/multiproc_executor.py`）。GPU 线程将 `AsyncModelRunnerOutput` 放入队列后立即返回，后台线程负责调用 `get_output()` 等待 D2H 完成并序列化发送到 response MQ。

### CUDA Stream 使用一览（V2 runner + 主 runner）

| 位置 | Stream | 用途 |
|------|--------|------|
| `prepare_inputs` 中的 H2D 拷贝 | default stream | 准备输入 tensor |
| `model.forward()` | default stream | 模型前向 |
| `sample()` / `compute_logits()` | default stream | 采样 |
| `AsyncOutput` / `AsyncGPUModelRunnerOutput` | `output_copy_stream` / `async_output_copy_stream` | D2H 拷贝 sampled tokens 和 logprobs |
| `StructuredOutputsWorker` | 专用 `copy_stream` | H2D 拷贝 grammar bitmask |
| `DraftTokensHandler` | 专用 `copy_stream` | D2H 拷贝 draft tokens |

### CUDA Graph 与 prepare_inputs 的关系

开启 CUDA graph 后，**只有 `model(...)` 的前向推理被捕获到图中，`prepare_inputs` 不在图中**。

#### 捕获阶段

`CudaGraphManager.capture_graph()`（`vllm/vllm/v1/worker/gpu/cudagraph_utils.py`）中，`torch.cuda.graph(graph, self.pool)` 的 context 只包裹了 `model(input_ids=..., positions=..., inputs_embeds=...)` 和一行 `self.hidden_states[:num_tokens] = hidden_states`。`prepare_inputs_to_capture()` 在捕获上下文之前调用，不在图中。

#### 运行阶段

`execute_model()` 中：
1. 先执行 `prepare_inputs()` —— 常规 Python/CUDA 代码，不在图中
2. 然后 `self.cudagraph_manager.run(num_tokens)` —— 内部调用 `graph.replay()`

#### 为什么 prepare_inputs 不能被融入 CUDA Graph

CUDA graph 的本质限制：
- **固定的 GPU 操作序列和控制流**：`prepare_inputs` 中有数据依赖的动态逻辑（每步请求数量不同、每个请求的 `num_scheduled_tokens` 不同、numpy 排序和条件分支等），无法被固化到图中
- **固定的 tensor 形状**：`prepare_inputs` 的中间 tensor（`idx_mapping`、`query_start_loc`、`slot_mappings`、`attn_metadata`）的有效长度每步都在变
- **固定的内存地址**：这正是 `InputBuffers` 的设计意义——预分配最大尺寸的 GPU buffer（`input_ids`、`positions`、`seq_lens`、`query_start_loc` 等），地址在整个生命周期内固定

#### InputBuffers 是动态 prepare_inputs 和静态 CUDA Graph 之间的桥梁

`InputBuffers` 充当参数化接口：`prepare_inputs` 每步往固定地址 buffer 里写入新内容，CUDA graph replay 时 GPU kernel 从相同地址读取最新数据。效果等价于图被"参数化"了。

### 理论上能否让 prepare_inputs 和 forward GPU 并行

理论上有意义的是 step N 的 forward 与 step N+1 的 prepare_inputs 的 GPU 并行（而非同一步内的并行，因为同步步内存在数据依赖）。这需要 double-buffering 设计，类似 pipeline parallelism 的思路。vLLM 目前选择通过 batch queue 和 async output copy 实现 CPU-GPU overlap，而不是 GPU-GPU stream overlap。

## 易混淆点
- `prefill` 和 `decode` 的边界不在 scheduler，而在 runner 侧的 `prefill_len` 判断。
- `query_len` 是“本轮调度的 token 数”，不是“这个请求的总长度”。
- `seq_len < prefill_len` 表示当前还是 chunked prefill，这一轮即使跑了 `sample()`，最终也会被强制视为“没有新 token 输出”。
- `block_tables` 在 v2 里不是只用于取 block id；它还参与 `compute_slot_mappings()`，决定当前 token 会把 KV 写到哪个物理 slot。
- v2 下被抢占恢复的请求，不是继续沿用旧的 runner request state，而是会作为 `scheduled_new_reqs` 重新灌入 `RequestState`。
- async scheduler 的 `num_output_placeholders` 是“未来即将产出的 token 占位”，它会让 scheduler 在真实 token 返回前先把请求往前推进一拍。
- `DraftTokensHandler` 在 v2 中主要用于 structured output 场景下把 draft token 回传给 scheduler 做 grammar 校验；不是所有 spec decode step 都需要把 draft token 真正回传到 scheduler。
- CUDA graph 模式下 `cudagraph_manager.run()` 只是 `graph.replay()`，不传入任何参数。输入数据通过 `InputBuffers` 的固定地址 buffer 传递——`prepare_inputs` 覆写 buffer 内容，graph replay 读取同一地址。这不是"图外传参"，而是利用了 CUDA graph 录制时绑定的内存地址在 replay 时仍然有效的特性。
