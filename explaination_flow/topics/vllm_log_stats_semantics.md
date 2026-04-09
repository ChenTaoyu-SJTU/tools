---
tags: [log, stats, throughput, metrics, prompt-throughput, generation-throughput, cache-hit-rate, spec-decode-metrics, VLLM_LOG_STATS_INTERVAL]
---

# vLLM Log Stats Semantics

## 结论
- `Avg prompt throughput` 和 `Avg generation throughput` 不是服务启动以来的累计平均值，而是最近一个日志统计周期内的平均吞吐。
- `SpecDecoding metrics` 中的 `Accepted throughput`、`Drafted throughput`、`Accepted`、`Drafted`、`Mean acceptance length`、`Per-position acceptance rate` 也都是最近一个日志统计周期内聚合后输出的结果，打印后会清零重新统计。
- `Running`、`Waiting`、`GPU KV cache usage` 是打印当下的瞬时状态。
- `Prefix cache hit rate`、`MM cache hit rate` 不是单次日志周期值，而是最近 N 个请求的滑动窗口命中率；默认窗口大小是 1000 个 request。
- `Avg prompt throughput` 的 prompt token 口径只统计“本地实际 prefill 计算的 token”，不包含从 prefix cache / 外部 KV transfer 直接命中的 token，因此在缓存命中较多时可能出现 prompt throughput 偏低但整体服务吞吐不低的情况。

## 代码依据
- `vllm/vllm/v1/metrics/loggers.py`
  - `_track_iteration_stats()` 会在当前日志周期内累加 token 数。
    - prompt 侧使用的是 `iteration_stats.prompt_token_stats.computed`（排除 cached/transferred），generation 侧使用 `iteration_stats.num_generation_tokens`。
  - `_get_throughput()` 用 `tracked_stats / (now - self.last_log_time)` 计算吞吐。
  - `_update_stats()` 调用后会执行 `_reset(now)`，把 `num_prompt_tokens`、`num_generation_tokens` 等周期计数清零。
- `vllm/vllm/v1/spec_decode/metrics.py`
  - 类注释明确写明：按时间区间聚合，`log()` 输出后 `reset()` 清零。
- `vllm/vllm/v1/metrics/stats.py`
  - `CachingMetrics.hit_rate` 注释说明是 `the past N requests`。
  - `CachingMetrics` 默认 `max_recent_requests=1000`，因此 cache hit rate 是滑动窗口，不是全生命周期累计。
- `vllm/vllm/envs.py`
  - `VLLM_LOG_STATS_INTERVAL` 默认是 `10.0` 秒。
- `vllm/vllm/v1/engine/llm_engine.py`
  - `do_log_stats_with_interval()` 按 `VLLM_LOG_STATS_INTERVAL` 控制打印节奏；每次触发后会调用 `logger_manager.log()` 进而触发上述清零逻辑。

## 解释样例
如果某次日志打印为：
- `Avg generation throughput: 306.5 tokens/s`
- `Accepted: 2145 tokens`

那它更接近“最近约 10 秒内”的统计结果，而不是“服务从启动到现在的总平均/总累计”。
