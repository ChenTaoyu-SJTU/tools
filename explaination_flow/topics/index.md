# Topic Index

| topic | 关键词 | 一句话摘要 |
|-------|--------|-----------|
| [vllm_ascend_worker_init_flow_mp_backend](vllm_ascend_worker_init_flow_mp_backend.md) | worker, init, mp-backend, NPUWorker, distributed, kv-cache, Phase A/B, ACLGraph | mp backend 下 NPUWorker 初始化两阶段流程（同步 init+load → RPC 驱动 KV cache） |
| [vllm_scheduling_overview](vllm_scheduling_overview.md) | scheduling, async, scheduler, AsyncScheduler, RecomputeScheduler, executor | vllm/vllm-ascend 的 async scheduling 默认行为、推导逻辑与生效路径 |
| [vllm_v1_model_runner_v2_execution_path](vllm_v1_model_runner_v2_execution_path.md) | v2-runner, execute_model, prepare_inputs, sample, async-scheduling, CUDA-graph, InputBatch, chunked-prefill, spec-decode | V2 GPU ModelRunner 的完整 step 链路：schedule → prepare → forward → sample → update |
| [prepare_inputs_attn_metadata](prepare_inputs_attn_metadata.md) | prepare_inputs, attention-metadata, token-positions, slot-mapping, block-table, input_ids, prefill, chunked-prefill | model_runner_v1 架构下 inputs 和 attention metadata 的构建过程（含图解示例） |
| [vllm_distributed_dp_tp_ep_topology](vllm_distributed_dp_tp_ep_topology.md) | distributed, DP, TP, EP, PP, PCP, rank, world_size, parallel_state, MoE, ExternalDP, headless | DP/TP/EP 拓扑、rank 语义、init_distributed_environment 与多机部署路径 |
| [xllm_vllm_vllm_ascend_runner_and_profiling](xllm_vllm_vllm_ascend_runner_and_profiling.md) | xLLM, vLLMModelRunner, profiling, Ascend_Profiling, launcher, DP, enable_attention_dp, NPUModelRunner, inputs_sharing | xLLM 如何通过 vanilla_vllm backend 接线到 vllm-ascend NPUModelRunner，以及 profiling 路径差异 |
| [vllm_ascend_feature_switches_and_memory_optimization](vllm_ascend_feature_switches_and_memory_optimization.md) | HCCL_BUFFSIZE, FUSED_MC2, layer_sharding, MoE, memory, DeepSeek-V3, W8A8, FlashComm2 | HCCL_BUFFSIZE、FUSED_MC2、layer_sharding 三类 Ascend 特性开关的真实语义与适用场景 |
| [vllm_ascend_runtime_graph_and_padding](vllm_ascend_runtime_graph_and_padding.md) | profile_run, dummy_run, ACLGraph, graph-capture, DP-padding, cudagraph_capture_sizes, FlashComm1, num_tokens_across_dp | profile run / graph capture / DP padding 的关系，以及 Ascend 侧 ACLGraph shape 决策流程 |
| [vllm_log_stats_semantics](vllm_log_stats_semantics.md) | log, stats, throughput, metrics, prompt-throughput, generation-throughput, cache-hit-rate, spec-decode-metrics | vLLM 日志指标的统计口径：周期平均 vs 瞬时状态 vs 滑动窗口 |
| [xllm_repo_transfer_via_tos](xllm_repo_transfer_via_tos.md) | xLLM, rclone, TOS, transfer, tar, sha256, migration | xLLM 仓库通过对象存储（rclone）在远程机器间迁移的标准流程 |
