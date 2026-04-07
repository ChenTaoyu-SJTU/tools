# Objective
当前的工作总目标为通过优化算子来让 xLLM 在 NPU(Ascend) 芯片上的性能提高。xLLM 需要 import vllm, vllm_ascend 以能够在 NPU 芯片上运行。

# 任务拆解
1. 需要通过适配代码让 xLLM 的 standalone 代码来让 xLLM 能够在 NPU 芯片上运行。
2. 需要做 Profiling 来分析 xLLM 在 NPU 芯片上的性能，找到性能瓶颈。
3. 需要根据 Profiling 结果，优化算子，提高 xLLM 在 NPU 芯片上的性能。

# 业务产出
1. 适配vllm-ascend开源大语言模型框架到火山方舟的xLLM框架，具体工作有：
    + 将vllm-ascend的最底层接口model_runner接入xLLM，让xLLM能够在NPU芯片上运行，为DeepSeek-v4模型接入做好准备。
    + 校准xLLM框架的精度和性能，确保两者与vllm-ascend上的精度和性能一致。
    + 修复关键bug，优化代码可读性与可维护性。修复的关键的bug和关键适配工作包括有：
        1. 让xLLM的acl_graph模式能够在NPU上正确运行，达到性能和精度都需一致。
        2. 定位main分支的某些算子报错问题，解决方案是重写分布式初始化过程，该算子只有正确初始化分布式环境才能让其成功运行。
        3. 构建了单机，双机的dp+tp+ep的分布式服务系统，支持大ep场景。
        4. 产出了详细的文档，包括适配代码，操作说明，profiling结果，优化建议等。
2. 为高性能计算团队在算子侧提供算子性能profiling服务，具体工作有：
    + 解决了vllm在PD分离下，其TTFT和TPOT与理论性能不一致的问题。
    + 发现了 vllm bench serve 在进行在线压测时，无法测准在给定数据下的真实性能，需要使用standalone来使用固定数据进行profiling。
    + 为团队构建了一套自动化的profiling流程，包括数据准备，profiling运行，结果分析等。
        1. 支持多种分布式并行场景，包括dp, tp, ep, 单机, 双机，这些情景下均可进行profiling。
        2. 支持多种profiling模式，包括prefill_only, chunk_prefill, decode_only等。
        3. 支持多种profiling数据，包括丰富的batch_size, context_len, query_len等不同组合。
        4. 智能化分布profiling数据，可针对pd并行场景下，自动分配profiling数据，确保每个Node的profiling数据量相等。
        5. 产出了详细的使用说明文档，包括适配代码，操作说明，profiling结果，优化建议等。
        6. 做了性能计算器的解释说明，为后续的性能优化提供了理论基础。