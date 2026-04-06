# Objective
当前的工作总目标为通过优化算子来让 xLLM 在 NPU(Ascend) 芯片上的性能提高。xLLM 需要 import vllm, vllm_ascend 以能够在 NPU 芯片上运行。

# 任务拆解
1. 需要通过适配代码让 xLLM 的 standalone 代码来让 xLLM 能够在 NPU 芯片上运行。
2. 需要做 Profiling 来分析 xLLM 在 NPU 芯片上的性能，找到性能瓶颈。
3. 需要根据 Profiling 结果，优化算子，提高 xLLM 在 NPU 芯片上的性能。