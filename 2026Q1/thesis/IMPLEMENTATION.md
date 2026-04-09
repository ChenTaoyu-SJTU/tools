# 总则
demo的简洁性约束：
    + 只考虑语言模型，当前例子为Qwen3-0.6B。其他所有架构暂不考虑。
    + 暂不考虑speculative decoding
    + 不考虑PD分离架构。
    + 基于vllm 0.16.0版本进行实现。
    + 需要支持acl_graph模式。
    + 不考虑pcp模式
    + 只考虑tp, dp, ep的分布式场景。pp不考虑。
复用vllm的model_runner v1的结构。

## 实验环境
本试验使用vllm, vllm-ascend两个代码库作为baseline和基础仓库。
在`910c_2`服务器的`cty_vllm_main`容器内进行实验，代码仓库为`/vllm-workspace`。执行命令的位置为`/workspace`。
在vllm和vllm-ascend仓库的`RS_scheduler`分支上进行实验。如果没有该分支，则创建出来。


## 实现Input_batch
Input_batch的角色：包含inputs_ids和attn_metadata等等模型所需的输入信息。
实现方式：
    + 设计为一个dataclass，包含inputs_ids和attn_metadata等等模型所需的输入信息。
关键的forward函数在此：`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`的`execute_model()`函数内：
```py
with (
    record_function_or_nullcontext("forward"),
    set_ascend_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        aclgraph_runtime_mode=cudagraph_mode,
        batch_descriptor=batch_desc,
        num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
        model_instance=self.model,
        max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
        skip_compiled=has_encoder_input,
    ),
    self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
):
    hidden_states = self._model_forward(
        num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
    )
```
所以你的input_batch需要包含：
    + input_batch.input_ids, 
    + input_batch.positions,
    + input_batch.attn_metadata
    + input_batch.num_tokens_padded
    + input_batch.num_tokens_across_dp
    + input_batch.cudagraph_mode
    + input_batch.batch_desc
    + input_batch.num_actual_tokens
规定这里的max_tokens_across_pcp为0，has_encoder_input为False。
