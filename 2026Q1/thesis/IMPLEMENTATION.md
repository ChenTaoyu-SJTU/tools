# Implementation

## 实现Input_batch
1. Input_batch看看能不能复用`vllm/vllm/v1/worker/gpu/input_batch.py`里的结构
2. 做一个test来测试, 这里的所有的变量都需要被mock掉，包括self.vllm_config, input_batch, num_tokens_across_dp。这里的self.kv_connector是NO_OP_KV_CONNECTOR。这里的self.uses_mrope是False。
    ```py
    # Run PyTorch model in eager mode.
    positions = input_batch.positions
    if self.uses_mrope:
        assert input_batch.mrope_positions is not None
        positions = input_batch.mrope_positions
    with set_forward_context(
        input_batch.attn_metadata,
        self.vllm_config,
        num_tokens=input_batch.num_tokens_after_padding,
        # TODO(woosuk): Support piecewise CUDA graph.
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        num_tokens_across_dp=num_tokens_across_dp,
        slot_mapping=input_batch.slot_mappings,
    ):
        self.kv_connector.pre_forward(scheduler_output)
        hidden_states = self.model(
            input_ids=input_batch.input_ids,
            positions=positions,
            inputs_embeds=input_batch.inputs_embeds,
        )
    ```