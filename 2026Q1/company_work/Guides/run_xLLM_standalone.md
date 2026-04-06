# 准备工作
1. 开发机配置 git config:
    https://bytedance.larkoffice.com/wiki/QQRxwIXfPiQKDfk9KcLcoHevn9f
2. 拉下 xLLM 的 feat/npu_model_runner_cty 分支
3. 安装一些依赖:
    ```bash
    pip install jaxtyping pynvml munch readerwriterlock
    pip install "lm-eval[api]"
    ```
# 进行测试
1. 在`xLLM/examples/model_runner/vllm/xllm_npu_script.sh`中会跑 real_input 和 profiling 测试，如果你只需要跑其中一个，把另一个注释掉即可。
2. 设置profiling数据参数：在`xLLM/examples/model_runner/vllm/Ascend_Profiling.py`中，你可以设置profiling的数据参数，在 251 行开始。
    ```python
    profiling_datas = [
        [2, 0, 10240, "prefill_only"],
        [2, 10240, 10240, "chunk_prefill"],
        [40, 4096, 1, "decode_only"],
        [40, 8192, 1, "decode_only"],
        [40, 10240, 1, "decode_only"],
        [160, 10240, 1, "decode_only"],
    ]
    ```