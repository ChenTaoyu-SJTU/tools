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
阅读`xLLM/agents_docs/specification/Ascend_Profiling.md`，根据其中的说明进行测试。