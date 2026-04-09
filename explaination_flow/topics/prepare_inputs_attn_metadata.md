---
tags: [prepare_inputs, attention-metadata, token-positions, slot-mapping, block-table, input_ids, prefill, chunked-prefill, model_runner_v1]
---

model_runner_v1架构
整体概念
执行模型前向传播所需信息：
- 输入（inputs）
- 输入对应的注意力元数据（attention metadata）
下图展示了model_forward需要准备的内容。
              +---------------+
  inputs  --> |               |
              |     model     |  --> output
attn_meta --> |               |
              +---------------+
1. 对于被调度完后的一批token(可能属于不同的请求)，当前的inputs有哪些：
  - inputs_ids: token对应的int表示
  - token_positions: 每个 token 在其请求序列中的相对位置
  - token_indices: 每个token在token id table上的位置
2. 构建输入的 attention metadata
模型在前向计算过程中需要以下 attention 元数据：
- query start location
每个请求在当前 batch 中对应 token 的起止位置
- sequence length
每个请求的序列长度（包含已计算 token + 当前调度 token）
- number of computed tokens
每个请求已经计算过的 token 数量
- number of requests
当前 batch 中的请求数量
- number of tokens
当前 batch 调度的 token 总数
- block table
block级的位置索引：将序列中的逻辑 block 地址映射到设备内存中的物理 block 地址
- max query len
当前 batch 中请求的最大 token 数
- slot mapping
token级的位置索引：input token 将被存储到 KV cache 的位置索引
- attention mask
在 softmax 前应用于 attention score 的 mask，用于控制 token 的可见性（通常是 causal attention）

---
开始之前
主要有三类变量：
1. Token 级变量（token level）
表示每个调度 token 的属性，变量长度 = 调度 token 数量

---
2. Request 级变量（request level）
表示每个请求的属性，变量长度 = 调度请求数量
特殊情况：
- query start location 的长度会 多一个元素

---
3. 系统级变量（system level）
Token IDs table
存储每个请求的 token ids（模型输入）。
形状：
(max num request, max model len)
其中：
- max num request
一个 forward batch 中允许的最大并发请求数
- max model len
单个请求序列最大 token 数

---
Block table
用于将序列中的 逻辑 block 地址 映射到 设备内存中的物理 block 地址
形状：
(max num request, max model len / block size)

---
注意：
这两个表都是在 _update_states 方法中生成的，然后再进行 输入准备阶段。

---
小提示
Token ID 本质是一个 整数（通常为 int32），表示一个 token。
示例：
| Token ID     | Token         | 
|--------------|---------------|
| 0            | [PAD]         |
| 1            | <|endoftext|> |
| 2            | <|start|>     |
| 3            | [SEP]         |
| 4            | I             |
| 5            | the           |
| 6            | be            |
| 7            | of            |
| 8            | and           |     
| ...          | ...           |     
| vocab_size-1 | <|im_end|>    |

---
详细过程
假设条件：
- 每次最多调度 token 数：10
- block size：2
- 总共有 3 个请求
它们的 prompt 长度分别为：
3, 2, 8
- max model length：12
（单个请求序列可处理的最大 token 数）
这些参数在 启动 vLLM 时配置，不是固定值。

---
Step 1：所有请求都处于 prefill 阶段
获取 inputs
最多调度 token 数为 10，因此本次调度：
{'0': 3, '1': 2, '2': 5}
注意：
request_2 使用 chunked prefill，剩余 3 个 prompt token 未调度。

---
1 获取 token positions
首先确定 token 属于哪个请求：
tokens 0–2 → request_0
tokens 3–4 → request_1
tokens 5–9 → request_2
用 request indices 表示：
[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
每个 token 的位置计算方式：
position = computed_tokens + 当前token在请求中的相对位置
例子：
request_0: [0+0,0+1,0+2]
request_1: [0+0,0+1]
request_2: [0+0,0+1,0+2,0+3,0+4]
拼接后：
[0,1,2,0,1,0,1,2,3,4]
最终：
token positions = [0,1,2,0,1,0,1,2,3,4]
这是 token-level 变量。

---
2 获取 token indices
当前 Token IDs table 形状：
(max num request, max model len)
为什么表中存在未调度 token？
因为：
- 一个请求的 token ids 会 一次性写入 table
- 每次 forward 只取出当前调度的 token
示意：
| T_0_0 | T_0_1 | T_0_2 | ? | ? | ? | ... |
| T_1_0 | T_1_1 | ? | ? | ? | ... |
| T_2_0 | T_2_1 | T_3_2 | T_3_3 | ... |
设：
M = max model len
则：
token_index = position + request_index * M
计算得到：
[0,1,2,12,13,24,25,26,27,28]

---
3 获取 Token IDs
通过 token indices 访问 token table：在vllm中，block table被取名为token_ids_cpu
input_ids = token_table[token_indices]
得到：
Input IDs =
[T_0_0,T_0_1,T_0_2,T_1_0,T_1_1,T_2_0,T_2_1,T_3_2,T_3_3,T_3_4]

---
构建 attention metadata
当前 Block Table
(max num request , 6)
|1|2|0|0|0|0|
|3|0|0|0|0|0|
|4|5|6|0|0|0|
设备 KV cache block：
|0|1|2|3|4|5|6|7|8|9|...
设：
K = max_model_len / block_size = 6

---
slot mapping 构建流程
1 计算 block table indices
公式：
request_indices * K + positions / block_size
得到：
[0,0,1,6,6,12,12,13,13,14]

---
2 获取 device block number
block_numbers = block_table[block_table_indices]
得到：
[1,1,2,3,3,4,4,5,5,6]

---
3 计算 block offsets
positions % block_size
得到：
[0,1,0,0,1,0,1,0,1,0]

---
4 构建 slot mapping
device_block_number * block_size + block_offset
得到：
[2,3,4,6,7,8,9,10,11,12]

---
Request-level metadata
调度 token：
[3,2,5]
得到：
query start location = [0,3,5,10]
sequence length = [3,2,5]
computed tokens = [0,0,0]
number of requests = 3
number of tokens = [3,2,5]
max query len = 5
slot mapping = [2,3,4,6,7,8,9,10,11,12]

---
attention mask
prefill 阶段：
创建 一个共享 mask
形状：
5 × 5
block table, slot mapping和device memory之间的关系：
0. scheduled tokens:
{'0': 3, '1': 2, '2': 5}

1. block table:
|1|2|0|0|0|0|
|3|0|0|0|0|0|
|4|5|6|0|0|0|

2. slot mapping:
[2,3,4,6,7,8,9,10,11,12]

3. devoce memory:
block view：|   0   |   1   |   2   |   3   |   4   |    5    |    6    |    7    |
token view：| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |



---
Step 2：Chunked Prefill
获取 inputs
调度 token：
{'0':1,'1':1,'2':3}
request indices
[0,1,2,2,2]
token positions
[3,2,5,6,7]

---
当前 Token IDs table：
T_0_3
T_1_2
是模型生成的新 token。

---
token indices
[3,14,29,30,31]

---
Input IDs
[T_0_3,T_1_2,T_3_5,T_3_6,T_3_7]

---
Attention metadata
新增 block：
7 → request_1
8 → request_2
Block table：
|1|2|0|0|0|0|
|3|7|0|0|0|0|
|4|5|6|8|0|0|

---
token-level
block table indices
[1,7,14,15,15]
device block number
[2,7,6,8,8]
block offsets
[1,0,1,0,1]
slot mapping
[5,14,13,16,17]

---
request-level
调度 token：
[1,1,3]
query start location = [0,1,2,5]
sequence length = [4,3,8]
computed tokens = [3,2,5]
number of requests = 3
max query len = 3
slot mapping = [5,14,13,16,17]

---
attention mask
形状：
5 × 8
每个 token：
1 × 8
block table, slot mapping和device memory之间的关系：
0. scheduled tokens:
{"0": 1, "1": 1, "2": 3}

1. block table:
|1|2|0|0|0|0|
|3|7|0|0|0|0|
|4|5|6|8|0|0|

2. slot mapping:
[5,14,13,16,17]

3. devoce memory:
block view：|   0   |   1   |   2   |   3   |   4   |    5    |    6    |    7    |    8    |
token view：| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
