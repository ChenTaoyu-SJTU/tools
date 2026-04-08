System:

- apt and yum
    
    ```
    apt-get update -y
    ```
    
- kill process:
    
    ```bash
    kill <process_id>
    
    # force kill
    kill -9 <process_id>
    ```
    
- Cpu information:
    - `lscpu`
    - `uname -i`
- Os information:
    - `cat /etc/os-release`
- cann version:
    - `cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info`
    - `cat /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info`

ssh

- Add remote machine
    
    ```
    Host RTX
        HostName 202.120.32.244
        User chentaoyu
        Port 8822
        RemoteForward 7876 localhost:7897
    
    Host Ascend
        HostName 111.186.61.2
        User fushengqi
        Port 10001
        RemoteForward 7876 localhost:7897
    ```
    
- Generate keys and then send the public key to remote
    
    ```python
    # generate pair of key
    ssh-keygen -t ed25519 -C "ctynb@qq.com"
    
    # display the public key in local machine
    cat ~/.ssh/id_ed25519.pub
    
    # put it into the server
    echo "paste_your_public_key_here" >> ~/.ssh/authorized_keys
    
    # example:
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIP86baQlX87om/iESbFQGnKmDmq7cwld4gWaL9ozYQyu ctynb@qq.com" >> ~/.ssh/authorized_keys
    ```
    

Git

- Set user name and email:
    
    ```bash
    git config --global user.name "ChenTaoyu-SJTU"
    git config --global user.email "ctynb@qq.com"
    
    # After the above commands, the content of ~/.gitconfig:
    [user]
    email = ctynb@qq.com
    name = ChenTaoyu-SJTU
    ```
    
- check the configuration:
    
    `git config --list` 
    
- set github ssh private key
    
    ```bash
    # 若未安装 ssh-keygen：
    sudo apt update
    sudo apt install openssh-client -y   # apt
    sudo yum install openssh-clients -y   # yum
    
    ssh-keygen -t ed25519 -C "ctynb@qq.com"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    cat ~/.ssh/id_ed25519.pub
    # Add xxx.pub github ssh-key website: https://github.com/settings/keys
    # example: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIuQx6cBDetFy0DRRJyAtz6HRSENtSaiTSA11vkqUh60 ctynb@qq.com
    
    # change the read/writ permission of the ~/.ssh/id_ed25519 and ~/.ssh more strict for user only
    chmod 600 ~/.ssh/id_ed25519
    chmod 700 ~/.ssh
    ```
    
- git clone using

```bash
# Recommand ssh version:
git clone git@github.com:ChenTaoyu-SJTU/xDiT.git

git clone https://github.com/vllm-project/vllm.git
```

- Set remote URL
    - Add remote:
    
    ```bash
    # Add remote url:
    git remote add upstream <url>
    
    # Reset remote url:
    git remote set-url origin <new-url>
    
    # Remote remote url:
    git remote remove upstream
    ```
    
- Checkout to an existing branch from a remote repository
    
    ```bash
    # Fetch all remote branches
    git fetch origin
    
    # Checkout to the remote branch (this creates a local tracking branch)
    git checkout <branch-name>
    
    # Or just using one command:
    git checkout -b <branch-name> origin/<branch-name>
    ```
    
- Rebase the upstream/main
    
    ```bash
    git fetch upstream
    
    git rebase upstream/main
    
    # and you should merge your commit first to avoid some unessasary conflict
    # Squash last 3 commits
    git rebase -i HEAD~3
    
    #change pick to squash
    pick abc1234 First commit message
    s def5678 Second commit message
    s ghi9012 Third commit message
    
    # abort rebase if you need another change
    git rebase --abort
    
    # After resolve conflict, add these files:
    git add <resolved-files>
    
    # continue rebase
    git rebase --continue
    
    git push origin current_branch
    ```
    
- Push a new branch to origin:
    
    `git push -u origin my-new-branch`
    
- check before pull request
    
    `pre-commit run --all-files`     
    

Docker

- commit the container current status to a image: (first we should stop it)
    
    `docker stop <container_name_or_id>`
    
    `docker commit <container_name_or_id> <my-custom-image-name>:latest`
    

Conda

- Install anaconda
    
    ```bash
    # check the cpu architecture
    uname -i
    
    # e.g. aarch64
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    bash Miniconda3-latest-Linux-aarch64.sh
    source ~/.bashrc
    
    ```
    
- new environment, enter, exit
    
    ```bash
    conda create --name venv_name python=3.11
    conda activate venvname
    conda deactivate
    ```
    

Pip

- install uv
    
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
- pip upgrade
    
    `python -m pip install --upgrade pip`
    
- pip install requirements.txt
    
    `pip install -r requirements.txt`
    
- mirror source
    - set global index-url:
    
    ```bash
    pip config set global.index-url "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    ```
    
    - After set global index-url we can see: `vim /home/coder/.config/pip/pip.conf`
    
    ```
    [global]
    index-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    ```
    
    - In uv environment set the index-url:
    
    ```bash
    VLLM_USE_PRECOMPILED=1 uv pip install --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ --editable .
    ```
    
- Using org when tsinghua mirror occur SSL error:
    
    `PIP_INDEX_URL=https://pypi.org/simple pre-commit run --all-files`
    

Cann installation:

```
# Install required python packages.
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs 'numpy<2.0.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# Download and install the CANN package.
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-toolkit_8.1.RC1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-toolkit_8.1.RC1_linux-"$(uname -i)".run
./Ascend-cann-toolkit_8.1.RC1_linux-"$(uname -i)".run --full

source /usr/local/Ascend/ascend-toolkit/set_env.sh
or
source /home/cty/Ascend/ascend-toolkit/set_env.sh

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-kernels-910b_8.1.RC1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-kernels-910b_8.1.RC1_linux-"$(uname -i)".run
./Ascend-cann-kernels-910b_8.1.RC1_linux-"$(uname -i)".run --install

wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run
./Ascend-cann-nnal_8.1.RC1_linux-"$(uname -i)".run --install

source /usr/local/Ascend/nnal/atb/set_env.sh
or
source /home/cty/Ascend/nnal/atb/set_env.sh

# 查看 cann 环境：
cat /home/cty/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
```

VLLM start code:

`example.py`

```python
from vllm import LLM, SamplingParams
import os

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

llm = LLM(model="Qwen/Qwen3-0.6B", 
          enforce_eager=True,
          distributed_executor_backend="mp", 
          # tensor_parallel_size=2,
          max_model_len=4096,
          cpu_offload_gb=10,  # Offload 10GB of model weights to CPU
          )

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`serve.py`

```python
import requests
import json
import argparse

# # curl the model server
# curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
#   "model": "Qwen/Qwen3-30B-A3B",
#   "messages": [
#     {"role": "user", "content": "Give me a short introduction to large language models."}
#   ],
#   "temperature": 0.6,
#   "top_p": 0.95,
#   "top_k": 20,
#   "max_tokens": 4096
# }'

def main():
    parser = argparse.ArgumentParser(description='Process some parameters')

    parser.add_argument('--query', '-q', type=str, help='Your query')

    args = parser.parse_args()
    query: str = args.query
    
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 8192,
        "stream": True
    }

    with requests.post(url, headers=headers, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]  # Remove 'data: ' prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            print(content, end="", flush=True)
                    except Exception as e:
                        print(f'Some error {e} happens')
                        pass
    print()

if __name__ == '__main__':
    main()

# Example usage:
# python tmp.py --query "What is large launguage model?"
```

`env.sh`

```bash
export VLLM_USE_V1=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=True
export VLLM_LOGGING_LEVEL=DEBUG
# pytorch:
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export HF_ENDPOINT=https://hf-mirror.com
# visible
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export http_proxy=http://127.0.0.1:7876
export https_proxy=http://127.0.0.1:7876
```

vscode

- `launch.json` for debug. e.g.
    
    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "debugpy",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "justMyCode": false,
                "env": {
                    "VLLM_USE_V1":"1",
                    "VLLM_USE_MODELSCOPE":"True",
                    "PYTORCH_NPU_ALLOC_CONF":"max_split_size_mb:256",
                    "VLLM_LOGGING_LEVEL":"DEBUG",
                },
                "cwd": "${workspaceFolder}",
                "args": [],
            }
        ]
    }
    ```
    
- debug the vllm serve
    
    ```json
    {
        "name": "Debug vllm serve",
        "type": "debugpy",
        "request": "launch",
        "module": "vllm.entrypoints.cli.main",
        "cwd": "/data01/cty/xLLM",
        "justMyCode": false,
        "subProcess": true,
        "console": "integratedTerminal",
        "args": [
            "serve",
            "/data00/download/deepseek-v3.2-w8a8-ascend",
            "--host", "0.0.0.0",
            "--port", "8077",
            "-dp", "1",
            "-tp", "16",
            "--seed", "1024",
            "--quantization", "ascend",
            "--enable-expert-parallel",
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--gpu-memory-utilization", "0.92",
            "--speculative-config", "{\"num_speculative_tokens\": 3, \"method\":\"deepseek_mtp\"}",
            "--compilation-config", "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\":[8,16,24,32,40,48,56,64,72,80]}",
            "--max-model-len", "12000",
            "--max-num-batched-tokens", "12288",
            "--tokenizer-mode", "deepseek_v32",
            "--reasoning-parser", "deepseek_v3",
            "--load-format", "dummy"
        ],
        "env": {
                "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${env:LD_PRELOAD}",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "OMP_PROC_BIND": "false",
                "OMP_NUM_THREADS": "10",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "VLLM_USE_V1": "1",
                "HCCL_BUFFSIZE": "256",
    
                "ASCEND_AGGREGATE_ENABLE": "1",
                "ASCEND_TRANSPORT_PRINT": "1",
                "ACL_OP_INIT_MODE": "1",
                "ASCEND_A3_ENABLE": "1",
                "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": "300000",
                "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                "VLLM_ASCEND_ENABLE_MLAPO": "1",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    
                "VLLM_USE_MODELSCOPE": "True"
        }
    }
    ```
    
- debug `pytest` related file:
    
    ```json
    {
      "version": "0.2.0",
      "configurations": [
        {
          "name": "Debug pytest: test_simple.py",
          "type": "debugpy",
          "request": "launch",
          "module": "pytest",
          "justMyCode": false,
          "env": {
                "VLLM_USE_V1":"1",
                "VLLM_USE_MODELSCOPE":"True",
                "PYTORCH_NPU_ALLOC_CONF":"max_split_size_mb:256",
                "VLLM_LOGGING_LEVEL":"DEBUG"
            },
          "args": [
            "-sv",
            "${file}",
          ],
          "cwd": "${workspaceFolder}" 
        }
      ]
    }
    ```
    
- debug using `debugpy`
    
    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Attach Process 1",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": "localhost",
                    "port": 8812
                }
            },
            {
                "name": "Python: Attach Process 2",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": "localhost",
                    "port": 8813
                }
            }
        ]
    }
    ```
    
    ```python
        if local_rank == 0:
            debugpy.listen(8812)
            print(f"INFO LOCAL RANK {local_rank}: Waiting for debugger attach...")
            debugpy.wait_for_client()  # Optional: wait until the debugger is attached
            print(f"INFO LOCAL RANK {local_rank}: is attach successfully")
        else:
            debugpy.listen(8813)
            print(f"INFO LOCAL RANK {local_rank}: Waiting for debugger attach...")
            debugpy.wait_for_client()
            print(f"INFO LOCAL RANK {local_rank}: is attach successfully")
    
    ```
    
- debug the torchrun`torchrun --nproc_per_node=16 example.py`
    
    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Debug xLLM NPU torchrun (16 NPUs)",
                "type": "debugpy",
                "request": "launch",
                "module": "torch.distributed.run",
                "console": "integratedTerminal",
                "justMyCode": false,
                "env": {
                    "VLLM_USE_V1": "1",
                    "HCCL_OP_EXPANSION_MODE": "AIV",
                    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                    "OMP_NUM_THREADS": "10"
                },
                "args": [
                    "--nproc_per_node=16",
                    "/data01/cty/xLLM/examples/model_runner/vllm/xllm_model_runner_standalone_main.py",
                    "--model_dir", "/data00/download/deepseek-v3.2-w8a8-ascend",
                    "--forward_engine_type", "npu.v0.15.0",
                    "--tp_size", "16",
                    "--quantization", "ascend",
                    "--prompt", "作为一个新闻发言人，我认为我们应该"
                ],
                "cwd": "/data01/cty/xLLM"
            },
        ]
    }
    ```
    
- xllm_vllm.code-workspace
    
    ```json
    {
      "folders": [
        {
          "path": "/vllm-workspace/vllm"  
        },
        {
          "path": "/vllm-workspace/vllm-ascend" 
        },
        {
            "path": "/data01/cty/xLLM/"  
        },
        {
            "path": "/data01/cty/gitspace/bench"
        },
        {
            "path": "/data01/cty/gitspace/agents_markdowns"
        }
      ],
        "settings": {
          "terminal.integrated.cwd": "/data01/cty/xLLM",
          "terminal.integrated.scrollback": 10000,
          "python.analysis.extraPaths": [
            "/data01/cty/xLLM",
            "/vllm-workspace/vllm",
            "/vllm-workspace/vllm-ascend"
          ]
      }
    }
    ```
    

vLLM

- vllm serve and bench serve
    
    ```bash
    # vllm serve
    vllm serve <model> --port 8000 --tensor-parallel-size 4 \
    --data-parallel-size 4 --enable-expert-parallel --max-model-len 8192 \
    --gpu-memory-utilization 0.3 --enforce-eager \
    --max-num-batched-tokens 8192 # (or --no-enable-chunked-prefill)
    
    # bench serve
    vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 \
    --port 8000 --dataset-name random --random-input-len 100 \
    --random-output-len 400 --num-prompts 500 --max-concurrency 50
    
    vllm bench serve --port 8000 --seed $(date +%s) \
    --model /home/cty/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
    --dataset-name random --random-input-len 256 \
    --random-output-len 200 --num-prompts 30 --burstiness 100 \
    --request-rate 3 --ignore-eos --metric-percentiles 90
    
    # curl the model server
    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "Qwen/Qwen3-30B-A3B",
      "messages": [
        {"role": "user", "content": "Give me a short introduction to large language models."}
      ],
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "max_tokens": 4096,
      "stream": true
    }'
    ```
    
- PD 分离：
    
    ```bash
    #!/bin/bash
    set -xe
    
    # Function to clean up previous instances
    cleanup_instances() {
      echo "Cleaning up any running vLLM instances..."
      pkill -f "vllm serve" || true
      sleep 2
    }
    
    # Waits for vLLM to start.
    wait_for_server() {
      local port=$1
      timeout 1200 bash -c "
        until curl -s localhost:${port}/v1/completions > /dev/null; do
          sleep 1
        done" && return 0 || return 1
    }
    
    # Start Prefill Instance
    CUDA_VISIBLE_DEVICES=5 VLLM_NIXL_SIDE_CHANNEL_PORT=5559 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --port 8100 \
     --enforce-eager \
     --gpu-memory-utilization 0.3 \
     --max-model-len 1024 \
     --tensor-parallel-size 1 \
     --distributed-executor-backend mp \
     --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
    
    # Start Decode Instance
    CUDA_VISIBLE_DEVICES=6 VLLM_NIXL_SIDE_CHANNEL_PORT=5659 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --port 8200 \
     --enforce-eager \
     --gpu-memory-utilization 0.3 \
     --max-model-len 1024 \
     --tensor-parallel-size 1 \
     --distributed-executor-backend mp \
     --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
    
    # Waiting until the Prefill and Decode Instances ready for serving
    wait_for_server 8100
    wait_for_server 8200
    
    # Start the proxy server
    python /home/chentaoyu/git_place/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
     --port 8192 \
     --prefiller-hosts localhost \
     --prefiller-ports 8100 \
     --decoder-hosts localhost \
     --decoder-ports 8200 &
    
    # Wait for the proxy to start
    sleep 3600
    
    # Clean up before running next model
    cleanup_instances
    sleep 3
    
    echo "All finished"
    ```
    

Filesystem

- Check the size of a Directory:
    
    `du xxx -sh`
    

Claude code:

- claude code with GLM configuration:

```bash
# first install claude code, this may use export https_proxy=http://127.0.0.1:xxxx
curl -fsSL https://claude.ai/install.sh | bash

# add it to environment
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# And then initialize environment using GLM model
curl -O "https://cdn.bigmodel.cn/install/claude_code_env.sh" && bash ./claude_code_env.sh

# zhipu API key: 1f42ddcc32c942f088dada4cd3a1ca65.Nrs2XUoFdSAJz1ck
```

## CODEX

- 安装最新的`nodejs`，否则会出错
    
    ```bash
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
    
    # update npm
    npm install -g npm@11.12.0
    ```
    
- install and login codex
    
    ```bash
    # we should use the newest nodejs version, but if using
    # default apt install, it will install a old version, so we need
    # ask AI for how to installing a new nodejs version
    npm install -g @openai/codex
    
    # login: directly enter the codex
    codex
    
    # logout inside codex:
    /logout
    
    # login in the bash:
    codex login
    ```
    
- ISSUE in the login step:
    - in the local mac machine, we need turn off the `clash verge`, otherwise, there will be a proxy problem in exchanging the token in localhost.
    - After finish the local machine login, there will be a big problem in remote ssh machine, we should copy the `~/.codex/auth.json` from local machine to the remote machine. This is the only right way to login the codex in remote machine.