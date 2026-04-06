# Remote Workflow Rules

## 适用范围
- 远程机器：`910c_1`、`910c_2`、`910c_3`、`910c_4`
- SSH 配置位置：本地 `~/.ssh/config`
- 默认工作目录：远程 `Host` 上的 `/data01/cty`

## 执行硬规则
1. 执行任务前，先收集本地 `tools`、远程 `agents_tools` 和目标机器的相关信息，再给出计划，不要直接开始执行。
2. 涉及远程操作时，优先查看远程服务器上 `/data01/cty/agents_tools` 下的说明，再执行具体动作。
3. 涉及文件同步时，先明确这是“全新传输”还是“更新覆盖”。
4. 更新已有文件或目录时，先处理重名冲突；需要时先删除旧内容，再覆盖新内容。
5. 收尾时提供简洁总结，说明执行结果、验证结果和后续注意事项。

## 本地与远程说明目录
- 本地 `agents_tools`：用于指导 agent 如何操作本地 Mac。
- 远程 `agents_tools`：位于各远程机器的 `/data01/cty/agents_tools`，用于指导 agent 如何操作远程服务器。
- 执行远程任务时，经常需要先阅读远程 `agents_tools` 中的说明。

## Host、container 与中转规则
- 对每个 container 而言，`/data00`、`/data01`、`/data02`、`/data03` 都挂载到 Host 的对应目录。
- 从本地 Mac 上传内容到 container 时，优先先传到 Host 的 `/data01/cty/Downloads`，再进入 container。
- 需要在 Host 和 container 之间中转文件时，默认使用 `/data01/cty/Downloads`。

## 文件传输原则
### 小型文件
- 从远程下载到本地 Mac：使用 `scp`，默认下载到本地 `~/Downloads`。
- 从本地上传到远程：使用 `scp`，默认上传到远程 `/data01/cty/Downloads`。

### 大型文件
- 大文件优先通过对象存储中转。
- `rclone` 使用方法见 `tools/remote_workflow/rclone.md`。
- 镜像等大文件的服务器侧默认落盘目录为 `/data01/cty/Downloads`。

## agents_tools 更新模板
当需要将某台机器上的 `agents_tools` 同步到其他机器时，按以下顺序执行：

1. 在源机器上打包 `agents_tools`。
2. 将压缩包拉回本地 Mac。
3. 在本地解压到 `/Users/bytedance/gitspace/remote_agent_tools`。
4. 将压缩包分发到其他目标机器。
5. 在目标机器上覆盖旧目录并解压。

说明：
- 如果目标位置已经存在旧文件或旧目录，先删除或覆盖，避免残留重复内容。
- 临时中转目录仍使用本文件定义的 `Downloads` 规则。

## 初始化环境的通用检查
### Docker container
- 先检查目标机器上的 container 是否已经初始化完成。
- 验收基准：`cty_vllm_v0.13.0` 和 `cty_vllm_v0.16.0rc1` 都已启动。

### Codex 安装与权限文件
- `codex` 只安装在 container 内，不安装在 Host 上。
- 如果 container 内已经安装 `codex`，不要重复安装。
- 本地权限文件 `~/.codex/auth.json` 上传时，目标是 container 内的对应路径，不是 Host 用户目录。
- 最稳妥路径是：本地文件 -> 远程 `/data01/cty/Downloads` 中转 -> `docker exec` 或 `docker cp` 覆盖 container 内 `/root/.codex/auth.json` -> SHA-256 校验。

## 示例任务
### 镜像下载工作流
位于`tools/remote_workflow/images_related.md`

### 拉起 vLLM PD 分离服务工作流
准备阶段重点检查：

1. 先用 `pkill VLLM` 清理机器上的旧服务。
2. 确认每台机器都存在 `/data01/cty/pd_example`；如不存在，从已有机器中转复制。
3. 根据任务要求核对模型并行配置，并同步修改脚本和文档。
4. 阅读 `/data01/cty/pd_example/README.md`，确认机器角色、脚本参数、IP 和网卡配置一致。
5. 准备完成后，汇总各台服务器的关键信息，交给用户验收。

易错点：
- 先查 container 和 `pd_example`，再查 README、脚本、网卡和 IP 是否一致。
- 最容易出错的是“文档角色写错”和“模型并行配置不一致”。
- `run_dp_template.sh` 中 Prefill 与 Decode 节点的 `--additional-config` 不同。
- Prefill 节点使用 eager 模式，Decode 节点使用 cuda graph 模式。

### 下载模型权重
准备：先检查权重是否已经下载过了，下载的目录在`~/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B`。
1. 使用本地 Mac 上的 `modelscope` 下载模型权重：`modelscope download --model Qwen/Qwen3.5-0.8B`。
2. 将下载的模型权重上传到对象存储 `volces-tos` 中。 
3. 在服务器上从对象存储下载模型权重到`/data01/cty/Downloads`目录。
4. 验收模型权重是否下载成功。
## 常用任务提示词
### 初始化各服务器的 Docker container 环境
根据位于 `/Users/bytedance/gitspace/tools/remote_workflow/utils.md` 中的初始化规则，结合远程服务器的 `agents_tools` 配置，为四台远程服务器执行 Docker container 环境初始化，并在完成后提供每台机器的状态、配置详情和验证结果。

### 将对象存储中的镜像传到服务器
根据文档规则，将对象存储 `volces-tos` 中的镜像传入服务器，并按默认中转和落盘规则完成下载与验收。

## 常见错误
1. SSH 沙箱环境下权限失败
   - 沙箱环境可能无法解析跳板机域名，也无法使用真实网络。执行远程任务时不要使用 sandbox。
2. 在跳板机 `jumpecs-lf.byted.org` 阶段出现 `Permission denied (gssapi-with-mic)`
   - 首先尝试新开一个`terminal`进行`ssh`尝试重连，如果仍然失败则提醒用户执行 `kinit`.
   - 提醒用户执行 `kinit chentaoyu.0@BYTEDANCE.COM`。
   - 用户确认完成后，会重新开启一个真实终端环境再继续执行。
