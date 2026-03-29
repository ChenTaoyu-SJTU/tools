# 背景
目前我会主要在`910c_1`, `910c_2`, `910c_3`, `910c_4`远程机器上进行开发工作。相关ssh配置在本地的`~/.ssh/config`. 该服务器已经实现了免密登陆。
该机器为华为的Ascend 16卡910C A3服务器。

我在这台服务器上，需要经常跑vllm和vllm-ascend的相关代码，运行deepseek等相关大语言模型。

# agent行动规则
+ 你必须关注文档中的**常见错误**来规避一些风险！
+ 在执行任务前，你需要根据任务内容去收集本地机器上的相关文档和规则还需要收集服务器上的相关信息，然后先做好计划，确定好任务的执行顺序，然后回复给我，让我确认一下，不要直接开始执行。
+ 你需要在收尾工作的时候进行一次简洁干练的总结，这份总结以后可以帮助你提高执行效率。
+ 在传输或更新一些文件时，会遇到原来的文件已经存在的问题，你需要先删除或者覆盖掉那些重复的文件或目录。如果涉及到文件传输或更新，你可以先明确这是一次全新的传输操作还是更新操作。

# 本地机器上的agents_tools说明
## 定义
本地的`agents_tools`目录是给agent使用的说明工具目录。用于让agents根据这里面的规范操作**本地mac机器**。

# 远程机器上给agents使用的说明工具
## 定义
远程的`agents_tools`目录是给agent使用的说明工具目录。用于让agents根据这里面的规范操作**远程服务器**。

## 远程机器上的agents_tools
在每个远程机器上的`/data01/cty/agents_tools`目录下，有我自定义的工具，给agent做使用说明。**你经常需要查看这些工具说明来执行我给你下发的指令。**

## agents_tools的更新
我可能经常会在`910c_2`上更新`agents_tools`。然后我有时候会给你派发命令来让你同步这些更新。你先用zip命令将服务器上的`agents_tools`打包，然后通过`scp`命令将压缩文件传到mac上，然后解压成`remote_agent_tools`在`/Users/bytedance/gitspace`目录下，然后依次上传zip文件，并解压到其他远程机器上。注意，如果目录下有因为之前的更新操作而存在的重复的文件或目录，你需要删除或者覆盖掉那些重复的文件或目录。暂存/中转目录的规则在本文档中有介绍。

# remote 机器信息
一共有四台远程机器，分别是`910c_1`, `910c_2`, `910c_3`, `910c_4`。相关配置在`~/.ssh/config`内。
对于任何一台机器，我的工作目录都在`/data01/cty`目录下。

## remote的Host和container的mount关系
重要：对于每个container来说，`/data00`, `/data01`, `/data02`, `data03`一定是挂载上了Host的对应目录。如果想要从mac上传东西到container内，可以通过`/data01/cty/Downloads`目录进行中转。

# 小型文件的上传或下载
## 从远程机器上下载小型文件
如果需要从远程机器上下载一些文件到本地mac上，请使用scp命令，将远程机器上的文件下载到本地mac的`~/Downloads`目录下。

## 从本地mac上传小型文件
从本地上传到远程机器上，使用scp命令，将本地mac上的文件上传到远程机器的`/data01/cty/Downloads`目录下。

# 大型文件的上传或下载
## vllm-ascend的官方镜像网站的拉取和上传流程
该流程将镜像从官网拉入本地mac机器，然后用mac通过volces-tos对象存储中转，将镜像送入远程机器`910c_2`，最后在远程机器上把镜像给载入。

https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest
使用的platform是`linux/arm64`, a3版本的镜像。不是openeuler，是ubuntu的镜像。

### 常用的仓库版本
releases-v0.13.0-a3, linux on arm64
v0.16.0rc1-a3, linux on arm64

### mac上拉取镜像，给rclone中转
我的定制`rclone`用法在`tools/remote_workflow/rclone.md`内。
首先，我们需要在本地的mac上去拉取镜像。我们将使用podman进行拉取镜像。podman如果当前没在运行，则首先要启动：`podman machine start`，然后检查状态`podman images`

然后再将它`podman save -o`到本地mac的`~/tmp`目录中。

再通过rclone将它上传到volces-tos上，暂存位置为`yupeng-dev-hb2/chentaoyu/images`，最后再从volces-tos下载到服务器上。

注意，这些镜像文件大小有17~20GiB，而我网速不是特别快，所以download, save和upload时间比较长，不要急，可以等比较久的时间。

然后你可以ssh到服务器上去，将东西下载至`/data01/cty/Downloads`。

最后通过`docker load`将镜像加载到docker中(注意我们在mac上用podman去下载，但是在服务器上，我们使用docker来加载和使用容器)。

# 初始化各个服务器上的环境

## 初始化container
首先可以检查服务器上的docker container是否已经完成了初始化。验收标准是cty_vllm_v0.13.0和cty_vllm_v0.16.0rc1这两个container都已经启动了。

## 初始化codex安装
1. 在每个服务器的对应的container内部去安装codex，不要安装在Host上。如果container内已经存在codex则不要重复安装。
    ```
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs

    # update npm
    npm install -g npm@11.12.0

    # install codex
    npm install -g @openai/codex
    ```
2. 更新权限文件：把本地mac上的`~/.codex/auth.json`文件给拷贝到container的对应目录里面，不要拷贝到Host的对应目录里。如果里面已经有了`~/.codex/auth.json`，那么可能是过时的文件，你需要用新的覆盖掉它。
3. 验收标准: 我的每个container中都成功安装了codex，并且mac上的`~/.codex/auth.json`也正确上传了
简洁总结：
  这类权限文件更新最稳妥的路径是“本地文件 -> 远程 `/data01/cty/Downloads` 中转 -> `docker exec` 覆盖 `container` 内 `/root/.codex/auth.json` -> SHA-256 校验”。这样可以明确规避把文件误写到 Host 用户目录的常见错误。
# 一些使用示例

## 拉起vllm PD分离服务

### 1. 准备工作
1. 确保服务器上的docker container已经初始化完成。请用`pkill VLLM`来清除这个机器上所有的VLLM服务。
2. 确保每个服务器上都存在`/data01/cty/pd_example`文件夹，如果没有，请将`910c_1`上的`/data01/cty/pd_example`文件夹复制到其他服务器上，可以使用中转的方法来操作。中转规则在本文档中有介绍。
3. 检查当前任务给出的模型并行设置，并且通过这个设置去更改服务器上的脚本和文档配置。一定要根据指令去严谨的修改！
4. 确保每个服务器上对应的`/data01/cty/pd_example`的文件夹中的各个文档内的特化说明和脚本文件的特化参数都是正确的，你需要根据每个服务器的情况来进行配置。你可以通过阅读`/data01/cty/pd_example/README.md`文件来了解每个服务器的角色，每个脚本文件的参数，和如何在每个服务器上配置正确的参数。**易错点**: 请一定让文档内容，机器角色，脚本参数它们保持一致, 特别是关注`run_dp_template.sh`的里的并行模式和其他脚本中的参数是否一致，文档内的角色和机器角色是否一致，脚本配置和机器配置是否一致(ip地址，nic网卡等等)
- 先查容器和 pd_example，再查 README/脚本/网卡/IP 一致性。
- 这套环境里最容易踩坑的是"文档角色写错"和"模型并行配制"。
- 对于Prefill节点和Decode节点来说，`run_dp_template.sh`中的`--additional-config`是不一样的，请注意区别！
- 对于Prefill节点来说，使用eager模式，而Decode节点使用cuda graph模式。
5. 准备好一切就绪后，请列出各个服务器上的具体情况信息，交给我来验收。

## 命令agent给服务器初始化docker container环境
给agent提示词：根据位于"/Users/bytedance/gitspace/tools/remote_workflow/utils.md"文件中指定的初始化规则，基于远程服务器的`agents_tools`配置，为四台远程服务器执行Docker容器环境的初始化工作。确保每台服务器的Docker容器环境都能成功初始化并满足运行要求。完成所有服务器的初始化后，需提供详细的执行报告，包括每台服务器的初始化状态、配置详情及验证结果，以便进行验收确认。

## 把volces-tos上的镜像传到到服务器上
给agent提示词：
根据文档给出的规则，将对象存储volces-tos里的镜像传入到服务器上。

# 执行任务时常见错误
1. ssh沙箱环境下权限失败
    - 沙箱里没法解析跳板机域名，所以会被远程探测拦住。请使用真实环境和真实网络，别用`sandbox`
2. 在跳板机 jumpecs-lf.byted.org 阶段收到 Permission denied (gssapi-with-
  mic)
    - 提醒我输入`kinit chentaoyu.0@BYTEDANCE.COM`，等我确认完成后，我会杀死整个terminal，然后重启一个codex来继续执行。