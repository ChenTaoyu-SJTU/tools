# Codex 更新速记

## 目标
- 机器：`910c_1`、`910c_2`、`910c_3`、`910c_4`
- 容器：不要写死容器名，按当前运行中的目标容器动态发现
- 要求：只在 container 内安装或更新 `codex`，并覆盖容器内的 `~/.codex/auth.json`

## 固定规则
- 不要把 `codex` 装在 Host 上
- 本地权限文件来源：`/Users/bytedance/.codex/auth.json`
- Host 中转目录：`/data01/cty/Downloads/auth.json`
- 容器内目标路径：`/root/.codex/auth.json`
- 执行远程命令时不要走 sandbox，要用真实网络环境
- 新容器如果还没装 `codex`，默认自动安装
- 先发现容器，再执行安装和覆盖，不依赖固定镜像名

## 容器发现
- 先看当前运行中的容器：
```bash
ssh 910c_X "docker ps --format '{{.Names}}'"
```
- 如果只想处理你的业务容器，优先用名称过滤，例如：
```bash
ssh 910c_X "docker ps --format '{{.Names}}' | grep '^cty_'"
```
- 如果后续你的命名规则变了，只需要替换过滤条件，不要改主流程

## 最快流程
1. 上传权限文件到 4 台机器：
```bash
scp /Users/bytedance/.codex/auth.json 910c_1:/data01/cty/Downloads/auth.json
scp /Users/bytedance/.codex/auth.json 910c_2:/data01/cty/Downloads/auth.json
scp /Users/bytedance/.codex/auth.json 910c_3:/data01/cty/Downloads/auth.json
scp /Users/bytedance/.codex/auth.json 910c_4:/data01/cty/Downloads/auth.json
```

2. 每台机器执行一次：
```bash
ssh 910c_X '
containers=$(docker ps --format "{{.Names}}" | grep "^cty_" || true)
[ -n "$containers" ] || { echo "no target containers"; exit 1; }
for c in $containers; do
  docker exec "$c" bash -lc "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v codex >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
      apt install -y nodejs
      npm install -g npm@11.12.0
      npm install -g @openai/codex
    fi
    mkdir -p /root/.codex
    chmod 700 /root/.codex
  "
  docker cp /data01/cty/Downloads/auth.json "$c":/root/.codex/auth.json
  docker exec "$c" bash -lc "
    chmod 600 /root/.codex/auth.json
    node -v
    npm -v
    codex --version
    sha256sum /root/.codex/auth.json
  "
done
'
```

## 验收
- 本地摘要：
```bash
shasum -a 256 /Users/bytedance/.codex/auth.json
```
- 容器摘要必须与本地一致
- 每个发现到的目标容器都要满足：
  - `node` 可用
  - `npm` 版本为 `11.12.0`
  - `codex` 可用
  - `/root/.codex/auth.json` 存在且摘要一致

## 建议
- 只更新权限文件时，不需要重装 `codex`，直接执行 `scp` + `docker cp` + 摘要校验
- 如果某台机器已有旧版 `codex`，想统一版本，去掉 `if ! command -v codex ...` 判断，强制重装即可
- 默认建议先人工确认一次容器发现结果，再执行批量更新
