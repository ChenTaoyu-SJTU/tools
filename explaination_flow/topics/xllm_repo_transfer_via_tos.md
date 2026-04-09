---
tags: [xLLM, rclone, TOS, transfer, tar, sha256, migration, remote, pigz]
---

# xLLM 仓库通过对象存储中转（rclone）迁移

## 适用场景
- 需要在多台远程机器之间搬运较大的仓库目录（例如 data01/cty 下的 xLLM）
- 机器间无法直接高速互传，或希望通过对象存储做“中转落地 + 校验”

## 前置条件
- 本地已配置 SSH Host（例如 910c_2、910c_4），可直接 ssh 连接
- 远程机器已安装并配置 rclone，且存在 remote：volces-tos
- 约定默认中转目录：volces-tos:yupeng-dev-hb2/chentaoyu
- 约定远程落盘目录：data01/cty/Downloads

相关速查可参考：tools/remote_workflow/rclone.md

## 迁移类型确认
- 全新传输：目标机不存在 xLLM，或希望完全覆盖（推荐）
- 更新覆盖：目标机已有 xLLM，需要明确“覆盖策略”（先备份或先删除再解压）

## 推荐工作流（全新传输/覆盖）
### 1) 在源机器打包并上传到对象存储
1. 在源机器确认仓库目录存在（例如 data01/cty/xLLM），并记录 git HEAD（若有 .git）
2. 在 data01/cty/Downloads 生成压缩包与校验文件（并行 gzip 依赖 pigz）
3. 将 tar.gz 与 sha256 文件上传到对象存储目录（例如 chentaoyu/xLLM_transfer）

示例（在源机器执行）：
```bash
TS=$(date +%Y%m%d_%H%M%S)
SRC=data01/cty/xLLM
OUTDIR=data01/cty/Downloads
OBJ=volces-tos:yupeng-dev-hb2/chentaoyu/xLLM_transfer
ARCHIVE=${OUTDIR}/xLLM_${TS}.tar.gz

tar -C data01/cty -I pigz -cf "${ARCHIVE}" xLLM
sha256sum "${ARCHIVE}" | tee "${ARCHIVE}.sha256"
rclone copy -P "${ARCHIVE}" "${OBJ}"
rclone copy "${ARCHIVE}.sha256" "${OBJ}"
```

### 2) 在目标机器下载、校验并解压
1. 从对象存储下载 tar.gz 与 sha256 到 data01/cty/Downloads
2. sha256 校验通过后再解压
3. 若目标位置已有同名目录，先移动到备份目录再落新内容
4. 解压完成后对比 git HEAD（若有 .git）

示例（在目标机器执行）：
```bash
TS=20260404_165800
OBJ=volces-tos:yupeng-dev-hb2/chentaoyu/xLLM_transfer
OUTDIR=data01/cty/Downloads
ARCHIVE=${OUTDIR}/xLLM_${TS}.tar.gz

rclone copy -P "${OBJ}/xLLM_${TS}.tar.gz" "${OUTDIR}"
rclone copy "${OBJ}/xLLM_${TS}.tar.gz.sha256" "${OUTDIR}"
(cd "${OUTDIR}" && sha256sum -c "xLLM_${TS}.tar.gz.sha256")

if [ -e data01/cty/xLLM ]; then
  mv data01/cty/xLLM "${OUTDIR}/xLLM.bak_${TS}"
fi
tar -C data01/cty -I pigz -xf "${ARCHIVE}"

cd data01/cty/xLLM
git rev-parse HEAD
```

## 验收清单
- 目标机 sha256sum -c 通过
- 目标机 data01/cty/xLLM 存在且文件数量合理
- 两端 git rev-parse HEAD 一致（如果迁移包含 .git）

