# Rclone 中转速查

## 默认路径
- 默认对象存储中转目录：`volces-tos:yupeng-dev-hb2/chentaoyu`
- 推荐镜像目录：`volces-tos:yupeng-dev-hb2/chentaoyu/images`
- 默认下载服务器别名：`ark-910c-2-huabei2`
- 服务器默认下载目录：`/data01/cty/Downloads`

## 从本地上传到对象存储
用于将本地 Mac 上的文件或目录上传到 `volces-tos` 中转：

```bash
rclone copy /Users/dir volces-tos:yupeng-dev-hb2/chentaoyu/dir -P --s3-upload-cutoff 80M --s3-upload-concurrency 8 --s3-chunk-size 16M --s3-disable-checksum
```

## 从对象存储下载到服务器
用于在服务器上将对象存储内容下载到本地目录：

```bash
rclone copy -P --transfers=32 --checkers=32 volces-tos:yupeng-dev-hb2/chentaoyu/xxxx /data01/xxx
```

## 删除对象存储中的文件
用于清理对象存储上的单个文件：

```bash
rclone delete volces-tos:yupeng-dev-hb2/hefei/file.txt
```

## 创建镜像目录
如需在对象存储上存放镜像文件，可先创建推荐目录：

```bash
rclone mkdir volces-tos:yupeng-dev-hb2/chentaoyu/images
```
