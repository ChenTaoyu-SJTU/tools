# Images Related

1. 常用的镜像仓库位于`https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest`
2. 服务器`910c_1`, `910c_2`, `910c_3`, `910c_4`的平台为 linux, aarch64

## 镜像下载的workflow
1. 首先在 Mac 上用 podman 下载镜像，镜像很大，这一步会耗时较长。
2. 然后用`podman save`保存镜像到`~/Downloads/images`目录，耗时同样较长。
3. 使用 `rclone`将`~/Downloads/images`目录下的`tar`文件像上传到对象存储, rclone手册位于`tools/remote_workflow/rclone.md`
4. 在服务器上从对象存储下载镜像
5. 在服务器上用`docker load`加载镜像
