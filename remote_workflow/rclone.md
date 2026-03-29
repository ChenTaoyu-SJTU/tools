# 背景
我本地的电脑是macos，服务器是linux，想要上传和下载文件，使用rclone工具进行中转。
文件先需要从mac上传到volces-tos(对象存储)，然后从volces-tos下载到服务器上。
我使用的服务器是`180.184.87.219`, 用户是root。已经配好了免密登陆。
ssh到服务器上的命令为: `ssh ark-910c-2-huabei2`, 相关配置写在了~/.ssh/config里。


# 从本地上传
rclone copy /Users/dir volces-tos:yupeng-dev-hb2/chentaoyu/dir -P --s3-upload-cutoff 80M --s3-upload-concurrency 8 --s3-chunk-size 16M --s3-disable-checksum

# 从服务器下载(注意xxx如果是目录要对应起来)
rclone copy -P --transfers=32 --checkers=32 volces-tos:yupeng-dev-hb2/chentaoyu/xxxx /data01/xxx

# 删除
rclone delete volces-tos:yupeng-dev-hb2/hefei/file.txt

# 在chentaoyu目录下创建images文件夹
rclone mkdir volces-tos:yupeng-dev-hb2/chentaoyu/images

# 在volces-tos上使用的默认中转目录
我在volces-tos上使用的默认中转目录是:`rclone lsf volces-tos:yupeng-dev-hb2/chentaoyu`

# 在服务器上的下载目录
在`ark-910c-2-huabei2`服务器上用`/data01/cty/Downloads`来存储从rclone上下载过来的东西。