# old-system-debian

Debian Stretch 已经 EOL（生命周期结束），官方主镜像已经删除，所以 apt update 会出现 404。

tretch 发布时间 2017，官方支持已结束，现在所有包被移动到 archive.debian.org。

所以需要把 apt 源换成 archive 源。

## 一、修改 Debian Stretch 源



```bash
编辑源文件：

nano /etc/apt/sources.list

把内容改成：

deb http://archive.debian.org/debian stretch main contrib non-free
deb http://archive.debian.org/debian-security stretch/updates main

保存退出。
```



## 二、关闭 Release 时间校验



```bash
因为 archive 仓库时间已经过期，需要关闭检查：

创建文件：

nano /etc/apt/apt.conf.d/99no-check-valid

写入：

Acquire::Check-Valid-Until "false";
Acquire::AllowInsecureRepositories "true";
Acquire::AllowDowngradeToInsecureRepositories "true";
```


## 三、更新 apt



```bash
执行：

apt update

如果出现 GPG 警告，可以忽略。

四、验证源是否生效

运行：

apt-cache policy

应该看到：

archive.debian.org
```


## 五、Stretch 还有一个坑



```bash
有时候 security 源会报错，可以直接删掉：

deb http://archive.debian.org/debian stretch main contrib non-free

只保留这一行。
```

