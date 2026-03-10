# fstab

背景：
```bash
使用Debian系统，出现
Job dev-disk-by-partlabel-xxx.device/start running (1min 30s)
```
## 方法一（最推荐）：



```bash
修改 /etc/fstab

先查看：

cat /etc/fstab

可能会看到类似：

PARTLABEL=oem      /oem      ext4 defaults 0 2
PARTLABEL=userdata /userdata ext4 defaults 0 2




如果这些分区不存在，就会卡住。

解决

改成：

PARTLABEL=oem      /oem      ext4 defaults,nofail 0 2
PARTLABEL=userdata /userdata ext4 defaults,nofail 0 2




关键参数：

nofail
```



## 方法二：


```bash
完全注释掉

如果确定不用这些分区：

sudo nano /etc/fstab

注释：

# PARTLABEL=oem      /oem      ext4 defaults 0 2
# PARTLABEL=userdata /userdata ext4 defaults 0 2
```


## 方法三：


减少等待时间

systemd 默认等 90秒。

可以改为：

```bash
x-systemd.device-timeout=1
```



例如：

```bash
PARTLABEL=oem /oem ext4 defaults,nofail,x-systemd.device-timeout=1 0 2
```


## 方法四：


```bash
直接 mask device（不太推荐）

找到 device unit：

systemctl list-units | grep disk

例如：

dev-disk-by\x2dpartlabel-oem.device

然后：

sudo systemctl mask dev-disk-by\\x2dpartlabel-oem.device

```

