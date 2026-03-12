# ptrsc

## 一、最简单：scrot（命令行截图）



```bash
适合 轻量系统 / Linaro / 没有桌面工具栏。

安装：

apt install scrot
```

问题：
root@linaro-alip:/home/linaro/Desktop# scrot giblib error: Can't open X display. It *is* running, yeah?

```bash
普通用户下执行 
DISPLAY=:0 scrot
```

