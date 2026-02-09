# blade3-load

创建负载测试启动脚本## 标题


```bash
sudo vim /usr/local/bin/load-test.sh
```
### 脚本内容


```bash
#!/bin/bash
set -e

# systemd 环境下必须显式设置
export DISPLAY=:0
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# 强制 Mesa 使用 surfaceless EGL
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=0

echo "[load-test] Starting CPU stress..."
stress-ng --cpu 8 --cpu-method all &

echo "[load-test] Starting GPU stress..."
glmark2-es2 \
  --benchmark terrain \
  --run-forever \
  --off-screen \
  &

wait
```

### 赋予执行权限

```bash
sudo chmod +x /usr/local/bin/load-test.sh
```
## 创建 systemd 服务文件

```bash
sudo vim /etc/systemd/system/load-test.service
```

### service 内容

```bash
[Unit]
Description=CPU and GPU Load Test Service
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/load-test.sh

# 停止服务时，杀掉所有子进程
KillMode=control-group

# 自动重启（防止异常退出）
Restart=always
RestartSec=5

# 提高资源限制（防止被 systemd 限制）
LimitNOFILE=infinity
LimitNPROC=infinity

# 日志直接进 journalctl
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## 启用并启动服务


```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

sudo systemctl enable load-test.service
sudo systemctl start load-test.service
```
## 管理与验证

### 查看服务状态

```bash
systemctl status load-test.service
```
### 查看运行日志

```bash
journalctl -u load-test.service -f
```

### 停止负载测试

```bash
sudo systemctl stop load-test.service
```

### 查看 CPU GPU 负载


```bash
htop
cat /sys/class/devfreq/fb000000.gpu/load
```






