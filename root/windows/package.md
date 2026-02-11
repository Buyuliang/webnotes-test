# package

## 背景：


环境是 Windows Server（无 GUI，只能用 cmd）
那么我们要做的就是 在纯命令行下安装 Git

### 一、安装 Chocolatey（命令行即可）



```bash
# 在 cmd 里运行（PowerShell 也可以执行）：

@powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin
```

这条命令会下载并安装 Chocolatey，并把它的路径临时加到环境变量。

### 二、用 Chocolatey 安装 Git


```bash
choco install git -y
```



### 三、在当前 cmd 会话里刷新环境变量



```bash
refreshenv
```
### 四、验证安装


```bash
git --version
```