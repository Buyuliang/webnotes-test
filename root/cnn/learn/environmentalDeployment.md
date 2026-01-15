# environmentalDeployment

## 一、macOS 环境总览（先给结论）



### ✅ 推荐方案（强烈）


```bash
Conda / Miniconda + Python 3.9/3.10 + CPU / MPS PyTorch

原因：

macOS 系统 Python 容易炸

pip +系统 Python 权限问题多

Conda 能 100% 隔离环境

PyTorch 官方支持 Apple MPS（Metal）
```


## 二、安装 Conda（推荐 Miniconda）


### 1️⃣ 下载 Miniconda



### 👉 打开官网

```bash
https://docs.conda.io/en/latest/miniconda.html

选择：

Apple Silicon (arm64) → M1 / M2 / M3

Intel (x86_64) → 老 Mac

文件名类似：

Miniconda3-latest-MacOSX-arm64.sh
```




### 2️⃣ 安装

```bash
bash Miniconda3-latest-MacOSX-arm64.sh


一路 yes
最后选择：

Do you wish the installer to initialize Miniconda3? [yes]

```



### 3️⃣ 验证

```bash
conda --version
python --version
```

## 三、创建实验专用环境（强烈建议）


### 4️⃣ 创建环境

```bash
conda create -n seg python=3.9
```


（3.9 / 3.10 都行，别用 3.12）

### 5️⃣ 激活环境

```bash
conda activate seg

终端前面应该看到：

(seg) tom@MacBook ~ %
```







## 四、安装 PyTorch（macOS 官方方式）


### 6️⃣ 安装 PyTorch（CPU / MPS）

```bash
Apple Silicon（推荐）
pip install torch torchvision torchaudio


PyTorch 会自动启用 MPS（Metal GPU）

Intel Mac（CPU）

pip install torch torchvision torchaudio
```

### 7️⃣ 验证 PyTorch

```bash
Apple Silicon：True ✅

Intel：False（正常）
```



## 五、安装实验所需库

```bash
pip install numpy matplotlib tqdm
```