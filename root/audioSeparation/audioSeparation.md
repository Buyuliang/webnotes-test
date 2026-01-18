# audioSeparation

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

## 环境配置

```bash
#!/bin/bash
# ============================================================
# 脚本功能：创建 Python 3.7 + TF1 + PyTorch + ONNX + Spleeter 环境
# 适用：Linux / conda 环境
# ============================================================

# 设置环境名
ENV_NAME="spleeter"

echo "==== 创建 conda 环境 ===="
conda create -n $ENV_NAME python=3.7 -y
conda activate $ENV_NAME

echo "==== 升级 pip ===="
pip install --upgrade pip

echo "==== 安装 TensorFlow 1.15.5 ===="
pip install tensorflow==1.15.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==== 安装 protobuf 3.20.3 ===="
pip uninstall -y protobuf
pip install protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==== 安装 PyTorch 1.13.1 CPU 版本 ===="
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
  --index-url https://download.pytorch.org/whl/cpu \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==== 安装 ONNX 及相关工具 ===="
pip install onnx==1.14.1 \
            onnxmltools==1.12.0 \
            onnxconverter_common==1.13.0 \
            onnxruntime==1.14.1 \
            -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==== 安装 ffmpeg（系统 + Python） ===="
conda install -y -c conda-forge ffmpeg
pip install ffmpeg-python -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install soundfile \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==== 安装辅助依赖 ===="
pip install packaging six typing_extensions -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "==== 验证安装 ===="
python - <<EOF
import tensorflow as tf
import torch
import onnx
import onnxmltools
import onnxconverter_common
import onnxruntime as ort
import packaging, six, typing_extensions

print("TF:", tf.__version__)
print("Torch:", torch.__version__)
print("Torch nn:", torch.nn)
print("ONNX:", onnx.__version__)
print("onnxmltools:", onnxmltools.__version__)
print("onnxconverter_common:", onnxconverter_common.__version__)
print("onnxruntime:", ort.__version__)
print("packaging:", packaging.__version__)
print("six:", six.__version__)
print("typing_extensions:", typing_extensions.__version__)
EOF

echo "==== 环境创建完成 ===="
echo "激活环境：conda activate $ENV_NAME"

```



