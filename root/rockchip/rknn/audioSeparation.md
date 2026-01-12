# audioSeparation

## 测试 spleeter

[spleeter GitHub 地址](https://github.com/deezer/spleeter.git)

### 2stems 测试：


```bash
python -m spleeter -p spleeter:2stems -o output audio_example.mp3
```

### 5stems 测试：


```bash
python -m spleeter separate -p spleeter:5stems  -o output-5stems/ ../yequ.mp3 
```

<span style="color: #ff0000">**注意：**
</span>模型放在 pretrained_models 目录下