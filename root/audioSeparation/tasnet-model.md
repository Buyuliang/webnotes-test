# tasnet-model

# TASNET 模型



```bash
class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N, audio_channels):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w
```

## 参数

| 参数 | 含义 |
| --- | --- |
| mixture | M = batch size（批大小）一次同时处理多少条音频 |
| T | 音频的采样点总数/这条音频一共有多少个数字 |
| K | 卷积在时间轴上走了多少步 K = {T - L}{L/2} + 1 |
| audio_channels | 输入通道 |
| N | 输出通道 = 卷积核数量 |
| kernel_size=L | 每次看L个点 |
| stride=L//2 | 每次移动L/2 |
| bias=False | 偏置=0 |

## 总结
原始声音，切成一小段一小段，用很多“模板”去匹配，看每一小段声音更像哪种模板。
想象你在分析一段音乐。

你准备了：

N 个“声音模板”
模板1：像鼓声
模板2：像人声
模板3：像钢琴
…
然后你做三件事

✅ 第一步：切片
把整段音频：

```bash
[整段声音]
```



切成：

```bash
[第一小段]
        [第二小段]
                [第三小段]
```

每段长度是 L

每次移动 L/2（50% 重叠）
```bash
|----|
    |----|
        |----|
```

✅ 第二步：拿模板去“对比”
对每一小段声音：

用模板1比一比 → 得到一个分数
用模板2比一比 → 得到一个分数
...
用模板N比一比 → 得到一个分数
这个“比一比”就是卷积。

✅ 第三步：只保留正分
```bash
F.relu(...)

```

意思是：

如果匹配分数 < 0 → 变成 0
如果 > 0 → 保留
意思就是：

只保留“匹配强度”，不要负数。

✅ 所以整个过程就是：
```bash
声音 → 切片 → 模板匹配 → 得到强度图
```

✅ 输出长什么样？
输出：

```bash
mixture_w: [M, N, K]
```



意思是：

M：几条音频
N：多少个模板
K：切了多少段时间
可以理解为：
```bash
        时间 →
模板1  [  2   5   1   0   3 ]
模板2  [  0   1   4   6   2 ]
模板3  [  3   0   2   1   0 ]
   ↓
  模板
```



这就是一张：

学习出来的“时频图”

✅ 那这些模板是什么？
在代码里：
```bash
self.conv1d_U = nn.Conv1d(...)
```



这些卷积核：

✅ 是自动学出来的

✅ 不再是固定的傅里叶基

✅ 比 STFT 更灵活

可以理解为：

网络自己发明了一套“声音字典”

✅ 再压缩成一句更简单的话
这个 Encoder 做的是：

把波形变成一种“可分解的表示”

就像：

STFT 把声音变成频谱
这个 Encoder 把声音变成 N 个基函数的组合强度
✅ 用 3 行话总结
1️⃣ 把音频切成很多小块

2️⃣ 用 N 个模板去匹配每一块

3️⃣ 得到一个“匹配强度图”

✅ 为什么要这样做？
因为后面要做：

语音分离

而分离更容易在：

✅ 这种“特征空间”做

❌ 原始波形直接做

✅ 超简单最终版本
这个类干的事：

把一维波形 → 变成一个“声音特征地图”

就像：

```bash
图片 → CNN → 特征图
声音 → Conv1d → 声音特征图
```











