# tasnet-model-TemporalConvNet

```bash
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y
```

## 参数
| 参数 | 含义 |
| --- | --- |
| M | batch |
| N | 通道数 |
| K  | 时间长度 |

## 简述

```bash
固定时间
在不同通道之间做归一化
```
✅ 举一个超小例子

```bash
M=1
N=2
K=3
```
输入：

$y =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}$

shape:
```bash
[1,2,3]
```

| 通道1 | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- |
| 通道2 | 4 | 5 | 6| 

## ✅ 四、第一步：计算 mean

```bash
mean = torch.mean(y, dim=1)
```

✅ dim=1 表示：

在通道维度求平均

也就是：

每个时间点单独算。

✅ 时间点 t=1
$$[1,4]$$
平均：
$$(1+4)/2 = 2.5$$
✅ t=2
$$(2+5)/2 = 3.5$$

✅ t=3
$$(3+6)/2 = 4.5$$

✅ mean 结果：
$$mean =
\begin{bmatrix}
2.5 & 3.5 & 4.5
\end{bmatrix}$$

shape:
```bash
[1,1,3]
```

## ✅ 五、第二步：计算方差

```bash
var = torch.var(y, dim=1)
```

✅ t=1
数据：
$$[1,4]$$
方差：
$$((1-2.5)^2 + (4-2.5)^2)/2$$
$$= (2.25 + 2.25)/2
= 2.25$$

✅ t=2
一样结构：
$$2.25$$

✅ t=3
也是：
$$2.25$$

✅ var：
$$[2.25, 2.25, 2.25]$$

## ✅ 六、第三步：标准化

公式：
$\hat y = \frac{y - mean}{\sqrt{var + \epsilon}}$

标准差：
$$\sqrt{2.25} = 1.5$$

✅ t=1
原始：
$$[1,4]$$
减均值：
$$[-1.5, 1.5]$$
除以1.5：
$$[-1, 1]$$

✅ t=2
$$[2,5]$$
→
$$[-1,1]$$

✅ t=3
$$[-1,1]$$

✅ 结果：
$$\hat y =
\begin{bmatrix}
-1 & -1 & -1 \\
1 & 1 & 1
\end{bmatrix}$$

## ✅ 七、gamma 和 beta
```bash
self.gamma = [1,N,1]
self.beta = [1,N,1]
```

初始：
```bash
gamma = 1
beta = 0
```

所以现在输出不变。
但训练时会变成：
$$output = \gamma \hat y + \beta$$
比如：
假设：
$$\gamma =
\begin{bmatrix}
2 \\
0.5
\end{bmatrix}$$
那输出变成：
通道1：
$$-1 × 2 = -2$$
通道2：
$$1 × 0.5 = 0.5$$

## ✅ 八、矩阵形式总结

在时间 t：
设：
$$y_t =
\begin{bmatrix}
y_{1t} \\
y_{2t} \\
\vdots \\
y_{Nt}
\end{bmatrix}$$

均值：
$$\mu_t = \frac{1}{N} \mathbf{1}^T y_t$$

标准化：
$$\hat y_t =
\frac{y_t - \mu_t \mathbf{1}}{\sqrt{\sigma_t^2}}$$

再乘：
$$output_t =
\Gamma \hat y_t + \beta$$
其中：
$$\Gamma = diag(\gamma_1,...,\gamma_N)$$

✅ 九、它到底是干嘛的？（直觉解释）
它做的是：

在每个时间点，把不同通道“拉到同一尺度”。

# TemporalBlock

```python
class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding,
                                        dilation, norm_type, causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)
```


## ✅ 一、它整体在做什么？（一句话）



对时间序列做一次“非线性变换 + 时序卷积”，
然后再加回原输入（残差连接）。

公式就是：
$$\text{输出} = F(x) + x$$

## ✅ 二、设定极小参数（方便手算）


设：
```bash
M = 1
B = 2   (in_channels)
H = 2   (out_channels)
K = 4   (时间长度)
kernel_size = 3
dilation = 1
padding = 1
```
输入：
$$x =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
2 & 1 & 0 & 1
\end{bmatrix}$$

## ✅ 三、第一步：1×1 卷积（通道变换）

```bash
conv1x1 = nn.Conv1d(in_channels=2, out_channels=2, 1)
```

这是：

每个时间点做一次矩阵乘法


✅ 设权重
$$W_{1×1} =
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}$$

✅ 在每个时间点做乘法
t=1
输入：
$$[1,2]$$
输出：
$$\begin{aligned}
y_1 &= 1×1 + 1×2 = 3 \\
y_2 &= 0×1 + 1×2 = 2
\end{aligned}$$

t=2
$$[2,1]$$
$$[3,1]$$

t=3
$$[3,0]$$
$$[3,0]$$

t=4
$$[4,1]$$
$$[5,1]$$

✅ 结果：
$$\begin{bmatrix}
3 & 3 & 3 & 5 \\
2 & 1 & 0 & 1
\end{bmatrix}$$

## ✅ 四、PReLU


PReLU：
$$f(x) =
\begin{cases}
x & x>0 \\
a x & x<0
\end{cases}$$
现在没有负数 → 不变。

## ✅ 五、归一化（假设不变）


为了简化演算，假设 norm 输出一样。



## ✅ 六、Depthwise Separable Conv


这是关键。
它分两步：

✅ Depthwise（每个通道单独卷积）
✅ Pointwise（1×1混合通道）


## ✅ 七、Depthwise 卷积


kernel=3, padding=1

✅ 设卷积核
通道1核：
$$[1,0,-1]$$
通道2核：
$$[1,1,1]$$

✅ 通道1卷积
输入：
$$[3,3,3,5]$$
padding后：
$$[0,3,3,3,5,0]$$

t1
$$0×1 + 3×0 + 3×(-1)
= -3$$
t2
$$3×1 + 3×0 + 3×(-1)
= 0$$
t3
$$3×1 + 3×0 + 5×(-1)
= -2$$
t4
$$3×1 + 5×0 + 0×(-1)
= 3$$

✅ 通道1输出：
$$[-3,0,-2,3]$$

✅ 通道2卷积
输入：
$$[2,1,0,1]$$
padding：
$$[0,2,1,0,1,0]$$

t1
$$0+2+1=3$$
t2
$$2+1+0=3$$
t3
$$1+0+1=2$$
t4
$$0+1+0=1$$

✅ 通道2输出：
$$[3,3,2,1]$$

## ✅ 八、Pointwise 1×1（通道混合）


设权重：
$$W_{pw} =
\begin{bmatrix}
1 & 0 \\
1 & 1
\end{bmatrix}$$

t1
输入：
$$[-3,3]$$
输出：
$$\begin{aligned}
y_1 &= -3 \\
y_2 &= -3+3=0
\end{aligned}$$

对全部时间点算完：
结果：
$$out =
\begin{bmatrix}
-3 & 0 & -2 & 3 \\
0 & 3 & 0 & 4
\end{bmatrix}$$

## ✅ 九、残差连接


原输入：
$$\begin{bmatrix}
1 & 2 & 3 & 4 \\
2 & 1 & 0 & 1
\end{bmatrix}$$

相加：

通道1：
$$[-2, 2, 1, 7]$$
通道2：
$$[2, 4, 0, 5]$$

✅ 最终输出：
$$\begin{bmatrix}
-2 & 2 & 1 & 7 \\
2 & 4 & 0 & 5
\end{bmatrix}$$

## ✅ 十、它到底在干嘛？（通俗理解）


TemporalBlock 做了三件事：

✅ 1️⃣ 通道混合
1×1卷积：

把不同通道的信息混在一起


✅ 2️⃣ 时间建模
Depthwise 卷积：

看当前时间附近的点


✅ 3️⃣ 保留原信息
残差：
$$output = F(x) + x$$
作用：
✅ 防止梯度消失
✅ 保留原始特征
✅ 更稳定

## ✅ 十一、用一句话总结


TemporalBlock 就是：

先重新组合通道 →
再在时间上做卷积 →
再把原始输入加回来。


## ✅ 十二、为什么要残差？


如果没有残差：
深层网络容易：

信息丢失
梯度消失
训练困难

加残差等于：

给网络一条“高速公路”


## ✅ 十三、结构本质


$$x \rightarrow 1×1 → 激活 → 卷积 → 1×1 → +x$$
本质是：

学习一个“增量修正”


## ✅ 十四、直觉类比


把 x 看成：

原始声音特征

TemporalBlock 学的是：

“我帮你微调一下”

不是重做，而是“修正”。

✅ 最终总结（超直观）
TemporalBlock 是：
✅ 一个小型时序处理单元
✅ 能看局部时间
✅ 能混合通道
✅ 能稳定训练
它是 Conv‑TasNet 的“核心积木”。

```bash
class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(B,
                                  H,
                                  P,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  norm_type=norm_type,
                                  causal=causal)
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net,
                                     mask_conv1x1)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask
```
## ✅ 一、整体流程（先建立大图）


输入：
$$mixture\_w: [M, N, K]$$
含义：

M = batch
N = 编码器滤波器数（特征通道）
K = 时间长度

输出：
$$est\_mask: [M, C, N, K]$$

C = 说话人数量

整体流程：
```bash
LayerNorm
→ 1x1 Bottleneck
→ 多个 TemporalBlock (带 dilation)
→ 1x1
→ reshape
→ 生成 mask
```
## ✅ 二、设一个超级小例子
```bash
M = 1
N = 2
B = 2
H = 2
P = 3
X = 2
R = 1
C = 2
K = 4
```

输入：
$$mixture_w =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
2 & 1 & 0 & 1
\end{bmatrix}$$

## ✅ 三、第一步：ChannelwiseLayerNorm

我们之前算过：
对每个时间点做通道归一化。

例如 t=1：
$$[1,2]$$
mean = 1.5
var = 0.25
标准化：
$$[-1,1]$$
假设最终得到：
$$\begin{bmatrix}
-1 & -1 & 1 & 1 \\
1 & 1 & -1 & -1
\end{bmatrix}$$
（简化结果）

✅ 含义：

消除通道之间尺度差异
## ✅ 四、Bottleneck 1×1卷积

```bash
[N=2] → [B=2]
```
就是：
每个时间点做矩阵乘法。
假设权重：
$$W =
\begin{bmatrix}
1 & 0 \\
1 & 1
\end{bmatrix}$$

例如 t=1：
输入：
$$[-1,1]$$
输出：
$$\begin{aligned}
y_1 &= -1 \\
y_2 &= -1+1=0
\end{aligned}$$

✅ 含义：

把编码器特征重新组合

## ✅ 五、Temporal Blocks（核心）

因为：
```bash
X=2
R=1
```



所以：
我们有 2 个 block：

第1个 dilation=1
第2个 dilation=2


✅ Block1（d=1）
看：
```bash
t-1, t, t+1
```



✅ Block2（d=2）
看：
```bash
t-2, t, t+2
```




✅ 两个叠加后：
感受野：
$$1 + 2(1+2) = 7$$
意思：

每个时间点可以看到前后3个时间点


✅ 物理意义：
网络可以利用更长时间上下文来判断：

谁在说话
声音连续性
音调变化

## ✅ 六、mask 1×1 卷积


```bash
[B=2] → [C*N=4]
```



因为：
```bash
C=2
N=2
```



所以：
输出通道数 = 4

假设某时间点输出：
$$[1,2,3,4]$$
reshape 成：
$$[2,2]$$
表示：
```bash
说话人1 mask: [1,2]
说话人2 mask: [3,4]
```
## ✅ 七、ReLU 或 Softmax

如果用 ReLU：
$$mask = max(0, score)$$
保证 mask ≥ 0
如果用 softmax：
$$mask_1 + mask_2 = 1$$
表示“分配比例”

✅ 含义：
mask 是：

告诉模型每个说话人在每个时间、每个通道占多少比例。


## ✅ 八、整个网络在干嘛？（最通俗解释）


TCN 做的是：

观察整段时间的编码特征
学习每个说话人的“分离规则”
输出每个说话人的掩码


## ✅ 九、再通俗一点


想象：
输入是：
```bash
两个人同时说话的混合特征
```



TCN 学的是：
```bash
哪些特征属于说话人1？
哪些属于说话人2？
```



输出：
```bash
两张“滤镜”
```



一乘就分开。

## ✅ 十、参数意义解释



| 参数 | 意义 |
| --- | --- |
| N | 编码特征维度 | 
| B | bottleneck通道 |
| H | 隐藏层通道 |
| P | 卷积核大小 |
| X | 每组block数量 |
| R | 重复次数 |
| C | 说话人数量 |

## ✅ 十一、为什么要 R 次 repeat？


每个 repeat 感受野：
$$≈ (P-1)(2^X -1)$$
多次 repeat：

✅ 增强建模能力

✅ 让网络更深

✅ 提取更复杂模式

## ✅ 十二、整体数学表达


$$mask = f(mixture_w)$$
其中：
$$f = LN → 1×1 → TCN(dilated) → 1×1$$

## ✅ 十三、终极一句话总结


TemporalConvNet 是：

一个带扩张卷积的深层时序网络，
用来学习如何把混合语音拆成多个说话人的掩码。

