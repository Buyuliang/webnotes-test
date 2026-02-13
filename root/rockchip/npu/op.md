# op

# RK3588 NPU 支持算子
---

## 1. Add / Bias
- 支持：✅
- 数据类型：int8 / float16
- 广播：✅（支持 ONNX 四维广播）
- 量化：per-layer / per-channel
- 多核协同：✅

---

## 2. Sub
- 支持：✅
- 数据类型：int8 / float16
- 广播：✅
- 量化：per-layer / per-channel
- 多核协同：❌

---

## 3. Mul / Scale
- 支持：✅
- 数据类型：int8 / float16
- 广播：✅
- 量化：per-layer / per-channel
- 多核协同：❌

---

## 4. Div
- 支持：⚠ 部分支持
- 数据类型：float16
- 广播：✅（HW 广播仅 FP16）
- 量化：per-layer / per-channel
- 多核协同：❌

---

## 5. Max
- 支持：✅
- 数据类型：int8 / float16
- 广播：✅
- 量化：per-layer / per-channel
- 多核协同：❌

---

## 6. Min
- 支持：✅
- 数据类型：int8 / float16
- 广播：✅
- 量化：per-layer / per-channel
- 多核协同：❌

---

## 7. GlobalAveragePool
- 支持：✅
- 数据类型：int8 / float16
- batch：1
- 量化：per-layer
- 多核协同：❌

---

## 8. GlobalMaxPool
- 支持：✅
- 数据类型：int8 / float16
- batch：1
- 多核协同：❌

---

## 9. AveragePool
- 支持：✅
- 数据类型：int8 / float16
- kernel：[1,7]
- stride：[1,8]
- batch：1
- 多核协同：❌

---

## 10. MaxPool
- 支持：✅
- 数据类型：int8 / float16
- kernel：[1,7]
- stride：[1,8]
- batch：1
- 多核协同：❌

---

## 11. BatchNormalization
- 支持：✅
- 数据类型：int8 / float16
- batch：1
- 量化：per-layer / per-channel
- 多核协同：❌

---

## 12. LayerNormalization
- 支持：⚠ 部分支持
- 数据类型：float16
- normalized_shape：除 batch 外所有维度
- 多核协同：❌

---

## 13. Clip / ReLU6
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：✅

---

## 14. Elu
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：❌

---

## 15. Gelu
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：❌

---

## 16. Relu
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：✅

---

## 17. LeakyRelu
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：✅

---

## 18. PRelu
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：✅

---

## 19. GRU
- 支持：⚠ 部分支持
- 数据类型：float16
- batch：1
- 多核协同：❌

---

## 20. LSTM
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- batch > 1 需 4 的倍数
- 多核协同：❌

---

## 21. Concat
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- channel 方向需对齐
- 多核协同：✅

---

## 22. Mish
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：❌

---

## 23. Pad
- 支持：✅
- 数据类型：int8 / float16
- batch：1
- mode：constant
- 多核协同：❌

---

## 24. ReduceMean
- 支持：❌（CPU 实现）

---

## 25. ReduceSum
- 支持：❌（CPU 实现）

---

## 26. Resize
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- scale：1–8 整数倍
- mode：nearest / linear
- 多核协同：❌

---

## 27. Reshape
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- 多核协同：❌

---

## 28. ReverseSequence
- 支持：❌

---

## 29. Sigmoid
- 支持：✅
- 数据类型：int8 / float16

---

## 30. HardSigmoid
- 支持：✅
- 数据类型：int8 / float16

---

## 31. Swish
- 支持：✅
- 数据类型：int8 / float16

---

## 32. HardSwish
- 支持：✅
- 数据类型：int8 / float16

---

## 33. Softplus
- 支持：✅
- 数据类型：int8 / float16

---

## 34. Softmax
- 支持：✅
- 数据类型：int8 / float16
- axis：1 或 3

---

## 35. Slice
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- channel 对齐要求

---

## 36. Split
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- channel 对齐要求

---

## 37. Tanh
- 支持：✅
- 数据类型：int8 / float16

---

## 38. Transpose
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- 仅指定 perm 支持

---

## 39. Convolution
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：✅

---

## 40. Depthwise Convolution
- 支持：✅
- 数据类型：int8 / float16
- 多核协同：✅

---

## 41. ConvTranspose
- 支持：✅
- 数据类型：int8 / float16

---

## 42. Gemm
- 支持：❌（转 MatMul）

---

## 43. MatMul
- 支持：⚠ 部分支持
- 数据类型：int8 / float16
- 支持 4D

---

## 44. Expand
- 支持：✅

---

## 45. Where
- 支持：✅
- 数据类型：int8 / float16 / int64

---

## 46. exSoftmaxMask
- 支持：⚠ 部分支持

---

## 47. exGlu
- 支持：✅

---

## 48. 融合算子

### 支持
- Conv + Relu
- Conv + Clip
- Conv + PRelu
- Conv + LeakyRelu
- Conv + Add

### 不支持
- Conv + Mul
- ConvTranspose + *
- Depthwise + *