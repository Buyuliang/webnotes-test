# gpu-note

# Panfrost、Bifrost 和 Valhall 指南

## 概述

**Panfrost** 是开源 GPU 驱动

**Bifrost**、**Valhall** 是 ARM Mali GPU 的两代架构

Panfrost 同时支持 Bifrost 和 Valhall

---

## 一、Panfrost 是什么（驱动）

### ✅ Panfrost

- Mesa 的开源 Mali GPU 驱动
- 工作在：
  - Linux Kernel DRM
  - Mesa（OpenGL / OpenGL ES / Vulkan）
- 目标：替代 ARM 官方闭源 Mali 驱动

### ✅ 谁用它？

- Linux 桌面（Wayland / X11）
- 开源 SBC / 开发板
- ChromiumOS
- Android（AOSP + Mesa）

---

## 二、Bifrost 是什么（GPU 架构）

### ✅ Bifrost（Mali-G 系列）

**发布时间：** 2016–2019

**代表 GPU：**

| GPU | 常见 SoC |
|-----|----------|
| Mali-G31 | RK3326 / RK3566 |
| Mali-G52 | RK3399Pro / Amlogic |
| Mali-G76 | Kirin 980 |

### ✅ 特点

- 指令集 Bifrost ISA
- 支持 OpenGL ES 3.x
- Vulkan 1.0 / 1.1（驱动支持决定）

### ✅ 驱动支持情况

| 驱动 | 状态 |
|------|------|
| ARM 官方 | ✅ 完整 |
| Panfrost | ✅ 非常成熟 |

> 📌 **Bifrost 是 Panfrost 最成熟、最稳定的部分**

---

## 三、Valhall 是什么（GPU 架构）

### ✅ Valhall（Mali-G / Mali-G 系列）

**发布时间：** 2019–至今

**代表 GPU：**

| GPU | 常见 SoC |
|-----|----------|
| Mali-G57 | RK3568 |
| Mali-G610 | RK3588 |
| Mali-G710 | Dimensity / Exynos |
| Mali-G310 | 新低功耗 |

### ✅ 特点

- 全新 Valhall ISA
- 更适合并行计算
- 面向 Vulkan / 现代 GPU pipeline

### ✅ 驱动支持情况

| 驱动 | 状态 |
|------|------|
| ARM 官方 | ✅ 完整 |
| Panfrost | ⚠️ 发展中（已可用） |

> 📌 **Valhall 的 Panfrost 支持仍在快速演进**

---

## 四、三者的"使用场景"对照（重点）

### ✅ 场景 1：Linux 桌面 / Wayland / KDE

| GPU 架构 | 推荐驱动 |
|----------|----------|
| Bifrost | ✅ Panfrost（非常稳定） |
| Valhall | ✅ Panfrost（新内核 + 新 Mesa） |

> 📌 RK3566 / RK3588 用 Panfrost 已很常见

### ✅ 场景 2：Android 系统

| 架构 | 推荐 |
|------|------|
| Bifrost | ARM 官方 Mali |
| Valhall | ARM 官方 Mali |

> 📌 Android 上 Panfrost 仍是实验性质

### ✅ 场景 3：嵌入式 / 开源系统（Yocto / Buildroot）

| 架构 | 推荐 |
|------|------|
| Bifrost | ✅ Panfrost |
| Valhall | ✅ Panfrost（Mesa ≥ 23.x） |

### ✅ 场景 4：Vulkan / 3D / Compute

| 架构 | Panfrost 状态 |
|------|---------------|
| Bifrost | ✅ Vulkan 1.1 稳定 |
| Valhall | ⚠️ Vulkan 仍在补特性 |

### ✅ 场景 5：AI / NPU 以外的 GPGPU

- Panfrost ≠ CUDA / OpenCL
- Valhall 在 Vulkan Compute 上潜力更大
- 仍不适合重度 AI 推理

---

## 五、典型 SoC → 驱动选择（实战）

| SoC | GPU | 架构 | 建议 |
|-----|-----|------|------|
| RK3399 | Mali-G52 | Bifrost | ✅ Panfrost |
| RK3566 | Mali-G52 | Bifrost | ✅ Panfrost |
| RK3568 | Mali-G57 | Valhall | ✅ Panfrost |
| RK3588 | Mali-G610 | Valhall | ✅ Panfrost（新内核） |

---

## 六、版本要求（非常重要）

### ✅ 推荐组合

| GPU | Kernel | Mesa |
|-----|--------|------|
| Bifrost | ≥ 5.10 | ≥ 21.x |
| Valhall | ≥ 6.1 | ≥ 23.x |
