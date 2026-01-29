# upload

# Nextcloud WebDAV 文件上传指南

## 概述

本指南介绍如何在 Shell 中使用命令行将文件上传到 Nextcloud（silo.focalcrest.com）的 WebDAV 服务。

**目标目录：** `/tom/audioSeparation`

**WebDAV 地址：** `https://silo.focalcrest.com/remote.php/dav/files/<用户名>/`

---

## 一、前提条件

### ✅ 必需信息

- ✅ Nextcloud 账号用户名
- ✅ 密码 或 **App Password**（强烈推荐）
- ✅ 对目标目录有写权限

### 🔐 获取 App Password（推荐）

1. 登录 Nextcloud
2. 进入 **Settings → Security → App passwords**
3. 创建新的 App Password
4. 保存密码（只显示一次）

**优势：**
- 更安全（不需要主密码）
- 可以单独撤销
- 适合自动化脚本

---

## 二、WebDAV 地址格式

### 标准格式

```
https://silo.focalcrest.com/remote.php/dav/files/<用户名>/
```

### 重要提示

⚠️ **网页 URL 不等于 WebDAV 路径**

网页链接：
```
https://silo.focalcrest.com/apps/files/files/1451567?dir=/tom/audioSeparation
```

❌ **不能直接使用** `dir=/tom/audioSeparation` 作为 WebDAV 路径

✅ **需要先确认真实的 WebDAV 目录结构**

---

## 三、方式 1：使用 curl 上传（最简单）

### 3.1 上传单个文件

**基本命令：**

```bash
curl -u USERNAME \
  -T 文件名 \
  "https://silo.focalcrest.com/remote.php/dav/files/USERNAME/路径/文件名"
```

**示例：**

```bash
curl -u tom \
  -T test.wav \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation/test.wav"
```

**使用 App Password（推荐）：**

```bash
curl -u tom:APP_PASSWORD \
  -T test.wav \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation/test.wav"
```

### 3.2 批量上传

```bash
curl -u tom:APP_PASSWORD \
  -T *.wav \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation/"
```

### 3.3 成功/失败判断

- ✅ **成功：** 无输出（静默成功）
- ❌ **失败：** 显示 HTTP 状态码或错误信息

---

## 四、方式 2：使用 davfs2 挂载（像本地目录）

### 4.1 安装

```bash
sudo apt install davfs2
```

### 4.2 创建挂载点

```bash
mkdir -p ~/silo
```

### 4.3 挂载

```bash
sudo mount -t davfs \
  https://silo.focalcrest.com/remote.php/dav/files/USERNAME/ \
  ~/silo
```

输入用户名和密码后，即可像本地目录一样使用：

```bash
# 查看目录
ls ~/silo/tom/audioSeparation

# 复制文件
cp test.wav ~/silo/tom/audioSeparation/

# 使用其他命令
mv file.txt ~/silo/tom/audioSeparation/
```

### 4.4 卸载

```bash
sudo umount ~/silo
```

---

## 五、方式 3：使用 rclone（强烈推荐）

### 5.1 安装

```bash
sudo apt install rclone
```

### 5.2 配置（一次配置，永久使用）

```bash
rclone config
```

**配置选项：**

```
n) New remote
name> silo
Storage> webdav
URL> https://silo.focalcrest.com/remote.php/dav/files/USERNAME/
Vendor> nextcloud
User> USERNAME
Password> APP_PASSWORD
```

### 5.3 使用

**列出目录：**

```bash
rclone ls silo:
rclone ls silo:tom/audioSeparation
```

**上传文件：**

```bash
rclone copy test.wav silo:tom/audioSeparation
```

**上传整个目录：**

```bash
rclone copy /local/directory silo:tom/audioSeparation -P
```

**同步目录：**

```bash
rclone sync /local/directory silo:tom/audioSeparation -P
```

### 5.4 rclone 优势

- ✅ 稳定可靠
- ✅ 支持断点续传
- ✅ 支持同步整个目录
- ✅ 自动处理路径问题
- ✅ 进度显示（使用 `-P` 参数）

---

## 六、确认 WebDAV 目录结构

### 6.1 列出根目录

```bash
curl -u tom -X PROPFIND \
  -H "Depth: 1" \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/"
```

### 6.2 解析输出

在 XML 输出中查找 `<d:href>` 标签，例如：

```xml
<d:href>/remote.php/dav/files/tom/tom/</d:href>
<d:href>/remote.php/dav/files/tom/Documents/</d:href>
<d:href>/remote.php/dav/files/tom/audioSeparation/</d:href>
```

### 6.3 列出子目录

```bash
curl -u tom -X PROPFIND \
  -H "Depth: 1" \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/"
```

---

## 七、创建目录（如果不存在）

### 使用 MKCOL 方法

```bash
curl -u tom -X MKCOL \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation"
```

**创建多级目录：**

```bash
# 先创建父目录
curl -u tom -X MKCOL \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/"

# 再创建子目录
curl -u tom -X MKCOL \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation"
```

---

## 八、常见错误及解决方案

### 8.1 401 Unauthorized

**原因：**
- 用户名/密码错误
- 未使用 App Password

**解决方案：**
- 检查用户名和密码
- 使用 App Password 替代主密码

### 8.2 404 Not Found

**原因：**
- 路径写错
- 用户名不匹配
- 目录不存在

**解决方案：**
1. 使用 `PROPFIND` 确认真实路径
2. 检查用户名是否正确
3. 使用 `MKCOL` 创建目录

**示例错误：**

```xml
<?xml version="1.0" encoding="utf-8"?>
<d:error xmlns:d="DAV:" xmlns:s="http://sabredav.org/ns">
<s:exception>Sabre\DAV\Exception\NotFound</s:exception>
<s:message>File with name //audioSeparation could not be located</s:message>
</d:error>
```

**解决方法：**
- 确认路径是否为 `/tom/tom/audioSeparation/` 而不是 `/tom/audioSeparation/`
- 先列出目录确认结构

### 8.3 403 Forbidden

**原因：**
- 没有该目录的写权限

**解决方案：**
- 联系管理员授予权限
- 检查目录权限设置

### 8.4 路径不匹配问题

**网页 URL：**
```
https://silo.focalcrest.com/apps/files/files/1451567?dir=/tom/audioSeparation
```

**WebDAV 真实路径：**
```
/remote.php/dav/files/tom/tom/audioSeparation/
```

⚠️ **注意：** `dir=/tom/audioSeparation` 中的 `/tom/` 是子目录，不是用户名

---

## 九、实际案例

### 案例 1：上传图片文件

**错误命令：**

```bash
curl -u tom -T sample_1_digit_0_prediction.png \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/audioSeparation/sample.png"
```

**错误原因：** 路径错误，应该是 `/tom/tom/audioSeparation/`

**正确命令：**

```bash
curl -u tom -T sample_1_digit_0_prediction.png \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation/sample.png"
```

### 案例 2：确认目录结构

**步骤 1：列出根目录**

```bash
curl -u tom -X PROPFIND -H "Depth: 1" \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/"
```

**步骤 2：查找目标目录**

在输出中查找包含 `audioSeparation` 或 `tom` 的路径

**步骤 3：列出子目录（如果需要）**

```bash
curl -u tom -X PROPFIND -H "Depth: 1" \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/"
```

**步骤 4：使用正确路径上传**

---

## 十、推荐方案选择

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 临时上传 1-2 个文件 | `curl -T` | 简单快速，无需安装 |
| 经常需要上传文件 | `rclone` | 稳定可靠，支持断点续传 |
| 想当网盘使用 | `davfs2` | 像本地目录一样操作 |
| 自动化脚本 | `rclone` + App Password | 最稳定，支持批量操作 |

---

## 十一、最佳实践

### 11.1 使用 App Password

```bash
# 不推荐（使用主密码）
curl -u tom:主密码 ...

# 推荐（使用 App Password）
curl -u tom:APP_PASSWORD ...
```

### 11.2 先确认路径再上传

```bash
# 1. 先列出目录
curl -u tom -X PROPFIND -H "Depth: 1" \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/"

# 2. 确认路径后上传
curl -u tom -T file.txt \
  "https://silo.focalcrest.com/remote.php/dav/files/tom/tom/audioSeparation/file.txt"
```

### 11.3 使用 rclone 避免路径问题

```bash
# rclone 会自动处理路径
rclone copy file.txt silo:tom/audioSeparation
```

### 11.4 批量上传使用 rclone

```bash
# 上传整个目录
rclone copy /local/directory silo:tom/audioSeparation -P

# 同步目录（删除目标中不存在的文件）
rclone sync /local/directory silo:tom/audioSeparation -P
```

---

## 十二、快速参考

### 12.1 curl 快速上传

```bash
curl -u USERNAME:APP_PASSWORD \
  -T 文件名 \
  "https://silo.focalcrest.com/remote.php/dav/files/USERNAME/tom/audioSeparation/文件名"
```

### 12.2 rclone 快速上传

```bash
rclone copy 文件名 silo:tom/audioSeparation
```

### 12.3 确认路径

```bash
curl -u USERNAME -X PROPFIND -H "Depth: 1" \
  "https://silo.focalcrest.com/remote.php/dav/files/USERNAME/"
```

### 12.4 创建目录

```bash
curl -u USERNAME -X MKCOL \
  "https://silo.focalcrest.com/remote.php/dav/files/USERNAME/tom/audioSeparation"
```

---

## 十三、注意事项

1. ⚠️ **网页 URL ≠ WebDAV 路径**：网页中的 `dir=` 参数不能直接用作 WebDAV 路径
2. ✅ **先确认路径**：使用 `PROPFIND` 确认真实的目录结构
3. 🔐 **使用 App Password**：更安全，适合自动化
4. 📁 **路径层级**：注意 `/tom/tom/audioSeparation/` 中的两个 `tom`，第一个是用户名，第二个是子目录
5. ✅ **成功无输出**：curl 上传成功时通常没有输出，这是正常的

---

## 十四、故障排查流程

1. ✅ **确认认证**：使用 `PROPFIND` 列出目录，确认用户名密码正确
2. ✅ **确认路径**：在输出中查找目标目录的真实路径
3. ✅ **确认权限**：检查是否有写权限
4. ✅ **创建目录**：如果目录不存在，使用 `MKCOL` 创建
5. ✅ **使用 rclone**：如果路径问题复杂，使用 rclone 自动处理

---

## 总结

- ✅ **最简单**：`curl -T` 上传单个文件
- ✅ **最稳定**：`rclone` 配置一次，永久使用
- ✅ **最方便**：`davfs2` 挂载后像本地目录
- ✅ **最重要**：先确认 WebDAV 真实路径，不要直接使用网页 URL

**推荐工作流：**

1. 使用 `rclone config` 配置一次
2. 使用 `rclone copy` 上传文件
3. 使用 `rclone sync` 同步目录

这样最简单、最稳定、最不容易出错！
