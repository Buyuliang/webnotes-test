# GitHub Pages 记事本应用

一个基于 GitHub Pages 的网页记事本应用，支持文本编辑、文件上传和图片粘贴功能。上传的文件和图片会自动存储到 GitHub Release 中。

## 功能特性

- ✏️ **文本编辑**: 创建、编辑、删除笔记
- 📎 **文件上传**: 支持上传任意类型的文件到 GitHub Release
- 🖼️ **图片粘贴**: 支持直接粘贴图片（Ctrl+V），自动上传到 GitHub Release
- 🔍 **搜索功能**: 快速搜索笔记内容
- 💾 **自动保存**: 编辑内容自动保存到本地存储
- 📱 **响应式设计**: 支持桌面和移动设备

## 使用说明

### 1. 部署到 GitHub Pages

1. 将代码推送到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages
3. 选择主分支（main/master）作为源
4. 访问 `https://yourusername.github.io/repo-name` 查看应用

### 2. 配置 GitHub Token

在使用文件上传和图片粘贴功能前，需要配置 GitHub Personal Access Token：

1. 点击右上角的 **⚙️ 配置** 按钮
2. 获取 GitHub Personal Access Token：
   - 访问 [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
   - 点击 "Generate new token (classic)"
   - 勾选 `repo` 权限（需要完整仓库访问权限）
   - 生成并复制 Token（格式：`ghp_xxxxxxxxxxxx`）
3. 填写配置信息：
   - **GitHub Token**: 粘贴你的 Personal Access Token
   - **仓库名称**: 填写 `username/repo-name` 格式（例如：`octocat/Hello-World`）
4. 点击 **保存配置**

> ⚠️ **安全提示**: Token 会存储在浏览器本地存储中，请妥善保管。建议使用最小权限的 Token，仅用于此仓库。

### 3. 使用功能

#### 创建笔记
- 点击 **+ 新建笔记** 按钮创建新笔记
- 在侧边栏点击笔记标题打开笔记

#### 编辑笔记
- 在标题栏输入笔记标题
- 在编辑区输入笔记内容
- 支持富文本编辑（HTML 格式）
- 内容会自动保存（2秒延迟）

#### 上传文件
- 在编辑笔记时，点击 **📎 上传文件** 按钮
- 选择要上传的文件（支持多选）
- 文件会自动上传到 GitHub Release
- 上传的文件会显示在编辑器底部

#### 粘贴图片
- 在编辑区直接粘贴图片（Ctrl+V 或 Cmd+V）
- 图片会自动上传到 GitHub Release
- 图片会直接插入到编辑器中

#### 搜索笔记
- 在侧边栏搜索框输入关键词
- 实时过滤笔记列表

#### 删除笔记
- 在编辑笔记时，点击 **🗑️ 删除** 按钮
- 确认删除操作

## 文件结构

```
webnotes-test/
├── index.html          # 主页面
├── styles.css          # 样式文件
├── app.js              # 应用主逻辑
├── github-api.js       # GitHub API 交互
└── README.md           # 说明文档
```

## 技术实现

### 存储方式

- **笔记数据**: 存储在浏览器 `localStorage` 中
- **文件和图片**: 上传到 GitHub Release 的 `attachments` release 中

### GitHub API 使用

应用使用 GitHub REST API v3 来：
- 创建/获取 Release
- 上传文件到 Release
- 管理 Release Assets

### 浏览器兼容性

- Chrome/Edge (推荐)
- Firefox
- Safari
- 需要支持 ES6+ 和 Fetch API

## 注意事项

1. **Token 安全**: GitHub Token 存储在本地，请勿在公共设备上使用
2. **文件大小**: GitHub Release 单个文件限制为 2GB，但建议上传较小的文件
3. **Release 数量**: 应用会使用或创建一个名为 `attachments` 的 Release
4. **CORS 限制**: 由于使用 GitHub API，需要确保 Token 有正确的权限

## 故障排除

### 上传失败
- 检查 Token 是否有 `repo` 权限
- 确认仓库名称格式正确（`username/repo-name`）
- 检查网络连接

### 配置无法保存
- 确认 Token 格式正确（以 `ghp_` 开头）
- 确认仓库名称格式正确
- 检查浏览器控制台错误信息

### 图片无法粘贴
- 确认已配置 GitHub Token
- 检查图片格式是否支持
- 查看浏览器控制台错误信息

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
