# GitHub Pages 记事本应用

一个基于 GitHub Pages 的网页记事本应用，支持文本编辑、文件上传和图片粘贴功能。所有数据直接存储在 GitHub 仓库中。

## 功能特性

- ✏️ **Markdown 编辑**: 支持 Markdown 格式编辑，实时预览
- 📎 **文件上传**: 支持上传任意类型的文件到仓库目录
- 🖼️ **图片粘贴**: 支持直接粘贴图片（Ctrl+V），自动上传到仓库并插入 Markdown 格式
- 🔍 **搜索功能**: 快速搜索笔记内容
- 💾 **自动保存**: 编辑内容自动保存到本地存储
- 🔄 **自动同步**: 每 30 秒自动同步到 GitHub，无需手动操作
- 🌐 **跨设备同步**: 笔记数据存储在 GitHub 仓库，支持多设备访问
- 📱 **响应式设计**: 支持桌面和移动设备

## 使用说明

### 0. 本地调试

在部署到 GitHub Pages 之前，可以在本地调试：

#### 🚀 最简单的方法：VS Code Live Server
1. 安装 VS Code
2. 安装 "Live Server" 扩展（搜索 "Live Server"）
3. 右键点击 `index.html` → 选择 "Open with Live Server"
4. 浏览器会自动打开

#### ⚡ 快速启动脚本
- **Windows**: 双击 `start-server.bat` 文件
- **Linux/Mac**: 
  ```bash
  chmod +x start-server.sh
  ./start-server.sh
  ```

#### 📝 手动启动（需要 Python）
```bash
# 进入项目目录
cd g:\webnotes-test

# 启动服务器
python -m http.server 8000

# 然后在浏览器访问: http://localhost:8000
```

#### 💡 如果没有 Python
- 使用 VS Code Live Server（推荐，无需安装）
- 或直接打开 `index.html`（功能可能受限）

> 详细说明请查看 `本地调试说明.md` 或 `本地调试-无需安装.md` 文件

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

#### 同步笔记
- 点击 **🔄 同步** 按钮从 GitHub 同步笔记数据
- 首次配置 GitHub 后会自动从 GitHub 加载笔记
- 笔记数据存储在仓库的 `notes/` 目录中，每个笔记是一个独立的 Markdown 文件
- 支持多设备访问，数据自动同步

#### 创建笔记
- 点击 **+ 新建笔记** 按钮创建新笔记
- 在侧边栏点击笔记标题打开笔记

#### 编辑笔记
- 在标题栏输入笔记标题
- 在编辑区输入 Markdown 格式文本
- 点击 **👁️ 预览** 按钮切换预览模式
- 支持 Markdown 语法：标题、列表、代码块、链接、图片等
- 内容会自动保存到本地（1秒延迟）
- **自动同步**：每 30 秒自动同步到 GitHub，无需手动操作
- 页面切换或关闭时也会自动同步

#### 上传文件
- 在编辑笔记时，点击 **📎 上传文件** 按钮
- 选择要上传的文件（支持多选）
- 文件会自动上传到仓库的 `{笔记名}/files/` 目录
- 上传的文件会显示在编辑器底部

#### 粘贴图片
- 在编辑区直接粘贴图片（Ctrl+V 或 Cmd+V）
- 图片会自动上传到仓库的 `{笔记名}/image/` 目录
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

- **笔记数据**: 
  - 本地存储：浏览器 `localStorage`（快速访问，用于兼容）
  - 云端存储：GitHub 仓库的 `notes/` 目录（跨设备同步）
  - 每个笔记存储为独立的 Markdown 文件（`.md`）
  - 自动同步：保存笔记时直接保存到对应的文件
- **文件和图片**: 
  - 文件存储在 `{笔记名}/files/` 目录
  - 图片存储在 `{笔记名}/image/` 目录（以时间戳命名）
  - 例如：笔记 `note1.md` 的文件在 `note1/files/`，图片在 `note1/image/`

### GitHub API 使用

应用使用 GitHub REST API v3 来：
- 读取/写入仓库文件（笔记、文件和图片）
- 管理仓库目录结构
- 同步笔记数据

### Markdown 支持

- 使用 [marked.js](https://marked.js.org/) 库渲染 Markdown
- 支持标准 Markdown 语法
- 实时预览功能
- 图片粘贴自动转换为 Markdown 格式

### 自动同步机制

- **智能同步**：只在内容真正改变时才创建 commit，避免产生大量无用 commit
- **定时同步**：每 60 秒自动检查并同步到 GitHub（如果内容有改变）
- **页面可见性同步**：页面从隐藏状态恢复时自动同步
- **页面关闭同步**：页面关闭前自动同步
- **静默同步**：同步过程不显示加载提示，不影响使用体验
- **简洁 Commit**：使用简洁的 commit 消息（"更新笔记"），减少仓库历史噪音

### 浏览器兼容性

- Chrome/Edge (推荐)
- Firefox
- Safari
- 需要支持 ES6+ 和 Fetch API

## 注意事项

1. **Token 安全**: GitHub Token 存储在本地，请勿在公共设备上使用
2. **文件大小**: GitHub 单个文件限制为 100MB，但建议上传较小的文件
3. **存储结构**: 
   - 笔记文件存储在仓库根目录（如 `note1.md`）
   - 文件存储在 `{笔记名}/files/` 目录
   - 图片存储在 `{笔记名}/image/` 目录（以时间戳命名）
4. **CORS 限制**: 由于使用 GitHub API，需要确保 Token 有正确的权限
5. **数据同步**: 
   - 笔记数据存储在仓库的 `notes/` 目录中，每个笔记是一个独立的 Markdown 文件
   - 首次配置后会自动从 GitHub 加载笔记
   - **自动同步**：每 30 秒自动同步到 GitHub，无需手动操作
   - 页面切换或关闭时也会自动同步
   - 点击"同步"按钮可以手动立即同步（智能合并本地和远程数据）
   - 代码会自动检测仓库的默认分支（main 或 master）
6. **Markdown 编辑**:
   - 支持标准 Markdown 语法
   - 点击"预览"按钮查看渲染效果
   - 粘贴图片会自动转换为 Markdown 格式

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
