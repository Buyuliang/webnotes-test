# GitHub API 测试命令

## 环境变量设置

```bash
# 设置 GitHub Token
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"

# 设置仓库名称
export GITHUB_REPO="username/repo-name"
```

## 基础测试命令

### 1. 获取仓库信息

```bash
curl -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}" | jq
```

### 2. 获取默认分支

```bash
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}" | jq -r '.default_branch'
```

### 3. 检查 notes-data.json 是否存在

```bash
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json?ref=main" | jq
```

### 4. 读取 notes-data.json 内容

```bash
# 获取文件信息（包含 SHA）
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json?ref=main" | jq -r '.content' | gbase64 -d

# 或者使用 base64 命令（Linux）
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json?ref=main" | jq -r '.content' | tr -d '\n' | base64 -d
```

### 5. 创建 notes-data.json 文件（首次创建）

```bash
# 准备内容（JSON 格式）
CONTENT='[]'

# 转换为 base64
ENCODED=$(echo -n "$CONTENT" | base64 | tr -d '\n')

# 创建文件
curl -X PUT \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     -H "Content-Type: application/json" \
     -d "{
       \"message\": \"创建笔记数据文件\",
       \"content\": \"${ENCODED}\",
       \"branch\": \"main\"
     }" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json" | jq
```

### 6. 更新 notes-data.json 文件

```bash
# 先获取文件的 SHA
SHA=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github.v3+json" \
            "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json?ref=main" | jq -r '.sha')

# 准备新内容
NEW_CONTENT='[{"id":"1","title":"测试笔记","content":"这是测试内容","createdAt":"2024-01-01T00:00:00.000Z","updatedAt":"2024-01-01T00:00:00.000Z","attachments":[]}]'

# 转换为 base64
ENCODED=$(echo -n "$NEW_CONTENT" | base64 | tr -d '\n')

# 更新文件
curl -X PUT \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     -H "Content-Type: application/json" \
     -d "{
       \"message\": \"更新笔记\",
       \"content\": \"${ENCODED}\",
       \"branch\": \"main\",
       \"sha\": \"${SHA}\"
     }" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json" | jq
```

### 7. 获取最新的 Release

```bash
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" | jq
```

### 8. 创建 Release（如果不存在）

```bash
curl -X POST \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     -H "Content-Type: application/json" \
     -d '{
       "tag_name": "attachments",
       "name": "附件存储",
       "body": "用于存储记事本上传的文件和图片",
       "draft": false,
       "prerelease": false
     }' \
     "https://api.github.com/repos/${GITHUB_REPO}/releases" | jq
```

### 9. 上传文件到 Release

```bash
# 先获取 Release 的 upload_url
UPLOAD_URL=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
                   -H "Accept: application/vnd.github.v3+json" \
                   "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" | jq -r '.upload_url' | sed 's/{.*}$//')

# 创建测试文件
echo "这是测试文件内容" > test.txt

# 上传文件
curl -X POST \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     -H "Content-Type: text/plain" \
     --data-binary @test.txt \
     "${UPLOAD_URL}?name=test.txt" | jq
```

### 10. 删除 Release Asset

```bash
# 获取 Asset ID
ASSET_ID=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
                 -H "Accept: application/vnd.github.v3+json" \
                 "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" | jq -r '.assets[] | select(.name=="test.txt") | .id')

# 删除 Asset
curl -X DELETE \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/releases/assets/${ASSET_ID}"
```

## 完整测试流程

### 测试文件读写

```bash
#!/bin/bash

# 设置变量
GITHUB_TOKEN="your_token"
GITHUB_REPO="username/repo-name"
BRANCH="main"

# 1. 检查文件是否存在
echo "检查文件..."
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json?ref=${BRANCH}" > /tmp/file_info.json

SHA=$(jq -r '.sha // empty' /tmp/file_info.json)

if [ -z "$SHA" ] || [ "$SHA" = "null" ]; then
    echo "文件不存在，创建新文件..."
    # 创建文件
    ENCODED=$(echo -n '[]' | base64 | tr -d '\n')
    curl -X PUT \
         -H "Authorization: token ${GITHUB_TOKEN}" \
         -H "Accept: application/vnd.github.v3+json" \
         -H "Content-Type: application/json" \
         -d "{\"message\":\"创建笔记数据\",\"content\":\"${ENCODED}\",\"branch\":\"${BRANCH}\"}" \
         "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json"
else
    echo "文件存在，SHA: ${SHA}"
    # 读取内容
    jq -r '.content' /tmp/file_info.json | tr -d '\n' | base64 -d | jq
fi
```

## 常见问题

### Windows PowerShell 版本

```powershell
# 设置变量
$env:GITHUB_TOKEN = "your_token"
$env:GITHUB_REPO = "username/repo-name"

# 获取仓库信息
$headers = @{
    "Authorization" = "token $env:GITHUB_TOKEN"
    "Accept" = "application/vnd.github.v3+json"
}
Invoke-RestMethod -Uri "https://api.github.com/repos/$env:GITHUB_REPO" -Headers $headers
```

### Base64 编码（Windows）

```powershell
# PowerShell 中编码
$content = '[]'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($content)
$encoded = [Convert]::ToBase64String($bytes)
```

## 注意事项

1. **Token 权限**：确保 Token 有 `repo` 权限
2. **Base64 编码**：GitHub API 要求内容必须是 base64 编码
3. **SHA 值**：更新文件时必须提供正确的 SHA 值
4. **分支名称**：确保使用正确的分支名称（main 或 master）
5. **Content-Type**：上传文件时注意设置正确的 Content-Type

## 调试技巧

```bash
# 查看完整响应（包括 HTTP 状态码）
curl -v -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}"

# 只查看 HTTP 状态码
curl -s -o /dev/null -w "%{http_code}" \
     -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/${GITHUB_REPO}/contents/notes-data.json?ref=main"
```
