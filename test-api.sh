#!/bin/bash

# GitHub API 测试脚本
# 使用方法：
# 1. 设置环境变量
#    export GITHUB_TOKEN="your_token_here"
#    export GITHUB_REPO="username/repo-name"
# 2. 运行脚本: bash test-api.sh

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查环境变量
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}错误: 请设置 GITHUB_TOKEN 环境变量${NC}"
    echo "export GITHUB_TOKEN=\"your_token_here\""
    exit 1
fi

if [ -z "$GITHUB_REPO" ]; then
    echo -e "${RED}错误: 请设置 GITHUB_REPO 环境变量${NC}"
    echo "export GITHUB_REPO=\"username/repo-name\""
    exit 1
fi

BASE_URL="https://api.github.com"
REPO_URL="${BASE_URL}/repos/${GITHUB_REPO}"

echo -e "${BLUE}=== GitHub API 测试 ===${NC}\n"
echo "仓库: ${GITHUB_REPO}"
echo "Token: ${GITHUB_TOKEN:0:10}...\n"

# 1. 获取仓库信息
echo -e "${GREEN}1. 获取仓库信息${NC}"
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "${REPO_URL}" | jq -r '.name, .default_branch, .full_name' || echo "失败"
echo ""

# 2. 获取默认分支
echo -e "${GREEN}2. 获取默认分支${NC}"
DEFAULT_BRANCH=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
                      -H "Accept: application/vnd.github.v3+json" \
                      "${REPO_URL}" | jq -r '.default_branch')
echo "默认分支: ${DEFAULT_BRANCH}"
echo ""

# 3. 检查 notes-data.json 文件是否存在
echo -e "${GREEN}3. 检查 notes-data.json 文件${NC}"
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "${REPO_URL}/contents/notes-data.json?ref=${DEFAULT_BRANCH}" | jq -r '.sha // "文件不存在"' || echo "文件不存在"
echo ""

# 4. 获取最新的 Release
echo -e "${GREEN}4. 获取最新的 Release${NC}"
curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/vnd.github.v3+json" \
     "${REPO_URL}/releases/latest" | jq -r '.tag_name // "没有 Release"' || echo "没有 Release"
echo ""

# 5. 创建或获取 Release（用于文件上传）
echo -e "${GREEN}5. 创建/获取 Release${NC}"
LATEST_RELEASE=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
                       -H "Accept: application/vnd.github.v3+json" \
                       "${REPO_URL}/releases/latest")

RELEASE_ID=$(echo "$LATEST_RELEASE" | jq -r '.id // empty')

if [ -z "$RELEASE_ID" ] || [ "$RELEASE_ID" = "null" ]; then
    echo "创建新的 Release..."
    NEW_RELEASE=$(curl -s -X POST \
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
                       "${REPO_URL}/releases")
    RELEASE_ID=$(echo "$NEW_RELEASE" | jq -r '.id')
    UPLOAD_URL=$(echo "$NEW_RELEASE" | jq -r '.upload_url' | sed 's/{.*}$//')
    echo "Release ID: ${RELEASE_ID}"
    echo "Upload URL: ${UPLOAD_URL}"
else
    UPLOAD_URL=$(echo "$LATEST_RELEASE" | jq -r '.upload_url' | sed 's/{.*}$//')
    echo "使用现有 Release ID: ${RELEASE_ID}"
    echo "Upload URL: ${UPLOAD_URL}"
fi
echo ""

# 6. 测试上传文件到 Release（需要实际文件）
echo -e "${GREEN}6. 测试上传文件到 Release${NC}"
echo "提示: 需要准备一个测试文件，例如: echo 'test' > test.txt"
echo "上传命令示例:"
echo "curl -X POST \\"
echo "  -H \"Authorization: token \${GITHUB_TOKEN}\" \\"
echo "  -H \"Accept: application/vnd.github.v3+json\" \\"
echo "  -H \"Content-Type: text/plain\" \\"
echo "  --data-binary @test.txt \\"
echo "  \"\${UPLOAD_URL}?name=test.txt\""
echo ""

# 7. 读取 notes-data.json（如果存在）
echo -e "${GREEN}7. 读取 notes-data.json 内容${NC}"
FILE_CONTENT=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
                     -H "Accept: application/vnd.github.v3+json" \
                     "${REPO_URL}/contents/notes-data.json?ref=${DEFAULT_BRANCH}")

FILE_SHA=$(echo "$FILE_CONTENT" | jq -r '.sha // empty')

if [ -n "$FILE_SHA" ] && [ "$FILE_SHA" != "null" ]; then
    echo "文件 SHA: ${FILE_SHA}"
    DECODED=$(echo "$FILE_CONTENT" | jq -r '.content' | tr -d '\n' | base64 -d 2>/dev/null || echo "解码失败")
    echo "文件内容预览:"
    echo "$DECODED" | head -c 200
    echo "..."
else
    echo "文件不存在"
fi
echo ""

# 8. 测试保存文件（需要提供内容）
echo -e "${GREEN}8. 测试保存文件${NC}"
echo "提示: 保存文件需要提供 SHA（如果文件已存在）"
echo "保存命令示例:"
echo "curl -X PUT \\"
echo "  -H \"Authorization: token \${GITHUB_TOKEN}\" \\"
echo "  -H \"Accept: application/vnd.github.v3+json\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"message\": \"更新笔记\","
echo "    \"content\": \"$(echo '{"test": "data"}' | base64 | tr -d '\n')\","
echo "    \"branch\": \"${DEFAULT_BRANCH}\","
echo "    \"sha\": \"${FILE_SHA}\""
echo "  }' \\"
echo "  \"${REPO_URL}/contents/notes-data.json\""
echo ""

echo -e "${BLUE}=== 测试完成 ===${NC}"
