#!/bin/bash

echo "========================================"
echo "   GitHub Pages 记事本 - 本地服务器"
echo "========================================"
echo ""
echo "正在启动本地服务器..."
echo "访问地址: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "========================================"
echo ""

python3 -m http.server 8000
