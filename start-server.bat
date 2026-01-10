@echo off
echo ========================================
echo    GitHub Pages 记事本 - 本地服务器
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python！
    echo.
    echo 请选择以下方法之一：
    echo 1. 安装 Python: https://www.python.org/downloads/
    echo 2. 使用 VS Code Live Server 扩展
    echo 3. 直接打开 index.html（功能可能受限）
    echo.
    pause
    exit /b 1
)

echo 正在启动本地服务器...
echo 访问地址: http://localhost:8000
echo.
echo 按 Ctrl+C 停止服务器
echo ========================================
echo.

python -m http.server 8000

pause
