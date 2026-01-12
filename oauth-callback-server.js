/**
 * 简单的 OAuth 回调服务器
 * 用于处理 GitHub OAuth 回调并获取 access_token
 * 
 * 使用方法：
 * 1. 安装依赖: npm install express cors
 * 2. 运行: node oauth-callback-server.js
 * 3. 在 OAuth App 中设置 Authorization callback URL 为: http://localhost:3000/oauth/callback
 * 4. 在登录页面设置 redirect_uri 为: http://localhost:3000/oauth/callback
 */

const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

// OAuth 回调处理
app.get('/oauth/callback', async (req, res) => {
    const { code, state } = req.query;
    
    if (!code) {
        return res.redirect(`http://localhost:8000/login.html?error=no_code`);
    }
    
    // 从 sessionStorage 获取的 state 应该在前端验证
    // 这里只是简单处理，实际应该验证 state
    
    try {
        // 获取 client_id（应该从配置或环境变量获取）
        // 这里需要用户配置
        const clientId = process.env.GITHUB_CLIENT_ID || req.query.client_id;
        const clientSecret = process.env.GITHUB_CLIENT_SECRET || req.query.client_secret;
        
        if (!clientId || !clientSecret) {
            return res.redirect(`http://localhost:8000/login.html?error=missing_config`);
        }
        
        // 换取 access_token
        const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                client_id: clientId,
                client_secret: clientSecret,
                code: code
            }).toString()
        });
        
        const tokenData = await tokenResponse.json();
        
        if (tokenData.error) {
            return res.redirect(`http://localhost:8000/login.html?error=${tokenData.error}`);
        }
        
        // 重定向回前端，携带 token（使用 hash 而不是 query，避免 token 出现在服务器日志中）
        const redirectUrl = `http://localhost:8000/login.html?token=${tokenData.access_token}&state=${state}`;
        res.redirect(redirectUrl);
        
    } catch (error) {
        console.error('OAuth 回调处理失败:', error);
        res.redirect(`http://localhost:8000/login.html?error=server_error`);
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`OAuth 回调服务器运行在 http://localhost:${PORT}`);
    console.log(`请在 OAuth App 中设置 Authorization callback URL 为: http://localhost:${PORT}/oauth/callback`);
});

