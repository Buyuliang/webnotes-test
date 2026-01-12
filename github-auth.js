/**
 * GitHub OAuth 授权模块
 * 纯前端实现，使用 GitHub Web Application Flow (OAuth 2.0)
 * 
 * 注意：此方案为快速原型方案，Token 会在浏览器中可见
 * 仅用于演示或内部使用，不建议用于生产环境
 */

class GitHubAuth {
    constructor() {
        this.clientId = null;
        this.state = null;
    }

    /**
     * 启动 OAuth 授权流程（Web Application Flow）
     * @param {string} repoName - 仓库名称 (owner/repo)
     * @param {string} clientId - OAuth App Client ID
     * @param {string} redirectUri - 回调 URL（必须是 OAuth App 中配置的）
     */
    startOAuthFlow(repoName, clientId, redirectUri) {
        // 生成随机 state 用于防止 CSRF 攻击
        this.state = this.generateState();
        
        // 保存 state 和配置到 sessionStorage
        sessionStorage.setItem('oauth_state', this.state);
        sessionStorage.setItem('oauth_repo', repoName);
        sessionStorage.setItem('oauth_client_id', clientId);
        sessionStorage.setItem('oauth_redirect_uri', redirectUri);
        
        // 构建授权 URL
        const scope = 'repo';
        const authUrl = `https://github.com/login/oauth/authorize?` +
            `client_id=${encodeURIComponent(clientId)}&` +
            `redirect_uri=${encodeURIComponent(redirectUri)}&` +
            `scope=${encodeURIComponent(scope)}&` +
            `state=${encodeURIComponent(this.state)}`;
        
        // 重定向到 GitHub 授权页面
        window.location.href = authUrl;
    }

    /**
     * 处理 OAuth 回调（从 URL 参数中获取 code 并换取 token）
     * @param {string} code - 授权码
     * @param {string} state - 状态参数
     * @returns {Promise<string>} access_token
     */
    async handleCallback(code, state) {
        // 验证 state
        const savedState = sessionStorage.getItem('oauth_state');
        if (!savedState || savedState !== state) {
            throw new Error('State 验证失败，可能存在安全风险');
        }
        
        const clientId = sessionStorage.getItem('oauth_client_id');
        const redirectUri = sessionStorage.getItem('oauth_redirect_uri');
        
        if (!clientId || !redirectUri) {
            throw new Error('缺少必要的配置信息');
        }
        
        // 使用 CORS 代理来获取 token（因为 GitHub API 有 CORS 限制）
        // 注意：这里使用一个公开的 CORS 代理，生产环境应该使用自己的代理
        const proxyUrl = 'https://cors-anywhere.herokuapp.com/';
        const tokenUrl = 'https://github.com/login/oauth/access_token';
        
        try {
            // 尝试直接请求（可能会失败）
            const response = await this.requestTokenDirect(code, clientId, redirectUri);
            return response.access_token;
        } catch (error) {
            // 如果直接请求失败（CORS），提示用户使用代理或手动输入 token
            throw new Error('由于 CORS 限制，无法自动获取 token。请使用以下方法之一：\n' +
                '1. 使用 CORS 代理服务\n' +
                '2. 手动创建 Personal Access Token 并输入\n' +
                '3. 使用后端服务处理 OAuth 回调');
        }
    }

    /**
     * 直接请求 token（可能会因为 CORS 失败）
     */
    async requestTokenDirect(code, clientId, redirectUri) {
        const body = new URLSearchParams({
            client_id: clientId,
            code: code,
            redirect_uri: redirectUri
        });
        
        const response = await fetch('https://github.com/login/oauth/access_token', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: body.toString()
        });
        
        if (!response.ok) {
            throw new Error(`获取 token 失败: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error_description || data.error);
        }
        
        return data;
    }

    /**
     * 生成随机 state
     */
    generateState() {
        return Math.random().toString(36).substring(2, 15) + 
               Math.random().toString(36).substring(2, 15);
    }

    /**
     * 启动设备流授权（已废弃，因为 CORS 限制）
     * 保留此方法以保持向后兼容，但会提示使用 Web Application Flow
     */
    async startDeviceFlow(repoName, clientId, callbacks = {}) {
        // 设备流在浏览器中无法使用（CORS 限制）
        // 改用 Web Application Flow
        if (callbacks.onError) {
            callbacks.onError('设备流在浏览器中无法使用（CORS 限制）。请使用 Web Application Flow。');
        }
        throw new Error('设备流不支持浏览器环境，请使用 startOAuthFlow 方法');
    }

    /**
     * 请求设备代码
     * GitHub 设备流 API: POST https://github.com/login/device/code
     */
    async requestDeviceCode(clientId) {
        // 构建请求体
        const scope = 'repo'; // 需要 repo 权限
        const body = new URLSearchParams({
            client_id: clientId,
            scope: scope
        });
        
        try {
            const response = await fetch('https://github.com/login/device/code', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: body.toString()
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`请求设备代码失败: ${response.status} - ${errorText}`);
            }
            
            // GitHub 返回的是 text/plain 格式，需要手动解析
            const text = await response.text();
            const params = new URLSearchParams(text);
            
            return {
                device_code: params.get('device_code'),
                user_code: params.get('user_code'),
                verification_uri: params.get('verification_uri') || 'https://github.com/login/device',
                verification_uri_complete: params.get('verification_uri_complete'),
                expires_in: parseInt(params.get('expires_in') || '900', 10),
                interval: parseInt(params.get('interval') || '5', 10)
            };
        } catch (error) {
            console.error('请求设备代码失败:', error);
            throw new Error(`请求设备代码失败: ${error.message}`);
        }
    }

    /**
     * 轮询检查授权状态
     * GitHub 设备流 API: POST https://github.com/login/oauth/access_token
     */
    async pollForToken(deviceCode, clientId) {
        const body = new URLSearchParams({
            client_id: clientId,
            device_code: deviceCode,
            grant_type: 'urn:ietf:params:oauth:grant-type:device_code'
        });
        
        try {
            const response = await fetch('https://github.com/login/oauth/access_token', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: body.toString()
            });
            
            if (!response.ok) {
                throw new Error(`轮询授权状态失败: ${response.status}`);
            }
            
            // GitHub 返回的是 text/plain 格式，需要手动解析
            const text = await response.text();
            const params = new URLSearchParams(text);
            
            if (params.get('access_token')) {
                return {
                    access_token: params.get('access_token'),
                    token_type: params.get('token_type'),
                    scope: params.get('scope')
                };
            } else {
                return {
                    error: params.get('error'),
                    error_description: params.get('error_description'),
                    error_uri: params.get('error_uri')
                };
            }
        } catch (error) {
            console.error('轮询授权状态失败:', error);
            throw error;
        }
    }

    /**
     * 停止轮询
     */
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
}

// 创建全局实例
const githubAuth = new GitHubAuth();

