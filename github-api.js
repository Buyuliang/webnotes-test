/**
 * GitHub API 交互模块
 * 用于上传文件到 GitHub Release
 */

class GitHubAPI {
    constructor() {
        this.token = null;
        this.repo = null;
        this.baseURL = 'https://api.github.com';
    }

    /**
     * 初始化配置
     */
    init(token, repo) {
        this.token = token;
        this.repo = repo;
    }

    /**
     * 检查配置是否完整
     */
    isConfigured() {
        return this.token && this.repo;
    }

    /**
     * 发送 GitHub API 请求
     */
    async request(endpoint, options = {}) {
        if (!this.isConfigured()) {
            throw new Error('GitHub 配置未完成，请先配置 Token 和仓库名称');
        }

        const url = `${this.baseURL}${endpoint}`;
        const headers = {
            'Authorization': `token ${this.token}`,
            'Accept': 'application/vnd.github.v3+json',
            ...options.headers
        };

        const response = await fetch(url, {
            ...options,
            headers
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `GitHub API 错误: ${response.status}`);
        }

        return response.json();
    }

    /**
     * 获取最新的 Release
     */
    async getLatestRelease() {
        try {
            return await this.request(`/repos/${this.repo}/releases/latest`);
        } catch (error) {
            // 如果没有 release，返回 null
            if (error.message.includes('404')) {
                return null;
            }
            throw error;
        }
    }

    /**
     * 创建新的 Release
     */
    async createRelease(tagName = 'attachments', name = '附件存储', body = '用于存储记事本上传的文件和图片') {
        return await this.request(`/repos/${this.repo}/releases`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                tag_name: tagName,
                name: name,
                body: body,
                draft: false,
                prerelease: false
            })
        });
    }

    /**
     * 获取或创建 Release（如果不存在则创建）
     */
    async getOrCreateRelease() {
        let release = await this.getLatestRelease();
        
        if (!release) {
            // 创建新的 release
            release = await this.createRelease();
        }
        
        return release;
    }

    /**
     * 上传文件到 Release
     */
    async uploadFileToRelease(file, fileName = null) {
        const release = await this.getOrCreateRelease();
        const uploadUrl = release.upload_url.replace(/{.*}$/, '');
        const name = fileName || file.name || `file_${Date.now()}`;
        
        // 检查文件是否已存在
        const existingAsset = release.assets.find(asset => asset.name === name);
        if (existingAsset) {
            // 删除已存在的文件
            await this.deleteAsset(existingAsset.id);
        }

        // 上传文件
        // GitHub Release Assets API 需要将文件作为二进制数据上传
        // name 参数在 URL 查询字符串中，文件内容在请求体中
        const response = await fetch(`${uploadUrl}?name=${encodeURIComponent(name)}`, {
            method: 'POST',
            headers: {
                'Authorization': `token ${this.token}`,
                'Accept': 'application/vnd.github.v3+json',
                'Content-Type': file.type || 'application/octet-stream'
            },
            body: file
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `上传失败: ${response.status}`);
        }

        const asset = await response.json();
        return {
            id: asset.id,
            name: asset.name,
            url: asset.browser_download_url,
            size: asset.size
        };
    }

    /**
     * 删除 Release Asset
     */
    async deleteAsset(assetId) {
        return await this.request(`/repos/${this.repo}/releases/assets/${assetId}`, {
            method: 'DELETE'
        });
    }

    /**
     * 获取 Release 中的所有文件
     */
    async getReleaseAssets() {
        const release = await this.getOrCreateRelease();
        return release.assets.map(asset => ({
            id: asset.id,
            name: asset.name,
            url: asset.browser_download_url,
            size: asset.size,
            created_at: asset.created_at
        }));
    }

    /**
     * 将图片转换为 Blob
     */
    async imageToBlob(imageUrl) {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        return blob;
    }

    /**
     * 上传图片（从 base64 或 blob）
     */
    async uploadImage(imageData, fileName = null) {
        let file;
        
        if (typeof imageData === 'string') {
            // base64 数据
            const base64Data = imageData.split(',')[1] || imageData;
            const mimeType = imageData.match(/data:([^;]+);/)?.[1] || 'image/png';
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            
            const byteArray = new Uint8Array(byteNumbers);
            file = new Blob([byteArray], { type: mimeType });
        } else {
            file = imageData;
        }

        const name = fileName || `image_${Date.now()}.${this.getImageExtension(file.type)}`;
        return await this.uploadFileToRelease(file, name);
    }

    /**
     * 获取图片扩展名
     */
    getImageExtension(mimeType) {
        const extensions = {
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/gif': 'gif',
            'image/webp': 'webp',
            'image/svg+xml': 'svg'
        };
        return extensions[mimeType] || 'png';
    }
}

// 创建全局实例
const githubAPI = new GitHubAPI();
