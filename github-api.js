/**
 * GitHub API 交互模块
 * 用于管理仓库中的文件和目录
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
     * 已移除：不再使用 GitHub Release 存储文件
     * 文件现在直接存储在仓库目录中
     */

    /**
     * 上传文件到仓库目录（files/）
     * @param {File|Blob} file - 要上传的文件
     * @param {string} notePath - 笔记文件路径（如 'note1.md' 或 'notes/folder/note.md'）
     * @param {string} fileName - 文件名（可选，默认使用原文件名）
     * @returns {Promise<Object>} 返回文件信息 {name, url, path, size}
     */
    async uploadFileToDirectory(file, notePath, fileName = null) {
        const branch = await this.getDefaultBranch();
        
        // 确定文件存储路径
        // notePath 格式：'root/note1.md' 或 'root/folder/note1.md'
        // 文件存储在：'root/files/' 或 'root/folder/files/'（在笔记同级目录下）
        const noteDir = notePath.includes('/') 
            ? notePath.substring(0, notePath.lastIndexOf('/'))
            : 'root';
        
        // 构建 files 目录路径（在笔记同级目录下）
        const filesDir = `${noteDir}/files`;
        const name = fileName || file.name || `file_${Date.now()}`;
        const filePath = `${filesDir}/${name}`;
        
        // 确保 files 目录存在
        try {
            await this.createDirectory(filesDir);
        } catch (e) {
            // 目录可能已存在，忽略错误
        }
        
        // 检查文件是否已存在
        let existingFile = null;
        let sha = null;
        try {
            const fileInfoResponse = await fetch(
                `https://api.github.com/repos/${this.repo}/contents/${filePath}?ref=${branch}`,
                {
                    headers: {
                        'Authorization': `token ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json'
                    }
                }
            );
            if (fileInfoResponse.ok) {
                existingFile = await fileInfoResponse.json();
                sha = existingFile.sha;
            }
        } catch (e) {
            // 文件不存在，继续上传
        }
        
        // 如果文件已存在，返回信息让调用者决定是否覆盖
        if (existingFile) {
            return {
                exists: true,
                path: filePath,
                sha: sha,
                name: name
            };
        }
        
        // 读取文件内容并转换为 base64（二进制文件需要特殊处理）
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        // 对于二进制文件，直接转换为 base64，不使用 encodeURIComponent
        let binaryString = '';
        const chunkSize = 8192;
        for (let i = 0; i < uint8Array.length; i += chunkSize) {
            const chunk = uint8Array.subarray(i, i + chunkSize);
            binaryString += String.fromCharCode.apply(null, chunk);
        }
        const encodedContent = btoa(binaryString);
        
        // 上传文件
        const response = await fetch(
            `https://api.github.com/repos/${this.repo}/contents/${filePath}`,
            {
                method: 'PUT',
                headers: {
                    'Authorization': `token ${this.token}`,
                    'Accept': 'application/vnd.github.v3+json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: `Upload file: ${name}`,
                    content: encodedContent,
                    branch: branch
                })
            }
        );

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `上传失败: ${response.status}`);
        }

        const data = await response.json();
        // 构建文件的下载 URL（使用 raw.githubusercontent.com）
        const downloadUrl = `https://raw.githubusercontent.com/${this.repo}/${branch}/${filePath}`;

        return {
            exists: false,
            name: name,
            url: downloadUrl,
            path: filePath,
            size: file.size || 0
        };
    }

    /**
     * 强制覆盖上传文件（用于覆盖已存在的文件）
     */
    async uploadFileToDirectoryForce(file, notePath, fileName, sha) {
        const branch = await this.getDefaultBranch();
        
        // 确定文件存储路径
        const noteDir = notePath.includes('/') 
            ? notePath.substring(0, notePath.lastIndexOf('/'))
            : 'root';
        
        // 构建 files 目录路径（在笔记同级目录下）
        const filesDir = `${noteDir}/files`;
        const name = fileName || file.name || `file_${Date.now()}`;
        const filePath = `${filesDir}/${name}`;
        
        // 读取文件内容并转换为 base64
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        let binaryString = '';
        const chunkSize = 8192;
        for (let i = 0; i < uint8Array.length; i += chunkSize) {
            const chunk = uint8Array.subarray(i, i + chunkSize);
            binaryString += String.fromCharCode.apply(null, chunk);
        }
        const encodedContent = btoa(binaryString);
        
        // 上传文件（使用 SHA 强制覆盖）
        const response = await fetch(
            `https://api.github.com/repos/${this.repo}/contents/${filePath}`,
            {
                method: 'PUT',
                headers: {
                    'Authorization': `token ${this.token}`,
                    'Accept': 'application/vnd.github.v3+json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: `Upload file: ${name} (overwrite)`,
                    content: encodedContent,
                    branch: branch,
                    sha: sha
                })
            }
        );

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `上传失败: ${response.status}`);
        }

        const data = await response.json();
        const downloadUrl = `https://raw.githubusercontent.com/${this.repo}/${branch}/${filePath}`;

        return {
            exists: false,
            name: name,
            url: downloadUrl,
            path: filePath,
            size: file.size || 0
        };
    }

    /**
     * 上传图片到仓库目录（image/），支持 CTRL+V 粘贴
     * @param {File|Blob|string} imageData - 图片数据（可以是 File、Blob 或 base64 字符串）
     * @param {string} notePath - 笔记文件路径
     * @param {string} fileName - 文件名（可选，默认使用时间戳）
     * @returns {Promise<Object>} 返回图片信息 {name, url, path, size}
     */
    async uploadImageToDirectory(imageData, notePath, fileName = null) {
        const branch = await this.getDefaultBranch();
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

        // 确定图片存储路径
        // notePath 格式：'root/note1.md' 或 'root/folder/note1.md'
        // 图片存储在：'root/images/' 或 'root/folder/images/'（在笔记同级目录下）
        const noteDir = notePath.includes('/') 
            ? notePath.substring(0, notePath.lastIndexOf('/'))
            : 'root';
        
        // 构建 images 目录路径（在笔记同级目录下）
        const imageDir = `${noteDir}/images`;
        
        // 使用时间戳命名图片
        const timestamp = Date.now();
        const extension = this.getImageExtension(file.type);
        const name = fileName || `${timestamp}.${extension}`;
        const imagePath = `${imageDir}/${name}`;
        
        // 确保 image 目录存在
        try {
            await this.createDirectory(imageDir);
        } catch (e) {
            // 目录可能已存在，忽略错误
        }
        
        // 读取文件内容并转换为 base64（二进制文件需要特殊处理）
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        // 对于二进制文件，直接转换为 base64，不使用 encodeURIComponent
        let binaryString = '';
        const chunkSize = 8192;
        for (let i = 0; i < uint8Array.length; i += chunkSize) {
            const chunk = uint8Array.subarray(i, i + chunkSize);
            binaryString += String.fromCharCode.apply(null, chunk);
        }
        const encodedContent = btoa(binaryString);
        
        // 上传图片
        const response = await fetch(
            `https://api.github.com/repos/${this.repo}/contents/${imagePath}`,
            {
                method: 'PUT',
                headers: {
                    'Authorization': `token ${this.token}`,
                    'Accept': 'application/vnd.github.v3+json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: `Upload image: ${name}`,
                    content: encodedContent,
                    branch: branch
                })
            }
        );

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `上传失败: ${response.status}`);
        }

        const data = await response.json();
        // 构建图片的下载 URL
        const downloadUrl = `https://raw.githubusercontent.com/${this.repo}/${branch}/${imagePath}`;
        
        return {
            name: name,
            url: downloadUrl,
            path: imagePath,
            size: file.size || 0
        };
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

    /**
     * 获取仓库的默认分支
     */
    async getDefaultBranch() {
        try {
            const repoInfo = await this.request(`/repos/${this.repo}`);
            return repoInfo.default_branch || 'main';
        } catch (error) {
            // 如果获取失败，默认使用 main
            return 'main';
        }
    }

    /**
     * 获取文件内容（从仓库）
     */
    async getFileContent(path, branch = null) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        try {
            // 先尝试获取文件
            const response = await fetch(
                `https://api.github.com/repos/${this.repo}/contents/${path}?ref=${branch}`,
                {
                    headers: {
                        'Authorization': `token ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json'
                    }
                }
            );

            if (!response.ok) {
                if (response.status === 404) {
                    return null; // 文件不存在
                }
                throw new Error(`获取文件失败: ${response.status}`);
            }

            const data = await response.json();
            // GitHub API 返回的是 base64 编码的内容
            const content = atob(data.content.replace(/\n/g, ''));
            return JSON.parse(content);
        } catch (error) {
            if (error.message.includes('404')) {
                return null;
            }
            throw error;
        }
    }

    /**
     * 检查文件内容是否改变
     */
    async hasContentChanged(path, newContent, branch = null) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        try {
            const existingContent = await this.getFileContent(path, branch);
            if (existingContent === null) {
                // 文件不存在，认为有改变
                return true;
            }
            
            // 比较内容（转换为字符串比较）
            const newContentStr = typeof newContent === 'string' ? newContent : JSON.stringify(newContent, null, 2);
            const existingContentStr = typeof existingContent === 'string' ? existingContent : JSON.stringify(existingContent, null, 2);
            
            return newContentStr !== existingContentStr;
        } catch (error) {
            // 获取失败，认为有改变（安全起见）
            return true;
        }
    }

    /**
     * 保存文件内容到仓库（带重试机制和冲突解决）
     */
    async saveFileContent(path, content, branch = null, message = 'Update notes data', retryCount = 5, checkChanges = true) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }

        // 如果启用了变更检查，先检查内容是否真的改变了
        if (checkChanges) {
            const hasChanged = await this.hasContentChanged(path, content, branch);
            if (!hasChanged) {
                // 内容没有改变，直接返回，不创建 commit
                console.log('内容未改变，跳过保存');
                return { content: content, skipped: true };
            }
        }

        for (let attempt = 0; attempt < retryCount; attempt++) {
            try {
                // 如果是重试，先等待一段时间（让 GitHub 的更新完成）
                if (attempt > 0) {
                    const waitTime = Math.min(Math.pow(2, attempt) * 200, 2000); // 最多等待 2 秒
                    console.log(`等待 ${waitTime}ms 后重试 (${attempt + 1}/${retryCount})...`);
                    await new Promise(resolve => setTimeout(resolve, waitTime));
                }

                // 每次重试都重新获取最新的 SHA 和内容
                let sha = null;
                let existingContent = null;
                
                try {
                    const existingFileResponse = await fetch(
                        `https://api.github.com/repos/${this.repo}/contents/${path}?ref=${branch}`,
                        {
                            headers: {
                                'Authorization': `token ${this.token}`,
                                'Accept': 'application/vnd.github.v3+json'
                            }
                        }
                    );
                    if (existingFileResponse.ok) {
                        const data = await existingFileResponse.json();
                        sha = data.sha;
                        // 如果是重试或第一次失败后，读取现有内容用于合并（仅对 JSON 文件）
                        if (attempt > 0 && path.endsWith('.json')) {
                            try {
                                const decodedContent = atob(data.content.replace(/\n/g, ''));
                                existingContent = JSON.parse(decodedContent);
                            } catch (e) {
                                // 解析失败，忽略
                                console.warn('解析现有内容失败:', e);
                            }
                        }
                    }
                } catch (e) {
                    // 文件不存在，sha 保持为 null
                    console.warn('获取文件信息失败:', e);
                }

                // 将内容转换为 base64
                let contentToSave = content;
                
                // 如果是重试且获取到了现有内容，尝试智能合并（仅对 JSON 数组）
                if (attempt > 0 && existingContent && path.endsWith('.json') && Array.isArray(existingContent) && Array.isArray(content)) {
                    console.log('检测到冲突，开始合并数据...');
                    // 合并策略：保留最新的笔记（根据 updatedAt）
                    const mergedMap = new Map();
                    
                    // 先添加远程笔记
                    existingContent.forEach(note => {
                        mergedMap.set(note.id, note);
                    });
                    
                    // 合并本地笔记：如果本地更新则保留本地，否则保留远程
                    content.forEach(localNote => {
                        const remoteNote = mergedMap.get(localNote.id);
                        if (!remoteNote) {
                            // 本地独有的笔记
                            mergedMap.set(localNote.id, localNote);
                        } else {
                            // 比较更新时间，保留最新的
                            const localTime = new Date(localNote.updatedAt || 0).getTime();
                            const remoteTime = new Date(remoteNote.updatedAt || 0).getTime();
                            if (localTime > remoteTime) {
                                mergedMap.set(localNote.id, localNote);
                            }
                        }
                    });
                    
                    // 转换为数组并按更新时间排序
                    contentToSave = Array.from(mergedMap.values()).sort((a, b) => {
                        return new Date(b.updatedAt || 0).getTime() - new Date(a.updatedAt || 0).getTime();
                    });
                    console.log(`合并完成: 远程 ${existingContent.length} 条，本地 ${content.length} 条，合并后 ${contentToSave.length} 条`);
                }
                
                const contentStr = typeof contentToSave === 'string' ? contentToSave : JSON.stringify(contentToSave, null, 2);
                // 使用 TextEncoder 确保正确的 UTF-8 编码
                let encodedContent;
                try {
                    const encoder = new TextEncoder();
                    const bytes = encoder.encode(contentStr);
                    const binaryString = String.fromCharCode.apply(null, bytes);
                    encodedContent = btoa(binaryString);
                } catch (e) {
                    encodedContent = btoa(unescape(encodeURIComponent(contentStr)));
                }

                // 创建或更新文件
                const response = await fetch(
                    `https://api.github.com/repos/${this.repo}/contents/${path}`,
                    {
                        method: 'PUT',
                        headers: {
                            'Authorization': `token ${this.token}`,
                            'Accept': 'application/vnd.github.v3+json',
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            content: encodedContent,
                            branch: branch,
                            ...(sha ? { sha: sha } : {})
                        })
                    }
                );

                if (response.ok) {
                    return await response.json();
                }

                // 处理错误
                const errorData = await response.json().catch(() => ({ message: response.statusText }));
                const errorMessage = errorData.message || `保存文件失败: ${response.status}`;
                
                // 如果是 409 冲突错误且还有重试次数，继续重试（等待已在循环开始处处理）
                if (response.status === 409 && attempt < retryCount - 1) {
                    console.log(`文件冲突 (SHA 不匹配)，将在下次重试时获取最新 SHA...`);
                    continue;
                }

                // 其他错误或重试次数用完，抛出错误
                throw new Error(errorMessage);
            } catch (error) {
                // 如果是最后一次尝试，抛出错误
                if (attempt === retryCount - 1) {
                    throw error;
                }
                // 如果是 409 错误或 SHA 不匹配，继续重试（等待已在循环开始处处理）
                if (error.message && (error.message.includes('does not match') || error.message.includes('409'))) {
                    console.log(`检测到冲突错误，将在下次重试时获取最新 SHA (${attempt + 1}/${retryCount})...`);
                    continue;
                }
                // 其他错误也重试，但先等待
                const waitTime = Math.min(Math.pow(2, attempt) * 200, 2000);
                console.log(`遇到错误，等待 ${waitTime}ms 后重试 (${attempt + 1}/${retryCount})...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
        
        // 理论上不会到达这里，但为了安全起见
        throw new Error('保存文件失败：重试次数已用完');
    }


    /**
     * 获取目录内容
     * @param {string} path - 目录路径，默认为 'root' 目录
     */
    async getDirectoryContents(path = 'root', branch = null, bypassCache = false) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        try {
            // 如果需要绕过缓存，添加时间戳参数
            let url = `https://api.github.com/repos/${this.repo}/contents/${path}?ref=${branch}`;
            if (bypassCache) {
                url += `&_t=${Date.now()}`;
            }
            
            const response = await fetch(url, {
                headers: {
                    'Authorization': `token ${this.token}`,
                    'Accept': 'application/vnd.github.v3+json'
                }
            });

            if (!response.ok) {
                if (response.status === 404) {
                    return []; // 目录不存在，返回空数组
                }
                throw new Error(`获取目录失败: ${response.status}`);
            }

            const data = await response.json();
            // GitHub API 返回的是数组
            return Array.isArray(data) ? data : [];
        } catch (error) {
            if (error.message.includes('404')) {
                return [];
            }
            throw error;
        }
    }

    /**
     * 创建目录（通过创建一个 .gitkeep 文件）
     */
    async createDirectory(path, branch = null) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        // GitHub 不支持直接创建空目录，通过创建 .gitkeep 文件来实现
        const gitkeepPath = `${path}/.gitkeep`;
        
        try {
            // 检查目录是否已存在
            const contents = await this.getDirectoryContents(path, branch);
            if (contents.length > 0) {
                return true; // 目录已存在
            }
            
            // 创建 .gitkeep 文件
            const content = '# 此文件用于保持目录结构';
            // 使用 TextEncoder 确保正确的 UTF-8 编码
            let encodedContent;
            try {
                const encoder = new TextEncoder();
                const bytes = encoder.encode(content);
                const binaryString = String.fromCharCode.apply(null, bytes);
                encodedContent = btoa(binaryString);
            } catch (e) {
                encodedContent = btoa(unescape(encodeURIComponent(content)));
            }
            
            const response = await fetch(
                `https://api.github.com/repos/${this.repo}/contents/${gitkeepPath}`,
                {
                    method: 'PUT',
                    headers: {
                        'Authorization': `token ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json',
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: 'Create directory',
                        content: encodedContent,
                        branch: branch
                    })
                }
            );

            return response.ok;
        } catch (error) {
            // 如果文件已存在，也算成功
            if (error.message && error.message.includes('already exists')) {
                return true;
            }
            throw error;
        }
    }

    /**
     * 删除文件或目录
     */
    async deleteFile(path, branch = null, message = 'Delete file') {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        try {
            // 先获取文件的 SHA
            const fileInfo = await fetch(
                `https://api.github.com/repos/${this.repo}/contents/${path}?ref=${branch}`,
                {
                    headers: {
                        'Authorization': `token ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json'
                    }
                }
            );

            if (!fileInfo.ok) {
                if (fileInfo.status === 404) {
                    return true; // 文件不存在，认为删除成功
                }
                throw new Error(`获取文件信息失败: ${fileInfo.status}`);
            }

            const data = await fileInfo.json();
            
            // 检查返回的数据类型
            // 如果是数组，说明是目录，需要递归删除
            if (Array.isArray(data)) {
                return await this.deleteDirectory(path, branch, message);
            }
            
            // 如果是对象且 type 为 'dir'，也需要递归删除
            if (data.type === 'dir') {
                return await this.deleteDirectory(path, branch, message);
            }
            
            // 否则是文件，直接删除
            const sha = data.sha;

            // 删除文件
            const response = await fetch(
                `https://api.github.com/repos/${this.repo}/contents/${path}`,
                {
                    method: 'DELETE',
                    headers: {
                        'Authorization': `token ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json',
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        sha: sha,
                        branch: branch
                    })
                }
            );

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: response.statusText }));
                throw new Error(errorData.message || `删除失败: ${response.status}`);
            }

            return true;
        } catch (error) {
            if (error.message && error.message.includes('404')) {
                return true; // 文件不存在，认为删除成功
            }
            throw error;
        }
    }

    /**
     * 递归删除目录（删除目录下的所有文件和子目录）
     */
    async deleteDirectory(path, branch = null, message = 'Delete folder') {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }

        try {
            // 获取目录下的所有内容
            const contents = await this.getDirectoryContents(path, branch);
            
            if (contents.length === 0) {
                // 目录为空，尝试删除 .gitkeep 文件（如果存在）
                const gitkeepPath = `${path}/.gitkeep`;
                try {
                    await this.deleteFile(gitkeepPath, branch, message);
                } catch (e) {
                    // .gitkeep 可能不存在（404），忽略错误
                    if (!e.message || !e.message.includes('404')) {
                        console.warn('删除 .gitkeep 失败:', e);
                    }
                }
                return true;
            }

            // 递归删除所有文件和子目录
            // 使用 Promise.allSettled 确保即使某些删除失败也继续
            const deletePromises = contents.map(async (item) => {
                try {
                    if (item.type === 'file') {
                        // 删除文件
                        return await this.deleteFile(item.path, branch, message);
                    } else if (item.type === 'dir') {
                        // 递归删除子目录
                        return await this.deleteDirectory(item.path, branch, message);
                    }
                } catch (error) {
                    // 记录错误但继续删除其他文件
                    console.warn(`删除 ${item.path} 失败:`, error);
                    throw error; // 重新抛出，让 Promise.allSettled 处理
                }
            });

            // 等待所有删除操作完成（使用 allSettled 确保所有操作都执行）
            const results = await Promise.allSettled(deletePromises);
            
            // 检查是否有失败的操作（除了 404）
            const failures = results.filter(r => 
                r.status === 'rejected' && 
                r.reason && 
                !r.reason.message.includes('404')
            );
            
            if (failures.length > 0) {
                console.warn('部分文件删除失败:', failures);
                // 如果所有失败都是 404，认为删除成功
                const non404Failures = failures.filter(f => 
                    !f.reason.message.includes('404')
                );
                if (non404Failures.length > 0) {
                    throw new Error(`删除目录失败: ${non404Failures[0].reason.message}`);
                }
            }

            // 所有内容删除后，尝试删除 .gitkeep 文件（如果存在）
            const gitkeepPath = `${path}/.gitkeep`;
            try {
                await this.deleteFile(gitkeepPath, branch, message);
            } catch (e) {
                // .gitkeep 可能不存在（404），忽略错误
                if (!e.message || !e.message.includes('404')) {
                    console.warn('删除 .gitkeep 失败:', e);
                }
            }

            // 验证目录是否真的被删除了（等待一小段时间让 GitHub 处理）
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // 再次检查目录是否还存在
            const remainingContents = await this.getDirectoryContents(path, branch);
            if (remainingContents.length > 0) {
                // 如果还有内容，可能是 .gitkeep 或其他文件，再次尝试删除
                console.warn(`目录 ${path} 仍有内容，尝试再次清理`);
                for (const item of remainingContents) {
                    try {
                        if (item.type === 'file') {
                            await this.deleteFile(item.path, branch, message);
                        } else if (item.type === 'dir') {
                            await this.deleteDirectory(item.path, branch, message);
                        }
                    } catch (e) {
                        console.warn(`清理 ${item.path} 失败:`, e);
                    }
                }
            }

            return true;
        } catch (error) {
            if (error.message && error.message.includes('404')) {
                return true; // 目录不存在，认为删除成功
            }
            throw error;
        }
    }

    /**
     * 读取笔记文件内容
     */
    async readNoteFile(filePath, branch = null) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        try {
            const response = await fetch(
                `https://api.github.com/repos/${this.repo}/contents/${filePath}?ref=${branch}`,
                {
                    headers: {
                        'Authorization': `token ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json'
                    }
                }
            );

            if (!response.ok) {
                if (response.status === 404) {
                    return null;
                }
                throw new Error(`读取文件失败: ${response.status}`);
            }

            const data = await response.json();
            // GitHub API 返回的是 base64 编码的内容
            // 使用正确的 UTF-8 解码方式处理中文
            const base64Content = data.content.replace(/\n/g, '');
            let content;
            try {
                // 先解码 base64
                const binaryString = atob(base64Content);
                // 转换为 UTF-8 字符串
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                // 使用 TextDecoder 解码 UTF-8
                const decoder = new TextDecoder('utf-8');
                content = decoder.decode(bytes);
            } catch (e) {
                // 降级到传统方法
                content = decodeURIComponent(escape(atob(base64Content)));
            }
            
            return {
                content: content,
                sha: data.sha,
                path: data.path,
                name: data.name,
                size: data.size
            };
        } catch (error) {
            if (error.message && error.message.includes('404')) {
                return null;
            }
            throw error;
        }
    }

    /**
     * 保存笔记文件
     */
    async saveNoteFile(filePath, content, branch = null, message = 'Update note', sha = null) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        // 确保内容不为 null 或 undefined
        const contentStr = content || '';
        
        // 对内容进行 base64 编码
        // 使用 UTF-8 编码确保中文字符正确处理
        // 使用 TextEncoder 确保正确的 UTF-8 编码
        let encodedContent;
        try {
            // 优先使用 TextEncoder（更可靠）
            const encoder = new TextEncoder();
            const bytes = encoder.encode(contentStr);
            const binaryString = String.fromCharCode.apply(null, bytes);
            encodedContent = btoa(binaryString);
        } catch (e) {
            // 降级到传统方法
            encodedContent = btoa(unescape(encodeURIComponent(contentStr)));
        }
        
        const body = {
            message: message,
            content: encodedContent,
            branch: branch
        };
        
        if (sha) {
            body.sha = sha;
        }

        const response = await fetch(
            `https://api.github.com/repos/${this.repo}/contents/${filePath}`,
            {
                method: 'PUT',
                headers: {
                    'Authorization': `token ${this.token}`,
                    'Accept': 'application/vnd.github.v3+json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            }
        );

        if (!response.ok) {
            // 获取详细的错误信息
            const errorData = await response.json().catch(() => ({ message: response.statusText }));
            const errorMessage = errorData.message || `保存文件失败: ${response.status}`;
            
            // 422 错误通常是验证失败，可能是文件已存在但缺少 SHA
            if (response.status === 422) {
                // 尝试获取现有文件的 SHA（如果文件存在）
                try {
                    // 直接获取文件信息（不解析内容）
                    const fileInfoResponse = await fetch(
                        `https://api.github.com/repos/${this.repo}/contents/${filePath}?ref=${branch}`,
                        {
                            headers: {
                                'Authorization': `token ${this.token}`,
                                'Accept': 'application/vnd.github.v3+json'
                            }
                        }
                    );
                    
                    if (fileInfoResponse.ok) {
                        const fileInfo = await fileInfoResponse.json();
                        if (fileInfo && fileInfo.sha) {
                            // 使用现有 SHA 重试
                            return await this.saveNoteFile(filePath, content, branch, message, fileInfo.sha);
                        }
                    }
                } catch (e) {
                    // 获取 SHA 失败，继续处理错误
                    console.warn('获取文件 SHA 失败:', e);
                }
                
                // 如果无法获取 SHA 或文件不存在，检查错误详情
                const details = errorData.errors ? JSON.stringify(errorData.errors) : '';
                if (errorData.errors && Array.isArray(errorData.errors)) {
                    const validationErrors = errorData.errors.map(err => err.message || err).join('; ');
                    throw new Error(`${errorMessage}: ${validationErrors}`);
                }
                
                throw new Error(`${errorMessage}${details ? ` (${details})` : ''}`);
            }
            
            throw new Error(errorMessage);
        }

        return await response.json();
    }

    /**
     * 递归获取目录树结构
     * @param {string} path - 目录路径，默认为 'root' 目录
     */
    async getDirectoryTree(path = 'root', branch = null, loadTitles = false, bypassCache = false) {
        if (!branch) {
            branch = await this.getDefaultBranch();
        }
        
        const tree = [];
        const contents = await this.getDirectoryContents(path, branch, bypassCache);
        
        // 先快速构建目录结构（不读取文件内容）
        const filePromises = [];
        
        for (const item of contents) {
            if (item.type === 'file') {
                // 跳过 .gitkeep 文件
                if (item.name === '.gitkeep') {
                    continue;
                }
                
                // 如果是 Markdown 文件且需要加载标题，异步读取
                if (item.name.endsWith('.md') && loadTitles) {
                    // 异步读取标题，不阻塞目录结构显示
                    const titlePromise = this.readNoteFile(item.path, branch)
                        .then(fileData => {
                            if (fileData && fileData.content) {
                                const firstLine = fileData.content.split('\n')[0].trim();
                                if (firstLine.startsWith('#')) {
                                    return firstLine.replace(/^#+\s*/, '').trim();
                                }
                            }
                            return null;
                        })
                        .catch(e => {
                            console.warn('读取文件标题失败:', e);
                            return null;
                        });
                    
                    filePromises.push({
                        path: item.path,
                        promise: titlePromise
                    });
                }
                
                // 先使用文件名作为标题，后续会更新
                tree.push({
                    type: 'file',
                    name: item.name,
                    path: item.path,
                    size: item.size,
                    sha: item.sha,
                    title: item.name.replace(/\.md$/, '') // 默认使用文件名
                });
            } else if (item.type === 'dir') {
                // 递归获取子目录（不加载标题，加快速度）
                const children = await this.getDirectoryTree(item.path, branch, loadTitles, bypassCache);
                tree.push({
                    type: 'dir',
                    name: item.name,
                    path: item.path,
                    children: children
                });
            }
        }
        
        // 如果需要加载标题，等待所有标题加载完成并更新
        if (loadTitles && filePromises.length > 0) {
            const titleResults = await Promise.allSettled(
                filePromises.map(fp => fp.promise)
            );
            
            titleResults.forEach((result, index) => {
                if (result.status === 'fulfilled' && result.value) {
                    const fileItem = tree.find(item => item.path === filePromises[index].path);
                    if (fileItem) {
                        fileItem.title = result.value;
                    }
                }
            });
        }
        
        return tree;
    }
    
    /**
     * 快速获取目录树（不读取文件标题，用于快速显示）
     */
    async getDirectoryTreeFast(path = 'root', branch = null, bypassCache = false) {
        return this.getDirectoryTree(path, branch, false, bypassCache);
    }
    
    /**
     * 完整获取目录树（读取文件标题，用于完整显示）
     */
    async getDirectoryTreeWithTitles(path = 'root', branch = null, bypassCache = false) {
        return this.getDirectoryTree(path, branch, true, bypassCache);
    }
}

// 创建全局实例
const githubAPI = new GitHubAPI();
