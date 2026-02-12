/**
 * 记事本应用主逻辑
 */

class NotesApp {
    constructor() {
        this.notes = [];
        this.currentNoteId = null;
        this.currentNotePath = null; // 当前打开的笔记文件路径，null 表示新笔记未保存
        this.tempParentPath = 'root'; // 临时保存新建笔记的父目录路径
        this.pendingImages = []; // 待上传的图片列表（用于新笔记）
        this.isPreviewMode = false;
        this.autoSyncInterval = null;
        this.syncInProgress = false;
        this.noteCache = new Map(); // 笔记缓存，key 为文件路径或 'new_note'
        this.hasUnsavedChanges = false; // 是否有未保存的修改
        this.init();
    }

    /**
     * 初始化应用
     */
    async init() {
        try {
            // 0. 初始化主题（必须在其他初始化之前，确保样式正确应用）
            this.initTheme();
            
            // 0.1. 从 localStorage 恢复笔记缓存
            try {
                const cacheDataStr = localStorage.getItem('note_cache');
                if (cacheDataStr) {
                    const cacheData = JSON.parse(cacheDataStr);
                    Object.entries(cacheData).forEach(([key, value]) => {
                        this.noteCache.set(key, value);
                    });
                }
            } catch (e) {
                console.warn('恢复笔记缓存失败:', e);
            }
            
            // 0.2. 检查目录树是否已显示（可能在应用初始化前已显示）
            const container = document.getElementById('directoryTree') || document.getElementById('notesList');
            const treeAlreadyShown = container && container.querySelector('[data-path="root"]');
            
            // 如果目录树未显示，立即显示
            if (!treeAlreadyShown) {
                const rootTree = [{
                    type: 'dir',
                    name: 'root',
                    path: 'root',
                    children: []
                }];
                this.renderDirectoryTree(rootTree);
            }

            // 1. 加载配置（可能失败：DOM 未加载、localStorage 不可用）
            try {
                this.loadConfig();
            } catch (error) {
                console.warn('加载配置失败:', error);
                // 继续执行，不影响其他功能
            }

            // 2. 设置事件监听器（可能失败：DOM 元素不存在）
            try {
                this.setupEventListeners();
            } catch (error) {
                console.error('设置事件监听器失败:', error);
                // 部分功能可能不可用，但应用仍可运行
                this.showMessage('部分功能初始化失败，请刷新页面重试', 'error');
            }

            // 3. 异步加载笔记和目录树数据（不阻塞界面显示）
            // 使用 Promise.all 并行加载，提高速度
            Promise.all([
                // 加载笔记（可能失败：网络错误、GitHub API 错误）
                (async () => {
            try {
                await this.loadNotes();
            } catch (error) {
                console.error('加载笔记失败:', error);
                // 使用空数组，确保应用可以继续运行
                this.notes = [];
                // 尝试从本地存储加载
                try {
                    const saved = localStorage.getItem('notes');
                    if (saved) {
                        this.notes = JSON.parse(saved);
                    }
                } catch (e) {
                    console.warn('从本地存储加载失败:', e);
                }
            }
                })(),
                // 加载目录树数据（异步，不阻塞）
                (async () => {
                    // 确保容器存在后再加载
                    const container = document.getElementById('directoryTree') || document.getElementById('notesList');
                    if (!container) {
                        // 如果容器不存在，等待一下再试
                        await new Promise(resolve => setTimeout(resolve, 10));
                    }
                    
                    try {
                        // 如果配置了 GitHub，加载目录树数据
                    if (githubAPI.isConfigured()) {
                            await this.loadDirectoryTree(false);
                        } else {
                            // 未配置 GitHub，尝试渲染笔记列表
                            this.renderNotesList();
                    }
                } catch (error) {
                        console.warn('加载目录树失败:', error);
                        // 如果目录树加载失败，保持显示空的 root 目录（已经显示了）
                    }
                })()
            ]).catch(error => {
                console.error('初始化数据加载失败:', error);
            });

            // 5. 启动自动同步（可能失败：但不应阻止应用运行）
            try {
                this.startAutoSync();
            } catch (error) {
                console.warn('启动自动同步失败:', error);
                // 自动同步失败不影响主要功能
            }
        } catch (error) {
            // 捕获所有未预期的错误
            console.error('应用初始化失败:', error);
            // 显示用户友好的错误提示
            setTimeout(() => {
                this.showMessage('应用初始化失败，请刷新页面重试', 'error');
            }, 500);
        }
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 主题切换按钮
        const themeToggleBtn = document.getElementById('themeToggleBtn');
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                this.toggleTheme();
            });
        }

        // 配置按钮
        document.getElementById('configBtn').addEventListener('click', () => {
            this.toggleConfigPanel();
        });

        // 保存配置
        document.getElementById('saveConfigBtn').addEventListener('click', () => {
            this.saveConfig();
        });

        // 同步笔记
        document.getElementById('syncBtn').addEventListener('click', () => {
            this.syncNotesFromGitHub();
        });

        // 切换预览模式
        document.getElementById('viewModeBtn').addEventListener('click', () => {
            this.togglePreviewMode();
        });

        // 全屏模式
        document.getElementById('fullscreenBtn').addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // 新建笔记
        document.getElementById('newNoteBtn').addEventListener('click', () => {
            this.createNewNote();
        });

        // 保存笔记
        document.getElementById('saveNoteBtn').addEventListener('click', () => {
            this.saveCurrentNote();
        });

        // 删除笔记
        document.getElementById('deleteNoteBtn').addEventListener('click', () => {
            this.deleteCurrentNote();
        });

        // 文件上传
        document.getElementById('uploadFileBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // 搜索
        document.getElementById('searchInput').addEventListener('input', async (e) => {
            const query = e.target.value.trim();
            if (query.length > 0) {
                await this.searchNotes(query);
            } else {
                // 清空搜索，恢复目录树显示
                await this.loadDirectoryTree();
            }
        });

        // 刷新目录树
        document.getElementById('refreshTreeBtn').addEventListener('click', () => {
            this.loadDirectoryTree();
        });

        // 退出登录
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                if (confirm('确定要退出登录吗？')) {
                    // 清除本地存储的认证信息
                    localStorage.removeItem('github_token');
                    localStorage.removeItem('github_repo');
                    localStorage.removeItem('github_client_id');
                    // 跳转到登录页面
                    window.location.href = 'login.html';
                }
            });
        }

        // Markdown 工具栏按钮
        this.setupMarkdownToolbar();

        // 图片粘贴
        const noteContent = document.getElementById('noteContent');
        
        // TAB 键功能（类似 VSCode）
        noteContent.addEventListener('keydown', (e) => {
            this.handleTabKey(e);
        });
        
        noteContent.addEventListener('paste', (e) => {
            this.handleImagePaste(e);
        });

        // 文件拖拽
        const editorContainer = document.getElementById('editorContainer');
        editorContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        editorContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.handleFileDrop(e);
        });
        
        // 文本区域也支持拖拽
        noteContent.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
        
        noteContent.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.handleFileDrop(e);
        });

        // 自动保存到本地（防抖）
        let saveTimeout;
        const autoSaveHandler = () => {
            // 实时更新预览
            if (this.isPreviewMode) {
                this.updatePreview();
            }
            
            // 标记有未保存的修改
            this.hasUnsavedChanges = true;
            this.updateUnsavedIndicator();
            
            // 保存到缓存
            this.saveCurrentNoteToCache();
            
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                this.autoSaveLocal();
            }, 1000);
        };
        
        noteContent.addEventListener('input', autoSaveHandler);
        document.getElementById('noteTitle').addEventListener('input', autoSaveHandler);

        // 新建笔记对话框按钮
        document.getElementById('cancelNoteBtn').addEventListener('click', () => {
            this.hideNewNoteDialog();
        });

        document.getElementById('createNoteBtn').addEventListener('click', () => {
            this.createNote();
        });

        // 新建文件夹对话框按钮
        document.getElementById('cancelFolderBtn').addEventListener('click', () => {
            this.hideNewFolderDialog();
        });

        document.getElementById('createFolderBtn').addEventListener('click', () => {
            this.createFolder();
        });
    }

    /**
     * 加载配置
     */
    loadConfig() {
        const token = localStorage.getItem('github_token');
        const repo = localStorage.getItem('github_repo');
        
        if (token && repo) {
            document.getElementById('githubToken').value = token;
            document.getElementById('githubRepo').value = repo;
            githubAPI.init(token, repo);
        }
    }

    /**
     * 保存配置
     */
    async saveConfig() {
        const token = document.getElementById('githubToken').value.trim();
        const repo = document.getElementById('githubRepo').value.trim();

        if (!token || !repo) {
            this.showMessage('请填写完整的配置信息', 'error');
            return;
        }

        this.showLoading(true);
        try {
            githubAPI.init(token, repo);
            localStorage.setItem('github_token', token);
            localStorage.setItem('github_repo', repo);
            
            // 配置成功后，重新加载数据并渲染
            await this.loadNotes();
            
            // 尝试加载目录树（优先使用目录结构）
            try {
                await this.loadDirectoryTree();
            } catch (error) {
                console.warn('加载目录树失败，使用旧的笔记列表:', error);
                // 如果目录树加载失败，使用旧的笔记列表
                this.renderNotesList();
            }
            
            // 重新启动自动同步
            this.startAutoSync();
            
            this.showMessage('配置保存成功！已从 GitHub 同步数据', 'success');
            this.toggleConfigPanel();
        } catch (error) {
            this.showMessage(`配置保存失败: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 切换配置面板
     */
    toggleConfigPanel() {
        const panel = document.getElementById('configPanel');
        panel.classList.toggle('hidden');
    }

    /**
     * 初始化主题
     */
    initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        this.applyTheme(savedTheme);
    }

    /**
     * 应用主题
     */
    applyTheme(theme) {
        const html = document.documentElement;
        if (theme === 'dark') {
            html.setAttribute('data-theme', 'dark');
        } else {
            html.removeAttribute('data-theme');
        }
        
        // 更新按钮文本
        const themeToggleBtn = document.getElementById('themeToggleBtn');
        if (themeToggleBtn) {
            themeToggleBtn.textContent = theme === 'dark' ? '☀️ 明亮' : '🌙 暗黑';
        }
        
        // 保存主题设置
        localStorage.setItem('theme', theme);
    }

    /**
     * 切换主题
     */
    toggleTheme() {
        const currentTheme = localStorage.getItem('theme') || 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
    }

    /**
     * 加载笔记（已迁移到目录结构）
     * 现在笔记数据通过目录树管理
     */
    async loadNotes() {
        // 已迁移到目录结构
        // 数据通过 loadDirectoryTree() 加载
        // 这里只保留本地存储的兼容性
        const saved = localStorage.getItem('notes');
        if (saved) {
            try {
                this.notes = JSON.parse(saved);
            } catch (e) {
                this.notes = [];
            }
        } else {
            this.notes = [];
        }
    }

    /**
     * 保存笔记到本地存储和 GitHub
     */
    async saveNotes(syncToGitHub = true) {
        // 保存到本地存储（用于兼容）
        localStorage.setItem('notes', JSON.stringify(this.notes));

        // 已迁移到目录结构
        // 笔记通过 saveNoteFile() 直接保存到目录中的文件
    }

    /**
     * 创建新笔记（仅在本地缓存，不提交到 GitHub）
     */
    createNewNote() {
        // 在创建新笔记前，先保存当前笔记到缓存
        if (this.currentNotePath || this.currentNoteId) {
            this.saveCurrentNoteToCache();
        }
        
        // 检查是否有未保存的修改
        if (this.hasUnsavedChangesInCurrentNote()) {
            const shouldContinue = confirm('当前笔记有未保存的修改，是否继续创建新笔记？\n\n未保存的内容已自动缓存，稍后可以恢复。');
            if (!shouldContinue) {
                return;
            }
        }
        
        // 尝试从缓存恢复新笔记
        const cached = this.restoreNoteFromCache('new_note');
        
        // 新建笔记只在本地缓存，不提交到 GitHub
        // 设置 currentNotePath 为 null，表示这是新笔记，还未保存
        this.currentNotePath = null;
        this.currentNoteId = null;
        
        // 更新 UI
        if (cached) {
            document.getElementById('noteTitle').value = cached.title;
            document.getElementById('noteContent').value = cached.content;
            document.getElementById('notePath').textContent = '路径: 未保存（新笔记，已缓存）';
        } else {
        document.getElementById('noteTitle').value = '未命名笔记';
        document.getElementById('noteContent').value = '# 未命名笔记\n\n';
        document.getElementById('notePath').textContent = '路径: 未保存（新笔记）';
        }
        
        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('editorContainer').classList.remove('hidden');
    }

    /**
     * 选择笔记
     */
    selectNote(noteId) {
        this.currentNoteId = noteId;
        const note = this.notes.find(n => n.id === noteId);
        
        if (!note) return;

        // 更新 UI
        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('editorContainer').classList.remove('hidden');
        
        document.getElementById('noteTitle').value = note.title;
        document.getElementById('noteContent').value = note.content || '';
        
        // 更新预览
        if (this.isPreviewMode) {
            this.updatePreview();
        }
        
        this.renderAttachments(note.attachments);
        this.updateLastSaved(note.updatedAt);
        this.renderNotesList();
    }

    /**
     * 保存当前笔记
     */
    async saveCurrentNote() {
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        const title = document.getElementById('noteTitle').value || '未命名笔记';
        const content = document.getElementById('noteContent').value || '';

        // 确保内容的第一行是标题（Markdown 格式）
        let contentToSave = content;
        const firstLine = content.trim().split('\n')[0];
        if (!firstLine.startsWith('#')) {
            // 如果第一行不是标题，添加标题
            contentToSave = `# ${title}\n\n${content}`;
        } else {
            // 如果第一行是标题，更新标题
            const lines = content.split('\n');
            lines[0] = `# ${title}`;
            contentToSave = lines.join('\n');
        }

        // 确定文件路径
        let filePath = this.currentNotePath;
        
        // 如果是新笔记（currentNotePath 为 null），根据标题生成文件名
        if (!filePath) {
            // 使用标题作为文件名（去除特殊字符，避免路径问题）
            const sanitizedTitle = title.replace(/[\/\\:*?"<>|]/g, '_');
            const fileName = sanitizedTitle.endsWith('.md') ? sanitizedTitle : `${sanitizedTitle}.md`;
            let parentPath = this.tempParentPath || 'root';
            
            // 清理 parentPath：去除末尾斜杠，去除双斜杠
            parentPath = parentPath.replace(/\/+/g, '/').replace(/\/$/, '');
            if (!parentPath) parentPath = 'root';
            
            filePath = `${parentPath}/${fileName}`;
        } else {
            // 如果是已存在的笔记，根据标题更新文件名（如果标题改变了）
            const pathParts = filePath.split('/');
            const oldFileName = pathParts[pathParts.length - 1];
            const sanitizedTitle = title.replace(/[\/\\:*?"<>|]/g, '_');
            const newFileName = sanitizedTitle.endsWith('.md') ? sanitizedTitle : `${sanitizedTitle}.md`;
            
            // 如果文件名改变了，需要删除旧文件并创建新文件
            if (oldFileName !== newFileName) {
                const oldFilePath = filePath;
                pathParts[pathParts.length - 1] = newFileName;
                filePath = pathParts.join('/');
                
                // 保存旧文件路径，稍后删除
                this.oldFilePathToDelete = oldFilePath;
            }
        }

        // 验证并修复路径
        const pathParts = filePath.split('/');
        const hasInvalidNotePath = pathParts.some(part => part.startsWith('note_') && !part.includes('.'));
        
        if (hasInvalidNotePath) {
            const validParts = [];
            
            for (let i = 0; i < pathParts.length; i++) {
                const part = pathParts[i];
                // 跳过空字符串和看起来像笔记文件名但没有扩展名的部分（note_ 开头且没有点）
                if (!part || (part.startsWith('note_') && !part.includes('.'))) {
                    continue;
                }
                validParts.push(part);
            }
            
            // 确保最后一个部分是 .md 文件
            const lastPart = validParts[validParts.length - 1];
            if (!lastPart.endsWith('.md')) {
                // 使用标题作为文件名
                const fileName = title ? `${title.replace(/[\/\\:*?"<>|]/g, '_')}.md` : 'untitled.md';
                validParts[validParts.length - 1] = fileName;
            }
            
            filePath = validParts.join('/');
        }
        
        // 清理路径：去除双斜杠、去除空部分、去除末尾斜杠
        filePath = filePath.replace(/\/+/g, '/').replace(/\/$/, '');
        
        // 确保以 root/ 开头
        if (!filePath.startsWith('root/')) {
            filePath = `root/${filePath}`;
        }
        
        // 再次清理路径（防止 root//xxx 的情况）
        filePath = filePath.replace(/\/+/g, '/').replace(/\/$/, '');

        // 确保 root 目录存在
        try {
            await githubAPI.createDirectory('root');
        } catch (error) {
            // 目录可能已存在，忽略错误
        }
        
        // 检查并创建所有必要的父目录
        const pathPartsForDirs = filePath.split('/');
        if (pathPartsForDirs.length > 2) { // root + filename，如果超过2个部分说明有子目录
            // 从 root 开始，逐级创建目录
            let currentPath = 'root';
            for (let i = 1; i < pathPartsForDirs.length - 1; i++) { // 跳过 root 和文件名
                const dirName = pathPartsForDirs[i];
                if (dirName && dirName.trim()) { // 确保目录名不为空
                    currentPath = `${currentPath}/${dirName}`;
                    try {
                        await githubAPI.createDirectory(currentPath);
                    } catch (error) {
                        // 目录可能已存在，忽略错误
                    }
                }
            }
        }

        this.showLoading(true);
        try {
            // 如果是新笔记且有待上传的图片，先上传图片并替换 content 中的 data URI
            if (!this.currentNotePath && this.pendingImages.length > 0) {
                let updatedContent = contentToSave;
                
                // 上传所有待上传的图片
                for (const pendingImage of this.pendingImages) {
                    try {
                        const asset = await githubAPI.uploadImageToDirectory(pendingImage.file, filePath);
                        
                        // 替换 content 中的 data URI 为 GitHub URL
                        const dataUriPattern = new RegExp(`!\\[([^\\]]*)\\]\\(${pendingImage.dataUri.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`, 'g');
                        updatedContent = updatedContent.replace(dataUriPattern, `![$1](${asset.url})`);
                    } catch (error) {
                        console.warn(`上传图片失败: ${error.message}`, pendingImage);
                        // 继续处理其他图片
                    }
                }
                
                // 更新内容
                contentToSave = updatedContent;
                
                // 清空待上传图片列表
                this.pendingImages = [];
            }
            
            // 保存到 GitHub（如果是新笔记使用 Create，否则使用 Update）
            const isNewNote = !this.currentNotePath;
            const commitMessage = isNewNote ? 'Create new note' : 'Update note';
            await githubAPI.saveNoteFile(filePath, contentToSave, null, commitMessage);
            
            // 如果文件名改变了，删除旧文件
            if (this.oldFilePathToDelete && this.oldFilePathToDelete !== filePath) {
                try {
                    await githubAPI.deleteFile(this.oldFilePathToDelete, null, 'Rename note');
                } catch (error) {
                    console.warn('删除旧文件失败:', error);
                }
                this.oldFilePathToDelete = null;
            }
            
            // 更新 currentNotePath
            this.currentNotePath = filePath;
            this.currentNoteId = filePath;
            document.getElementById('notePath').textContent = `路径: ${filePath}`;
            
            // 更新编辑器内容（如果内容被修改了）
            if (contentToSave !== content) {
                document.getElementById('noteContent').value = contentToSave;
            }
            
            // 保存成功后，清除缓存中的该笔记（因为已保存到服务器）
            if (this.currentNotePath) {
                this.noteCache.delete(this.currentNotePath);
                // 更新 localStorage
                try {
                    const cacheData = {};
                    this.noteCache.forEach((value, key) => {
                        cacheData[key] = value;
                    });
                    localStorage.setItem('note_cache', JSON.stringify(cacheData));
                } catch (e) {
                    console.warn('更新缓存失败:', e);
                }
            }
            
            this.updateLastSaved(new Date().toISOString());
            this.hasUnsavedChanges = false;
            this.updateUnsavedIndicator();
            this.showMessage('笔记已保存', 'success');
            
            // 等待一小段时间，确保 GitHub API 的更改已生效
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // 强制刷新目录树（重新从 GitHub 获取）
            await this.loadDirectoryTree(true);
        } catch (error) {
            this.showMessage(`保存失败: ${error.message}`, 'error');
            // 即使保存失败，也刷新目录树，确保 UI 状态正确
            await new Promise(resolve => setTimeout(resolve, 500));
            await this.loadDirectoryTree(true);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 自动保存到本地（快速保存）
     */
    async autoSaveLocal() {
        // 保存到缓存
        this.saveCurrentNoteToCache();
        
        if (!this.currentNoteId) return;

        const note = this.notes.find(n => n.id === this.currentNoteId);
        if (!note) return;

        note.title = document.getElementById('noteTitle').value || '未命名笔记';
        note.content = document.getElementById('noteContent').value || '';
        note.updatedAt = new Date().toISOString();

        // 只保存到本地，不同步到 GitHub（避免频繁请求）
        localStorage.setItem('notes', JSON.stringify(this.notes));
        this.updateLastSaved(note.updatedAt);
        
        // 更新预览
        if (this.isPreviewMode) {
            this.updatePreview();
        }
    }

    /**
     * 更新未保存标识
     */
    updateUnsavedIndicator() {
        const lastSavedEl = document.getElementById('lastSaved');
        if (!lastSavedEl) return;
        
        if (this.hasUnsavedChanges) {
            lastSavedEl.textContent = '未保存 *';
            lastSavedEl.style.color = '#e74c3c';
            lastSavedEl.style.fontWeight = 'bold';
        } else {
            const date = new Date();
            lastSavedEl.textContent = `最后保存: ${date.toLocaleString('zh-CN')}`;
            lastSavedEl.style.color = '#888';
            lastSavedEl.style.fontWeight = 'normal';
        }
    }

    /**
     * 删除当前笔记
     */
    async deleteCurrentNote() {
        if (!this.currentNoteId) return;

        if (!confirm('确定要删除这个笔记吗？')) return;

        this.notes = this.notes.filter(n => n.id !== this.currentNoteId);
        await this.saveNotes();
        this.currentNoteId = null;

        document.getElementById('emptyState').classList.remove('hidden');
        document.getElementById('editorContainer').classList.add('hidden');
        
        this.renderNotesList();
        this.showMessage('笔记已删除', 'success');
    }

    /**
     * 渲染笔记列表
     */
    renderNotesList() {
        // 兼容新旧版本：优先使用 directoryTree，如果没有则使用 notesList
        const list = document.getElementById('directoryTree') || document.getElementById('notesList');
        if (!list) {
            console.warn('找不到目录树或笔记列表容器');
            return false;
        }
        
        try {
            list.innerHTML = '';

            if (this.notes.length === 0) {
                list.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">暂无笔记，点击"新建笔记"开始</div>';
                return true;
            }

            this.notes.forEach(note => {
                const item = document.createElement('div');
                item.className = `note-item ${note.id === this.currentNoteId ? 'active' : ''}`;
                item.addEventListener('click', () => this.selectNote(note.id));

                const title = document.createElement('div');
                title.className = 'note-item-title';
                title.textContent = note.title || '未命名笔记';

                const preview = document.createElement('div');
                preview.className = 'note-item-preview';
                const textContent = note.content.replace(/<[^>]*>/g, '').substring(0, 50);
                preview.textContent = textContent || '空笔记';

                const date = document.createElement('div');
                date.className = 'note-item-date';
                date.textContent = new Date(note.updatedAt).toLocaleString('zh-CN');

                item.appendChild(title);
                item.appendChild(preview);
                item.appendChild(date);
                list.appendChild(item);
            });
            
            return true;
        } catch (error) {
            console.error('渲染笔记列表失败:', error);
            if (list) {
                list.innerHTML = '<div style="padding: 20px; text-align: center; color: #e74c3c;">渲染失败，请刷新页面</div>';
            }
            return false;
        }
    }

    /**
     * 过滤笔记
     */
    filterNotes(query) {
        const items = document.querySelectorAll('.note-item');
        const lowerQuery = query.toLowerCase();

        items.forEach(item => {
            const title = item.querySelector('.note-item-title').textContent.toLowerCase();
            const preview = item.querySelector('.note-item-preview').textContent.toLowerCase();
            
            if (title.includes(lowerQuery) || preview.includes(lowerQuery)) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    }

    /**
     * 搜索笔记内容（搜索 root 目录下所有笔记，不包括 files 和 images 目录）
     */
    async searchNotes(query) {
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        if (!query || query.trim().length === 0) {
            await this.loadDirectoryTree();
            return;
        }

        this.showLoading(true);
        try {
            // 获取所有笔记文件（递归搜索，排除 files 和 images 目录）
            const allNotes = await this.getAllNoteFiles('root');
            
            // 搜索匹配的笔记
            const searchResults = [];
            const lowerQuery = query.toLowerCase();

            for (const note of allNotes) {
                try {
                    const fileData = await githubAPI.readNoteFile(note.path);
                    if (fileData && fileData.content) {
                        const content = fileData.content.toLowerCase();
                        const title = note.title || note.name.replace(/\.md$/, '');
                        
                        // 搜索标题和内容
                        if (title.toLowerCase().includes(lowerQuery) || content.includes(lowerQuery)) {
                            // 提取匹配的上下文
                            const lines = fileData.content.split('\n');
                            let preview = '';
                            for (let i = 0; i < lines.length; i++) {
                                if (lines[i].toLowerCase().includes(lowerQuery)) {
                                    const start = Math.max(0, i - 1);
                                    const end = Math.min(lines.length, i + 2);
                                    preview = lines.slice(start, end).join('\n');
                                    break;
                                }
                            }
                            
                            searchResults.push({
                                path: note.path,
                                title: title,
                                preview: preview || fileData.content.substring(0, 200),
                                matchCount: (content.match(new RegExp(lowerQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length
                            });
                        }
                    }
                } catch (error) {
                    console.warn(`搜索笔记失败 (${note.path}):`, error);
                }
            }

            // 按匹配次数排序
            searchResults.sort((a, b) => b.matchCount - a.matchCount);

            // 显示搜索结果
            this.renderSearchResults(searchResults, query);
        } catch (error) {
            this.showMessage(`搜索失败: ${error.message}`, 'error');
            await this.loadDirectoryTree();
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 递归获取所有笔记文件（排除 files 和 images 目录）
     */
    async getAllNoteFiles(path) {
        const notes = [];
        const contents = await githubAPI.getDirectoryContents(path);
        
        for (const item of contents) {
            // 跳过 files 和 images 目录
            if (item.type === 'dir' && (item.name === 'files' || item.name === 'images')) {
                continue;
            }
            
            if (item.type === 'file' && item.name.endsWith('.md')) {
                notes.push({
                    path: item.path,
                    name: item.name,
                    title: null
                });
            } else if (item.type === 'dir') {
                // 递归获取子目录中的笔记
                const subNotes = await this.getAllNoteFiles(item.path);
                notes.push(...subNotes);
            }
        }
        
        return notes;
    }

    /**
     * 渲染搜索结果
     */
    renderSearchResults(results, query) {
        const container = document.getElementById('directoryTree') || document.getElementById('notesList');
        if (!container) return;

        if (results.length === 0) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: #999;">未找到包含 "${query}" 的笔记</div>`;
            return;
        }

        container.innerHTML = '';
        
        const resultsDiv = document.createElement('div');
        resultsDiv.className = 'search-results';
        resultsDiv.innerHTML = `<div style="padding: 10px; color: #666; font-size: 14px; border-bottom: 1px solid #eee;">找到 ${results.length} 个匹配的笔记</div>`;

        results.forEach(result => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'tree-item tree-item-file search-result-item';
            itemDiv.style.cursor = 'pointer';
            itemDiv.style.padding = '10px';
            itemDiv.style.borderBottom = '1px solid #eee';

            itemDiv.addEventListener('mouseenter', () => {
                itemDiv.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
            });
            itemDiv.addEventListener('mouseleave', () => {
                itemDiv.style.backgroundColor = '';
            });

            const titleDiv = document.createElement('div');
            titleDiv.style.fontWeight = 'bold';
            titleDiv.style.marginBottom = '5px';
            titleDiv.style.color = '#333';
            titleDiv.textContent = result.title;

            const pathDiv = document.createElement('div');
            pathDiv.style.fontSize = '12px';
            pathDiv.style.color = '#999';
            pathDiv.style.marginBottom = '5px';
            pathDiv.textContent = result.path;

            const previewDiv = document.createElement('div');
            previewDiv.style.fontSize = '13px';
            previewDiv.style.color = '#666';
            previewDiv.style.maxHeight = '60px';
            previewDiv.style.overflow = 'hidden';
            previewDiv.textContent = result.preview;

            itemDiv.appendChild(titleDiv);
            itemDiv.appendChild(pathDiv);
            itemDiv.appendChild(previewDiv);

            // 点击打开笔记
            itemDiv.addEventListener('click', () => {
                this.openNoteFile(result.path);
            });

            resultsDiv.appendChild(itemDiv);
        });

        container.appendChild(resultsDiv);
    }

    /**
     * 处理文件拖拽
     */
    async handleFileDrop(e) {
        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;

        // 如果没有打开的笔记，创建一个新笔记
        const noteContent = document.getElementById('noteContent');
        const hasNote = noteContent && (noteContent.value.trim().length > 0 || this.currentNotePath || this.currentNoteId);
        if (!hasNote) {
            this.createNewNote();
        }

        // 分离图片和文件
        const imageFiles = [];
        const otherFiles = [];

        Array.from(files).forEach(file => {
            if (file.type.startsWith('image/')) {
                imageFiles.push(file);
            } else {
                otherFiles.push(file);
            }
        });

        // 处理图片（类似粘贴图片）
        for (const file of imageFiles) {
            await this.handleImageFile(file);
        }

        // 处理其他文件
        if (otherFiles.length > 0) {
            await this.handleFileUpload(otherFiles);
        }
    }

    /**
     * 处理单个图片文件（用于拖拽和粘贴）
     */
    async handleImageFile(file) {
        // 如果是新笔记（未保存），将图片转换为 data URI 存储在本地
        if (!this.currentNotePath) {
            // 读取文件为 data URI
            const reader = new FileReader();
            reader.onload = (event) => {
                const dataUri = event.target.result;
                const imageId = `pending_image_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                
                // 保存到待上传列表
                this.pendingImages.push({
                    id: imageId,
                    file: file,
                    dataUri: dataUri,
                    name: file.name || `image_${Date.now()}.png`
                });
                
                // 插入 Markdown 格式的图片到编辑器（使用 data URI）
                const textarea = document.getElementById('noteContent');
                const markdownImage = `![${file.name || 'image'}](${dataUri})`;
                const cursorPos = textarea.selectionStart;
                const textBefore = textarea.value.substring(0, cursorPos);
                const textAfter = textarea.value.substring(cursorPos);
                textarea.value = textBefore + markdownImage + '\n\n' + textAfter;
                
                // 设置光标位置
                const newPos = cursorPos + markdownImage.length + 2;
                textarea.setSelectionRange(newPos, newPos);
                textarea.focus();
                
                // 更新预览
                if (this.isPreviewMode) {
                    this.updatePreview();
                }
                
                this.showMessage('图片已添加到笔记（将在保存时上传）', 'success');
            };
            reader.readAsDataURL(file);
        } else {
            // 如果笔记已保存，立即上传
            if (!githubAPI.isConfigured()) {
                this.showMessage('请先配置 GitHub Token 和仓库', 'error');
                return;
            }
            
            this.showLoading(true);
            try {
                const asset = await githubAPI.uploadImageToDirectory(file, this.currentNotePath);
                
                // 插入 Markdown 格式的图片到编辑器
                const textarea = document.getElementById('noteContent');
                const markdownImage = `![${asset.name}](${asset.url})`;
                const cursorPos = textarea.selectionStart;
                const textBefore = textarea.value.substring(0, cursorPos);
                const textAfter = textarea.value.substring(cursorPos);
                textarea.value = textBefore + markdownImage + '\n\n' + textAfter;
                
                // 设置光标位置
                const newPos = cursorPos + markdownImage.length + 2;
                textarea.setSelectionRange(newPos, newPos);
                textarea.focus();
                
                // 更新预览
                if (this.isPreviewMode) {
                    this.updatePreview();
                }
                
                this.showMessage('图片已上传', 'success');
            } catch (error) {
                this.showMessage(`图片上传失败: ${error.message}`, 'error');
            } finally {
                this.showLoading(false);
            }
        }
    }

    /**
     * 处理文件上传
     */
    async handleFileUpload(files) {
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        if (!this.currentNotePath) {
            this.showMessage('请先选择或创建一个笔记', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const uploadResults = [];
            
            // 先检查所有文件是否存在
            for (const file of files) {
                const checkResult = await githubAPI.uploadFileToDirectory(file, this.currentNotePath);
                
                if (checkResult.exists) {
                    // 文件已存在，询问是否覆盖
                    const shouldOverwrite = confirm(`文件 "${checkResult.name}" 已存在，是否覆盖？`);
                    if (shouldOverwrite) {
                        // 强制覆盖
                        const asset = await githubAPI.uploadFileToDirectoryForce(
                            file, 
                            this.currentNotePath, 
                            checkResult.name, 
                            checkResult.sha
                        );
                        uploadResults.push({
                            name: asset.name,
                            url: asset.url,
                            path: asset.path,
                            size: file.size,
                            type: file.type
                        });
                    } else {
                        // 跳过此文件
                        this.showMessage(`已跳过文件: ${checkResult.name}`, 'info');
                    }
                } else {
                    // 文件不存在，直接使用上传结果
                    uploadResults.push({
                        name: checkResult.name,
                        url: checkResult.url,
                        path: checkResult.path,
                        size: file.size,
                        type: file.type
                    });
                }
            }
            
            if (uploadResults.length === 0) {
                this.showMessage('没有文件被上传', 'info');
                return;
            }
            
            // 在笔记中插入文件链接（Markdown 格式）
            const textarea = document.getElementById('noteContent');
            if (textarea) {
                let fileLinks = '';
                uploadResults.forEach(attachment => {
                    fileLinks += `[${attachment.name}](${attachment.url})\n`;
                });
                
                // 在光标位置插入文件链接
                const cursorPos = textarea.selectionStart;
                const textBefore = textarea.value.substring(0, cursorPos);
                const textAfter = textarea.value.substring(cursorPos);
                textarea.value = textBefore + (textBefore.endsWith('\n') ? '' : '\n') + fileLinks + textAfter;
                
                // 设置光标位置
                const newPos = cursorPos + fileLinks.length + (textBefore.endsWith('\n') ? 0 : 1);
                textarea.setSelectionRange(newPos, newPos);
                textarea.focus();
                
                // 更新预览
                if (this.isPreviewMode) {
                    this.updatePreview();
                }
            }
            
            this.showMessage(`成功上传 ${uploadResults.length} 个文件`, 'success');
        } catch (error) {
            this.showMessage(`上传失败: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }

        // 清空文件输入
        document.getElementById('fileInput').value = '';
    }

    /**
     * 处理图片粘贴（支持 CTRL+V）
     */
    async handleImagePaste(e) {
        const items = e.clipboardData?.items;
        if (!items) return;

        // 如果没有打开的笔记，创建一个新笔记
        // 检查是否有编辑器内容，如果有说明已经有笔记了（即使未保存）
        const noteContent = document.getElementById('noteContent');
        const hasNote = noteContent && (noteContent.value.trim().length > 0 || this.currentNotePath || this.currentNoteId);
        if (!hasNote) {
            this.createNewNote();
        }

        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                e.preventDefault();
                
                try {
                    const file = item.getAsFile();
                    
                    // 如果是新笔记（未保存），将图片转换为 data URI 存储在本地
                    if (!this.currentNotePath) {
                        // 读取文件为 data URI
                        const reader = new FileReader();
                        reader.onload = (event) => {
                            const dataUri = event.target.result;
                            const imageId = `pending_image_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                            
                            // 保存到待上传列表
                            this.pendingImages.push({
                                id: imageId,
                                file: file,
                                dataUri: dataUri,
                                name: file.name || `image_${Date.now()}.png`
                            });
                            
                            // 插入 Markdown 格式的图片到编辑器（使用 data URI）
                            const textarea = document.getElementById('noteContent');
                            const markdownImage = `![${file.name || 'image'}](${dataUri})`;
                            const cursorPos = textarea.selectionStart;
                            const textBefore = textarea.value.substring(0, cursorPos);
                            const textAfter = textarea.value.substring(cursorPos);
                            textarea.value = textBefore + markdownImage + '\n\n' + textAfter;
                            
                            // 设置光标位置
                            const newPos = cursorPos + markdownImage.length + 2;
                            textarea.setSelectionRange(newPos, newPos);
                            textarea.focus();
                            
                            // 更新预览
                            if (this.isPreviewMode) {
                                this.updatePreview();
                            }
                            
                            this.showMessage('图片已添加到笔记（将在保存时上传）', 'success');
                        };
                        reader.readAsDataURL(file);
                    } else {
                        // 如果笔记已保存，立即上传
                        if (!githubAPI.isConfigured()) {
                            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
                            return;
                        }
                        
                        this.showLoading(true);
                        try {
                            const asset = await githubAPI.uploadImageToDirectory(file, this.currentNotePath);
                            
                            // 插入 Markdown 格式的图片到编辑器
                            const textarea = document.getElementById('noteContent');
                            const markdownImage = `![${asset.name}](${asset.url})`;
                            const cursorPos = textarea.selectionStart;
                            const textBefore = textarea.value.substring(0, cursorPos);
                            const textAfter = textarea.value.substring(cursorPos);
                            textarea.value = textBefore + markdownImage + '\n\n' + textAfter;
                            
                            // 设置光标位置
                            const newPos = cursorPos + markdownImage.length + 2;
                            textarea.setSelectionRange(newPos, newPos);
                            textarea.focus();
                            
                            // 更新预览
                            if (this.isPreviewMode) {
                                this.updatePreview();
                            }
                            
                            this.showMessage('图片已上传', 'success');
                        } catch (error) {
                            this.showMessage(`图片上传失败: ${error.message}`, 'error');
                        } finally {
                            this.showLoading(false);
                        }
                    }
                } catch (error) {
                    this.showMessage(`图片处理失败: ${error.message}`, 'error');
                }
            }
        }
    }

    /**
     * 处理文件拖拽
     */
    async handleFileDrop(e) {
        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;

        // 如果没有打开的笔记，创建一个新笔记
        const noteContent = document.getElementById('noteContent');
        const hasNote = noteContent && (noteContent.value.trim().length > 0 || this.currentNotePath || this.currentNoteId);
        if (!hasNote) {
            this.createNewNote();
        }

        // 分离图片和文件
        const imageFiles = [];
        const otherFiles = [];

        Array.from(files).forEach(file => {
            if (file.type.startsWith('image/')) {
                imageFiles.push(file);
            } else {
                otherFiles.push(file);
            }
        });

        // 处理图片（类似粘贴图片）
        for (const file of imageFiles) {
            await this.handleImageFile(file);
        }

        // 处理其他文件
        if (otherFiles.length > 0) {
            await this.handleFileUpload(otherFiles);
        }
    }

    /**
     * 处理单个图片文件（用于拖拽和粘贴）
     */
    async handleImageFile(file) {
        // 如果是新笔记（未保存），将图片转换为 data URI 存储在本地
        if (!this.currentNotePath) {
            // 读取文件为 data URI
            const reader = new FileReader();
            reader.onload = (event) => {
                const dataUri = event.target.result;
                const imageId = `pending_image_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                
                // 保存到待上传列表
                this.pendingImages.push({
                    id: imageId,
                    file: file,
                    dataUri: dataUri,
                    name: file.name || `image_${Date.now()}.png`
                });
                
                // 插入 Markdown 格式的图片到编辑器（使用 data URI）
                const textarea = document.getElementById('noteContent');
                const markdownImage = `![${file.name || 'image'}](${dataUri})`;
                const cursorPos = textarea.selectionStart;
                const textBefore = textarea.value.substring(0, cursorPos);
                const textAfter = textarea.value.substring(cursorPos);
                textarea.value = textBefore + markdownImage + '\n\n' + textAfter;
                
                // 设置光标位置
                const newPos = cursorPos + markdownImage.length + 2;
                textarea.setSelectionRange(newPos, newPos);
                textarea.focus();
                
                // 更新预览
                if (this.isPreviewMode) {
                    this.updatePreview();
                }
                
                this.showMessage('图片已添加到笔记（将在保存时上传）', 'success');
            };
            reader.readAsDataURL(file);
        } else {
            // 如果笔记已保存，立即上传
            if (!githubAPI.isConfigured()) {
                this.showMessage('请先配置 GitHub Token 和仓库', 'error');
                return;
            }
            
            this.showLoading(true);
            try {
                const asset = await githubAPI.uploadImageToDirectory(file, this.currentNotePath);
                
                // 插入 Markdown 格式的图片到编辑器
                const textarea = document.getElementById('noteContent');
                const markdownImage = `![${asset.name}](${asset.url})`;
                const cursorPos = textarea.selectionStart;
                const textBefore = textarea.value.substring(0, cursorPos);
                const textAfter = textarea.value.substring(cursorPos);
                textarea.value = textBefore + markdownImage + '\n\n' + textAfter;
                
                // 设置光标位置
                const newPos = cursorPos + markdownImage.length + 2;
                textarea.setSelectionRange(newPos, newPos);
                textarea.focus();
                
                // 更新预览
                if (this.isPreviewMode) {
                    this.updatePreview();
                }
                
                this.showMessage('图片已上传', 'success');
            } catch (error) {
                this.showMessage(`图片上传失败: ${error.message}`, 'error');
            } finally {
                this.showLoading(false);
            }
        }
    }

    /**
     * 渲染附件列表
     */
    renderAttachments(attachments) {
        const list = document.getElementById('attachmentsList');
        list.innerHTML = '';

        if (!attachments || attachments.length === 0) {
            return;
        }

        attachments.forEach(attachment => {
            const item = document.createElement('div');
            item.className = 'attachment-item';
            
            const link = document.createElement('a');
            link.href = attachment.url;
            link.target = '_blank';
            link.textContent = attachment.name;
            
            const size = document.createElement('span');
            size.textContent = this.formatFileSize(attachment.size);
            size.style.color = '#999';
            size.style.marginLeft = '5px';

            item.appendChild(link);
            item.appendChild(size);
            list.appendChild(item);
        });
    }

    /**
     * 格式化文件大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * 更新最后保存时间
     */
    updateLastSaved(dateString) {
        const lastSavedEl = document.getElementById('lastSaved');
        if (!lastSavedEl) return;
        
        if (dateString) {
        const date = new Date(dateString);
            lastSavedEl.textContent = `最后保存: ${date.toLocaleString('zh-CN')}`;
            lastSavedEl.style.color = '#888';
            lastSavedEl.style.fontWeight = 'normal';
            this.hasUnsavedChanges = false;
        } else {
            lastSavedEl.textContent = '未保存';
            lastSavedEl.style.color = '#888';
            lastSavedEl.style.fontWeight = 'normal';
        }
    }

    /**
     * 显示加载提示
     */
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    /**
     * 显示消息提示
     */
    showMessage(message, type = 'success') {
        const toast = document.getElementById('messageToast');
        toast.textContent = message;
        toast.className = `toast ${type}`;
        toast.classList.remove('hidden');

        setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }

    /**
     * 从 GitHub 同步笔记（已迁移到目录结构）
     */
    async syncNotesFromGitHub() {
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        this.showLoading(true);

        try {
            // 已迁移到目录结构，重新加载目录树即可
            await this.loadDirectoryTree();
            this.showMessage('同步成功！', 'success');
        } catch (error) {
            this.showMessage(`同步失败: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 合并笔记（智能合并策略）
     */
    mergeNotes(localNotes, remoteNotes) {
        const notesMap = new Map();
        
        // 先添加远程笔记
        remoteNotes.forEach(note => {
            notesMap.set(note.id, note);
        });
        
        // 合并本地笔记：如果本地更新则保留本地，否则保留远程
        localNotes.forEach(localNote => {
            const remoteNote = notesMap.get(localNote.id);
            if (!remoteNote) {
                // 本地独有的笔记
                notesMap.set(localNote.id, localNote);
            } else {
                // 比较更新时间，保留最新的
                const localTime = new Date(localNote.updatedAt).getTime();
                const remoteTime = new Date(remoteNote.updatedAt).getTime();
                if (localTime > remoteTime) {
                    notesMap.set(localNote.id, localNote);
                }
            }
        });
        
        // 转换为数组并按更新时间排序
        return Array.from(notesMap.values()).sort((a, b) => {
            return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
        });
    }

    /**
     * 启动自动同步
     */
    startAutoSync() {
        // 如果已经启动，先清除
        if (this.autoSyncInterval) {
            clearInterval(this.autoSyncInterval);
        }

        // 每 60 秒自动同步到 GitHub（增加间隔，减少 commit 数量）
        // 注意：只在内容真正改变时才会创建 commit
        this.autoSyncInterval = setInterval(async () => {
            if (githubAPI.isConfigured() && !this.syncInProgress) {
                await this.autoSyncToGitHub();
            }
        }, 60000); // 60 秒

        // 页面可见性变化时同步（避免重复绑定）
        if (!this.visibilityHandler) {
            this.visibilityHandler = async () => {
                if (!document.hidden && githubAPI.isConfigured() && !this.syncInProgress) {
                    await this.autoSyncToGitHub();
                }
            };
            document.addEventListener('visibilitychange', this.visibilityHandler);
        }

        // 页面关闭前同步（避免重复绑定）
        if (!this.beforeUnloadHandler) {
            this.beforeUnloadHandler = (e) => {
                // 保存当前笔记到缓存
                this.saveCurrentNoteToCache();
                
                // 检查是否有未保存的修改
                if (this.hasUnsavedChangesInCurrentNote()) {
                    // 显示浏览器默认的退出提示
                    e.preventDefault();
                    e.returnValue = '您有未保存的修改，确定要离开吗？';
                    return e.returnValue;
                }
                
                if (githubAPI.isConfigured() && !this.syncInProgress) {
                    // 同步保存当前笔记到本地
                    if (this.currentNoteId) {
                        const note = this.notes.find(n => n.id === this.currentNoteId);
                        if (note) {
                            note.title = document.getElementById('noteTitle').value || '未命名笔记';
                            note.content = document.getElementById('noteContent').value || '';
                            note.updatedAt = new Date().toISOString();
                        }
                    }
                    localStorage.setItem('notes', JSON.stringify(this.notes));
                    // 尝试同步到 GitHub（不阻塞页面关闭）
                    this.autoSyncToGitHub().catch(() => {});
                }
            };
            window.addEventListener('beforeunload', this.beforeUnloadHandler);
        }
    }

    /**
     * 自动同步到 GitHub（静默同步，只在内容改变时创建 commit）
     */
    async autoSyncToGitHub() {
        if (this.syncInProgress) return;
        
        this.syncInProgress = true;
        try {
            // 先保存当前编辑的笔记到本地
            if (this.currentNoteId) {
                const note = this.notes.find(n => n.id === this.currentNoteId);
                if (note) {
                    const newTitle = document.getElementById('noteTitle').value || '未命名笔记';
                    const newContent = document.getElementById('noteContent').value || '';
                    
                    // 检查内容是否真的改变了
                    if (note.title === newTitle && note.content === newContent) {
                        // 内容没有改变，跳过同步
                        this.syncInProgress = false;
                        return;
                    }
                    
                    note.title = newTitle;
                    note.content = newContent;
                    note.updatedAt = new Date().toISOString();
                }
            }
            
            // 更新本地存储（用于兼容）
            localStorage.setItem('notes', JSON.stringify(this.notes));

            // 已迁移到目录结构，不再使用 saveNotesData()
            // 笔记通过 saveNoteFile() 直接保存到目录中的文件
        } catch (error) {
            console.error('自动同步失败:', error);
            // 静默失败，不显示错误提示（自动同步不应该打扰用户）
            // 但会在控制台记录错误，方便调试
        } finally {
            this.syncInProgress = false;
        }
    }

    /**
     * 设置 Markdown 工具栏
     */
    setupMarkdownToolbar() {
        const toolbar = document.querySelector('.markdown-toolbar');
        if (!toolbar) return;

        // 处理颜色选择器
        const colorBtn = toolbar.querySelector('[data-action="color"]');
        if (colorBtn) {
            const colorPicker = colorBtn.closest('.toolbar-dropdown').querySelector('.color-picker');
            colorBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isVisible = colorPicker.style.display !== 'none';
                // 关闭所有其他下拉菜单
                document.querySelectorAll('.color-picker, .fontsize-picker').forEach(p => {
                    if (p !== colorPicker) p.style.display = 'none';
                });
                colorPicker.style.display = isVisible ? 'none' : 'block';
            });

            // 颜色选项点击
            colorPicker.querySelectorAll('.color-option').forEach(option => {
                option.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const color = option.dataset.color;
                    this.insertColorOrSize('color', color);
                    colorPicker.style.display = 'none';
                });
            });

            // 颜色输入框
            const colorInput = colorPicker.querySelector('.color-input');
            colorInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const color = colorInput.value.trim();
                    if (color) {
                        this.insertColorOrSize('color', color);
                        colorPicker.style.display = 'none';
                        colorInput.value = '';
                    }
                }
            });
        }

        // 处理换行符转换
        const linebreakBtn = toolbar.querySelector('[data-action="linebreak"]');
        if (linebreakBtn) {
            const linebreakPicker = linebreakBtn.closest('.toolbar-dropdown').querySelector('.linebreak-picker');
            const linebreakStatus = linebreakPicker.querySelector('#linebreakStatus');
            
            // 检测当前换行符类型
            const detectLineBreak = () => {
                const textarea = document.getElementById('noteContent');
                const content = textarea.value;
                
                let lfCount = 0;
                let crlfCount = 0;
                let crCount = 0;
                
                for (let i = 0; i < content.length - 1; i++) {
                    if (content[i] === '\r' && content[i + 1] === '\n') {
                        crlfCount++;
                        i++; // 跳过下一个字符
                    } else if (content[i] === '\n') {
                        lfCount++;
                    } else if (content[i] === '\r') {
                        crCount++;
                    }
                }
                
                // 检查最后一个字符
                if (content.length > 0) {
                    const lastChar = content[content.length - 1];
                    if (lastChar === '\n' && content[content.length - 2] !== '\r') {
                        lfCount++;
                    } else if (lastChar === '\r') {
                        crCount++;
                    }
                }
                
                let currentType = '未知';
                let total = lfCount + crlfCount + crCount;
                
                if (total === 0) {
                    currentType = '无换行符';
                } else if (crlfCount > lfCount && crlfCount > crCount) {
                    currentType = 'CRLF (Windows)';
                } else if (lfCount > crCount) {
                    currentType = 'LF (Unix/Linux/Mac)';
                } else if (crCount > 0) {
                    currentType = 'CR (旧版 Mac)';
                } else {
                    currentType = '混合';
                }
                
                linebreakStatus.innerHTML = `
                    <strong>当前换行符类型：</strong>${currentType}<br>
                    <small>LF: ${lfCount} | CRLF: ${crlfCount} | CR: ${crCount}</small>
                `;
            };
            
            linebreakBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                detectLineBreak();
                const isVisible = linebreakPicker.style.display !== 'none';
                // 关闭所有其他下拉菜单
                document.querySelectorAll('.color-picker, .fontsize-picker, .linebreak-picker').forEach(p => {
                    if (p !== linebreakPicker) p.style.display = 'none';
                });
                linebreakPicker.style.display = isVisible ? 'none' : 'block';
            });
            
            // 换行符转换选项
            linebreakPicker.querySelectorAll('.linebreak-option').forEach(option => {
                option.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const targetType = option.dataset.type;
                    this.convertLineBreaks(targetType);
                    linebreakPicker.style.display = 'none';
                });
            });
        }

        // 处理字体大小选择器
        const fontSizeBtn = toolbar.querySelector('[data-action="fontsize"]');
        if (fontSizeBtn) {
            const fontSizePicker = fontSizeBtn.closest('.toolbar-dropdown').querySelector('.fontsize-picker');
            fontSizeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isVisible = fontSizePicker.style.display !== 'none';
                // 关闭所有其他下拉菜单
                document.querySelectorAll('.color-picker, .fontsize-picker').forEach(p => {
                    if (p !== fontSizePicker) p.style.display = 'none';
                });
                fontSizePicker.style.display = isVisible ? 'none' : 'block';
            });

            // 字体大小选项点击
            fontSizePicker.querySelectorAll('.size-option').forEach(option => {
                option.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const size = option.dataset.size;
                    this.insertColorOrSize('fontsize', size);
                    fontSizePicker.style.display = 'none';
                });
            });

            // 字体大小输入框
            const sizeInput = fontSizePicker.querySelector('.size-input');
            sizeInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const size = sizeInput.value.trim();
                    if (size) {
                        this.insertColorOrSize('fontsize', size);
                        fontSizePicker.style.display = 'none';
                        sizeInput.value = '';
                    }
                }
            });
        }

        // 点击外部关闭下拉菜单
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.toolbar-dropdown')) {
                document.querySelectorAll('.color-picker, .fontsize-picker, .linebreak-picker').forEach(p => {
                    p.style.display = 'none';
                });
            }
        });

        // 其他工具栏按钮
        toolbar.addEventListener('click', (e) => {
            const btn = e.target.closest('.toolbar-btn');
            if (!btn || btn.dataset.action === 'color' || btn.dataset.action === 'fontsize' || btn.dataset.action === 'linebreak') return;

            const action = btn.dataset.action;
            const level = btn.dataset.level;
            const type = btn.dataset.type;

            const textarea = document.getElementById('noteContent');
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const selectedText = textarea.value.substring(start, end);
            const beforeText = textarea.value.substring(0, start);
            const afterText = textarea.value.substring(end);

            let insertText = '';
            let newCursorPos = start;

            switch (action) {
                case 'heading':
                    if (selectedText) {
                        // 有选中文本：将选中文本转换为标题，光标定位到标题文本末尾
                        const headingPrefix = '#'.repeat(parseInt(level)) + ' ';
                        insertText = `${headingPrefix}${selectedText}\n\n`;
                        newCursorPos = start + headingPrefix.length + selectedText.length;
                    } else {
                        // 无选中文本：插入标题模板，光标定位到"标题"文本中间
                        insertText = `${'#'.repeat(parseInt(level))} 标题\n\n`;
                        newCursorPos = start + '#'.repeat(parseInt(level)).length + 2; // 定位到"标题"前
                    }
                    break;

                case 'bold':
                    if (selectedText) {
                        // 有选中文本：添加粗体标记，光标定位到文本末尾
                        insertText = `**${selectedText}**`;
                        newCursorPos = start + 2 + selectedText.length;
                    } else {
                        // 无选中文本：插入粗体模板，光标定位到文本中间
                        insertText = `**粗体文本**`;
                        newCursorPos = start + 2; // 定位到"粗体文本"前
                    }
                    break;

                case 'italic':
                    if (selectedText) {
                        // 有选中文本：添加斜体标记，光标定位到文本末尾
                        insertText = `*${selectedText}*`;
                        newCursorPos = start + 1 + selectedText.length;
                    } else {
                        // 无选中文本：插入斜体模板，光标定位到文本中间
                        insertText = `*斜体文本*`;
                        newCursorPos = start + 1; // 定位到"斜体文本"前
                    }
                    break;

                case 'strikethrough':
                    if (selectedText) {
                        // 有选中文本：添加删除线标记，光标定位到文本末尾
                        insertText = `~~${selectedText}~~`;
                        newCursorPos = start + 2 + selectedText.length;
                    } else {
                        // 无选中文本：插入删除线模板，光标定位到文本中间
                        insertText = `~~删除文本~~`;
                        newCursorPos = start + 2; // 定位到"删除文本"前
                    }
                    break;

                case 'code':
                    if (selectedText) {
                        // 有选中文本：添加代码标记，光标定位到文本末尾
                        insertText = `\`${selectedText}\``;
                        newCursorPos = start + 1 + selectedText.length;
                    } else {
                        // 无选中文本：插入代码模板，光标定位到文本中间
                        insertText = `\`代码\``;
                        newCursorPos = start + 1; // 定位到"代码"前
                    }
                    break;

                case 'codeblock':
                    const language = prompt('请输入编程语言（可选，直接回车跳过）:', '');
                    const lang = language ? language.trim() : '';
                    if (selectedText) {
                        // 有选中文本：将文本放入代码块，光标定位到代码块末尾（在结束标记前）
                        insertText = `\`\`\`${lang}\n${selectedText}\n\`\`\`\n\n`;
                        newCursorPos = start + `\`\`\`${lang}\n`.length + selectedText.length;
                    } else {
                        // 无选中文本：插入空代码块，光标定位到代码块内部
                        insertText = `\`\`\`${lang}\n\n\`\`\`\n\n`;
                        newCursorPos = start + `\`\`\`${lang}\n`.length; // 定位到代码块内部（第一行）
                    }
                    break;

                case 'bash':
                    if (selectedText) {
                        // 有选中文本：将文本放入 bash 代码块，光标定位到代码块末尾（在结束标记前）
                        insertText = `\`\`\`bash\n${selectedText}\n\`\`\`\n\n`;
                        newCursorPos = start + '```bash\n'.length + selectedText.length;
                    } else {
                        // 无选中文本：插入空 bash 代码块，光标定位到代码块内部
                        insertText = `\`\`\`bash\n\n\`\`\`\n\n`;
                        newCursorPos = start + '```bash\n'.length; // 定位到代码块内部（第一行）
                    }
                    break;

                case 'link':
                    const linkText = selectedText || prompt('链接文本:', '') || '链接文本';
                    const linkUrl = prompt('链接地址:', 'https://');
                    if (linkUrl) {
                        insertText = `[${linkText}](${linkUrl})`;
                        newCursorPos = start + insertText.length;
                    } else {
                        return; // 用户取消
                    }
                    break;

                case 'image':
                    const altText = selectedText || prompt('图片描述:', '') || '图片';
                    const imageUrl = prompt('图片地址:', 'https://');
                    if (imageUrl) {
                        insertText = `![${altText}](${imageUrl})`;
                        newCursorPos = start + insertText.length;
                    } else {
                        return; // 用户取消
                    }
                    break;

                case 'table':
                    const rows = prompt('表格行数（不包括表头）:', '3') || '3';
                    const cols = prompt('表格列数:', '3') || '3';
                    const rowCount = parseInt(rows) || 3;
                    const colCount = parseInt(cols) || 3;
                    
                    let table = '|';
                    for (let i = 0; i < colCount; i++) {
                        table += ` 列${i + 1} |`;
                    }
                    table += '\n|';
                    for (let i = 0; i < colCount; i++) {
                        table += ' --- |';
                    }
                    for (let r = 0; r < rowCount; r++) {
                        table += '\n|';
                        for (let c = 0; c < colCount; c++) {
                            table += ` 内容 |`;
                        }
                    }
                    table += '\n\n';
                    insertText = table;
                    newCursorPos = start + insertText.length;
                    break;

                case 'list':
                    if (selectedText) {
                        // 有选中文本：转换为列表项，光标定位到文本末尾
                        if (type === 'ul') {
                            insertText = `- ${selectedText}\n`;
                            newCursorPos = start + 2 + selectedText.length;
                        } else {
                            insertText = `1. ${selectedText}\n`;
                            newCursorPos = start + 3 + selectedText.length;
                        }
                    } else {
                        // 无选中文本：插入列表模板，光标定位到文本中间
                        if (type === 'ul') {
                            insertText = `- 列表项\n`;
                            newCursorPos = start + 2; // 定位到"列表项"前
                        } else {
                            insertText = `1. 列表项\n`;
                            newCursorPos = start + 3; // 定位到"列表项"前
                        }
                    }
                    break;

                case 'quote':
                    if (selectedText) {
                        // 有选中文本：转换为引用，光标定位到文本末尾
                        insertText = `> ${selectedText}\n\n`;
                        newCursorPos = start + 2 + selectedText.length;
                    } else {
                        // 无选中文本：插入引用模板，光标定位到文本中间
                        insertText = `> 引用文本\n\n`;
                        newCursorPos = start + 2; // 定位到"引用文本"前
                    }
                    break;

                case 'hr':
                    insertText = '\n---\n\n';
                    newCursorPos = start + 1; // 定位到分隔线后
                    break;

                case 'color':
                case 'fontsize':
                    // 这些由下拉菜单处理
                    return;
            }

            // 插入文本
            textarea.value = beforeText + insertText + afterText;
            
            // 确保光标位置在有效范围内
            const maxPos = textarea.value.length;
            newCursorPos = Math.min(Math.max(0, newCursorPos), maxPos);
            
            // 设置焦点和光标位置
            textarea.focus();
            textarea.setSelectionRange(newCursorPos, newCursorPos);
            
            // 滚动到光标位置（确保光标可见）
            try {
                // 计算光标所在行
                const textBeforeCursor = textarea.value.substring(0, newCursorPos);
                const linesBeforeCursor = textBeforeCursor.split('\n').length;
                const lineHeight = parseInt(window.getComputedStyle(textarea).lineHeight) || 20;
                
                // 滚动到光标位置（保留3行可见区域）
                const scrollTo = Math.max(0, (linesBeforeCursor - 3) * lineHeight);
                textarea.scrollTop = scrollTo;
            } catch (e) {
                // 如果滚动失败，至少确保焦点在文本框
                textarea.scrollTop = 0;
            }

            // 更新预览
            if (this.isPreviewMode) {
                this.updatePreview();
            }
        });
    }

    /**
     * 处理 TAB 键（类似 VSCode 的缩进功能）
     */
    handleTabKey(e) {
        const textarea = e.target;
        
        // 只处理 TAB 键
        if (e.key !== 'Tab') return;
        
        e.preventDefault();
        
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const value = textarea.value;
        const tabSize = 4; // 使用 4 个空格作为缩进
        const indent = ' '.repeat(tabSize);
        const isShiftTab = e.shiftKey;
        
        // 获取选中的文本
        const selectedText = value.substring(start, end);
        const beforeText = value.substring(0, start);
        const afterText = value.substring(end);
        
        // 检查是否选中了多行（通过检查是否包含换行符）
        const hasMultipleLines = selectedText.includes('\n');
        
        if (hasMultipleLines) {
            // 多行处理
            const lines = selectedText.split('\n');
            const linesBefore = beforeText.split('\n');
            const firstLineIndex = linesBefore.length - 1;
            const firstLineStart = beforeText.lastIndexOf('\n') + 1;
            
            // 处理每一行
            const modifiedLines = lines.map((line, index) => {
                if (isShiftTab) {
                    // 减少缩进：移除开头的空格（最多移除 tabSize 个）
                    if (line.startsWith(' ')) {
                        const leadingSpaces = line.match(/^ */)[0].length;
                        const spacesToRemove = Math.min(tabSize, leadingSpaces);
                        return line.substring(spacesToRemove);
                    }
                    return line;
                } else {
                    // 增加缩进：在行首添加缩进
                    return indent + line;
                }
            });
            
            // 重新构建文本
            const newSelectedText = modifiedLines.join('\n');
            const newValue = beforeText + newSelectedText + afterText;
            
            // 计算新的光标位置
            let newStart, newEnd;
            if (isShiftTab) {
                // 减少缩进：光标位置需要调整
                const firstLineOriginal = lines[0];
                const firstLineModified = modifiedLines[0];
                const firstLineDiff = firstLineOriginal.length - firstLineModified.length;
                
                // 计算总减少的字符数
                let totalRemoved = 0;
                lines.forEach((line, idx) => {
                    if (idx === 0) {
                        totalRemoved += firstLineDiff;
                    } else {
                        const original = line;
                        const modified = modifiedLines[idx];
                        totalRemoved += (original.length - modified.length);
                    }
                });
                
                newStart = Math.max(firstLineStart, start - firstLineDiff);
                newEnd = Math.max(newStart, end - totalRemoved);
            } else {
                // 增加缩进：每行都增加了 indent.length 个字符
                const addedChars = lines.length * indent.length;
                newStart = start + (firstLineIndex === 0 ? indent.length : 0);
                newEnd = end + addedChars;
            }
            
            textarea.value = newValue;
            textarea.setSelectionRange(newStart, newEnd);
        } else {
            // 单行处理
            const lineStart = beforeText.lastIndexOf('\n') + 1;
            const lineEnd = afterText.indexOf('\n');
            const lineEndPos = lineEnd === -1 ? value.length : end + lineEnd;
            const currentLine = value.substring(lineStart, lineEndPos);
            
            if (isShiftTab) {
                // 减少缩进
                if (currentLine.startsWith(' ')) {
                    const leadingSpaces = currentLine.match(/^ */)[0].length;
                    const spacesToRemove = Math.min(tabSize, leadingSpaces);
                    const newLine = currentLine.substring(spacesToRemove);
                    const newValue = value.substring(0, lineStart) + newLine + value.substring(lineEndPos);
                    const newStart = Math.max(lineStart, start - spacesToRemove);
                    const newEnd = Math.max(newStart, end - spacesToRemove);
                    
                    textarea.value = newValue;
                    textarea.setSelectionRange(newStart, newEnd);
                }
            } else {
                // 增加缩进
                const newLine = indent + currentLine;
                const newValue = value.substring(0, lineStart) + newLine + value.substring(lineEndPos);
                const newStart = start + indent.length;
                const newEnd = end + indent.length;
                
                textarea.value = newValue;
                textarea.setSelectionRange(newStart, newEnd);
            }
        }
        
        // 更新预览
        if (this.isPreviewMode) {
            this.updatePreview();
        }
    }

    /**
     * 转换换行符
     */
    convertLineBreaks(targetType) {
        const textarea = document.getElementById('noteContent');
        let content = textarea.value;
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        
        // 定义目标换行符
        let targetBreak = '';
        let targetName = '';
        switch (targetType) {
            case 'lf':
                targetBreak = '\n';
                targetName = 'LF (Unix/Linux/Mac)';
                break;
            case 'crlf':
                targetBreak = '\r\n';
                targetName = 'CRLF (Windows)';
                break;
            case 'cr':
                targetBreak = '\r';
                targetName = 'CR (旧版 Mac)';
                break;
            default:
                return;
        }
        
        // 统一转换为目标换行符
        // 先处理 CRLF（必须在 LF 和 CR 之前）
        content = content.replace(/\r\n/g, '\0'); // 临时标记
        // 处理单独的 CR
        content = content.replace(/\r/g, '\0');
        // 处理单独的 LF
        content = content.replace(/\n/g, '\0');
        // 替换为目标换行符
        content = content.replace(/\0/g, targetBreak);
        
        // 更新文本
        textarea.value = content;
        
        // 保持光标位置（如果可能）
        try {
            textarea.setSelectionRange(start, end);
        } catch (e) {
            // 如果位置无效，将光标移到末尾
            textarea.setSelectionRange(content.length, content.length);
        }
        
        // 显示成功消息
        this.showMessage(`已转换为 ${targetName} 换行符`, 'success');
        
        // 更新预览
        if (this.isPreviewMode) {
            this.updatePreview();
        }
    }

    /**
     * 插入颜色或字体大小
     */
    insertColorOrSize(type, value) {
        const textarea = document.getElementById('noteContent');
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const selectedText = textarea.value.substring(start, end);
        const beforeText = textarea.value.substring(0, start);
        const afterText = textarea.value.substring(end);

        let insertText = '';
        let newCursorPos = start;

        if (type === 'color') {
            insertText = `<span style="color: ${value}">${selectedText || '彩色文本'}</span>`;
            newCursorPos = start + (selectedText ? insertText.length : 33);
        } else if (type === 'fontsize') {
            insertText = `<span style="font-size: ${value}">${selectedText || '文本'}</span>`;
            newCursorPos = start + (selectedText ? insertText.length : 31);
        }

        textarea.value = beforeText + insertText + afterText;
        textarea.focus();
        textarea.setSelectionRange(newCursorPos, newCursorPos);

        if (this.isPreviewMode) {
            this.updatePreview();
        }
    }

    /**
     * 切换预览模式
     */
    togglePreviewMode() {
        this.isPreviewMode = !this.isPreviewMode;
        const editor = document.getElementById('noteContent');
        const preview = document.getElementById('markdownPreview');
        const viewBtn = document.getElementById('viewModeBtn');

        if (this.isPreviewMode) {
            // 切换到预览模式
            editor.classList.add('hidden');
            preview.classList.remove('hidden');
            viewBtn.textContent = '✏️ 编辑';
            this.updatePreview();
        } else {
            // 切换到编辑模式
            editor.classList.remove('hidden');
            preview.classList.add('hidden');
            viewBtn.textContent = '👁️ 预览';
        }
    }

    /**
     * 切换全屏模式
     */
    toggleFullscreen() {
        const editorContainer = document.getElementById('editorContainer');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        
        if (!document.fullscreenElement && !document.webkitFullscreenElement && !document.mozFullScreenElement && !document.msFullscreenElement) {
            // 进入全屏
            if (editorContainer.requestFullscreen) {
                editorContainer.requestFullscreen();
            } else if (editorContainer.webkitRequestFullscreen) {
                editorContainer.webkitRequestFullscreen();
            } else if (editorContainer.mozRequestFullScreen) {
                editorContainer.mozRequestFullScreen();
            } else if (editorContainer.msRequestFullscreen) {
                editorContainer.msRequestFullscreen();
            }
        } else {
            // 退出全屏
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                document.mozCancelFullScreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            }
        }
    }

    /**
     * 处理全屏状态变化
     */
    handleFullscreenChange() {
        const editorContainer = document.getElementById('editorContainer');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement);
        
        if (isFullscreen) {
            // 全屏时隐藏侧边栏和头部
            document.querySelector('.sidebar')?.classList.add('hidden');
            document.querySelector('header')?.classList.add('hidden');
            document.querySelector('.config-panel')?.classList.add('hidden');
            
            // 更新按钮文本
            if (fullscreenBtn) {
                fullscreenBtn.textContent = '⛶ 退出全屏';
                fullscreenBtn.title = '退出全屏';
            }
            
            // 添加全屏样式类
            editorContainer?.classList.add('fullscreen-mode');
            document.body.classList.add('fullscreen-active');
        } else {
            // 恢复侧边栏和头部
            document.querySelector('.sidebar')?.classList.remove('hidden');
            document.querySelector('header')?.classList.remove('hidden');
            
            // 更新按钮文本
            if (fullscreenBtn) {
                fullscreenBtn.textContent = '⛶ 全屏';
                fullscreenBtn.title = '全屏模式';
            }
            
            // 移除全屏样式类
            editorContainer?.classList.remove('fullscreen-mode');
            document.body.classList.remove('fullscreen-active');
        }
    }

    /**
     * 更新 Markdown 预览
     */
    updatePreview() {
        const content = document.getElementById('noteContent').value;
        const preview = document.getElementById('markdownPreview');
        
        if (typeof marked !== 'undefined') {
            // 抽取所有数学块为占位符，避免 marked 修改 TeX 的反斜线或换行
            const mathBlocks = [];
            let idx = 0;
            const mathRegex = /\$\$[\s\S]+?\$\$|\$[\s\S]+?\$/g;
            const contentWithPlaceholders = content.replace(mathRegex, (match) => {
                const key = `@@MATH${idx}@@`;
                mathBlocks.push({key, text: match});
                idx += 1;
                return key;
            });

            // 使用 marked 渲染含占位符的内容
            preview.innerHTML = marked.parse(contentWithPlaceholders);

            // 将占位符替换回原始 TeX 文本（作为文本/HTML 片段），以便 auto-render 能识别
            try {
                mathBlocks.forEach(({key, text}) => {
                    // 直接替换字符串中的占位符为原始 TeX（包含 $ 或 $$）
                    preview.innerHTML = preview.innerHTML.split(key).join(text);
                });

                // 使用 KaTeX auto-render 渲染替换后的 DOM
                if (typeof renderMathInElement !== 'undefined') {
                    renderMathInElement(preview, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false}
                        ],
                        throwOnError: false
                    });
                } else if (typeof katex !== 'undefined') {
                    // 回退：没有 auto-render 时，尽量用 katex 渲染所有 $$...$$ 和 $...$
                    let html = preview.innerHTML;
                    html = html.replace(/\$\$([\s\S]+?)\$\$/g, (m, expr) => {
                        try { return katex.renderToString(expr, {displayMode: true, throwOnError: false}); } catch (e) { return m; }
                    });
                    html = html.replace(/\$([\s\S]+?)\$/g, (m, expr) => {
                        try { return katex.renderToString(expr, {displayMode: false, throwOnError: false}); } catch (e) { return m; }
                    });
                    preview.innerHTML = html;
                }
            } catch (e) {
                console.warn('数学渲染出错:', e);
            }

            // 为代码块添加复制功能
            this.addCopyButtonToCodeBlocks(preview);
        } else {
            // 如果没有 marked.js，显示原始文本
            preview.textContent = content;
        }
    }

    /**
     * 为预览中的代码块添加复制按钮
     */
    addCopyButtonToCodeBlocks(preview) {
        // 查找所有代码块（pre 元素）
        const preElements = preview.querySelectorAll('pre');
        
        preElements.forEach((preElement) => {
            // 检查是否已经添加了复制按钮
            if (preElement.querySelector('.copy-code-btn')) {
                return;
            }
            
            // 获取代码内容
            const codeElement = preElement.querySelector('code');
            const codeText = codeElement ? (codeElement.textContent || codeElement.innerText) : (preElement.textContent || preElement.innerText);
            
            // 创建复制按钮
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-code-btn';
            copyBtn.innerHTML = '📋 复制';
            copyBtn.title = '复制代码';
            
            // 设置按钮样式
            copyBtn.style.cssText = `
                position: absolute;
                top: 8px;
                right: 8px;
                padding: 6px 12px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                opacity: 0.7;
                transition: all 0.2s;
                z-index: 10;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            `;
            
            // 鼠标悬停效果
            copyBtn.addEventListener('mouseenter', () => {
                copyBtn.style.opacity = '1';
                copyBtn.style.transform = 'translateY(-2px)';
                copyBtn.style.boxShadow = '0 2px 8px rgba(102, 126, 234, 0.4)';
            });
            
            copyBtn.addEventListener('mouseleave', () => {
                copyBtn.style.opacity = '0.7';
                copyBtn.style.transform = 'translateY(0)';
                copyBtn.style.boxShadow = 'none';
            });
            
            // 复制功能
            copyBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                e.preventDefault();
                
                try {
                    await navigator.clipboard.writeText(codeText);
                    
                    // 显示成功提示
                    const originalText = copyBtn.innerHTML;
                    const originalBg = copyBtn.style.background;
                    copyBtn.innerHTML = '✅ 已复制';
                    copyBtn.style.background = '#27ae60';
                    
                    setTimeout(() => {
                        copyBtn.innerHTML = originalText;
                        copyBtn.style.background = originalBg;
                    }, 2000);
                } catch (err) {
                    // 降级方案：使用传统方法
                    const textArea = document.createElement('textarea');
                    textArea.value = codeText;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-9999px';
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    textArea.select();
                    
                    try {
                        const successful = document.execCommand('copy');
                        if (successful) {
                            copyBtn.innerHTML = '✅ 已复制';
                            copyBtn.style.background = '#27ae60';
                            setTimeout(() => {
                                copyBtn.innerHTML = '📋 复制';
                                copyBtn.style.background = '#667eea';
                            }, 2000);
                        } else {
                            throw new Error('复制失败');
                        }
                    } catch (err) {
                        copyBtn.innerHTML = '❌ 失败';
                        copyBtn.style.background = '#e74c3c';
                        setTimeout(() => {
                            copyBtn.innerHTML = '📋 复制';
                            copyBtn.style.background = '#667eea';
                        }, 2000);
                    }
                    
                    document.body.removeChild(textArea);
                }
            });
            
            // 将代码块容器设置为相对定位
            preElement.style.position = 'relative';
            preElement.appendChild(copyBtn);
        });
    }

    /**
     * 加载目录树（从 root 目录）
     */
    async loadDirectoryTree(forceRefresh = false) {
        if (!githubAPI.isConfigured()) {
            // 如果未配置，使用旧的笔记列表
            this.renderNotesList();
            return;
        }

        // 标记是否正在加载，避免重复加载
        if (this._loadingTree) {
            return;
        }
        this._loadingTree = true;

        try {
            // 如果强制刷新，先清空容器
            if (forceRefresh) {
                const container = document.getElementById('directoryTree') || document.getElementById('notesList');
                if (container) {
                    container.innerHTML = '';
                    // 立即显示空的 root 目录
                    const emptyRootTree = [{
                        type: 'dir',
                        name: 'root',
                        path: 'root',
                        children: []
                    }];
                    this.renderDirectoryTree(emptyRootTree);
                }
            } else {
                // 非强制刷新时，确保至少显示 root 目录（如果还没有显示）
                const container = document.getElementById('directoryTree') || document.getElementById('notesList');
                if (container && !container.querySelector('[data-path="root"]')) {
            const emptyRootTree = [{
                type: 'dir',
                name: 'root',
                path: 'root',
                children: []
            }];
            this.renderDirectoryTree(emptyRootTree);
                }
            }

            // 异步检查并创建 root 目录（不阻塞显示）
            githubAPI.createDirectory('root').catch(error => {
                // 目录可能已存在，忽略错误
            });

            // 从 GitHub 获取目录结构
            // 如果是强制刷新，绕过缓存获取最新数据
            let tree = [];
            try {
                tree = await githubAPI.getDirectoryTreeFast('root', null, forceRefresh);
            } catch (error) {
                // 如果获取失败，使用空数组（确保 root 目录仍然显示）
                console.warn('获取 root 目录内容失败，使用空目录:', error);
                tree = [];
            }
            
            // 确保 tree 是数组
            if (!Array.isArray(tree)) {
                tree = [];
            }
            
            // 过滤掉 files 和 images 目录
            const filteredTree = this.filterTree(tree, ['files', 'images']);
            
            // 将 root 目录本身也显示出来，包装成一个包含 root 的树结构
            // 即使 filteredTree 为空，root 目录也要显示
            const rootTree = [{
                type: 'dir',
                name: 'root',
                path: 'root',
                children: Array.isArray(filteredTree) ? filteredTree : []
            }];
            
            // 只在非强制刷新时渲染（强制刷新时已经渲染了空目录）
            if (!forceRefresh) {
            // 渲染目录结构（使用文件名作为标题）
            this.renderDirectoryTree(rootTree);
            } else {
                // 强制刷新时，更新现有目录树（增量更新）
                this.updateDirectoryTree(rootTree);
            }
            
            // 然后异步加载文件标题并更新显示（不阻塞初始显示）
            // 使用防抖，避免频繁更新
            if (this._titleLoadTimeout) {
                clearTimeout(this._titleLoadTimeout);
            }
            this._titleLoadTimeout = setTimeout(() => {
            if (forceRefresh) {
                    this.loadDirectoryTreeTitles(true).finally(() => {
                        this._loadingTree = false;
                    });
            } else {
                    this.loadDirectoryTreeTitles(false).finally(() => {
                        this._loadingTree = false;
                    });
            }
            }, 300); // 延迟300ms加载标题，避免立即刷新
            
        } catch (error) {
            console.warn('加载目录树失败:', error);
            this._loadingTree = false;
            // 即使出错，也显示 root 目录（空目录）
            const rootTree = [{
                type: 'dir',
                name: 'root',
                path: 'root',
                children: []
            }];
            this.renderDirectoryTree(rootTree);
        }
    }

    /**
     * 异步加载目录树中的文件标题（不阻塞初始显示）
     */
    async loadDirectoryTreeTitles(bypassCache = false) {
        if (!githubAPI.isConfigured()) {
            return;
        }

        try {
            // 加载带标题的完整目录树
            const tree = await githubAPI.getDirectoryTreeWithTitles('root', null, bypassCache);
            
            // 确保 tree 是数组
            if (!Array.isArray(tree)) {
                return; // 如果获取失败，保持现有显示
            }
            
            // 过滤掉 files 和 images 目录
            const filteredTree = this.filterTree(tree, ['files', 'images']);
            
            // 将 root 目录本身也显示出来
            // 即使 filteredTree 为空，root 目录也要显示
            const rootTree = [{
                type: 'dir',
                name: 'root',
                path: 'root',
                children: Array.isArray(filteredTree) ? filteredTree : []
            }];
            
            // 增量更新标题（只更新文件名显示，不重新渲染整个树）
            this.updateDirectoryTreeTitles(rootTree);
        } catch (error) {
            console.warn('加载目录树标题失败:', error);
            // 失败不影响使用，保持现有显示
        }
    }

    /**
     * 增量更新目录树标题（只更新文件名显示，不重新渲染整个树）
     */
    updateDirectoryTreeTitles(tree) {
        const container = document.getElementById('directoryTree') || document.getElementById('notesList');
        if (!container) return;

        // 递归更新标题
        const updateItemTitle = (item, element) => {
            if (!element) return;
            
            const nameSpan = element.querySelector('.tree-item-name');
            if (nameSpan && item.type === 'file' && item.name.endsWith('.md') && item.title) {
                // 只有当标题不同时才更新，避免闪烁
                if (nameSpan.textContent !== item.title) {
                    nameSpan.textContent = item.title;
                    nameSpan.setAttribute('title', `${item.title} (${item.name})`);
                }
            }
            
            // 递归处理子项
            if (item.children && item.children.length > 0) {
                const childrenDiv = element.nextElementSibling;
                if (childrenDiv && childrenDiv.classList.contains('tree-children')) {
                    const childElements = childrenDiv.querySelectorAll('.tree-item');
                    item.children.forEach((child, index) => {
                        if (index < childElements.length) {
                            updateItemTitle(child, childElements[index]);
                        }
                    });
                }
            }
        };

        // 更新根目录
        const rootElement = container.querySelector('[data-path="root"]');
        if (rootElement && tree.length > 0) {
            updateItemTitle(tree[0], rootElement);
        }
    }

    /**
     * 更新目录树（增量更新，避免完全重新渲染）
     */
    updateDirectoryTree(newTree) {
        const container = document.getElementById('directoryTree') || document.getElementById('notesList');
        if (!container) return;

        // 检查当前树结构是否相同（只比较路径）
        const currentItems = Array.from(container.querySelectorAll('.tree-item')).map(el => ({
            path: el.getAttribute('data-path'),
            type: el.getAttribute('data-type')
        }));

        const newItems = [];
        const collectItems = (tree, parentPath = '') => {
            tree.forEach(item => {
                newItems.push({ path: item.path, type: item.type });
                if (item.children && item.children.length > 0) {
                    collectItems(item.children, item.path);
                }
            });
        };
        collectItems(newTree);

        // 如果结构相同，只更新标题；否则完全重新渲染
        const structureChanged = currentItems.length !== newItems.length ||
            currentItems.some((item, index) => 
                !newItems[index] || 
                item.path !== newItems[index].path || 
                item.type !== newItems[index].type
            );

        if (structureChanged) {
            // 结构改变，需要重新渲染
            this.renderDirectoryTree(newTree);
        } else {
            // 结构相同，只更新标题
            this.updateDirectoryTreeTitles(newTree);
        }
    }

    /**
     * 过滤目录树，移除指定的目录（files 和 images）
     */
    filterTree(tree, excludeNames) {
        if (!tree || !Array.isArray(tree)) {
            return [];
        }
        
        return tree
            .filter(item => {
                // 过滤掉 files 和 images 目录（递归过滤所有层级）
                if (item.type === 'dir' && excludeNames.includes(item.name)) {
                    return false;
                }
                return true;
            })
            .map(item => {
                // 递归过滤子项
                if (item.type === 'dir' && item.children && item.children.length > 0) {
                    return {
                        ...item,
                        children: this.filterTree(item.children, excludeNames)
                    };
                }
                return item;
            });
    }

    /**
     * 渲染目录树
     */
    renderDirectoryTree(tree, container = null, level = 0) {
        const list = container || document.getElementById('directoryTree') || document.getElementById('notesList');
        if (!list) return;

        if (level === 0) {
            list.innerHTML = '';
        }

        tree.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = `tree-item ${item.type === 'dir' ? 'tree-item-folder' : 'tree-item-file'}`;
            itemDiv.style.paddingLeft = `${level * 20}px`;
            itemDiv.setAttribute('data-path', item.path);
            itemDiv.setAttribute('data-type', item.type);

            const name = item.name;

            // 转义路径中的特殊字符，避免 XSS
            const escapedPath = item.path.replace(/"/g, '&quot;');
            const escapedName = name.replace(/</g, '&lt;').replace(/>/g, '&gt;');

            // 创建图标容器
            const iconSpan = document.createElement('span');
            iconSpan.className = `tree-item-icon ${item.type === 'dir' ? 'icon-folder' : 'icon-file'}`;
            iconSpan.setAttribute('aria-label', item.type === 'dir' ? '文件夹' : '文件');
            
            // 如果是文件夹，添加展开/折叠图标
            if (item.type === 'dir') {
                const expandIcon = document.createElement('span');
                expandIcon.className = 'tree-expand-icon';
                expandIcon.textContent = item.children && item.children.length > 0 ? '▶' : ' ';
                expandIcon.setAttribute('data-expanded', 'false');
                iconSpan.appendChild(expandIcon);
                
                const folderIcon = document.createElement('span');
                folderIcon.textContent = '📁';
                iconSpan.appendChild(folderIcon);
            } else {
                iconSpan.textContent = '📄';
            }

            // 创建名称元素
            const nameSpan = document.createElement('span');
            nameSpan.className = 'tree-item-name';
            // 如果是笔记文件，显示标题而不是文件名
            if (item.type === 'file' && item.name.endsWith('.md') && item.title) {
                nameSpan.textContent = item.title;
                nameSpan.setAttribute('title', `${item.title} (${item.name})`);
            } else {
                nameSpan.textContent = escapedName;
                nameSpan.setAttribute('title', escapedName);
            }

            // 创建操作按钮容器
            const actionsSpan = document.createElement('span');
            actionsSpan.className = 'tree-item-actions';

            // 如果是文件夹，添加新建笔记按钮
            if (item.type === 'dir') {
                const newNoteBtn = document.createElement('button');
                newNoteBtn.className = 'btn-new-note';
                newNoteBtn.setAttribute('data-path', item.path);
                newNoteBtn.title = '在此文件夹新建笔记';
                newNoteBtn.textContent = '+';
                actionsSpan.appendChild(newNoteBtn);
            }

            // 添加删除按钮（root 目录不能删除）
            let deleteBtn = null;
            if (item.path !== 'root') {
                deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn-delete';
                deleteBtn.setAttribute('data-path', item.path);
                deleteBtn.setAttribute('data-type', item.type);
                deleteBtn.title = '删除';
                deleteBtn.textContent = '🗑️';
                actionsSpan.appendChild(deleteBtn);
            }

            // 组装元素
            itemDiv.appendChild(iconSpan);
            itemDiv.appendChild(nameSpan);
            itemDiv.appendChild(actionsSpan);

            // 点击事件
            itemDiv.addEventListener('click', (e) => {
                if (e.target.classList.contains('btn-delete') || e.target.classList.contains('btn-new-note')) {
                    return; // 让按钮处理自己的事件
                }
                if (item.type === 'file') {
                    this.openNoteFile(item.path);
                } else {
                    // 切换文件夹展开/折叠
                    const children = itemDiv.nextElementSibling;
                    if (children && children.classList.contains('tree-children')) {
                        const isHidden = children.classList.contains('hidden');
                        children.classList.toggle('hidden');
                        
                        // 更新展开/折叠图标
                        const expandIcon = iconSpan.querySelector('.tree-expand-icon');
                        if (expandIcon) {
                            expandIcon.textContent = isHidden ? '▼' : '▶';
                        }
                    }
                }
            });

            // 删除按钮事件（只有在 deleteBtn 存在时才添加）
            if (deleteBtn) {
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (confirm(`确定要删除${item.type === 'dir' ? '文件夹' : '文件'} "${name}" 吗？`)) {
                        await this.deleteFileOrFolder(item.path, item.type);
                        // deleteFileOrFolder 内部已经会刷新目录树
                    }
                });
            }

            // 新建笔记按钮事件（如果是文件夹，newNoteBtn 已经在上面创建了）
            if (item.type === 'dir') {
                const newNoteBtn = actionsSpan.querySelector('.btn-new-note');
                if (newNoteBtn) {
                    newNoteBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        // item.path 是目录路径，直接使用
                        this.showNewNoteDialog(item.path);
                    });
                }
            }

            list.appendChild(itemDiv);

            // 如果有子项，递归渲染（默认折叠）
            if (item.children && item.children.length > 0) {
                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children hidden';
                list.appendChild(childrenDiv);
                this.renderDirectoryTree(item.children, childrenDiv, level + 1);
            }
        });
    }

    /**
     * 保存当前笔记到缓存
     */
    saveCurrentNoteToCache() {
        const title = document.getElementById('noteTitle')?.value || '';
        const content = document.getElementById('noteContent')?.value || '';
        
        if (!title && !content) {
            return; // 空内容不缓存
        }
        
        const cacheKey = this.currentNotePath || 'new_note';
        this.noteCache.set(cacheKey, {
            title: title,
            content: content,
            path: this.currentNotePath,
            timestamp: Date.now()
        });
        
        // 保存到 localStorage（持久化）
        try {
            const cacheData = {};
            this.noteCache.forEach((value, key) => {
                cacheData[key] = value;
            });
            localStorage.setItem('note_cache', JSON.stringify(cacheData));
        } catch (e) {
            console.warn('保存缓存到 localStorage 失败:', e);
        }
    }

    /**
     * 从缓存恢复笔记
     */
    restoreNoteFromCache(filePath) {
        const cacheKey = filePath || 'new_note';
        const cached = this.noteCache.get(cacheKey);
        
        if (cached) {
            return cached;
        }
        
        // 尝试从 localStorage 恢复
        try {
            const cacheDataStr = localStorage.getItem('note_cache');
            if (cacheDataStr) {
                const cacheData = JSON.parse(cacheDataStr);
                if (cacheData[cacheKey]) {
                    this.noteCache.set(cacheKey, cacheData[cacheKey]);
                    return cacheData[cacheKey];
                }
            }
        } catch (e) {
            console.warn('从 localStorage 恢复缓存失败:', e);
        }
        
        return null;
    }

    /**
     * 检查是否有未保存的修改
     */
    hasUnsavedChangesInCurrentNote() {
        if (!this.currentNotePath && !this.currentNoteId) {
            // 新笔记，检查是否有内容
            const title = document.getElementById('noteTitle')?.value || '';
            const content = document.getElementById('noteContent')?.value || '';
            return title.trim() !== '' || content.trim() !== '# 未命名笔记\n\n';
        }
        
        // 已存在的笔记，检查是否与缓存或原始内容不同
        const title = document.getElementById('noteTitle')?.value || '';
        const content = document.getElementById('noteContent')?.value || '';
        
        const cached = this.restoreNoteFromCache(this.currentNotePath);
        if (cached) {
            return cached.title !== title || cached.content !== content;
        }
        
        // 如果没有缓存，认为有修改（保守策略）
        return true;
    }

    /**
     * 打开笔记文件
     */
    async openNoteFile(filePath) {
        // 在打开新笔记前，先保存当前笔记到缓存
        if (this.currentNotePath || this.currentNoteId) {
            this.saveCurrentNoteToCache();
        }
        
        // 检查是否有未保存的修改
        if (this.hasUnsavedChangesInCurrentNote()) {
            const shouldContinue = confirm('当前笔记有未保存的修改，是否继续打开新笔记？\n\n未保存的内容已自动缓存，稍后可以恢复。');
            if (!shouldContinue) {
                return;
            }
        }
        
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        // 先尝试从缓存恢复
        const cached = this.restoreNoteFromCache(filePath);
        if (cached) {
            // 从缓存恢复
            this.currentNotePath = filePath;
            this.currentNoteId = filePath;
            document.getElementById('noteTitle').value = cached.title;
            document.getElementById('noteContent').value = cached.content;
            document.getElementById('notePath').textContent = `路径: ${filePath} (已缓存)`;
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('editorContainer').classList.remove('hidden');
            
            if (this.isPreviewMode) {
                this.updatePreview();
            }
            
            // 在后台加载最新内容
            this.loadNoteFileInBackground(filePath);
            return;
        }

        this.showLoading(true);
        try {
            const fileData = await githubAPI.readNoteFile(filePath);
            if (!fileData) {
                this.showMessage('文件不存在', 'error');
                return;
            }

            // 显示文件内容
            this.currentNotePath = filePath;
            this.currentNoteId = filePath; // 使用路径作为 ID
            
            // 从内容中提取标题
            let title = fileData.name.replace(/\.md$/, '');
            const firstLine = fileData.content.trim().split('\n')[0];
            if (firstLine.startsWith('#')) {
                title = firstLine.replace(/^#+\s*/, '').trim();
            }
            
            document.getElementById('noteTitle').value = title;
            document.getElementById('noteContent').value = fileData.content;
            document.getElementById('notePath').textContent = `路径: ${filePath}`;

            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('editorContainer').classList.remove('hidden');

            // 保存到缓存
            this.saveCurrentNoteToCache();

            if (this.isPreviewMode) {
                this.updatePreview();
            }
        } catch (error) {
            this.showMessage(`打开文件失败: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 在后台加载笔记文件（用于更新缓存）
     */
    async loadNoteFileInBackground(filePath) {
        try {
            const fileData = await githubAPI.readNoteFile(filePath);
            if (fileData) {
                // 检查缓存内容是否与服务器内容不同
                const cached = this.restoreNoteFromCache(filePath);
                if (cached && cached.content !== fileData.content) {
                    // 内容不同，提示用户
                    const useCached = confirm('检测到服务器上的文件已更新，是否使用缓存的内容？\n\n点击"确定"使用缓存内容，点击"取消"使用服务器最新内容。');
                    if (!useCached) {
                        // 使用服务器内容
                        let title = fileData.name.replace(/\.md$/, '');
                        const firstLine = fileData.content.trim().split('\n')[0];
                        if (firstLine.startsWith('#')) {
                            title = firstLine.replace(/^#+\s*/, '').trim();
                        }
                        document.getElementById('noteTitle').value = title;
                        document.getElementById('noteContent').value = fileData.content;
                        this.saveCurrentNoteToCache();
                        if (this.isPreviewMode) {
                            this.updatePreview();
                        }
                    }
                } else if (!cached) {
                    // 没有缓存，直接更新
                    let title = fileData.name.replace(/\.md$/, '');
                    const firstLine = fileData.content.trim().split('\n')[0];
                    if (firstLine.startsWith('#')) {
                        title = firstLine.replace(/^#+\s*/, '').trim();
                    }
                    document.getElementById('noteTitle').value = title;
                    document.getElementById('noteContent').value = fileData.content;
                    this.saveCurrentNoteToCache();
                    document.getElementById('notePath').textContent = `路径: ${filePath}`;
                    if (this.isPreviewMode) {
                        this.updatePreview();
                    }
                }
            }
        } catch (error) {
            console.warn('后台加载文件失败:', error);
        }
    }

    /**
     * 显示新建文件夹对话框
     */
    showNewFolderDialog(parentPath = 'root') {
        // parentPath 默认为 'root' 目录
        document.getElementById('folderPath').value = parentPath || 'root';
        document.getElementById('folderName').value = '';
        document.getElementById('newFolderDialog').classList.remove('hidden');
    }

    /**
     * 隐藏新建文件夹对话框
     */
    hideNewFolderDialog() {
        document.getElementById('newFolderDialog').classList.add('hidden');
    }

    /**
     * 创建文件夹
     */
    async createFolder() {
        const folderName = document.getElementById('folderName').value.trim();
        let parentPath = document.getElementById('folderPath').value.trim() || 'root';

        if (!folderName) {
            this.showMessage('请输入文件夹名称', 'error');
            return;
        }

        // 验证 parentPath 是目录路径，不是文件路径
        if (parentPath.endsWith('.md')) {
            const lastSlash = parentPath.lastIndexOf('/');
            parentPath = lastSlash > 0 ? parentPath.substring(0, lastSlash) : 'root';
        }
        
        // 检查 parentPath 是否包含文件名（如 root/note_xxx），如果是则提取目录
        const pathParts = parentPath.split('/');
        const lastPart = pathParts[pathParts.length - 1];
        if (lastPart.startsWith('note_') && !lastPart.includes('.')) {
            parentPath = pathParts.slice(0, -1).join('/') || 'root';
        }
        
        // 验证 parentPath 必须在 root 目录下或其子目录下
        if (!parentPath.startsWith('root')) {
            this.showMessage('保存路径必须在 root 目录下或其子目录下', 'error');
            return;
        }
        
        // 文件夹创建在 root 目录下
        const folderPath = parentPath ? `${parentPath}/${folderName}` : `root/${folderName}`;

        this.showLoading(true);
        try {
            await githubAPI.createDirectory(folderPath);
            this.showMessage('文件夹创建成功', 'success');
            this.hideNewFolderDialog();
            
            // 等待一小段时间，确保 GitHub API 的更改已生效
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // 强制刷新目录树（重新从 GitHub 获取）
            await this.loadDirectoryTree(true);
        } catch (error) {
            this.showMessage(`创建文件夹失败: ${error.message}`, 'error');
            // 即使失败也刷新，确保 UI 状态正确
            await new Promise(resolve => setTimeout(resolve, 500));
            await this.loadDirectoryTree(true);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * 显示新建笔记对话框
     */
    showNewNoteDialog(parentPath = 'root') {
        // parentPath 默认为 'root' 目录
        // 验证 parentPath 是目录路径，不是文件路径
        let validatedPath = parentPath || 'root';
        
        // 如果 parentPath 以 .md 结尾，说明是文件路径，需要提取目录部分
        if (validatedPath.endsWith('.md')) {
            const lastSlash = validatedPath.lastIndexOf('/');
            validatedPath = lastSlash > 0 ? validatedPath.substring(0, lastSlash) : 'root';
        }
        
        // 检查 parentPath 是否包含文件名（如 root/note_xxx），如果是则提取目录
        const pathParts = validatedPath.split('/');
        const lastPart = pathParts[pathParts.length - 1];
        // 如果最后一部分看起来像笔记文件名（以 note_ 开头且没有扩展名），可能是误传
        if (lastPart.startsWith('note_') && !lastPart.includes('.')) {
            validatedPath = pathParts.slice(0, -1).join('/') || 'root';
        }
        
        document.getElementById('newNotePath').value = validatedPath;
        document.getElementById('newNoteName').value = '';
        document.getElementById('newNoteDialog').classList.remove('hidden');
    }

    /**
     * 隐藏新建笔记对话框
     */
    hideNewNoteDialog() {
        document.getElementById('newNoteDialog').classList.add('hidden');
    }

    /**
     * 创建笔记（从对话框，仅在本地缓存）
     */
    async createNote() {
        const noteName = document.getElementById('newNoteName').value.trim();
        let parentPath = document.getElementById('newNotePath').value.trim() || 'root';
        
        // 清理路径：去除双斜杠、去除空部分、去除末尾斜杠
        parentPath = parentPath.replace(/\/+/g, '/').replace(/\/$/, '');
        if (!parentPath) parentPath = 'root';
        
        // 验证 parentPath 是目录路径，不是文件路径
        if (parentPath.endsWith('.md')) {
            const lastSlash = parentPath.lastIndexOf('/');
            parentPath = lastSlash > 0 ? parentPath.substring(0, lastSlash) : 'root';
        }
        
        // 检查 parentPath 是否包含文件名（如 root/note_xxx），如果是则提取目录
        const pathParts = parentPath.split('/').filter(part => part && part.trim());
        const lastPart = pathParts[pathParts.length - 1];
        if (lastPart && lastPart.startsWith('note_') && !lastPart.includes('.')) {
            pathParts.pop();
            parentPath = pathParts.length > 0 ? pathParts.join('/') : 'root';
        } else {
            parentPath = pathParts.join('/') || 'root';
        }

        if (!noteName) {
            this.showMessage('请输入笔记名称', 'error');
            return;
        }

        // 验证 parentPath 必须在 root 目录下或其子目录下
        if (!parentPath.startsWith('root')) {
            parentPath = 'root';
        }
        
        // 再次清理路径
        parentPath = parentPath.replace(/\/+/g, '/').replace(/\/$/, '');

        // 新建笔记只在本地缓存，不提交到 GitHub
        // 保存 parentPath 到临时变量，用于后续保存时使用
        this.tempParentPath = parentPath;
        this.currentNotePath = null; // null 表示新笔记，还未保存
        this.currentNoteId = null;

        // 更新 UI
        const initialContent = `# ${noteName}\n\n`;
        document.getElementById('noteTitle').value = noteName;
        document.getElementById('noteContent').value = initialContent;
        document.getElementById('notePath').textContent = `路径: 未保存（新笔记）`;

        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('editorContainer').classList.remove('hidden');

        // 关闭对话框
        this.hideNewNoteDialog();

        this.showMessage('笔记已创建（本地），点击保存提交到 GitHub', 'success');
        
        // 刷新目录树（虽然新笔记还未保存，但刷新可以确保 UI 状态正确）
        await this.loadDirectoryTree();
    }

    /**
     * 删除文件或文件夹
     */
    async deleteFileOrFolder(path, type) {
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        this.showLoading(true);
        try {
            await githubAPI.deleteFile(path, null, type === 'dir' ? 'Delete folder' : 'Delete file');
            this.showMessage(`${type === 'dir' ? '文件夹' : '文件'}删除成功`, 'success');
            
            // 等待更长时间，确保 GitHub API 的更改已生效（GitHub 可能有缓存）
            // 对于目录删除，需要更长的等待时间
            const waitTime = type === 'dir' ? 2000 : 1000;
            await new Promise(resolve => setTimeout(resolve, waitTime));
            
            // 强制刷新目录树（绕过缓存，重新从 GitHub 获取）
            await this.loadDirectoryTree(true);
            
            // 再次等待并刷新一次，确保数据同步
            await new Promise(resolve => setTimeout(resolve, 500));
            await this.loadDirectoryTree(true);
        } catch (error) {
            this.showMessage(`删除失败: ${error.message}`, 'error');
            // 即使失败也刷新，确保 UI 状态正确
            await new Promise(resolve => setTimeout(resolve, 1000));
            await this.loadDirectoryTree(true);
        } finally {
            this.showLoading(false);
        }
    }
}

// 检查登录状态
function checkAuth() {
    const token = localStorage.getItem('github_token');
    const repo = localStorage.getItem('github_repo');
    
    // 如果未登录，跳转到登录页面
    if (!token || !repo) {
        // 保存当前页面路径，登录后可以返回
        const currentPath = window.location.pathname;
        if (currentPath !== '/login.html' && !currentPath.endsWith('login.html')) {
            window.location.href = 'login.html';
            return false;
        }
    }
    
    return true;
}

// 立即显示目录树框架（在应用初始化之前）
function showInitialTree() {
    const container = document.getElementById('directoryTree') || document.getElementById('notesList');
    if (container && container.innerHTML.trim() === '') {
        const rootTree = [{
            type: 'dir',
            name: 'root',
            path: 'root',
            children: []
        }];
        // 创建临时渲染函数
        const tempRender = (tree, cont = container, level = 0) => {
            if (level === 0) {
                cont.innerHTML = '';
            }
            tree.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = `tree-item ${item.type === 'dir' ? 'tree-item-folder' : 'tree-item-file'}`;
                itemDiv.style.paddingLeft = `${level * 20}px`;
                itemDiv.setAttribute('data-path', item.path);
                itemDiv.setAttribute('data-type', item.type);
                
                const iconSpan = document.createElement('span');
                iconSpan.className = `tree-item-icon ${item.type === 'dir' ? 'icon-folder' : 'icon-file'}`;
                if (item.type === 'dir') {
                    const expandIcon = document.createElement('span');
                    expandIcon.className = 'tree-expand-icon';
                    expandIcon.textContent = ' ';
                    iconSpan.appendChild(expandIcon);
                    const folderIcon = document.createElement('span');
                    folderIcon.textContent = '📁';
                    iconSpan.appendChild(folderIcon);
                } else {
                    iconSpan.textContent = '📄';
                }
                
                const nameSpan = document.createElement('span');
                nameSpan.className = 'tree-item-name';
                nameSpan.textContent = item.name;
                
                itemDiv.appendChild(iconSpan);
                itemDiv.appendChild(nameSpan);
                cont.appendChild(itemDiv);
                
                if (item.children && item.children.length > 0) {
                    const childrenDiv = document.createElement('div');
                    childrenDiv.className = 'tree-children hidden';
                    cont.appendChild(childrenDiv);
                    tempRender(item.children, childrenDiv, level + 1);
                }
            });
        };
        tempRender(rootTree);
    }
}

// 立即尝试显示目录树（如果 DOM 已准备好）
if (document.readyState !== 'loading') {
    showInitialTree();
} else {
    // 如果 DOM 还在加载，在 DOMContentLoaded 时立即显示
    document.addEventListener('DOMContentLoaded', showInitialTree, { once: true });
}

// 初始化应用（等待 DOM 加载完成）
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // 检查登录状态
        if (checkAuth()) {
        const app = new NotesApp();
        window.app = app; // 方便调试
        }
    });
} else {
    // 检查登录状态
    if (checkAuth()) {
    const app = new NotesApp();
    window.app = app; // 方便调试
    }
}
