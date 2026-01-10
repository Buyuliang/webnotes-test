/**
 * 记事本应用主逻辑
 */

class NotesApp {
    constructor() {
        this.notes = [];
        this.currentNoteId = null;
        this.init();
    }

    /**
     * 初始化应用
     */
    init() {
        this.loadConfig();
        this.loadNotes();
        this.setupEventListeners();
        this.renderNotesList();
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 配置按钮
        document.getElementById('configBtn').addEventListener('click', () => {
            this.toggleConfigPanel();
        });

        // 保存配置
        document.getElementById('saveConfigBtn').addEventListener('click', () => {
            this.saveConfig();
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
        document.getElementById('searchInput').addEventListener('input', (e) => {
            this.filterNotes(e.target.value);
        });

        // 图片粘贴
        const noteContent = document.getElementById('noteContent');
        noteContent.addEventListener('paste', (e) => {
            this.handleImagePaste(e);
        });

        // 自动保存（防抖）
        let saveTimeout;
        noteContent.addEventListener('input', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                this.autoSave();
            }, 2000);
        });

        document.getElementById('noteTitle').addEventListener('input', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                this.autoSave();
            }, 2000);
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

        try {
            githubAPI.init(token, repo);
            localStorage.setItem('github_token', token);
            localStorage.setItem('github_repo', repo);
            
            // 测试连接
            await githubAPI.getOrCreateRelease();
            
            this.showMessage('配置保存成功！', 'success');
            this.toggleConfigPanel();
        } catch (error) {
            this.showMessage(`配置保存失败: ${error.message}`, 'error');
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
     * 加载笔记
     */
    loadNotes() {
        const saved = localStorage.getItem('notes');
        if (saved) {
            this.notes = JSON.parse(saved);
        }
    }

    /**
     * 保存笔记到本地存储
     */
    saveNotes() {
        localStorage.setItem('notes', JSON.stringify(this.notes));
    }

    /**
     * 创建新笔记
     */
    createNewNote() {
        const newNote = {
            id: Date.now().toString(),
            title: '未命名笔记',
            content: '',
            attachments: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        this.notes.unshift(newNote);
        this.saveNotes();
        this.renderNotesList();
        this.selectNote(newNote.id);
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
        document.getElementById('noteContent').innerHTML = note.content;
        
        this.renderAttachments(note.attachments);
        this.updateLastSaved(note.updatedAt);
        this.renderNotesList();
    }

    /**
     * 保存当前笔记
     */
    async saveCurrentNote() {
        if (!this.currentNoteId) return;

        const note = this.notes.find(n => n.id === this.currentNoteId);
        if (!note) return;

        note.title = document.getElementById('noteTitle').value || '未命名笔记';
        note.content = document.getElementById('noteContent').innerHTML;
        note.updatedAt = new Date().toISOString();

        this.saveNotes();
        this.renderNotesList();
        this.updateLastSaved(note.updatedAt);
        this.showMessage('笔记已保存', 'success');
    }

    /**
     * 自动保存
     */
    autoSave() {
        if (!this.currentNoteId) return;

        const note = this.notes.find(n => n.id === this.currentNoteId);
        if (!note) return;

        note.title = document.getElementById('noteTitle').value || '未命名笔记';
        note.content = document.getElementById('noteContent').innerHTML;
        note.updatedAt = new Date().toISOString();

        this.saveNotes();
        this.updateLastSaved(note.updatedAt);
    }

    /**
     * 删除当前笔记
     */
    deleteCurrentNote() {
        if (!this.currentNoteId) return;

        if (!confirm('确定要删除这个笔记吗？')) return;

        this.notes = this.notes.filter(n => n.id !== this.currentNoteId);
        this.saveNotes();
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
        const list = document.getElementById('notesList');
        list.innerHTML = '';

        if (this.notes.length === 0) {
            list.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">暂无笔记，点击"新建笔记"开始</div>';
            return;
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
     * 处理文件上传
     */
    async handleFileUpload(files) {
        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        if (!this.currentNoteId) {
            this.showMessage('请先选择或创建一个笔记', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const note = this.notes.find(n => n.id === this.currentNoteId);
            if (!note) return;

            const uploadPromises = Array.from(files).map(async (file) => {
                const asset = await githubAPI.uploadFileToRelease(file);
                return {
                    name: asset.name,
                    url: asset.url,
                    size: asset.size,
                    type: file.type
                };
            });

            const attachments = await Promise.all(uploadPromises);
            note.attachments = note.attachments || [];
            note.attachments.push(...attachments);
            note.updatedAt = new Date().toISOString();

            this.saveNotes();
            this.renderAttachments(note.attachments);
            this.showMessage(`成功上传 ${attachments.length} 个文件`, 'success');
        } catch (error) {
            this.showMessage(`上传失败: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }

        // 清空文件输入
        document.getElementById('fileInput').value = '';
    }

    /**
     * 处理图片粘贴
     */
    async handleImagePaste(e) {
        const items = e.clipboardData?.items;
        if (!items) return;

        if (!githubAPI.isConfigured()) {
            this.showMessage('请先配置 GitHub Token 和仓库', 'error');
            return;
        }

        if (!this.currentNoteId) {
            this.createNewNote();
        }

        const note = this.notes.find(n => n.id === this.currentNoteId);
        if (!note) return;

        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                e.preventDefault();
                
                this.showLoading(true);

                try {
                    const file = item.getAsFile();
                    const asset = await githubAPI.uploadImage(file);
                    
                    // 插入图片到编辑器
                    const img = document.createElement('img');
                    img.src = asset.url;
                    img.alt = asset.name;
                    
                    const selection = window.getSelection();
                    if (selection.rangeCount > 0) {
                        const range = selection.getRangeAt(0);
                        range.deleteContents();
                        range.insertNode(img);
                    } else {
                        document.getElementById('noteContent').appendChild(img);
                    }

                    // 保存附件信息
                    note.attachments = note.attachments || [];
                    note.attachments.push({
                        name: asset.name,
                        url: asset.url,
                        size: asset.size,
                        type: 'image'
                    });
                    note.updatedAt = new Date().toISOString();

                    this.saveNotes();
                    this.renderAttachments(note.attachments);
                    this.showMessage('图片已上传', 'success');
                } catch (error) {
                    this.showMessage(`图片上传失败: ${error.message}`, 'error');
                } finally {
                    this.showLoading(false);
                }
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
        const date = new Date(dateString);
        document.getElementById('lastSaved').textContent = 
            `最后保存: ${date.toLocaleString('zh-CN')}`;
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
}

// 初始化应用
const app = new NotesApp();
