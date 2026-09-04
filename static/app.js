let currentMode = 'chat';
        let pollingInterval = null;

        function safeParseMarkdown(text) {
            try {
                if (typeof marked !== 'undefined' && marked && marked.parse) {
                    return marked.parse(text);
                }
            } catch (e) {
                console.warn("Markdown parser fallback:", e);
            }
            return String(text).replace(/\n/g, '<br>');
        }

        function switchMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
            if (window.event && window.event.currentTarget) {
                window.event.currentTarget.classList.add('active');
            }
            
            const title = mode === 'chat' ? 'Chat with Codebase' : 'Issue Resolver & Debugger';
            const modeElem = document.getElementById('modeTitle');
            if (modeElem) modeElem.innerText = title;
            
            const placeholder = mode === 'chat' 
                ? 'Ask a question about the code...' 
                : 'Describe the bug or issue. E.g., "Login button fails when clicking twice"...';
            const userIn = document.getElementById('userInput');
            if (userIn) userIn.placeholder = placeholder;
            
            addMessage('bot', `Switched to <b>${mode === 'chat' ? 'Standard Chat' : 'Issue Resolver'}</b> mode.`);
        }

        function handleEnter(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function startAnalysis() {
            const urlInput = document.getElementById('repoUrl');
            const url = urlInput ? urlInput.value.trim() : '';
            if (!url) {
                alert("Please enter a GitHub repository URL");
                return;
            }
            
            const useGraph = document.getElementById('useGraph') ? document.getElementById('useGraph').checked : true;
            const useFaiss = document.getElementById('useFaiss') ? document.getElementById('useFaiss').checked : true;
            const apiKeyInput = document.getElementById('geminiApiKey');
            const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
            
            if (!useGraph && !useFaiss) {
                alert("Please enable at least one retrieval method (Code Graph or FAISS)");
                return;
            }
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            const progressArea = document.getElementById('progressArea');
            const statusText = document.getElementById('statusText');
            const progressFill = document.getElementById('progressFill');
            
            if (analyzeBtn) analyzeBtn.disabled = true;
            if (progressArea) progressArea.classList.remove('hidden');
            if (progressFill) progressFill.style.width = '5%';
            if (statusText) statusText.innerText = "Connecting...";
            
            try {
                const payload = {
                    url: url,
                    use_graph: useGraph,
                    use_faiss: useFaiss
                };
                if (apiKey) {
                    payload.api_key = apiKey;
                }

                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                if (res.ok) {
                    const data = await res.json();
                    addMessage('bot', `🚀 Analysis started for <code>${url}</code> with CodeGraph=${data.config.use_graph}, FAISS=${data.config.use_faiss}`);
                    startPolling();
                } else {
                    const errData = await res.json().catch(() => ({}));
                    const errMsg = errData.detail || "Failed to start analysis";
                    alert("⚠️ " + errMsg);
                    addMessage('bot', `⚠️ <b>Error:</b> ${errMsg}`);
                    if (analyzeBtn) analyzeBtn.disabled = false;
                    if (progressArea) progressArea.classList.add('hidden');
                }
            } catch (e) {
                alert("Network Error: " + e.message);
                if (analyzeBtn) analyzeBtn.disabled = false;
                if (progressArea) progressArea.classList.add('hidden');
            }
        }

        function startPolling() {
            if (pollingInterval) {
                clearInterval(pollingInterval);
                pollingInterval = null;
            }
            
            pollingInterval = setInterval(async () => {
                try {
                    const res = await fetch('/api/status');
                    if (!res.ok) return;
                    const data = await res.json();
                    
                    const percent = Math.min(100, Math.max(0, (data.current / (data.total || 100)) * 100));
                    const fill = document.getElementById('progressFill');
                    const status = document.getElementById('statusText');
                    if (fill) fill.style.width = percent + '%';
                    if (status) status.innerText = `${data.message} (${Math.round(percent)}%)`;
                    
                    if (data.status === 'ready') {
                        clearInterval(pollingInterval);
                        pollingInterval = null;
                        const analyzeBtn = document.getElementById('analyzeBtn');
                        const btnText = document.getElementById('btnText');
                        const conn = document.getElementById('connectionStatus');
                        
                        if (analyzeBtn) analyzeBtn.disabled = false;
                        if (btnText) btnText.innerText = "Re-analyze / New Repo";
                        if (status) status.innerText = "Ready to Chat!";
                        if (conn) conn.innerHTML = '<span class="dot" style="background: #00b894; box-shadow: 0 0 10px #00b894;"></span> Connected';
                        
                        addMessage('bot', "✨ <b>Analysis Complete!</b> You can now ask questions in <b>Chat</b> or debug bugs in <b>Issue Resolver</b> mode. Click <i>'View Tutorial Report'</i> to inspect the full architecture.");
                    } else if (data.status === 'error') {
                        clearInterval(pollingInterval);
                        pollingInterval = null;
                        const analyzeBtn = document.getElementById('analyzeBtn');
                        if (analyzeBtn) analyzeBtn.disabled = false;
                        alert("Analysis failed: " + data.message);
                        addMessage('bot', `❌ <b>Analysis failed:</b> ${data.message}`);
                    }
                } catch (e) {
                    console.error("Polling error:", e);
                }
            }, 2500);
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            if (!input) return;
            const text = input.value.trim();
            if (!text) return;
            
            addMessage('user', text);
            input.value = '';
            
            const loadId = addMessage('bot', '<i class="fa-solid fa-spinner fa-spin"></i> Gemini is thinking...');
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question: text, mode: currentMode })
                });
                
                const data = await res.json();
                
                const loadElem = document.getElementById(loadId);
                if (loadElem) loadElem.remove();
                
                if (data.error) {
                    addMessage('bot', "⚠️ Error: " + data.error);
                } else {
                    let answer = safeParseMarkdown(data.answer || 'No answer returned');
                    
                    if (data.sources && data.sources.length > 0) {
                        answer += '<div style="margin-top:15px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;"><small>📚 Sources:</small><br>';
                        data.sources.forEach(s => {
                            answer += `<code style="font-size:0.8em; display:block; margin:2px 0;">${s.file}</code>`;
                        });
                        answer += '</div>';
                    }
                    
                    addMessage('bot', answer);
                }
            } catch (e) {
                const loadElem = document.getElementById(loadId);
                if (loadElem) loadElem.remove();
                addMessage('bot', "⚠️ Network Error: " + e.message);
            }
        }

        function addMessage(role, html) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            const msgId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substring(2, 7);
            div.id = msgId;
            
            const avatar = role === 'bot' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
            
            div.innerHTML = `
                <div class="avatar">${avatar}</div>
                <div class="content">${html}</div>
            `;
            
            const history = document.getElementById('chatHistory');
            if (history) {
                history.appendChild(div);
                history.scrollTop = history.scrollHeight;
            }
            return msgId;
        }

        function openReport() {
            const modal = document.getElementById('reportModal');
            const iframe = document.getElementById('reportFrame');
            if (iframe) iframe.src = '/api/report';
            if (modal) modal.classList.remove('hidden');
        }

        function closeReport() {
            const modal = document.getElementById('reportModal');
            if (modal) modal.classList.add('hidden');
        }

        // Automatic connection on page load
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const res = await fetch('/api/status');
                if (!res.ok) return;
                const data = await res.json();
                if (data.status === 'ready') {
                    const btnText = document.getElementById('btnText');
                    const conn = document.getElementById('connectionStatus');
                    const status = document.getElementById('statusText');
                    if (btnText) btnText.innerText = "Re-analyze / New Repo";
                    if (status) status.innerText = "Ready to Chat!";
                    if (conn) conn.innerHTML = '<span class="dot" style="background: #00b894; box-shadow: 0 0 10px #00b894;"></span> Connected';
                } else if (data.status === 'busy' || data.status === 'starting') {
                    const progressArea = document.getElementById('progressArea');
                    if (progressArea) progressArea.classList.remove('hidden');
                    startPolling();
                }
            } catch (e) {
                console.error("Init check error:", e);
            }
        });