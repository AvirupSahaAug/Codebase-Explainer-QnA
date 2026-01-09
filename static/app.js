let currentMode = 'chat'; // 'chat' or 'issue'
let isAnalyzing = false;
let pollingInterval = null;

function switchMode(mode) {
    currentMode = mode;
    
    // Update UI
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.currentTarget.classList.add('active');
    
    const title = mode === 'chat' ? 'Chat with Codebase' : 'Issue Resolver & Debugger';
    document.getElementById('modeTitle').innerText = title;
    
    const placeholder = mode === 'chat' 
        ? 'Ask a question about the code...' 
        : 'Describe the bug or issue. E.g., "Login button fails when clicking twice"...';
    document.getElementById('userInput').placeholder = placeholder;
    
    addMessage('bot', `Switched to <b>${mode === 'chat' ? 'Standard Chat' : 'Issue Resolver'}</b> mode.`);
}

function handleEnter(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function startAnalysis() {
    const url = document.getElementById('repoUrl').value;
    if (!url) return alert("Please enter a URL");
    
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('progressArea').classList.remove('hidden');
    
    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ url: url })
        });
        
        if (res.ok) {
            startPolling();
        } else {
            alert("Failed to start analysis");
            document.getElementById('analyzeBtn').disabled = false;
        }
    } catch (e) {
        alert("Error: " + e);
    }
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(async () => {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        // Update Bar
        const percent = (data.current / data.total) * 100;
        document.getElementById('progressFill').style.width = percent + '%';
        document.getElementById('statusText').innerText = `${data.message} (${Math.round(percent)}%)`;
        
        if (data.status === 'ready') {
            clearInterval(pollingInterval);
            document.getElementById('btnText').innerText = "Analysis Complete";
            document.getElementById('statusText').innerText = "Ready to Chat!";
            document.getElementById('connectionStatus').innerHTML = '<span class="dot" style="background: #00b894; box-shadow: 0 0 10px #00b894;"></span> Connected';
            addMessage('bot', "Analysis complete! I've read the code. How can I help?");
        } else if (data.status === 'error') {
            clearInterval(pollingInterval);
            document.getElementById('analyzeBtn').disabled = false;
            alert("Analysis failed: " + data.message);
        }
    }, 1000);
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text) return;
    
    addMessage('user', text);
    input.value = '';
    
    // Loading indicator
    const loadId = addMessage('bot', '<i class="fa-solid fa-spinner fa-spin"></i> Thinking...');
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ question: text, mode: currentMode })
        });
        
        const data = await res.json();
        
        // Remove loader
        document.getElementById(loadId).remove();
        
        if (data.error) {
            addMessage('bot', "⚠️ Error: " + data.error);
        } else {
            let answer = marked.parse(data.answer);
            
            // Add sources if any
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
        document.getElementById(loadId).remove();
        addMessage('bot', "⚠️ Network Error");
    }
}

function addMessage(role, html) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.id = 'msg-' + Date.now();
    
    const avatar = role === 'bot' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
    
    div.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="content">${html}</div>
    `;
    
    document.getElementById('chatHistory').appendChild(div);
    document.getElementById('chatHistory').scrollTop = document.getElementById('chatHistory').scrollHeight;
    return div.id;
}

function openReport() {
    const frame = document.getElementById('reportFrame');
    frame.src = '/api/report';
    document.getElementById('reportModal').classList.remove('hidden');
}

function closeReport() {
    document.getElementById('reportModal').classList.add('hidden');
}
