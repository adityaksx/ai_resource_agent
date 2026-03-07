// =====================================================
// STATE
// =====================================================
// Each item: { type, label, url?, file?, dataUrl? }
let attachments = [];
let chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
let currentSessionTitle = null; 
let currentChatSessionId = Date.now();

// =====================================================
// INIT
// =====================================================
window.addEventListener('DOMContentLoaded', renderHistory);

// =====================================================
// ATTACH MENU — toggle
// =====================================================
document.getElementById('attach-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    document.getElementById('attach-menu').classList.toggle('hidden');
});
document.addEventListener('click', () => {
    document.getElementById('attach-menu').classList.add('hidden');
});
document.getElementById('attach-menu').addEventListener('click', e => e.stopPropagation());

// =====================================================
// IMAGE UPLOAD via file picker
// =====================================================
document.getElementById('image-upload').addEventListener('change', function (e) {
    Array.from(e.target.files).forEach(file => {
        const reader = new FileReader();
        reader.onload = ev => {
            addAttachment({ type: 'image', label: file.name, file, dataUrl: ev.target.result });
        };
        reader.readAsDataURL(file);
    });
    e.target.value = '';
    document.getElementById('attach-menu').classList.add('hidden');
});

// =====================================================
// PASTE DETECTION — auto-convert URLs to chips
// =====================================================
document.getElementById('user-input').addEventListener('paste', (e) => {
    const pasted = (e.clipboardData || window.clipboardData).getData('text');
    const lines = pasted.split('\n').map(l => l.trim()).filter(Boolean);
    const urls = lines.filter(l => /^https?:\/\//i.test(l));
    const text = lines.filter(l => !/^https?:\/\//i.test(l));

    if (urls.length > 0) {
        e.preventDefault();
        urls.forEach(url => addAttachment({ type: detectUrlType(url), label: url, url }));
        if (text.length > 0) {
            const input = document.getElementById('user-input');
            input.value += (input.value ? '\n' : '') + text.join('\n');
            autoResize(input);
        }
    }
});

// =====================================================
// PROMPT LINK — called from menu buttons
// =====================================================
function promptLink(type) {
    document.getElementById('attach-menu').classList.add('hidden');
    const prefixes = {
        youtube: 'https://youtube.com/watch?v=',
        github: 'https://github.com/',
        instagram: 'https://instagram.com/p/',
        web: 'https://'
    };
    const url = window.prompt(`Paste your ${type} link:`, prefixes[type] || 'https://');
    if (url && /^https?:\/\//i.test(url.trim())) {
        addAttachment({ type: detectUrlType(url.trim()), label: url.trim(), url: url.trim() });
    }
}

// =====================================================
// ATTACHMENT MANAGEMENT
// =====================================================
function detectUrlType(url) {
    if (/youtu\.be|youtube\.com/i.test(url)) return 'youtube';
    if (/github\.com/i.test(url)) return 'github';
    if (/instagram\.com/i.test(url)) return 'instagram';
    return 'web';
}

function getIcon(type) {
    return { youtube: '▶', github: '⌥', instagram: '◈', web: '🌐', image: '🖼' }[type] || '📎';
}

function addAttachment(item) {
    attachments.push(item);
    renderAttachments();
}

function removeAttachment(idx) {
    attachments.splice(idx, 1);
    renderAttachments();
}

function renderAttachments() {
    const panel = document.getElementById('attachments-panel');
    panel.innerHTML = '';

    if (attachments.length === 0) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    attachments.forEach((item, idx) => {
        const chip = document.createElement('div');
        chip.className = `attachment-chip chip-${item.type}`;

        if (item.type === 'image') {
            chip.innerHTML = `
                <img src="${item.dataUrl}" class="chip-thumb" alt="${item.label}">
                <span class="chip-label">${truncate(item.label, 18)}</span>
                <button class="chip-remove" onclick="removeAttachment(${idx})">✕</button>`;
        } else {
            chip.innerHTML = `
                <span class="chip-icon">${getIcon(item.type)}</span>
                <span class="chip-label">${truncate(item.label, 40)}</span>
                <button class="chip-remove" onclick="removeAttachment(${idx})">✕</button>`;
        }
        panel.appendChild(chip);
    });
}

// =====================================================
// SEND
// =====================================================
async function send() {
    const inputEl = document.getElementById('user-input');
    const text = inputEl.value.trim();

    if (!text && attachments.length === 0) return;

    // Build user-facing display
    const displayParts = [];
    if (text) displayParts.push(text);
    attachments.forEach(a => {
        if (a.type === 'image') displayParts.push('📎 ' + a.label);
        else displayParts.push(getIcon(a.type) + ' ' + a.url);
    });

    const historyTitle = displayParts[0].slice(0, 60);  // ← capture title early

    addMessage('user', displayParts.join('\n'), attachments.filter(a => a.type === 'image'));
    // ← REMOVED saveToHistory() from here

    // Build FormData
    const formData = new FormData();
    const urlLines = attachments.filter(a => a.type !== 'image').map(a => a.url);
    const fullText = [text, ...urlLines].filter(Boolean).join('\n');
    if (fullText) formData.append('message', fullText);
    attachments.filter(a => a.type === 'image').forEach(a => {
        formData.append('images', a.file, a.file.name);
    });
    formData.append('session_id', currentChatSessionId); 

    // Clear state
    inputEl.value = '';
    inputEl.style.height = 'auto';
    attachments = [];
    renderAttachments();

    // Loading
    const loadingWrapper = addMessage('loading', '');

    try {
        const res = await fetch('/chat', { method: 'POST', body: formData });
        const data = await res.json();
        loadingWrapper.closest('.message-wrapper').remove();
        addMessage('bot', data.response || 'No response.');
        saveToHistory(historyTitle);   // ← MOVED HERE: saved after bot reply
    } catch (err) {
        loadingWrapper.closest('.message-wrapper').remove();
        addMessage('bot', '⚠️ Error: ' + err.message);
        saveToHistory(historyTitle);   // ← also save on error
    }
}

// =====================================================
// MESSAGE RENDERING
// =====================================================
function addMessage(type, text, imageAttachments = []) {
    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.remove();

    const messages = document.getElementById('messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'message-wrapper';

    const content = document.createElement('div');

    if (type === 'user') {
        content.className = 'message-user';

        // Show image thumbnails inline in user bubble
        if (imageAttachments.length > 0) {
            const thumbRow = document.createElement('div');
            thumbRow.className = 'user-img-row';
            imageAttachments.forEach(a => {
                const img = document.createElement('img');
                img.src = a.dataUrl;
                img.className = 'user-img-thumb';
                thumbRow.appendChild(img);
            });
            content.appendChild(thumbRow);
        }

        if (text) {
            const p = document.createElement('p');
            p.textContent = text;
            content.appendChild(p);
        }

    } else if (type === 'loading') {
        content.className = 'message-bot';
        content.innerHTML = `
            <div class="loading-indicator">
                <div class="dot-anim"><span></span><span></span><span></span></div>
                <span>Processing…</span>
            </div>`;
    } else {
        content.className = 'message-bot';
        content.innerHTML = `
            <div class="bot-header">🧠 Resource Agent</div>
            <div class="bot-body">${escapeHtml(text).replace(/\n/g, '<br>')}</div>`;
    }

    wrapper.appendChild(content);
    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
    return content;
}

// =====================================================
// UTILITIES
// =====================================================
function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function truncate(str, n) {
    return str.length > n ? str.slice(0, n) + '…' : str;
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
}

function newThread() {
    const messages = document.getElementById('messages');
    const hasContent = !messages.querySelector('#welcome-screen');
    if (hasContent) {
        const firstUserMsg = messages.querySelector('.message-user p');
        const title = firstUserMsg?.textContent?.slice(0, 60) || 'Untitled thread';
        saveToHistory(title);
    }

    currentSessionTitle = null;
    currentChatSessionId = Date.now();   // ← RESET session for new thread

    messages.innerHTML = `
        <div id="welcome-screen" class="welcome-screen">
            <h1>Where knowledge begins</h1>
            <p>Mix images, YouTube Links, GitHub repos, Instagram posts — all in one message</p>
        </div>`;
    document.getElementById('user-input').value = '';
    attachments = [];
    renderAttachments();
}

function saveToHistory(title) {
    const messages = document.getElementById('messages');

    const clone = messages.cloneNode(true);
    clone.querySelectorAll('img.user-img-thumb').forEach(img => {
        img.src = '';
        img.alt = '[image]';
    });
    const snapshot = clone.innerHTML;

    // Set session title on first message of this thread
    if (!currentSessionTitle) {
        currentSessionTitle = title;
    }

    // Always update the CURRENT session entry (not by title match)
    const existingIdx = chatHistory.findIndex(h => h.sessionTitle === currentSessionTitle);

    if (existingIdx !== -1) {
        chatHistory[existingIdx].html = snapshot;   // ← update existing
    } else {
        chatHistory.unshift({
            title:        currentSessionTitle,
            sessionTitle: currentSessionTitle,       // ← unique key for this session
            date:         new Date().toISOString(),
            html:         snapshot
        });
    }

    if (chatHistory.length > 30) chatHistory.pop();

    try {
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    } catch(e) {
        console.warn('History save failed (storage full):', e);
    }

    renderHistory();
}


function renderHistory() {
    const list = document.getElementById('history-list');
    list.innerHTML = '';
    chatHistory.slice(0, 20).forEach((item, idx) => {
        const li = document.createElement('li');
        li.textContent = item.title;
        li.title = item.title;
        li.addEventListener('click', () => loadThread(idx));   // ← ADD THIS
        list.appendChild(li);
    });
}

function loadThread(idx) {
    const item = chatHistory[idx];
    console.log('[LOAD] idx:', idx, 'item:', item);
    if (!item || !item.html) return;

    // Highlight selected thread in sidebar
    document.querySelectorAll('.history-list li').forEach((li, i) => {
        li.style.background = i === idx ? '#252727' : '';
        li.style.color      = i === idx ? '#e8e8e6' : '';
    });

    const messages = document.getElementById('messages');
    messages.innerHTML = item.html;
    console.log('[LOAD] restored html length:', item.html.length);
    messages.scrollTop = messages.scrollHeight;

    attachments = [];
    renderAttachments();
    document.getElementById('user-input').value = '';
    autoResize(document.getElementById('user-input'));
}

