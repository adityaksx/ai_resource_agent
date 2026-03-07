let allItems   = [];
let activeFilter = 'all';
let currentView  = 'grid';
let pendingDelId = null;
let currentModalId = null; 
let _originalAnswer  = ''; 

/* ── Source helpers ─────────────────────────────── */
const ICONS = {
  github_repo:'🐙', github_file:'🐙', github_gist:'🐙',
  youtube_video:'▶️', youtube_shorts:'▶️', youtube_playlist:'▶️',
  local_image:'🖼️', image_url:'🖼️',
  pdf_document:'📄', pdf_url:'📄',
  instagram_post:'📸', instagram_reel:'📸',
  web:'🌐', medium_article:'📝', substack_article:'📝',
  arxiv_paper:'🔬', reddit_post:'💬', reddit_subreddit:'💬',
  plain_text:'📋', plain_text_file:'📋',
  huggingface_model:'🤗', huggingface_dataset:'🤗',
};
const icon = s => ICONS[s] || '📎';

function badgeClass(s) {
  if (!s) return 'badge-default';
  if (s.startsWith('github'))    return 'badge-github';
  if (s.startsWith('youtube'))   return 'badge-youtube';
  if (s.includes('image'))       return 'badge-image';
  if (s.includes('text') || s === 'plain_text') return 'badge-text';
  if (s.includes('pdf'))         return 'badge-pdf';
  if (s.startsWith('instagram')) return 'badge-instagram';
  return 'badge-web';
}

function filterKey(s) {
  if (!s) return 'other';
  if (s.startsWith('github'))  return 'github';
  if (s.startsWith('youtube')) return 'youtube';
  if (s.includes('image'))     return 'image';
  if (s.includes('text') || s === 'plain_text') return 'text';
  return 'web';
}

function fmt(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString('en-IN',
    { day:'numeric', month:'short', year:'numeric' });
}

function esc(str) {
  return String(str||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
    .replace(/'/g,'&#039;');
}

/* ── Load ───────────────────────────────────────── */
async function loadResources() {
  try {
    const r = await fetch('/api/resources?limit=500');
    const d = await r.json();
    allItems = d.resources || [];
    computeStats();
    render();
  } catch(e) {
    document.getElementById('itemsWrap').innerHTML =
      `<div class="empty"><div class="icon">⚠️</div>
       <h3>Could not load resources</h3><p>${e.message}</p></div>`;
  }
}

function computeStats() {
  document.getElementById('totalCount').textContent  = allItems.length;
  document.getElementById('githubCount').textContent = allItems.filter(i=>i.source?.startsWith('github')).length;
  document.getElementById('imageCount').textContent  = allItems.filter(i=>i.source?.includes('image')).length;
  document.getElementById('webCount').textContent    = allItems.filter(i=>filterKey(i.source)==='web').length;
  document.getElementById('ytCount').textContent     = allItems.filter(i=>i.source?.startsWith('youtube')).length;
}

/* ── Filter / View ──────────────────────────────── */
function setFilter(f, btn) {
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  render();
}

function setView(v) {
  currentView = v;
  document.getElementById('btnGrid').classList.toggle('on', v==='grid');
  document.getElementById('btnList').classList.toggle('on', v==='list');
  const wrap = document.getElementById('itemsWrap');
  wrap.className = v === 'grid' ? 'items-wrap grid-view' : 'items-wrap list-view';
  render();
}

/* ── Render ─────────────────────────────────────── */
function render() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  let items = [...allItems].sort((a, b) => b.id - a.id);

  if (activeFilter !== 'all')
    items = items.filter(i => filterKey(i.source) === activeFilter);

  if (q)
    items = items.filter(i =>
      (i.vault_title||i.title||'').toLowerCase().includes(q) ||
      (i.url||'').toLowerCase().includes(q) ||
      (i.source||'').toLowerCase().includes(q)
    );

  const wrap = document.getElementById('itemsWrap');

  if (!items.length) {
    wrap.innerHTML = `<div class="empty">
      <div class="icon">🗄️</div>
      <h3>Nothing here yet</h3>
      <p>Start by pasting a link or text in the chat.</p></div>`;
    return;
  }

  // ── Build groups ──────────────────────────────────
  const groups     = new Map();   // sessionId → [items]
  const renderedIds = new Set();

  items.forEach(item => {
    const sid = item.session_id;
    if (sid) {
      if (!groups.has(sid)) groups.set(sid, []);
      groups.get(sid).push(item);
    }
  });

  // ── Build unified list of render-units ───────────
  // Each unit: { type: 'folder'|'solo', date, data }
  const units = [];

  groups.forEach((groupItems, sid) => {
    if (groupItems.length < 2) return;
    // Use newest item's created_at as folder date
    const newest = groupItems[0];
    units.push({
      type:   'folder',
      date:   newest.created_at,
      sid,
      items:  groupItems,
      newest,
    });
    groupItems.forEach(i => renderedIds.add(i.id));
  });

  items.forEach(item => {
    if (renderedIds.has(item.id)) return;
    units.push({
      type: 'solo',
      date: item.created_at,
      item,
    });
  });

  // ── Sort all units newest → oldest ───────────────
  units.sort((a, b) => new Date(b.date) - new Date(a.date));

  // ── Render with sequential display numbers ────────
  const isGrid = currentView === 'grid';
  const cards  = units.map((unit, idx) => {
    const displayNum = units.length - idx;   // 1-based sequential number

    if (unit.type === 'folder') {
      const { sid, items: groupItems, newest } = unit;
      return `
        <div class="card folder-card" onclick="openSessionPopup(${sid})">
          <div class="card-header">
            <div class="card-icon folder-icon">📁</div>
            <div class="card-meta">
              <div class="card-num">#${displayNum} · ${groupItems.length} items</div>
              <div class="card-title">${esc(newest.vault_title || newest.title || 'Session')}</div>
              ${!isGrid ? `<span class="card-date">${fmt(newest.created_at)}</span>` : ''}
            </div>
            <span class="card-badge badge-session">Folder</span>
            <button class="card-del" title="Delete folder"
              onclick="askDelFolder(event, ${sid})">🗑</button>
          </div>
          ${isGrid ? `
          <div class="folder-thumbs">
            ${groupItems.slice(0,4).map(i =>
              `<div class="folder-thumb-item">${icon(i.source||'unknown')}</div>`
            ).join('')}
            ${groupItems.length > 4
              ? `<div class="folder-thumb-item folder-more">+${groupItems.length - 4}</div>`
              : ''}
          </div>
          <div class="card-footer">
            <span>📁 ${groupItems.length} resources</span>
            <span>${fmt(newest.created_at)}</span>
          </div>` : ''}
        </div>`;
    }

    // solo card
    const { item } = unit;
    const src         = item.source || 'unknown';
    const cardTitle   = item.vault_title || item.title || item.url || 'Untitled';
    const cardSnippet = item.vault_snippet || (item.llm_output || '').slice(0, 160);

    return `
      <div class="card" data-id="${item.id}" onclick="openModal(${item.id})">
        <div class="card-header">
          <div class="card-icon">${icon(src)}</div>
          <div class="card-meta">
            <div class="card-num">#${displayNum}</div>
            <div class="card-title">${esc(cardTitle)}</div>
            ${!isGrid ? `<span class="card-date">${fmt(item.created_at)}</span>` : ''}
          </div>
          <span class="card-badge ${badgeClass(src)}">${src.replace(/_/g,' ')}</span>
          <button class="card-del" title="Delete" onclick="askDel(event,${item.id})">🗑</button>
        </div>
        ${isGrid && cardSnippet
          ? `<div class="card-snippet">${esc(cardSnippet)}…</div>` : ''}
        ${isGrid ? `
        <div class="card-footer">
          <span>
            <span class="status-dot ${item.status==='error'
              ? 'status-error' : 'status-success'}"></span>
            ${item.status || 'processed'}
          </span>
          <span>${fmt(item.created_at)}</span>
        </div>` : ''}
      </div>`;
  });

  wrap.innerHTML = cards.join('');
}


function openSessionPopup(sessionId) {
  const items = allItems
    .filter(i => i.session_id == sessionId)
    .sort((a,b) => b.id - a.id);

  const html = items.map(i => {
    const src     = i.source || 'unknown';
    const title   = esc(i.vault_title || i.title || i.url || 'Untitled');
    const snippet = esc((i.vault_snippet || '').slice(0, 80));
    return `
      <div class="session-item" onclick="openModal(${i.id})">
        <div class="session-icon">${icon(src)}</div>
        <div class="session-info">
          <div class="session-title">${title}</div>
          ${snippet ? `<div class="session-snippet">${snippet}…</div>` : ''}
        </div>
        <span class="card-badge ${badgeClass(src)}" style="margin-left:auto;flex-shrink:0">
          ${src.replace(/_/g,' ')}
        </span>
        <button class="session-del" title="Delete"
          onclick="event.stopPropagation(); askDel(event, ${i.id})">🗑</button>
      </div>`;
  }).join('');

  document.getElementById('mIcon').textContent  = '📁';
  document.getElementById('mTitle').textContent = `Session — ${items.length} items`;
  document.getElementById('mBadge').textContent = 'Folder';
  document.getElementById('mBadge').className   = 'card-badge badge-session';
  document.getElementById('mDate').textContent  = items.length ? fmt(items[0].created_at) : '';

  document.getElementById('tab-raw').innerHTML =
    `<div class="session-list">${html}</div>`;
  document.getElementById('tab-answer').innerHTML =
    '<div class="llm-answer" style="color:var(--muted)">No answer for session view.</div>';

  document.querySelectorAll('.tab').forEach((t,i)      => t.classList.toggle('active', i===0));
  document.querySelectorAll('.tab-panel').forEach((p,i) => p.classList.toggle('show',  i===0));

  document.getElementById('overlay').classList.add('show');
}


/* ── Modal popup ────────────────────────────────────── */
async function openModal(id) {

  // ── Change 4: error handling ──────────────────────
  let item;
  try {
    const r = await fetch(`/api/resources/${id}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    item = await r.json();
    currentModalId = item.id;
  } catch(e) {
    alert(`Could not load resource: ${e.message}`);
    return;
  }
  // ─────────────────────────────────────────────────

  const src = item.source || 'unknown';

  document.getElementById('mIcon').textContent  = icon(src);
  document.getElementById('mTitle').textContent = item.vault_title || item.title || item.url || 'Untitled';
  document.getElementById('mBadge').textContent = src.replace(/_/g,' ');
  document.getElementById('mBadge').className   = `card-badge ${badgeClass(src)}`;
  document.getElementById('mDate').textContent  = fmt(item.created_at);

  let rawHtml = '';
  let ri = null;
  try {
    ri = typeof item.raw_input === 'string' ? JSON.parse(item.raw_input) : item.raw_input;
  } catch(e) { ri = null; }

  if (ri) {
    if (ri.url) {
      const isRepo = src.startsWith('github');
      rawHtml += `
        <div class="section-label">URL</div>
        <div class="raw-block">
          ${isRepo
            ? `<a class="repo-path" href="${esc(ri.url)}" target="_blank" rel="noopener">🐙 ${esc(ri.url)}</a>`
            : `<a href="${esc(ri.url)}" target="_blank" rel="noopener">${esc(ri.url)}</a>`
          }
        </div>`;
    }

    if (item.files) {
      try {
        const f = JSON.parse(item.files);
        if (f.repo_path) {
          rawHtml += `
            <div class="section-label" style="margin-top:16px">Local Repo</div>
            <div class="raw-block"><span class="repo-path">📁 ${esc(f.repo_path)}</span></div>`;
        }
      } catch(e){}
    }

    if (ri.text) {
      rawHtml += `
        <div class="section-label" style="margin-top:16px">Text</div>
        <div class="raw-block">${esc(ri.text)}</div>`;
    }

    if (ri.image_path || ri.filename) {
      const filename = ri.filename || ri.image_path.split(/[\\/]/).pop();
      const imgSrc   = `/storage/images/${encodeURIComponent(filename)}`;
      rawHtml += `
        <div class="section-label" style="margin-top:16px">Image</div>
        <div class="raw-block">
          <code>${esc(ri.image_path || filename)}</code><br/>
          <img src="${imgSrc}" alt="${esc(filename)}"
            style="max-width:100%;border-radius:8px;margin-top:10px;border:1px solid var(--border)"
            onerror="this.style.display='none'" />
        </div>`;
    }
  } else if (item.raw_input) {
    rawHtml = `<div class="raw-block">${esc(String(item.raw_input))}</div>`;
  }

  if (!rawHtml && item.url) {
    rawHtml = `
      <div class="section-label">URL</div>
      <div class="raw-block">
        <a href="${esc(item.url)}" target="_blank" rel="noopener">${esc(item.url)}</a>
      </div>`;
  }

  document.getElementById('tab-raw').innerHTML =
    rawHtml || '<div class="raw-block" style="color:var(--muted)">No raw input stored.</div>';

  // ── Change 2: esc(item.error) to prevent XSS ─────
  const ans = item.llm_output || (item.error ? `⚠️ Error: ${esc(item.error)}` : 'No answer recorded.');
  document.getElementById('tab-answer').innerHTML = `<div class="llm-answer">${esc(ans)}</div>`;
  // ─────────────────────────────────────────────────

  document.querySelectorAll('.tab').forEach((t,i)       => t.classList.toggle('active', i===0));
  document.querySelectorAll('.tab-panel').forEach((p,i)  => p.classList.toggle('show',  i===0));

  document.getElementById('overlay').classList.add('show');
}


function maybeClose(e) {
  if (e.target === document.getElementById('overlay')) closeModal();
}
function closeModal() {
  document.getElementById('overlay').classList.remove('show');
}
function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('show'));
  document.getElementById('tab-'+name).classList.add('show');
}

/* ── Delete ─────────────────────────────────────── */
function askDel(e, id) {
  e.stopPropagation();
  pendingDelId = id;
  document.getElementById('delConfirm').style.display = 'flex';
}

function askDelFolder(e, sessionId) {
  e.stopPropagation();
  const groupItems = allItems.filter(i => i.session_id == sessionId);
  if (!groupItems.length) return;

  // Reuse del-confirm but store all IDs
  pendingDelId = groupItems.map(i => i.id);   // array of IDs
  document.getElementById('delConfirm').style.display = 'flex';
}

function cancelDel() {
  pendingDelId = null;
  document.getElementById('delConfirm').style.display = 'none';
}
async function confirmDel() {
  if (pendingDelId === null) return;

  const ids = Array.isArray(pendingDelId) ? pendingDelId : [pendingDelId];

  try {
    await Promise.all(ids.map(id =>
      fetch(`/api/resources/${id}`, { method: 'DELETE' })
    ));
    allItems = allItems.filter(i => !ids.includes(i.id));
    computeStats();
    render();
    closeModal();
  } catch(e) { alert('Delete failed: ' + e.message); }
  cancelDel();
}


document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeModal(); cancelDel(); }
});


function toggleEditAnswer() {
  const panel    = document.getElementById('tab-answer');
  const btn      = document.getElementById('editAnswerBtn');
  const existing = panel.querySelector('.llm-answer');

  if (panel.querySelector('.llm-edit-area')) return;  // already editing

  _originalAnswer = existing ? existing.textContent : '';

  panel.innerHTML = `
    <textarea class="llm-edit-area" id="llmEditTextarea">${esc(_originalAnswer)}</textarea>
    <div class="llm-edit-actions">
      <button class="btn-cancel-answer" onclick="cancelEditAnswer()">Cancel</button>
      <button class="btn-save-answer"   onclick="saveAnswer()">💾 Save</button>
    </div>`;

  btn.textContent = '✏️ Editing…';
  btn.disabled    = true;

  // Switch to answer tab
  document.querySelectorAll('.tab').forEach((t,i)      => t.classList.toggle('active', i===1));
  document.querySelectorAll('.tab-panel').forEach((p,i) => p.classList.toggle('show',   i===1));
}

function cancelEditAnswer() {
  document.getElementById('tab-answer').innerHTML =
    `<div class="llm-answer">${esc(_originalAnswer)}</div>`;
  const btn = document.getElementById('editAnswerBtn');
  btn.textContent = '✏️ Edit';
  btn.disabled    = false;
}


async function saveAnswer() {
  const textarea = document.getElementById('llmEditTextarea');
  if (!textarea || currentModalId === null) return;

  const newText = textarea.value;

  try {
    const res = await fetch(`/api/resources/${currentModalId}/answer`, {
      method:  'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ llm_output: newText }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    // Update in-memory allItems
    const item = allItems.find(i => i.id === currentModalId);
    if (item) item.llm_output = newText;

    document.getElementById('tab-answer').innerHTML =
      `<div class="llm-answer">${esc(newText)}</div>`;
    const btn = document.getElementById('editAnswerBtn');
    btn.textContent = '✏️ Edit';
    btn.disabled    = false;

  } catch(e) {
    alert('Save failed: ' + e.message);
  }
}


loadResources();
// auto refresh vault every 5 seconds
setInterval(() => {
  const modalOpen = document.getElementById('overlay').classList.contains('show');
  const searching = document.getElementById('searchInput').value.trim() !== '';
  if (!modalOpen && !searching) loadResources();
}, 5000);