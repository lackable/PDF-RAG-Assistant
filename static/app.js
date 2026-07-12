/**
 * PDF RAG Assistant - Conversational Frontend
 */

const API_BASE = "http://localhost:8000/api";

// State
let currentSessionId = null;
let sessions = [];
let isGenerating = false;

// DOM Elements
const els = {
  sessionsList: document.getElementById('sessions-list'),
  newChatBtn: document.getElementById('new-chat-btn'),
  chatEmptyState: document.getElementById('chat-empty-state'),
  chatMessages: document.getElementById('chat-messages'),
  chatInputArea: document.getElementById('chat-input-area'),
  chatInput: document.getElementById('chat-input'),
  sendBtn: document.getElementById('send-btn'),
  topkSelect: document.getElementById('topk-select'),
  chunksList: document.getElementById('chunks-list'),
  chunkCount: document.getElementById('chunk-count'),
  statusDot: document.getElementById('status-dot'),
  sidebarToggle: document.getElementById('sidebar-toggle'),
  sessionsSidebar: document.getElementById('sessions-sidebar'),
  
  // Modal
  renameModal: document.getElementById('rename-modal'),
  renameInput: document.getElementById('rename-input'),
  renameCancelBtn: document.getElementById('rename-cancel-btn'),
  renameConfirmBtn: document.getElementById('rename-confirm-btn')
};

// --- Initialization ---

async function init() {
  await loadSessions();
  if (sessions.length > 0) {
    selectSession(sessions[0].id);
  } else {
    showEmptyState();
  }
  setupEventListeners();
}

function setupEventListeners() {
  els.newChatBtn.addEventListener('click', createNewChat);
  
  els.sendBtn.addEventListener('click', handleSend);
  els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Auto-resize textarea
  els.chatInput.addEventListener('input', () => {
    els.chatInput.style.height = 'auto';
    els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 160) + 'px';
  });

  // Example queries
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', async () => {
      if (!currentSessionId) {
        await createNewChat();
      }
      els.chatInput.value = chip.dataset.query;
      handleSend();
    });
  });

  // Modal
  els.renameCancelBtn.addEventListener('click', closeRenameModal);
  
  // Sidebar toggle
  els.sidebarToggle.addEventListener('click', () => {
    els.sessionsSidebar.style.display = els.sessionsSidebar.style.display === 'none' ? 'flex' : 'none';
  });
}

// --- Session Management ---

async function loadSessions() {
  try {
    const res = await fetch(`${API_BASE}/sessions`);
    sessions = await res.json();
    renderSessions();
  } catch (err) {
    console.error("Failed to load sessions", err);
  }
}

function renderSessions() {
  els.sessionsList.innerHTML = '';
  sessions.forEach(session => {
    const item = document.createElement('div');
    item.className = `session-item ${session.id === currentSessionId ? 'active' : ''}`;
    item.dataset.id = session.id;
    
    const nameSpan = document.createElement('span');
    nameSpan.className = 'session-name';
    nameSpan.textContent = session.name || "New Chat";
    
    const actions = document.createElement('div');
    actions.className = 'session-actions';
    
    const renameBtn = document.createElement('button');
    renameBtn.className = 'session-rename-btn';
    renameBtn.innerHTML = '✎';
    renameBtn.title = 'Rename';
    renameBtn.onclick = (e) => { e.stopPropagation(); openRenameModal(session.id, session.name); };
    
    const delBtn = document.createElement('button');
    delBtn.className = 'session-delete-btn';
    delBtn.innerHTML = '×';
    delBtn.title = 'Delete';
    delBtn.onclick = async (e) => { 
      e.stopPropagation(); 
      if(confirm('Delete this chat?')) {
        await fetch(`${API_BASE}/sessions/${session.id}`, { method: 'DELETE' });
        if(currentSessionId === session.id) currentSessionId = null;
        await loadSessions();
        if(currentSessionId) selectSession(currentSessionId);
        else if(sessions.length > 0) selectSession(sessions[0].id);
        else showEmptyState();
      }
    };
    
    actions.appendChild(renameBtn);
    actions.appendChild(delBtn);
    
    item.appendChild(nameSpan);
    item.appendChild(actions);
    
    item.addEventListener('click', () => selectSession(session.id));
    els.sessionsList.appendChild(item);
  });
}

async function createNewChat() {
  try {
    const res = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const session = await res.json();
    currentSessionId = session.id;
    await loadSessions();
    selectSession(currentSessionId);
  } catch (err) {
    console.error("Failed to create chat", err);
  }
}

async function selectSession(id) {
  currentSessionId = id;
  renderSessions();
  els.chatEmptyState.style.display = 'none';
  els.chatMessages.style.display = 'flex';
  els.chatInputArea.style.display = 'block';
  
  try {
    const res = await fetch(`${API_BASE}/sessions/${id}/history`);
    const history = await res.json();
    renderHistory(history);
  } catch (err) {
    console.error("Failed to load history", err);
  }
}

function showEmptyState() {
  currentSessionId = null;
  renderSessions();
  els.chatEmptyState.style.display = 'flex';
  els.chatMessages.style.display = 'none';
  els.chatInputArea.style.display = 'none';
  els.chunksList.innerHTML = '<div class="no-retrieval-msg">Select or create a chat to view sources</div>';
  els.chunkCount.textContent = '—';
}

// --- Rename Modal ---

let renameTargetId = null;

function openRenameModal(id, currentName) {
  renameTargetId = id;
  els.renameInput.value = currentName || "";
  els.renameModal.style.display = 'flex';
  els.renameInput.focus();
  
  els.renameConfirmBtn.onclick = async () => {
    const newName = els.renameInput.value.trim();
    if(newName && renameTargetId) {
      await fetch(`${API_BASE}/sessions/${renameTargetId}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName })
      });
      closeRenameModal();
      await loadSessions();
    }
  };
}

function closeRenameModal() {
  els.renameModal.style.display = 'none';
  renameTargetId = null;
}

// --- Chat UI & Interaction ---

function renderHistory(history) {
  els.chatMessages.innerHTML = '';
  els.chunksList.innerHTML = '';
  els.chunkCount.textContent = '—';
  
  history.forEach(msg => {
    appendUserMessage(msg.input);
    const stats = {
      time_taken_s: msg.time_taken_s,
      input_tokens: msg.input_tokens,
      output_tokens: msg.output_tokens
    };
    appendAssistantMessage(msg.output, msg.agent_state, stats, msg.sources);
  });
  scrollToBottom();
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble user-bubble';
  bubble.textContent = text;
  
  row.appendChild(bubble);
  els.chatMessages.appendChild(row);
  scrollToBottom();
}

function appendAssistantMessage(text, state = null, stats = null, sources = null) {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = 'AI';
  
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble assistant-bubble';
  
  if (state) {
    const badge = document.createElement('div');
    badge.className = `state-badge state-${state}`;
    badge.textContent = state.toUpperCase().replace('_', ' ');
    bubble.appendChild(badge);
  }
  
  const content = document.createElement('div');
  content.className = 'bubble-text';
  content.innerHTML = formatMarkdown(text);
  bubble.appendChild(content);

  // Stats bar (time, tokens, and sources)
  const statsBar = document.createElement('div');
  statsBar.className = 'response-stats';
  statsBar.style.display = 'flex';
  
  let timeStr = '—';
  let tokensStr = '—';
  if (stats) {
    if (stats.time_taken_s !== undefined && stats.time_taken_s !== null) {
      timeStr = stats.time_taken_s;
    }
    if (stats.input_tokens !== undefined && stats.input_tokens !== null) {
      tokensStr = `${stats.input_tokens} in / ${stats.output_tokens} out`;
    }
  }

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const items = sources.map(src => {
      const pageText = src.page_numbers ? ` · Pg ${src.page_numbers}` : '';
      return `<li><span class="src-pdf" title="${src.pdf_name}">${src.pdf_name}</span>${pageText} · <span class="src-type pill-type-${src.element_type}">${src.element_type}</span></li>`;
    }).join('');
    sourcesHtml = `
      <details class="msg-sources-dropdown">
        <summary>📄 ${sources.length} Sources</summary>
        <ul class="msg-sources-list">
          ${items}
        </ul>
      </details>
    `;
  } else {
    sourcesHtml = `<span class="src-none">📄 No sources</span>`;
  }

  statsBar.innerHTML = `
    <span class="stat-item" title="Generation Time">⏱️ <span class="stat-val time-val">${timeStr}</span>s</span>
    <span class="stat-sep">·</span>
    <span class="stat-item" title="Tokens">🪙 <span class="stat-val tokens-val">${tokensStr}</span></span>
    <span class="stat-sep">·</span>
    <span class="stat-sources-slot">${sourcesHtml}</span>
  `;
  
  bubble.appendChild(statsBar);
  row.appendChild(avatar);
  row.appendChild(bubble);
  els.chatMessages.appendChild(row);
  scrollToBottom();
  
  return { contentEl: content, bubbleEl: bubble, statsBarEl: statsBar };
}

function scrollToBottom() {
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function handleSend() {
  if (isGenerating || !currentSessionId) return;
  
  const query = els.chatInput.value.trim();
  if (!query) return;
  
  isGenerating = true;
  els.chatInput.value = '';
  els.chatInput.style.height = 'auto';
  els.sendBtn.disabled = true;
  els.sendBtn.style.opacity = '0.5';
  els.statusDot.style.animation = 'none';
  els.statusDot.style.backgroundColor = 'var(--amber)';
  
  appendUserMessage(query);
  
  // Create placeholder for assistant message
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble assistant-bubble';
  
  const badgeContainer = document.createElement('div');
  bubble.appendChild(badgeContainer);
  
  const content = document.createElement('div');
  content.className = 'bubble-text';
  
  const cursor = document.createElement('div');
  cursor.className = 'cursor-blink';
  
  const statsBar = document.createElement('div');
  statsBar.className = 'response-stats';
  statsBar.style.display = 'flex'; // Visible immediately for active timer
  statsBar.innerHTML = `
    <span class="stat-item" title="Generation Time">⏱️ <span class="stat-val time-val">0.0</span>s</span>
    <span class="stat-sep">·</span>
    <span class="stat-item" title="Tokens">🪙 <span class="stat-val tokens-val">—</span></span>
    <span class="stat-sep">·</span>
    <span class="stat-sources-slot"></span>
  `;
  
  content.appendChild(cursor);
  bubble.appendChild(content);
  bubble.appendChild(statsBar);
  row.appendChild(avatar);
  row.appendChild(bubble);
  els.chatMessages.appendChild(row);
  scrollToBottom();
  
  els.chunksList.innerHTML = '';
  renderSkeletons(3);
  els.chunkCount.textContent = '...';
  
  let fullText = "";
  let pendingSources = [];
  const startTime = Date.now();
  let timeValEl = statsBar.querySelector('.time-val');
  let tokensValEl = statsBar.querySelector('.tokens-val');
  
  // Start active timer ticking
  const timerInterval = setInterval(() => {
    timeValEl.textContent = ((Date.now() - startTime) / 1000).toFixed(1);
  }, 100);
  
  try {
    const topK = parseInt(els.topkSelect.value) || 5;
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        query: query,
        top_k: topK
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP Error ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // keep incomplete line
      
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const dataStr = line.substring(6);
        if (!dataStr) continue;
        
        try {
          const data = JSON.parse(dataStr);
          
          if (data.type === 'state') {
            const badge = document.createElement('div');
            badge.className = `state-badge state-${data.state}`;
            badge.textContent = data.state.toUpperCase().replace('_', ' ');
            badgeContainer.appendChild(badge);
            
            if (data.state === 'greeting') {
              els.chunksList.innerHTML = '<div class="no-retrieval-msg">No retrieval performed</div>';
              els.chunkCount.textContent = '0';
            }
          }
          else if (data.type === 'chunks') {
            renderChunks(data.chunks);
            els.chunkCount.textContent = data.chunks.length;
            pendingSources = data.chunks.map(c => ({
              pdf_name: c.metadata?.source_file || 'Unknown',
              page_numbers: c.metadata?.page_numbers || '',
              element_type: c.metadata?.element_type || 'text'
            }));
          }
          else if (data.type === 'token') {
            fullText += data.token;
            content.innerHTML = formatMarkdown(fullText);
            content.appendChild(cursor);
            scrollToBottom();
          }
          else if (data.type === 'usage') {
            clearInterval(timerInterval);
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            timeValEl.textContent = elapsed;
            tokensValEl.textContent = `${data.input_tokens} in / ${data.output_tokens} out`;
          }
          else if (data.type === 'error') {
            fullText += `\n\n**Error:** ${data.message}`;
            content.innerHTML = formatMarkdown(fullText);
            content.appendChild(cursor);
          }
          else if (data.type === 'done') {
            clearInterval(timerInterval);
            // Re-load sessions to pick up any auto-renaming
            loadSessions();
          }
        } catch(e) {
          console.warn("Parse error on SSE line", line, e);
        }
      }
    }
    
  } catch (err) {
    console.error("Chat error:", err);
    fullText += `\n\n**Network Error:** ${err.message}`;
    content.innerHTML = formatMarkdown(fullText);
  } finally {
    clearInterval(timerInterval);
    
    // Render per-message sources dropdown
    const sourcesSlot = statsBar.querySelector('.stat-sources-slot');
    if (sourcesSlot) {
      if (pendingSources.length > 0) {
        const items = pendingSources.map(src => {
          const pageText = src.page_numbers ? ` · Pg ${src.page_numbers}` : '';
          return `<li><span class="src-pdf" title="${src.pdf_name}">${src.pdf_name}</span>${pageText} · <span class="src-type pill-type-${src.element_type}">${src.element_type}</span></li>`;
        }).join('');
        sourcesSlot.innerHTML = `
          <details class="msg-sources-dropdown">
            <summary>📄 ${pendingSources.length} Sources</summary>
            <ul class="msg-sources-list">
              ${items}
            </ul>
          </details>
        `;
      } else {
        sourcesSlot.innerHTML = `<span class="src-none">📄 No sources</span>`;
      }
    }
    
    cursor.remove();
    isGenerating = false;
    els.sendBtn.disabled = false;
    els.sendBtn.style.opacity = '1';
    els.statusDot.style.animation = 'pulse-dot 2s ease-in-out infinite';
    els.statusDot.style.backgroundColor = 'var(--emerald)';
    els.chatInput.focus();
  }
}

// --- Chunk Rendering ---

function renderSkeletons(count) {
  els.chunksList.innerHTML = '';
  for (let i = 0; i < count; i++) {
    els.chunksList.innerHTML += `
      <div class="chunk-skeleton">
        <div class="skel-line" style="width:20%"></div>
        <div class="skel-line" style="width:80%"></div>
        <div class="skel-line" style="width:60%"></div>
      </div>
    `;
  }
}

function renderChunks(chunks) {
  els.chunksList.innerHTML = '';
  if (!chunks || chunks.length === 0) {
    els.chunksList.innerHTML = '<div class="no-retrieval-msg">No relevant passages found</div>';
    return;
  }
  
  chunks.forEach((chunk, i) => {
    const meta = chunk.metadata || {};
    const type = meta.element_type || 'text';
    const source = meta.source_file || 'Unknown';
    const pages = meta.page_numbers || 'N/A';
    const rawScore = chunk.rerank_score !== undefined ? chunk.rerank_score : chunk.score;
    const score = (rawScore !== undefined && rawScore !== null) ? rawScore.toFixed(3) : null;
    const content = chunk.raw_content || chunk.content || '';
    
    // Identify retrieval source (dense vs bm25 vs both)
    let sourcePill = '';
    const retrievalSource = chunk.retrieval_source || '';
    if (retrievalSource) {
      const displayLabel = retrievalSource === 'both' ? 'Hybrid' : (retrievalSource === 'dense' ? 'Vector' : 'Keyword');
      sourcePill = `<span class="meta-pill pill-source-${retrievalSource}">${displayLabel}</span>`;
    }
    
    const card = document.createElement('div');
    card.className = `chunk-card type-${type}`;
    
    card.innerHTML = `
      <div class="chunk-header">
        <span class="chunk-rank">[${i + 1}]</span>
        <span class="chunk-source" title="${source}">${source}</span>
      </div>
      <div class="chunk-meta">
        <span class="meta-pill pill-page">Pg ${pages}</span>
        <span class="meta-pill pill-type-${type}">${type}</span>
        ${score ? `<span class="meta-pill pill-score">⭐ ${score}</span>` : ''}
        ${sourcePill}
      </div>
      <div class="chunk-preview" id="preview-${i}">
        ${escapeHtml(content)}
      </div>
      <button class="chunk-expand-btn" id="expand-${i}">Read more ▾</button>
    `;
    
    els.chunksList.appendChild(card);
    
    const expandBtn = document.getElementById(`expand-${i}`);
    const preview = document.getElementById(`preview-${i}`);
    expandBtn.addEventListener('click', () => {
      preview.classList.toggle('expanded');
      expandBtn.textContent = preview.classList.contains('expanded') ? 'Show less ▴' : 'Read more ▾';
    });
  });
}


// --- Utility ---

function formatMarkdown(text) {
  if (!text) return '';
  
  // Basic markdown (headers, bold, italic, code)
  let html = text
    .replace(/### (.*$)/gim, '<h3>$1</h3>')
    .replace(/## (.*$)/gim, '<h2>$1</h2>')
    .replace(/# (.*$)/gim, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    .replace(/`(.*?)`/gim, '<code>$1</code>')
    .replace(/\n\n/gim, '<br><br>')
    .replace(/\n/gim, '<br>');

  // Citations [1], [1][2]
  html = html.replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>');

  return html;
}

function escapeHtml(unsafe) {
  return (unsafe || '').toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Start
init();
