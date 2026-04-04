(function () {
  'use strict';

  const BASE = '';  // same origin
  let memoriesOffset = 0;
  const MEMORIES_LIMIT = 10;

  const SAMPLES = {
    meeting: `Met with the product team today (April 4, 2026) to discuss the Q2 roadmap. Key decisions: prioritize the memory retrieval layer over the UI polish, ship an internal alpha by end of April. Alice raised concerns about the entity extraction accuracy — we agreed to add a fallback regex layer. Action items: I'll benchmark ChromaDB vs Qdrant this week. Follow up with Alice on NER eval by Friday.`,
    journal: `Been thinking a lot about focus lately. I keep starting new projects before finishing old ones — the contextweave memory engine is the third thing I've picked up this month. But this one feels different. The core idea (that your past context should inform your present decisions) is something I genuinely believe in. I want to build something that outlasts the job application. Goal: ship a working demo by Thursday. Remember: done is better than perfect.`,
    learning: `Learning goals for this quarter: get deeper into vector databases — specifically how HNSW indexing works and why it outperforms flat cosine search at scale. Also want to understand temporal reasoning in LLMs better. Key takeaway from this week: importance scoring based on recency decay is underused in RAG systems. Most production RAG just does top-K cosine and calls it a day. The access frequency boost is the interesting part — the more you recall a memory, the more important it becomes.`,
  };

  // ── API ──────────────────────────────────────────────────────
  const api = {
    async get(path) {
      const r = await fetch(BASE + path);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    },
    async post(path, body) {
      const r = await fetch(BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `${r.status}`);
      }
      return r.json();
    },
    async upload(path, file) {
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(BASE + path, { method: 'POST', body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `${r.status}`);
      }
      return r.json();
    },
  };

  // ── TOAST ────────────────────────────────────────────────────
  function toast(msg, type = 'info', duration = 3500) {
    const c = document.getElementById('toast-container');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    c.appendChild(t);
    setTimeout(() => t.remove(), duration);
  }

  // ── HEALTH ───────────────────────────────────────────────────
  const statKeys = ['events', 'chunks', 'memories', 'vectors', 'entities', 'edges'];

  function animateCount(el, target) {
    const start = parseInt(el.textContent) || 0;
    if (start === target) return;
    const dur = 600, step = 16;
    const steps = dur / step;
    let i = 0;
    const tick = setInterval(() => {
      i++;
      el.textContent = Math.round(start + (target - start) * (i / steps));
      if (i >= steps) { el.textContent = target; clearInterval(tick); }
    }, step);
  }

  async function refreshHealth() {
    try {
      const h = await api.get('/api/health');
      statKeys.forEach(k => {
        const el = document.querySelector(`#stat-${k} .stat-num`);
        if (el) animateCount(el, h[k] ?? 0);
      });
      const dot = document.querySelector('.live-dot');
      const lbl = document.getElementById('live-label');
      dot.className = 'live-dot online';
      lbl.textContent = 'live';
      return h;
    } catch {
      const dot = document.querySelector('.live-dot');
      const lbl = document.getElementById('live-label');
      dot.className = 'live-dot error';
      lbl.textContent = 'unreachable';
      return null;
    }
  }

  // ── PIPELINE ANIMATION ───────────────────────────────────────
  function animatePipeline() {
    const steps = ['pipe-chunk', 'pipe-embed', 'pipe-entity', 'pipe-store'];
    steps.forEach(id => document.getElementById(id).className = 'pipe-step');
    steps.forEach((id, i) => {
      setTimeout(() => {
        const el = document.getElementById(id);
        el.classList.add('active');
        if (i === steps.length - 1) {
          setTimeout(() => steps.forEach(s => {
            document.getElementById(s).className = 'pipe-step done';
          }), 400);
        }
      }, i * 500);
    });
  }

  // ── INGEST TEXT ──────────────────────────────────────────────
  async function ingestText() {
    const content = document.getElementById('ingest-text').value.trim();
    if (!content) { toast('Please enter some text first.', 'error'); return; }
    const btn = document.getElementById('btn-ingest-text');
    btn.disabled = true; btn.textContent = 'Ingesting…';
    try {
      const res = await api.post('/api/ingest/text', { content });
      animatePipeline();
      toast(`✓ ${res.chunks_created} chunk${res.chunks_created !== 1 ? 's' : ''}, ${res.entities_extracted} entities extracted`, 'success');
      document.getElementById('ingest-text').value = '';
      await refreshHealth();
      suggestQuery(content);
    } catch (e) {
      toast(`Ingest failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = 'Ingest Text';
    }
  }

  // ── INGEST FILE ──────────────────────────────────────────────
  async function ingestFile() {
    const input = document.getElementById('file-input');
    if (!input.files.length) return;
    const btn = document.getElementById('btn-ingest-file');
    btn.disabled = true; btn.textContent = 'Uploading…';
    try {
      const res = await api.upload('/api/ingest', input.files[0]);
      animatePipeline();
      toast(`✓ ${res.chunks_created} chunks, ${res.entities_extracted} entities`, 'success');
      input.value = '';
      document.getElementById('file-name').textContent = '';
      btn.disabled = true; btn.textContent = 'Upload File';
      await refreshHealth();
    } catch (e) {
      toast(`Upload failed: ${e.message}`, 'error');
      btn.disabled = false; btn.textContent = 'Upload File';
    }
  }

  function suggestQuery(content) {
    const words = content.toLowerCase();
    let suggestion = 'What did I decide recently?';
    if (words.includes('pattern') || words.includes('trend')) suggestion = 'What patterns do you see?';
    else if (words.includes('goal') || words.includes('learn')) suggestion = 'What are my current learning goals?';
    else if (words.includes('meeting') || words.includes('team')) suggestion = 'What action items came out of my recent meetings?';
    else if (words.includes('decision') || words.includes('decided')) suggestion = 'What key decisions have I made?';
    const qi = document.getElementById('query-input');
    qi.value = suggestion;
    document.getElementById('section-query').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── QUERY ─────────────────────────────────────────────────────
  function renderMarkdown(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^(.+)$/, '<p>$1</p>');
  }

  async function runQuery() {
    const q = document.getElementById('query-input').value.trim();
    if (!q) { toast('Enter a query first.', 'error'); return; }
    const btn = document.getElementById('btn-query');
    btn.disabled = true; btn.textContent = 'Thinking…';
    const card = document.getElementById('response-card');
    card.classList.add('hidden');
    try {
      const res = await api.post('/api/query', { query: q, top_k: 8 });
      const body = document.getElementById('response-body');
      body.innerHTML = renderMarkdown(res.answer || 'No answer returned.');

      document.getElementById('query-type-badge').textContent = res.query_type || 'general';
      const cited = res.cited_chunks ? res.cited_chunks.length : (res.chunks_used || 0);
      document.getElementById('cited-badge').textContent = `${cited} source${cited !== 1 ? 's' : ''}`;

      const conf = typeof res.confidence === 'number' ? res.confidence : 0.7;
      const fill = document.getElementById('confidence-fill');
      fill.style.width = `${Math.round(conf * 100)}%`;
      fill.style.background = conf >= 0.7 ? 'var(--success)' : conf >= 0.4 ? 'var(--warning)' : 'var(--danger)';

      const pr = document.getElementById('patterns-row');
      pr.innerHTML = '';
      (res.patterns || []).forEach(p => {
        const pill = document.createElement('span');
        pill.className = 'entity-pill';
        pill.textContent = p;
        pr.appendChild(pill);
      });

      card.classList.remove('hidden');
    } catch (e) {
      toast(`Query failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = 'Ask';
    }
  }

  // ── MEMORIES ─────────────────────────────────────────────────
  function relativeTime(isoStr) {
    if (!isoStr) return '';
    const diff = Date.now() - new Date(isoStr).getTime();
    const m = Math.floor(diff / 60000);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  }

  function renderMemory(mem) {
    const imp = (mem.importance || 0).toFixed(2);
    const pct = Math.round((mem.importance || 0) * 100);
    const entities = (mem.entities || []).map(e =>
      `<span class="entity-pill" onclick="focusEntity('${e}')">${e}</span>`
    ).join('');
    return `
      <div class="memory-card">
        <div class="memory-header">
          <div class="memory-summary">${mem.summary || mem.content?.slice(0, 200) || '—'}</div>
          <div class="importance-wrap">
            <span class="importance-score">${imp}</span>
            <div class="importance-bar-wrap"><div class="importance-bar-fill" style="width:${pct}%"></div></div>
          </div>
        </div>
        <div class="memory-footer">
          <span class="source-badge">${mem.source || 'unknown'}</span>
          ${entities}
          <span class="time-label">${relativeTime(mem.timestamp)}</span>
        </div>
      </div>`;
  }

  async function loadMemories(reset = false) {
    if (reset) { memoriesOffset = 0; document.getElementById('memories-grid').innerHTML = ''; }
    const min = document.getElementById('importance-slider').value;
    const btn = document.getElementById('btn-load-memories');
    btn.disabled = true; btn.textContent = 'Loading…';
    try {
      const data = await api.get(`/api/memories?min_importance=${min}&limit=${MEMORIES_LIMIT}&offset=${memoriesOffset}`);
      const grid = document.getElementById('memories-grid');
      const mems = data.memories || data;
      if (!mems.length && memoriesOffset === 0) {
        grid.innerHTML = '<p class="muted" style="padding:20px 0">No memories yet. Ingest some text first.</p>';
      } else {
        mems.forEach(m => { grid.insertAdjacentHTML('beforeend', renderMemory(m)); });
        memoriesOffset += mems.length;
        const more = document.getElementById('btn-load-more');
        if (mems.length === MEMORIES_LIMIT) more.classList.remove('hidden');
        else more.classList.add('hidden');
      }
    } catch (e) {
      toast(`Could not load memories: ${e.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = 'Load Memories';
    }
  }

  // ── GRAPH ─────────────────────────────────────────────────────
  async function loadEntities() {
    const btn = document.getElementById('btn-load-entities');
    if (btn) { btn.disabled = true; btn.textContent = 'Loading…'; }
    try {
      const data = await api.get('/api/graph/entities');
      const list = document.getElementById('entity-list');
      const entities = data.entities || data;
      if (!entities.length) {
        list.innerHTML = '<p class="muted" style="padding:20px 0">No entities yet. Ingest some text to extract entities.</p>';
        return;
      }
      list.innerHTML = '';
      entities.forEach(e => {
        const card = document.createElement('div');
        card.className = 'entity-card';
        card.id = `entity-${e.name.replace(/\s+/g, '-')}`;
        card.innerHTML = `
          <div class="entity-header" onclick="toggleEntity(this)">
            <span class="entity-name">${e.name}</span>
            <span class="entity-type">${e.type || 'entity'}</span>
            <span class="entity-count">${e.mention_count || 1}×</span>
            <span class="entity-chevron">▶</span>
          </div>
          <div class="entity-chunks" id="chunks-${e.name.replace(/\s+/g, '-')}"></div>`;
        list.appendChild(card);
      });
    } catch (e) {
      toast(`Could not load entities: ${e.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = 'Load Entities'; }
    }
  }

  window.toggleEntity = async function (header) {
    const card = header.parentElement;
    const name = card.querySelector('.entity-name').textContent;
    const chunksEl = document.getElementById(`chunks-${name.replace(/\s+/g, '-')}`);
    if (card.classList.contains('open')) {
      card.classList.remove('open');
      return;
    }
    card.classList.add('open');
    if (chunksEl.children.length) return;
    chunksEl.innerHTML = '<div class="entity-chunk muted">Loading…</div>';
    try {
      const data = await api.get(`/api/graph/entity/${encodeURIComponent(name)}`);
      const chunks = data.chunks || [];
      chunksEl.innerHTML = chunks.length
        ? chunks.map(c => `<div class="entity-chunk">${c.content?.slice(0, 220) || c.summary || '—'}<br><span class="muted" style="font-size:0.75rem">${relativeTime(c.timestamp)}</span></div>`).join('')
        : '<div class="entity-chunk muted">No connected chunks found.</div>';
    } catch {
      chunksEl.innerHTML = '<div class="entity-chunk muted">Could not load chunks.</div>';
    }
  };

  window.focusEntity = function (name) {
    const id = `entity-${name.replace(/\s+/g, '-')}`;
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const header = el.querySelector('.entity-header');
      if (!el.classList.contains('open') && header) header.click();
    } else {
      document.getElementById('section-graph').scrollIntoView({ behavior: 'smooth' });
    }
  };

  // ── FADE-IN OBSERVER ─────────────────────────────────────────
  function setupFadeIn() {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); } });
    }, { threshold: 0.08 });
    document.querySelectorAll('.fade-in').forEach(el => obs.observe(el));
  }

  // ── DRAG & DROP ──────────────────────────────────────────────
  function setupDrop() {
    const zone = document.getElementById('drop-zone');
    const input = document.getElementById('file-input');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length) setFile(input.files[0]); });

    function setFile(f) {
      document.getElementById('file-name').textContent = f.name;
      document.getElementById('btn-ingest-file').disabled = false;
      // sync to the real file input
      const dt = new DataTransfer();
      dt.items.add(f);
      input.files = dt.files;
    }
  }

  // ── INIT ─────────────────────────────────────────────────────
  async function init() {
    setupFadeIn();
    setupDrop();

    document.getElementById('btn-ingest-text').addEventListener('click', ingestText);
    document.getElementById('btn-ingest-file').addEventListener('click', ingestFile);
    document.getElementById('btn-query').addEventListener('click', runQuery);
    document.getElementById('query-input').addEventListener('keydown', e => { if (e.key === 'Enter') runQuery(); });
    document.getElementById('btn-load-memories').addEventListener('click', () => loadMemories(true));
    document.getElementById('btn-load-more').addEventListener('click', () => loadMemories(false));

    document.getElementById('importance-slider').addEventListener('input', function () {
      document.getElementById('importance-val').textContent = parseFloat(this.value).toFixed(2);
    });

    document.querySelectorAll('[data-sample]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.getElementById('ingest-text').value = SAMPLES[btn.dataset.sample] || '';
        document.getElementById('section-ingest').scrollIntoView({ behavior: 'smooth' });
      });
    });

    document.querySelectorAll('[data-query]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.getElementById('query-input').value = btn.dataset.query;
        runQuery();
      });
    });

    const health = await refreshHealth();
    setInterval(refreshHealth, 30000);

    // Auto-load memories and entities if data exists
    if (health && health.memories > 0) loadMemories(true);
    if (health && health.entities > 0) loadEntities();
    else {
      document.getElementById('btn-load-entities').classList.remove('hidden');
      document.getElementById('btn-load-entities').addEventListener('click', loadEntities);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
