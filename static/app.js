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
  // A Supabase session outranks a pasted key — the signed-in identity is
  // what the Space panel shows, so requests must match it. A stored key
  // routes to that key's workspace; with neither, the shared demo space.
  function authHeaders() {
    const t = window.CWAuth && window.CWAuth.token();
    if (t) return { 'Authorization': 'Bearer ' + t };
    const k = localStorage.getItem('cw_api_key');
    return k ? { 'X-API-Key': k } : {};
  }

  const api = {
    async get(path) {
      const r = await fetch(BASE + path, { headers: authHeaders() });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    },
    async post(path, body) {
      const r = await fetch(BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const detail = err.detail;
        const msg = Array.isArray(detail)
          ? detail.map(d => d.msg || JSON.stringify(d)).join('; ')
          : (typeof detail === 'string' ? detail : `${r.status} ${r.statusText}`);
        throw new Error(msg);
      }
      return r.json();
    },
    async upload(path, file) {
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(BASE + path, { method: 'POST', headers: authHeaders(), body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const detail = err.detail;
        const msg = Array.isArray(detail)
          ? detail.map(d => d.msg || JSON.stringify(d)).join('; ')
          : (typeof detail === 'string' ? detail : `${r.status} ${r.statusText}`);
        throw new Error(msg);
      }
      return r.json();
    },
  };

  // ── HTML ESCAPING ────────────────────────────────────────────
  // All ingested content is user-controlled; anything rendered via
  // innerHTML must pass through here to prevent stored XSS.
  function esc(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

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
  const statKeys = ['events', 'memories', 'entities', 'vectors'];

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
        const el = document.getElementById(`stat-${k}`);
        if (el) animateCount(el, h[k] ?? 0);
      });
      const dot = document.getElementById('live-dot');
      const lbl = document.getElementById('live-label');
      dot.className = 'live-dot online';
      lbl.textContent = 'live';
      return h;
    } catch {
      const dot = document.getElementById('live-dot');
      const lbl = document.getElementById('live-label');
      dot.className = 'live-dot error';
      lbl.textContent = 'unreachable';
      return null;
    }
  }

  // ── NUDGE (proactive digest) ─────────────────────────────────
  function fillList(id, items, emptyText) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '';
    const list = items && items.length ? items : null;
    if (!list) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = emptyText;
      el.appendChild(li);
      return;
    }
    list.forEach(text => {
      const li = document.createElement('li');
      li.textContent = text;
      el.appendChild(li);
    });
  }

  async function loadDigest(force = false) {
    const section = document.getElementById('section-nudge');
    if (!section) return;
    try {
      const d = await api.get('/api/digest' + (force ? '?force=true' : ''));
      if (!d.memory_count) { section.classList.add('hidden'); return; }
      document.getElementById('nudge-headline').textContent = d.headline || '';
      fillList('nudge-focus', d.focus, 'Nothing surfaced yet');
      fillList('nudge-commitments', d.commitments, 'No open commitments');
      fillList('nudge-gaps', d.gaps, 'Nothing slipping');
      section.classList.remove('hidden');
    } catch {
      section.classList.add('hidden');
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
      const body = { content };
      const sourceSel = document.getElementById('source-select');
      if (sourceSel && sourceSel.value) body.source = sourceSel.value;
      const res = await api.post('/api/ingest/text', body);
      animatePipeline();
      toast(`✓ ${res.chunks_created} chunk${res.chunks_created !== 1 ? 's' : ''}, ${res.entities_extracted} entities extracted`, 'success');
      document.getElementById('ingest-text').value = '';
      await refreshHealth();
      suggestQuery(content);
    } catch (e) {
      toast(`Ingest failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false; btn.textContent = 'Ingest text';
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
      btn.disabled = true; btn.textContent = 'Upload file';
      await refreshHealth();
    } catch (e) {
      toast(`Upload failed: ${e.message}`, 'error');
      btn.disabled = false; btn.textContent = 'Upload file';
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
    return esc(text)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^(.+)$/, '<p>$1</p>');
  }

  async function runQuery(overrideQuery) {
    const q = overrideQuery || document.getElementById('query-input').value.trim();
    if (!q) { toast('Enter a query first.', 'error'); return; }
    if (overrideQuery) document.getElementById('query-input').value = overrideQuery;
    const btn = document.getElementById('btn-query');
    btn.disabled = true; btn.textContent = 'Thinking…';
    const card = document.getElementById('response-card');
    card.classList.add('hidden');
    try {
      const dateFrom = document.getElementById('date-from')?.value || null;
      const dateTo = document.getElementById('date-to')?.value || null;
      const res = await api.post('/api/query', {
        query: q, top_k: 8,
        date_from: dateFrom || null,
        date_to: dateTo || null,
      });
      const body = document.getElementById('response-body');
      body.innerHTML = renderMarkdown(res.answer || 'No answer returned.');

      document.getElementById('query-type-badge').textContent = res.query_type || 'general';
      const cited = (res.cited_memories || []).length || res.context_count || 0;
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

      // Expanded terms
      const et = document.getElementById('expanded-terms');
      if (et) {
        const terms = res.expanded_terms || [];
        et.innerHTML = terms.length
          ? `<span class="muted" style="font-size:.8rem">Also searched: </span>` +
            terms.map(t => `<span class="entity-pill" style="font-size:.75rem;opacity:.7">${esc(t)}</span>`).join('')
          : '';
      }

      // Suggested follow-up queries (DOM-built: inline onclick can't reach
      // the IIFE-scoped runQuery, and raw interpolation would be injectable)
      const sq = document.getElementById('suggested-queries');
      if (sq) {
        sq.innerHTML = '';
        const suggestions = res.suggested_queries || [];
        if (suggestions.length) {
          const wrap = document.createElement('div');
          wrap.style.marginTop = '1rem';
          const label = document.createElement('p');
          label.className = 'muted';
          label.style.cssText = 'font-size:.8rem;margin-bottom:.4rem';
          label.textContent = 'Follow-up questions:';
          wrap.appendChild(label);
          suggestions.forEach(s => {
            const b = document.createElement('button');
            b.className = 'btn-suggestion';
            b.textContent = s;
            b.addEventListener('click', () => runQuery(s));
            wrap.appendChild(b);
          });
          sq.appendChild(wrap);
        }
      }

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
      `<span class="entity-pill" data-entity="${esc(e)}">${esc(e)}</span>`
    ).join('');
    const accessCount = mem.access_count || 0;
    const accessBadge = accessCount > 0
      ? `<span class="access-badge" title="Recalled ${accessCount} time${accessCount !== 1 ? 's' : ''}">↩ ${accessCount}</span>`
      : '';
    return `
      <div class="memory-card">
        <div class="memory-header">
          <div class="memory-summary">${esc(mem.summary || mem.content?.slice(0, 200) || '—')}</div>
          <div class="importance-wrap">
            <span class="importance-score">${imp}</span>
            <div class="importance-bar-wrap"><div class="importance-bar-fill" style="width:${pct}%"></div></div>
          </div>
        </div>
        <div class="memory-footer">
          <span class="source-badge">${esc(mem.source || 'unknown')}</span>
          ${entities}
          ${accessBadge}
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
      btn.disabled = false; btn.textContent = 'Load memories';
    }
  }

  // ── GRAPH ─────────────────────────────────────────────────────
  const ENTITY_TYPE_COLORS = {
    person: 'entity-type-person',
    project: 'entity-type-project',
    topic: 'entity-type-topic',
    organization: 'entity-type-org',
    org: 'entity-type-org',
    place: 'entity-type-place',
    location: 'entity-type-place',
  };

  let _loadedEntities = [];

  async function loadEntities() {
    const btn = document.getElementById('btn-load-entities');
    if (btn) { btn.disabled = true; btn.textContent = 'Loading…'; }
    try {
      const data = await api.get('/api/graph/entities');
      const list = document.getElementById('entity-list');
      const entities = data.entities || data;
      _loadedEntities = entities;
      if (!entities.length) {
        list.innerHTML = '<p class="muted" style="padding:20px 0">No entities yet. Ingest some text to extract entities.</p>';
        return;
      }
      list.innerHTML = '';
      entities.forEach(e => {
        const rawType = (e.entity_type || e.type || 'entity').toLowerCase();
        const colorClass = ENTITY_TYPE_COLORS[rawType] || '';
        const card = document.createElement('div');
        card.className = 'entity-card';
        card.id = `entity-${e.name.replace(/\s+/g, '-')}`;
        card.innerHTML = `
          <div class="entity-header">
            <span class="entity-name">${esc(e.name)}</span>
            <span class="entity-type ${colorClass}">${esc(rawType)}</span>
            <span class="entity-count">${e.mention_count || 1}×</span>
            <span class="entity-chevron">▶</span>
          </div>
          <div class="entity-chunks" id="chunks-${esc(e.name.replace(/\s+/g, '-'))}"></div>`;
        list.appendChild(card);
      });
    } catch (e) {
      toast(`Could not load entities: ${e.message}`, 'error');
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = 'Load entities'; }
    }
  }

  async function surpriseMe() {
    if (!_loadedEntities.length) {
      try {
        const data = await api.get('/api/graph/entities?limit=50');
        _loadedEntities = data.entities || [];
      } catch {
        toast('Load entities first.', 'error'); return;
      }
    }
    if (!_loadedEntities.length) { toast('No entities yet — ingest some text first.', 'error'); return; }
    const pick = _loadedEntities[Math.floor(Math.random() * _loadedEntities.length)];
    const type = (pick.entity_type || pick.type || 'thing').toLowerCase();
    const templates = {
      person: [`What have I said or thought about ${pick.name}?`, `How does ${pick.name} connect to my current work?`],
      project: [`What's the status and history of ${pick.name}?`, `What decisions have I made about ${pick.name}?`],
      topic: [`What do I know about ${pick.name}?`, `How has my thinking on ${pick.name} evolved?`],
      organization: [`What's my relationship with ${pick.name}?`, `What have I noted about ${pick.name}?`],
      place: [`What do I associate with ${pick.name}?`, `What happened at or near ${pick.name}?`],
    };
    const options = templates[type] || [`Tell me everything about ${pick.name}.`, `What's the context around ${pick.name}?`];
    const query = options[Math.floor(Math.random() * options.length)];
    document.getElementById('section-query').scrollIntoView({ behavior: 'smooth', block: 'start' });
    setTimeout(() => runQuery(query), 400);
  }

  async function toggleEntity(header) {
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
      const chunks = data.connected_chunks || data.chunks || [];
      chunksEl.innerHTML = chunks.length
        ? chunks.map(c => `<div class="entity-chunk">${esc(c.content?.slice(0, 220) || c.summary || '—')}<br><span class="muted" style="font-size:0.75rem">${relativeTime(c.timestamp)}</span></div>`).join('')
        : '<div class="entity-chunk muted">No connected chunks found.</div>';
    } catch {
      chunksEl.innerHTML = '<div class="entity-chunk muted">Could not load chunks.</div>';
    }
  }

  function focusEntity(name) {
    const id = `entity-${name.replace(/\s+/g, '-')}`;
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      const header = el.querySelector('.entity-header');
      if (!el.classList.contains('open') && header) header.click();
    } else {
      document.getElementById('section-graph').scrollIntoView({ behavior: 'smooth' });
    }
  }

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

  // ── PRIVATE SPACE ────────────────────────────────────────────
  function spaceStatusRefresh() {
    const email = (window.CWAuth && window.CWAuth.email()) || null;
    const hasKey = !!localStorage.getItem('cw_api_key');
    const isPrivate = !!email || hasKey;
    document.getElementById('space-status').textContent = email
      ? `Signed in as ${email} — memories here are yours alone.`
      : hasKey
        ? 'Private space active — memories here are yours alone.'
        : 'Public demo space — anything you ingest is visible to other visitors.';
    document.getElementById('btn-wipe').classList.toggle('hidden', !isPrivate);
    document.getElementById('btn-leave-space').classList.toggle('hidden', !hasKey || !!email);
    document.getElementById('btn-create-space').classList.toggle('hidden', isPrivate);
    document.getElementById('btn-sign-out').classList.toggle('hidden', !email);
    // While signed in, the key-paste flow is hidden to avoid mixed identities
    document.getElementById('key-input').classList.toggle('hidden', !!email);
    document.getElementById('btn-use-key').classList.toggle('hidden', !!email);
    const authBox = document.getElementById('auth-box');
    if (authBox) {
      const authEnabled = !!(window.CWAuth && window.CWAuth.enabled);
      authBox.classList.toggle('hidden', !authEnabled || !!email);
    }
  }

  async function refreshAll() {
    const health = await refreshHealth();
    memoriesOffset = 0;
    document.getElementById('memories-grid').innerHTML = '';
    document.getElementById('entity-list').innerHTML = '';
    _loadedEntities = [];

    if (health && health.memories > 0) {
      loadMemories(true);
      loadDigest();
    } else {
      const nudge = document.getElementById('section-nudge');
      if (nudge) nudge.classList.add('hidden');
    }
    if (health && health.entities > 0) loadEntities();
    document.getElementById('btn-load-entities')
      .classList.toggle('hidden', !!(health && health.entities > 0));
    loadDigestSub();
    return health;
  }

  function initSpace() {
    spaceStatusRefresh();

    document.getElementById('btn-create-space').addEventListener('click', async () => {
      try {
        const res = await api.post('/api/auth/register', {});
        localStorage.setItem('cw_api_key', res.api_key);
        document.getElementById('key-value').textContent = res.api_key;
        document.getElementById('key-reveal').classList.remove('hidden');
        spaceStatusRefresh();
        toast('Private space created — copy your key somewhere safe, it is shown once.', 'success', 7000);
        await refreshAll();
      } catch (e) {
        toast(`Could not create space: ${e.message}`, 'error');
      }
    });

    document.getElementById('btn-copy-key').addEventListener('click', () => {
      const key = document.getElementById('key-value').textContent;
      navigator.clipboard.writeText(key)
        .then(() => toast('Key copied.', 'success'))
        .catch(() => toast('Copy failed — select the key manually.', 'error'));
    });

    document.getElementById('btn-use-key').addEventListener('click', async () => {
      const key = document.getElementById('key-input').value.trim();
      if (!key) { toast('Paste a key first.', 'error'); return; }
      const r = await fetch(BASE + '/api/me', { headers: { 'X-API-Key': key } });
      if (!r.ok) { toast('Invalid key.', 'error'); return; }
      localStorage.setItem('cw_api_key', key);
      document.getElementById('key-input').value = '';
      spaceStatusRefresh();
      toast('Private space activated.', 'success');
      await refreshAll();
    });

    document.getElementById('btn-leave-space').addEventListener('click', async () => {
      localStorage.removeItem('cw_api_key');
      document.getElementById('key-reveal').classList.add('hidden');
      spaceStatusRefresh();
      toast('Back in the public demo space.', 'info');
      await refreshAll();
    });

    document.getElementById('btn-export').addEventListener('click', async () => {
      try {
        const data = await api.get('/api/export');
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'contextweave-export.json';
        a.click();
        URL.revokeObjectURL(a.href);
        toast('Export downloaded.', 'success');
      } catch (e) {
        toast(`Export failed: ${e.message}`, 'error');
      }
    });

    document.getElementById('btn-wipe').addEventListener('click', async () => {
      if (!window.confirm('Erase ALL memories in your private space? This cannot be undone.')) return;
      try {
        const r = await fetch(BASE + '/api/memory', { method: 'DELETE', headers: authHeaders() });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || r.statusText);
        }
        toast('Memory wiped.', 'success');
        await refreshAll();
      } catch (e) {
        toast(`Wipe failed: ${e.message}`, 'error');
      }
    });

    document.getElementById('btn-refresh-nudge').addEventListener('click', () => loadDigest(true));
  }

  // ── DAILY DIGEST EMAIL ───────────────────────────────────────
  function isPrivateSpace() {
    return !!((window.CWAuth && window.CWAuth.email()) || localStorage.getItem('cw_api_key'));
  }

  async function loadDigestSub() {
    const box = document.getElementById('digest-sub');
    if (!box) return;
    if (!isPrivateSpace()) { box.classList.add('hidden'); return; }
    try {
      const s = await api.get('/api/digest/subscription');
      if (!s.available) { box.classList.add('hidden'); return; }
      box.classList.remove('hidden');
      const note = document.getElementById('digest-sub-note');
      const emailInput = document.getElementById('digest-email');
      document.getElementById('btn-digest-unsubscribe').classList.toggle('hidden', !s.subscribed);
      if (s.subscribed) {
        note.textContent = `Daily nudge goes to ${s.email} around ${String(s.send_hour_utc).padStart(2, '0')}:00 UTC.`;
        if (!emailInput.value) emailInput.value = s.email;
      } else {
        note.textContent = 'One email a day: your focus, commitments, and what’s slipping.';
        const signedIn = window.CWAuth && window.CWAuth.email();
        if (signedIn && !emailInput.value) emailInput.value = signedIn;
      }
    } catch {
      box.classList.add('hidden');
    }
  }

  function initDigestSub() {
    if (!document.getElementById('digest-sub')) return;

    document.getElementById('btn-digest-subscribe').addEventListener('click', async () => {
      const email = document.getElementById('digest-email').value.trim();
      if (!email) { toast('Enter an email address first.', 'error'); return; }
      // aim for ~8am in the visitor's timezone, expressed as a UTC hour
      const tzHours = Math.round(-new Date().getTimezoneOffset() / 60);
      const sendHour = ((8 - tzHours) % 24 + 24) % 24;
      try {
        await api.post('/api/digest/subscribe', { email, send_hour_utc: sendHour });
        toast('Subscribed — your first digest arrives tomorrow morning.', 'success', 6000);
        await loadDigestSub();
      } catch (e) {
        toast(`Could not subscribe: ${e.message}`, 'error');
      }
    });

    document.getElementById('btn-digest-unsubscribe').addEventListener('click', async () => {
      try {
        const r = await fetch(BASE + '/api/digest/subscribe', { method: 'DELETE', headers: authHeaders() });
        if (!r.ok) throw new Error(r.statusText);
        toast('Daily digest emails stopped.', 'info');
        await loadDigestSub();
      } catch (e) {
        toast(`Could not unsubscribe: ${e.message}`, 'error');
      }
    });
  }

  // ── SUPABASE SIGN-IN ─────────────────────────────────────────
  function initAuth() {
    if (!document.getElementById('auth-box')) return;
    const emailEl = document.getElementById('auth-email');
    const pwEl = document.getElementById('auth-password');

    async function doAuth(mode) {
      const email = emailEl.value.trim();
      const pw = pwEl.value;
      if (!email || !pw) { toast('Email and password required.', 'error'); return; }
      const btn = document.getElementById(mode === 'in' ? 'btn-sign-in' : 'btn-sign-up');
      btn.disabled = true;
      try {
        if (mode === 'in') {
          await window.CWAuth.signIn(email, pw);
          toast('Signed in — this space is yours alone.', 'success');
        } else {
          const active = await window.CWAuth.signUp(email, pw);
          if (!active) {
            toast('Account created — confirm via the email we sent, then sign in.', 'info', 8000);
            return;
          }
          toast('Account created — you are signed in.', 'success');
        }
        pwEl.value = '';
        spaceStatusRefresh();
        await refreshAll();
      } catch (e) {
        toast(`Sign-${mode === 'in' ? 'in' : 'up'} failed: ${e.message}`, 'error');
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById('btn-sign-in').addEventListener('click', () => doAuth('in'));
    document.getElementById('btn-sign-up').addEventListener('click', () => doAuth('up'));
    pwEl.addEventListener('keydown', e => { if (e.key === 'Enter') doAuth('in'); });

    document.getElementById('btn-sign-out').addEventListener('click', async () => {
      try {
        await window.CWAuth.signOut();
        toast('Signed out — back in the public demo space.', 'info');
        spaceStatusRefresh();
        await refreshAll();
      } catch (e) {
        toast(`Sign-out failed: ${e.message}`, 'error');
      }
    });
  }

  // ── INIT ─────────────────────────────────────────────────────
  function setupNav() {
    const nav = document.getElementById('nav');
    window.addEventListener('scroll', () => {
      nav.classList.toggle('scrolled', window.scrollY > 40);
    }, { passive: true });
  }

  async function init() {
    setupFadeIn();
    setupDrop();
    setupNav();

    document.getElementById('btn-ingest-text').addEventListener('click', ingestText);
    document.getElementById('btn-ingest-file').addEventListener('click', ingestFile);
    document.getElementById('btn-query').addEventListener('click', () => runQuery());
    document.getElementById('query-input').addEventListener('keydown', e => { if (e.key === 'Enter') runQuery(); });
    document.getElementById('btn-load-memories').addEventListener('click', () => loadMemories(true));
    document.getElementById('btn-load-more').addEventListener('click', () => loadMemories(false));

    // Entity pills inside memory cards (delegated — cards are re-rendered)
    document.getElementById('memories-grid').addEventListener('click', e => {
      const pill = e.target.closest('.entity-pill');
      if (pill && pill.dataset.entity) focusEntity(pill.dataset.entity);
    });

    // Entity card headers (delegated — no inline handlers, CSP-safe)
    document.getElementById('entity-list').addEventListener('click', e => {
      const header = e.target.closest('.entity-header');
      if (header) toggleEntity(header);
    });

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

    const surpriseBtn = document.getElementById('btn-surprise-me');
    if (surpriseBtn) surpriseBtn.addEventListener('click', surpriseMe);

    document.getElementById('btn-load-entities').addEventListener('click', loadEntities);

    // Resolve any existing Supabase session before the first data load so
    // the initial requests already carry the caller's identity.
    if (window.CWAuth) {
      await window.CWAuth.init();
      window.CWAuth.onChange(spaceStatusRefresh);
    }
    initSpace();
    initAuth();
    initDigestSub();
    if (window.CWCapture) {
      window.CWCapture.init(
        document.getElementById('btn-mic'),
        document.getElementById('ingest-text'),
        document.getElementById('source-select')
      );
    }

    setInterval(refreshHealth, 30000);
    await refreshAll();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
