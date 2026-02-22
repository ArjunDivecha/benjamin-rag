/* ==========================================================================
   Benjamin Maurice AI — Shared Application Logic
   This JS is identical for both French and Professional themes.
   ========================================================================== */

const form = document.getElementById('chatForm');
const objective = document.getElementById('objective');
const objectiveHint = document.getElementById('objectiveHint');
const message = document.getElementById('message');
const modeInputs = document.querySelectorAll('input[name="mode"]');
const modeDirectInput = document.getElementById('modeDirect');
const modeRagInput = document.getElementById('modeRag');
const directModeLabel = document.getElementById('directModeLabel');
const ragModeLabel = document.getElementById('ragModeLabel');
const useWebSearchInput = document.getElementById('useWebSearch');
const webSearchLabel = document.getElementById('webSearchLabel');
const webSearchNote = document.getElementById('webSearchNote');
const atelierButtons = document.querySelectorAll('.atelier[data-objective]');
const directSection = document.getElementById('directSection');
const ragSection = document.getElementById('ragSection');
const insightsExamples = document.getElementById('insightsExamples');
const industryLabel = document.getElementById('industryLabel');
const stakeholderLabel = document.getElementById('stakeholderLabel');
const projectObjectivesLabel = document.getElementById('projectObjectivesLabel');
const keyQuestionsLabel = document.getElementById('keyQuestionsLabel');
const projectObjectives = document.getElementById('projectObjectives');
const keyQuestions = document.getElementById('keyQuestions');
const industry = document.getElementById('industry');
const stakeholderType = document.getElementById('stakeholderType');
const sampleLinks = document.querySelectorAll('.sample-link');
const fileInput = document.getElementById('fileInput');
const fileArea = document.getElementById('fileArea');
const fileName = document.getElementById('fileName');
const fileClear = document.getElementById('fileClear');
const submitBtn = document.getElementById('submitBtn');
const responseEl = document.getElementById('response');
const compareResponseEl = document.getElementById('compareResponse');
const docList = document.getElementById('docList');
const refreshDocsBtn = document.getElementById('refreshDocsBtn');
const modelLeft = document.getElementById('modelLeft');
const modelRight = document.getElementById('modelRight');

const OBJECTIVE_META = {
  expert_network_brief: {
    hint: 'Compose an expert-network brief with ~5 screening questions that test true knowledge depth.',
    promptPlaceholder: 'Draft an expert network brief and include five screening questions that cover the key hypotheses.',
    forceRag: false,
    fieldLabels: {
      industry: 'industry',
      stakeholder: 'target expert profile',
      objective: 'engagement objective',
      questions: 'screening objectives',
    },
    fieldPlaceholders: {
      industry: 'e.g. semiconductor tooling',
      stakeholder: 'e.g. former hyperscaler procurement lead',
      objective: 'Describe the project objective and interview goals',
      questions: 'What are the key questions screening should pressure-test?',
    },
  },
  interview_guide: {
    hint: 'Build a stakeholder-specific guide with opening, core, sensitive, and closing questions.',
    promptPlaceholder: 'Draft an interview guide for [stakeholder] with careful phrasing for sensitive topics.',
    forceRag: false,
    fieldLabels: {
      industry: 'industry',
      stakeholder: 'stakeholder type',
      objective: 'interview objective',
      questions: 'hypotheses to test',
    },
    fieldPlaceholders: {
      industry: 'e.g. advanced manufacturing',
      stakeholder: 'e.g. enterprise customer, competitor, former employee',
      objective: 'What does this interview need to uncover?',
      questions: 'List the hypotheses the guide should validate or disprove',
    },
  },
  insights_qa: {
    hint: 'Runs side-by-side answers (local + Opus) over the same RAG context for transcript analysis.',
    promptPlaceholder: 'Ask for counts, quotes, speaker mentions, or consensus views with source citations.',
    forceRag: true,
    fieldLabels: {
      industry: 'market context (optional)',
      stakeholder: 'interview segment (optional)',
      objective: 'analysis focus',
      questions: 'proposition to test',
    },
    fieldPlaceholders: {
      industry: 'e.g. semiconductors, industrial software',
      stakeholder: 'e.g. buyers, former operators, channel partners',
      objective: 'What insight do you need from the interview set?',
      questions: 'State the proposition to test (e.g., "market growth is accelerating")',
    },
  },
};

function currentMode() {
  return [...modeInputs].find(i => i.checked)?.value || 'direct';
}

function objectiveSupportsWebSearch() {
  return objective.value === 'expert_network_brief' || objective.value === 'interview_guide';
}

function syncWebSearchToggle() {
  const supported = objectiveSupportsWebSearch();
  useWebSearchInput.disabled = !supported;
  if (!supported) {
    useWebSearchInput.checked = false;
  }
  webSearchLabel.classList.toggle('is-disabled', !supported);
  webSearchNote.classList.toggle('is-disabled', !supported);
  webSearchNote.textContent = supported
    ? 'When checked, the backend fetches recent web results before generation.'
    : 'Disabled for Insights Parlour (Mode 3).';
}

function applyObjectivePresentation() {
  const meta = OBJECTIVE_META[objective.value] || OBJECTIVE_META.expert_network_brief;
  objectiveHint.textContent = meta.hint;
  message.placeholder = meta.promptPlaceholder;

  const forceRag = Boolean(meta.forceRag);
  modeDirectInput.disabled = forceRag;
  directModeLabel.classList.toggle('is-disabled', forceRag);
  if (forceRag) {
    modeRagInput.checked = true;
  }

  industryLabel.textContent = meta.fieldLabels.industry;
  stakeholderLabel.textContent = meta.fieldLabels.stakeholder;
  projectObjectivesLabel.textContent = meta.fieldLabels.objective;
  keyQuestionsLabel.textContent = meta.fieldLabels.questions;

  industry.placeholder = meta.fieldPlaceholders.industry;
  stakeholderType.placeholder = meta.fieldPlaceholders.stakeholder;
  projectObjectives.placeholder = meta.fieldPlaceholders.objective;
  keyQuestions.placeholder = meta.fieldPlaceholders.questions;

  const showInsightsExamples = objective.value === 'insights_qa';
  insightsExamples.classList.toggle('hidden', !showInsightsExamples);
  syncWebSearchToggle();
}

function toggleModeUI() {
  const mode = currentMode();
  const isRag = mode === 'rag';
  directSection.classList.toggle('hidden', isRag);
  ragSection.classList.toggle('hidden', !isRag);
}

function syncAtelierState() {
  atelierButtons.forEach(btn => {
    const isActive = btn.dataset.objective === objective.value;
    btn.classList.toggle('active', isActive);
  });
}

function collectionLabel(vertical) {
  if (vertical === 'V1') return 'Brief Salon';
  if (vertical === 'V2') return 'Interview Atelier';
  if (vertical === 'V3') return 'Insights Parlour';
  return vertical ? `Collection ${vertical}` : 'Unsorted Collection';
}

function fragmentLabel(count) {
  const n = Number(count) || 0;
  return `${n} fragment${n === 1 ? '' : 's'} archived`;
}

function safePercent(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return 'n/a';
  return `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%`;
}

function confidenceClass(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return 'low';
  if (n >= 0.8) return 'high';
  if (n >= 0.65) return 'medium';
  return 'low';
}

function formatNumber(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'n/a';
  return n.toFixed(digits);
}

function shouldRunCompare(mode, objectiveValue) {
  return true;
}

function comparePlaceholderHtml(isActive) {
  return '<span class="side-placeholder">When you press Begin, the answer and stats will appear here.</span>';
}

function renderResponseHtml(payload, options = {}) {
  const data = payload || {};
  const content = String(data.content || '').trim() || 'No response text returned.';

  const promptTokens = Number(data.usage?.prompt_tokens || 0);
  const completionTokens = Number(data.usage?.completion_tokens || 0);
  const totalTokens = Number(data.metrics?.total_tokens || (promptTokens + completionTokens));
  const latencyMs = Number(data.metrics?.latency_ms || 0);
  const tokPerSec = Number(data.metrics?.tok_per_sec || 0);
  const retrievalMs = Number(data.metrics?.retrieval_ms || options.retrievalMs || 0);
  const webSearch = options.webSearch || null;

  let costValue = 'n/a';
  if (data.provider === 'bedrock' && data.model) {
    let inCostPerM = 0;
    let outCostPerM = 0;
    const lowModel = data.model.toLowerCase();
    if (lowModel.includes('opus')) { inCostPerM = 5.0; outCostPerM = 25.0; }
    else if (lowModel.includes('sonnet')) { inCostPerM = 3.0; outCostPerM = 15.0; }
    else if (lowModel.includes('haiku')) { inCostPerM = 1.0; outCostPerM = 5.0; }

    if (inCostPerM > 0 || outCostPerM > 0) {
      const cost = (promptTokens * inCostPerM + completionTokens * outCostPerM) / 1000000;
      costValue = cost === 0 ? '$0.00' : (cost < 0.0001 ? '<$0.0001' : '$' + cost.toFixed(4));
    }
  } else if (data.provider === 'ollama') {
    costValue = '$0.00';
  }

  let html = '<div class="metrics-grid">';
  html += `<div class="metric-chip"><span class="metric-label">Tokens</span><span class="metric-value">${totalTokens}</span><span style="font-size:0.7rem; opacity:0.6;">(${promptTokens} in / ${completionTokens} out)</span></div>`;
  html += `<div class="metric-chip"><span class="metric-label">Time</span><span class="metric-value">${formatNumber(latencyMs / 1000, 1)}s</span></div>`;
  if (tokPerSec > 0) html += `<div class="metric-chip"><span class="metric-label">Speed</span><span class="metric-value">${formatNumber(tokPerSec, 1)} t/s</span></div>`;
  if (costValue !== 'n/a') html += `<div class="metric-chip metric-cost"><span class="metric-label">Cost</span><span class="metric-value">${costValue}</span></div>`;
  if (retrievalMs > 0) html += `<div class="metric-chip"><span class="metric-label">RAG</span><span class="metric-value">${formatNumber(retrievalMs / 1000, 1)}s</span></div>`;
  if (webSearch?.requested) {
    const webValue = webSearch.enabled
      ? `${Number(webSearch.results_count || 0)} hits`
      : 'off';
    html += `<div class="metric-chip"><span class="metric-label">Web</span><span class="metric-value">${escapeHtml(webValue)}</span></div>`;
  }
  html += '</div>';

  html += `<div class="response-content">${escapeHtml(content)}</div>`;

  const rag = options.rag || null;
  if (rag) {
    const collection = escapeHtml(rag.collection || 'n/a');
    const chunks = Number(rag.chunks_retrieved || 0);
    const docs = Number(rag.documents_retrieved || 0);
    const avgScore = safePercent(rag.avg_score);
    html += `
      <div class="rag-summary">
        <span class="rag-summary-title">RAG Context Summary</span>
        <div class="rag-pills">
          <span class="rag-pill">Collection: ${collection}</span>
          <span class="rag-pill">${chunks} chunks</span>
          <span class="rag-pill">${docs} documents</span>
          <span class="rag-pill">Avg confidence: ${avgScore}</span>
        </div>
      </div>
    `;
  }

  const includeSources = options.includeSources !== false;
  const sources = Array.isArray(options.sources) ? options.sources : [];
  if (includeSources && sources.length) {
    html += '<div class="sources"><h4>Referenced Passages</h4><ul>';
    html += sources.map((s) => {
      const filename = s.filename || 'Untitled source';
      const chunkNum = Number.isInteger(s.chunk_id) ? s.chunk_id + 1 : '?';
      const confidence = safePercent(s.score);
      const klass = confidenceClass(s.score);
      const snippet = s.snippet ? `<div class="source-snippet">${escapeHtml(s.snippet)}</div>` : '';
      return `
        <li class="source-${klass}">
          <div class="source-row">
            <span class="source-title">${escapeHtml(filename)}</span>
            <span class="confidence-pill ${klass}">${confidence}</span>
          </div>
          <div class="source-subline">Archive fragment ${chunkNum}</div>
          ${snippet}
        </li>
      `;
    }).join('');
    html += '</ul></div>';
  }

  const webSources = Array.isArray(webSearch?.sources) ? webSearch.sources : [];
  if (webSearch?.requested && webSources.length) {
    html += '<div class="sources"><h4>Live Web Context</h4><ul>';
    html += webSources.map((s) => {
      const title = s.title || 'Untitled source';
      const url = s.url || '';
      const date = s.published_date || 'Unknown date';
      const snippet = s.snippet ? `<div class="source-snippet">${escapeHtml(s.snippet)}</div>` : '';
      return `
        <li>
          <div class="source-row">
            <span class="source-title">${escapeHtml(title)}</span>
          </div>
          <div class="source-subline">${escapeHtml(date)}</div>
          ${url ? `<a class="doc-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Open source</a>` : ''}
          ${snippet}
        </li>
      `;
    }).join('');
    html += '</ul></div>';
  }

  return html;
}

function resetComparePanel() {
  const active = shouldRunCompare(currentMode(), objective.value);
  compareResponseEl.classList.remove('error');
  compareResponseEl.innerHTML = comparePlaceholderHtml(active);
}

atelierButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    objective.value = btn.dataset.objective;
    syncAtelierState();
    applyObjectivePresentation();
    toggleModeUI();
    resetComparePanel();
  });
});

objective.addEventListener('change', () => {
  syncAtelierState();
  applyObjectivePresentation();
  toggleModeUI();
  resetComparePanel();
});
modeInputs.forEach(input => input.addEventListener('change', () => {
  toggleModeUI();
  resetComparePanel();
}));

syncAtelierState();
applyObjectivePresentation();
toggleModeUI();
resetComparePanel();

sampleLinks.forEach(link => {
  link.addEventListener('click', () => {
    const sample = link.getAttribute('data-sample') || '';
    if (sample) {
      message.value = sample;
      message.focus();
    }
  });
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    fileName.textContent = fileInput.files[0].name;
    fileName.classList.add('has-file');
    fileArea.classList.add('has-file');
    fileClear.style.display = 'block';
  }
});

fileClear.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.value = '';
  fileName.textContent = 'Select a transcript, memo, or draft';
  fileName.classList.remove('has-file');
  fileArea.classList.remove('has-file');
  fileClear.style.display = 'none';
});

async function loadDocuments() {
  docList.innerHTML = '<div class="doc-item">Loading library...</div>';
  try {
    const res = await fetch('/api/documents');
    const docs = await res.json().catch(() => []);
    if (!res.ok) {
      docList.innerHTML = `<div class="doc-item">Library unavailable (${res.status})</div>`;
      return;
    }
    if (!docs.length) {
      docList.innerHTML = '<div class="doc-item">No documents are in the library yet.</div>';
      return;
    }

    const grouped = docs.reduce((acc, doc) => {
      const folderName = collectionLabel(doc.vertical);
      if (!acc[folderName]) {
        acc[folderName] = [];
      }
      acc[folderName].push(doc);
      return acc;
    }, {});

    const preferredOrder = ['Brief Salon', 'Interview Atelier', 'Insights Parlour'];
    const folderNames = Object.keys(grouped).sort((a, b) => {
      const aIdx = preferredOrder.indexOf(a);
      const bIdx = preferredOrder.indexOf(b);
      const aRank = aIdx === -1 ? 999 : aIdx;
      const bRank = bIdx === -1 ? 999 : bIdx;
      if (aRank !== bRank) return aRank - bRank;
      return a.localeCompare(b);
    });

    docList.innerHTML = folderNames.map((folderName, index) => {
      const entries = grouped[folderName] || [];
      const itemsHtml = entries.map(d => `
        <div class="doc-item">
          <strong>${escapeHtml(d.filename || 'Untitled note')}</strong>
          <div class="doc-meta">${escapeHtml(fragmentLabel(d.chunk_count))}</div>
          ${d.doc_id
          ? `<a class="doc-link" href="/api/documents/${encodeURIComponent(d.doc_id)}/file" target="_blank" rel="noopener noreferrer">Open original document</a>`
          : ''
        }
        </div>
      `).join('');

      return `
        <details class="library-folder">
          <summary class="folder-summary">
            <span class="folder-title">${escapeHtml(folderName)}</span>
            <span class="folder-side">
              <span>${entries.length} file${entries.length === 1 ? '' : 's'}</span>
              <span class="folder-chevron">&#9654;</span>
            </span>
          </summary>
          <div class="folder-items">${itemsHtml}</div>
        </details>
      `;
    }).join('');
  } catch (err) {
    docList.innerHTML = `<div class="doc-item">${escapeHtml(err.message)}</div>`;
  }
}

refreshDocsBtn.addEventListener('click', async () => {
  const originalText = refreshDocsBtn.textContent;
  refreshDocsBtn.textContent = 'Processing...';
  refreshDocsBtn.disabled = true;

  try {
    const res = await fetch('/api/documents/sync', { method: 'POST' });
    if (!res.ok) {
      throw new Error(`Sync failed (${res.status})`);
    }
    const data = await res.json();
    if (data.ingested === 0 && data.removed === 0) {
      refreshDocsBtn.textContent = 'No New Files';
    } else if (data.ingested > 0 && data.removed > 0) {
      refreshDocsBtn.textContent = `${data.ingested} Added, ${data.removed} Removed`;
    } else if (data.ingested > 0) {
      refreshDocsBtn.textContent = `${data.ingested} File${data.ingested === 1 ? '' : 's'} Ingested`;
    } else if (data.removed > 0) {
      refreshDocsBtn.textContent = `${data.removed} File${data.removed === 1 ? '' : 's'} Removed`;
    }
    await loadDocuments();
  } catch (err) {
    console.error('Sync error:', err);
    refreshDocsBtn.textContent = 'Error syncing';
  } finally {
    setTimeout(() => {
      refreshDocsBtn.textContent = originalText;
      refreshDocsBtn.disabled = false;
    }, 3000);
  }
});

loadDocuments();

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const mode = currentMode();
  const compareMode = shouldRunCompare(mode, objective.value);
  submitBtn.disabled = true;
  responseEl.style.display = 'block';
  responseEl.classList.remove('error');
  responseEl.innerHTML = '<span class="response-header">preparing...</span>';
  compareResponseEl.classList.remove('error');
  compareResponseEl.innerHTML = '<span class="response-header">preparing...</span>';

  const fd = new FormData();
  fd.append('mode', mode);
  fd.append('objective', objective.value);
  fd.append('use_web_search', useWebSearchInput.checked ? 'true' : 'false');

  let userMessage = message.value.trim();
  if (mode === 'rag') {
    const ragContextParts = [];
    if (projectObjectives.value.trim()) ragContextParts.push(`Project objectives: ${projectObjectives.value.trim()}`);
    if (keyQuestions.value.trim()) ragContextParts.push(`Key questions: ${keyQuestions.value.trim()}`);
    if (industry.value.trim()) ragContextParts.push(`Industry: ${industry.value.trim()}`);
    if (stakeholderType.value.trim()) ragContextParts.push(`Stakeholder type: ${stakeholderType.value.trim()}`);
    ragContextParts.push(`User request: ${userMessage}`);
    userMessage = ragContextParts.join('\n');
  }
  fd.append('message', userMessage);

  if (mode === 'direct' && fileInput.files.length) {
    fd.append('file', fileInput.files[0]);
  }
  fd.append('model_left', modelLeft.value);
  fd.append('model_right', modelRight.value);

  try {
    const endpoint = compareMode ? '/api/chat/compare' : '/api/chat';
    const res = await fetch(endpoint, {
      method: 'POST',
      body: fd,
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      responseEl.classList.add('error');
      responseEl.innerHTML = `<span class="response-header">request error ${res.status}</span>${escapeHtml(data.detail || res.statusText)}`;
      compareResponseEl.classList.add('error');
      compareResponseEl.innerHTML = `<span class="response-header">request error ${res.status}</span>${escapeHtml(data.detail || res.statusText)}`;
      return;
    }

    if (!data.left || !data.right) {
      throw new Error('Compare response missing left/right payload');
    }
    const ragSummary = data.rag || null;
    const sharedRetrievalMs = Number(data.metrics?.retrieval_ms || 0);
    const sources = Array.isArray(data.sources) ? data.sources : [];
    const webSearch = data.web_search || null;

    responseEl.innerHTML = renderResponseHtml(data.left, {
      rag: ragSummary,
      retrievalMs: sharedRetrievalMs,
      sources,
      webSearch,
      includeSources: true,
    });
    compareResponseEl.innerHTML = renderResponseHtml(data.right, {
      rag: ragSummary,
      retrievalMs: sharedRetrievalMs,
      sources,
      webSearch,
      includeSources: true,
    });
    loadDocuments();
  } catch (err) {
    responseEl.classList.add('error');
    responseEl.innerHTML = `<span class="response-header">request error</span>${escapeHtml(err.message)}`;
    compareResponseEl.classList.add('error');
    compareResponseEl.innerHTML = `<span class="response-header">request error</span>${escapeHtml(err.message)}`;
  } finally {
    submitBtn.disabled = false;
  }
});

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = String(value ?? '');
  return div.innerHTML;
}

async function fetchModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const models = data.models || [];
    if (models.length === 0) {
      modelLeft.innerHTML = '<option value="">No models found</option>';
      modelRight.innerHTML = '<option value="">No models found</option>';
      return;
    }
    modelLeft.innerHTML = '';
    modelRight.innerHTML = '';
    models.forEach((m, i) => {
      const opt1 = document.createElement('option');
      opt1.value = m.provider + "|" + m.model;
      opt1.textContent = m.provider + " · " + m.model;

      const opt2 = document.createElement('option');
      opt2.value = m.provider + "|" + m.model;
      opt2.textContent = m.provider + " · " + m.model;

      modelLeft.appendChild(opt1);
      modelRight.appendChild(opt2);
    });

    if (models.length > 1) {
      modelRight.selectedIndex = 1;
    }
  } catch (e) {
    console.error("Failed to fetch models", e);
  }
}
fetchModels();
