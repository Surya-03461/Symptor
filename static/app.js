/**
 * AI Health Symptom Checker — Frontend JS
 * IBM Granite · IBM Cloud
 */

'use strict';

// ─── State ─────────────────────────────────────
const QUICK_SYMPTOMS = [
  { icon: '🤒', label: 'Fever', query: 'I have a fever and feel very warm' },
  { icon: '😷', label: 'Sore Throat', query: 'I have a sore throat and difficulty swallowing' },
  { icon: '🤕', label: 'Headache', query: 'I have a severe headache' },
  { icon: '😮‍💨', label: 'Cough', query: 'I have a persistent cough' },
  { icon: '💔', label: 'Chest Pain', query: 'I am experiencing chest pain' },
  { icon: '😮', label: 'Shortness of Breath', query: 'I have shortness of breath' },
  { icon: '🤢', label: 'Stomach Pain', query: 'I have stomach pain and nausea' },
  { icon: '😴', label: 'Fatigue', query: 'I feel extremely fatigued and tired all the time' },
  { icon: '🤮', label: 'Nausea', query: 'I feel nauseous and want to vomit' },
  { icon: '🩺', label: 'Rash', query: 'I have a rash on my skin' },
];

const EXAMPLE_QUERIES = [
  'I have a sore throat and fever',
  'I\'ve had a headache for 3 days',
  'Chest pain when I breathe',
  'Feeling very tired and dizzy',
];

// ─── DOM References ─────────────────────────────
const chatWindow     = document.getElementById('chatWindow');
const userInput      = document.getElementById('userInput');
const sendBtn        = document.getElementById('sendBtn');
const charCount      = document.getElementById('charCount');
const urgencyAlert   = document.getElementById('urgencyAlert');
const loadingOverlay = document.getElementById('loadingOverlay');
const clearChatBtn   = document.getElementById('clearChat');
const quickSymptomsEl = document.getElementById('quickSymptoms');

// ─── Initialization ─────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderQuickSymptoms();
  renderWelcomeMessage();
  userInput.focus();
});

function renderQuickSymptoms() {
  QUICK_SYMPTOMS.forEach(item => {
    const btn = document.createElement('button');
    btn.className = 'quick-symptom-btn';
    btn.innerHTML = `<span>${item.icon}</span>${item.label}`;
    btn.addEventListener('click', () => {
      userInput.value = item.query;
      updateCharCount();
      userInput.focus();
    });
    quickSymptomsEl.appendChild(btn);
  });
}

function renderWelcomeMessage() {
  const card = document.createElement('div');
  card.className = 'welcome-card';
  card.innerHTML = `
    <h2>👋 Welcome to AI Health Symptom Checker</h2>
    <p>Describe your symptoms in natural language. The AI, powered by <strong>IBM Granite</strong>, will analyze them and provide possible conditions, urgency levels, home care tips, and when to see a doctor — all sourced from <strong>WHO, CDC, and NIH</strong> guidelines.</p>
    <p><strong>Try an example:</strong></p>
    <div class="welcome-examples" id="welcomeExamples"></div>
  `;
  chatWindow.appendChild(card);

  const examplesEl = card.querySelector('#welcomeExamples');
  EXAMPLE_QUERIES.forEach(query => {
    const chip = document.createElement('button');
    chip.className = 'example-chip';
    chip.textContent = query;
    chip.addEventListener('click', () => {
      userInput.value = query;
      updateCharCount();
      userInput.focus();
    });
    examplesEl.appendChild(chip);
  });
}

// ─── Chat Logic ─────────────────────────────────

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Append user message
  appendMessage('user', text);
  userInput.value = '';
  updateCharCount();
  hideUrgencyAlert();

  // Show loading
  setLoading(true);

  try {
      const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
      });

      const data = await response.json();

      // Guardrail blocked the request
      if (!response.ok || data.guardrail_blocked) {
          appendGuardrailBlock(data.error || 'Your message was blocked by a safety guardrail.');
          return;
      }

      appendAssistantMessage(data);

      // Show non-blocking guardrail warnings (e.g. PII redaction notice)
      if (data.guardrail_warnings && data.guardrail_warnings.length > 0) {
          appendGuardrailWarnings(data.guardrail_warnings);
      }

      if (data.urgency) {
          showUrgencyAlert(data.urgency, data.urgency_info);
      }

  } catch (err) {
      appendMessage('assistant', `❌ Error: ${err.message}\n\nPlease try again or check your connection.`, null, 'error');
  } finally {
      setLoading(false);
  }
}

function appendMessage(role, text, meta = null, extraClass = '') {
  const msgEl = document.createElement('div');
  msgEl.className = `message ${role} ${extraClass}`;

  const avatarEl = document.createElement('div');
  avatarEl.className = 'msg-avatar';
  avatarEl.textContent = role === 'user' ? '👤' : '🏥';

  const contentEl = document.createElement('div');

  const bubbleEl = document.createElement('div');
  bubbleEl.className = 'msg-bubble';
  bubbleEl.innerHTML = formatMessageText(text);

  contentEl.appendChild(bubbleEl);

  if (meta) {
    const metaEl = document.createElement('div');
    metaEl.className = 'msg-meta';
    metaEl.innerHTML = meta;
    contentEl.appendChild(metaEl);
  }

  msgEl.appendChild(avatarEl);
  msgEl.appendChild(contentEl);

  chatWindow.appendChild(msgEl);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendGuardrailBlock(reason) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.innerHTML = `
        <div class="msg-avatar">🛡️</div>
        <div>
            <div class="msg-bubble guardrail-block">
                <strong>🚫 Message Blocked by Safety Guardrail</strong><br/><br/>
                ${escapeHtml(reason)}
            </div>
            <div class="msg-meta">Safety Guardrail · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
        </div>
    `;
    chatWindow.appendChild(msgEl);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendGuardrailWarnings(warnings) {
    const el = document.createElement('div');
    el.className = 'guardrail-warnings';
    el.innerHTML = `
        <span class="guardrail-warn-icon">⚠️ Safety Notice</span>
        <ul>${warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}</ul>
    `;
    chatWindow.appendChild(el);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function appendAssistantMessage(data) {
    const { response, urgency, matched_symptoms, source, is_emergency } = data;

  const extraClass = is_emergency ? 'emergency' : '';

  const sourceLabel = source === 'ibm_granite'
    ? '<span class="source-badge source-granite">IBM Granite</span>'
    : '<span class="source-badge source-db">Medical Database</span>';

  const symptomsLabel = matched_symptoms && matched_symptoms.length
    ? `Detected: <strong>${matched_symptoms.join(', ')}</strong> · `
    : '';

  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const meta = `${symptomsLabel}${sourceLabel} · ${time}`;

  appendMessage('assistant', response, meta, extraClass);
}

function formatMessageText(text) {
  // Convert markdown-style **bold** to <strong>
  let formatted = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>');
  return formatted;
}

// ─── Urgency Alert ───────────────────────────────

function showUrgencyAlert(urgency, urgencyInfo) {
  if (!urgencyInfo) return;
  urgencyAlert.className = `urgency-alert level-${urgency}`;
  urgencyAlert.innerHTML = `
    <span>${urgencyInfo.label || ''}</span>
    <span>${urgencyInfo.description || ''}</span>
  `;
  urgencyAlert.classList.remove('hidden');
}

function hideUrgencyAlert() {
  urgencyAlert.classList.add('hidden');
  urgencyAlert.className = 'urgency-alert hidden';
}

// ─── UI Helpers ─────────────────────────────────

function setLoading(show) {
  loadingOverlay.classList.toggle('hidden', !show);
  sendBtn.disabled = show;
  userInput.disabled = show;
}

function updateCharCount() {
  const len = userInput.value.length;
  charCount.textContent = `${len} / 2000`;
  charCount.style.color = len > 1800 ? '#da1e28' : '';
}

async function clearConversation() {
  try {
    await fetch('/api/clear', { method: 'POST' });
  } catch (_) {}
  chatWindow.innerHTML = '';
  hideUrgencyAlert();
  renderWelcomeMessage();
  userInput.value = '';
  updateCharCount();
  userInput.focus();
}

// ─── Event Listeners ─────────────────────────────

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

userInput.addEventListener('input', updateCharCount);

clearChatBtn.addEventListener('click', clearConversation);
