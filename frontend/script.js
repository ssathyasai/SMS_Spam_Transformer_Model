// API Configuration
const API_URL = 'http://localhost:5000';

// DOM Elements
const messageInput     = document.getElementById('message-input');
const predictBtn       = document.getElementById('predict-btn');
const resultSection    = document.getElementById('result-section');
const predictionResult = document.getElementById('prediction-result');
const confidenceFill   = document.getElementById('confidence-fill');
const confidenceValue  = document.getElementById('confidence-value');
const confidencePct    = document.getElementById('confidence-pct');
const probabilityPct   = document.getElementById('probability-pct');
const cleanedText      = document.getElementById('cleaned-text');
const spamIndicators   = document.getElementById('spam-indicators');
const indicatorTags    = document.getElementById('indicator-tags');
const fileInput        = document.getElementById('file-input');
const fileUploadArea   = document.getElementById('file-upload-area');
const processFileBtn   = document.getElementById('process-file-btn');
const regenerateFileBtn= document.getElementById('regenerate-file-btn');
const batchResults     = document.getElementById('batch-results');
const statsGrid        = document.getElementById('stats-grid');
const resultsTable     = document.getElementById('results-table');
const loadingOverlay   = document.getElementById('loading-overlay');

// ── Spam signal patterns for client-side indicator display ──
const SPAM_SIGNALS = [
    { pattern: /\bfree\b/i,           label: 'FREE offer' },
    { pattern: /\bwin(ner|ning)?\b/i, label: 'Winner claim' },
    { pattern: /\bcash\b/i,           label: 'Cash mention' },
    { pattern: /\bprize\b/i,          label: 'Prize' },
    { pattern: /\bclaim\b/i,          label: 'Claim now' },
    { pattern: /\bclick\b/i,          label: 'Click bait' },
    { pattern: /\burgent\b/i,         label: 'Urgent' },
    { pattern: /\blimited\b/i,        label: 'Limited time' },
    { pattern: /\bguaranteed?\b/i,    label: 'Guaranteed' },
    { pattern: /\bcongrat/i,          label: 'Congratulations' },
    { pattern: /\bverif/i,            label: 'Verification' },
    { pattern: /\bpassword\b/i,       label: 'Password' },
    { pattern: /\baccount\b/i,        label: 'Account' },
    { pattern: /\$\d+|\d+\$|£\d+/,   label: 'Money amount' },
    { pattern: /\d{3,}[-\s]?\d{3,}/,  label: 'Phone number' },
    { pattern: /http|www\.|\.com/i,   label: 'URL/Link' },
    { pattern: /\bsms\b|\btxt\b/i,    label: 'SMS/TXT' },
    { pattern: /\bstop\b/i,           label: 'STOP keyword' },
];

// ── Tab switching ──
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tabId}-tab`).classList.add('active');
        if (tabId !== 'single') resultSection.style.display = 'none';
        if (tabId !== 'batch')  batchResults.style.display  = 'none';
    });
});

// ── Single Message Prediction ──
predictBtn.addEventListener('click', async () => {
    const message = messageInput.value.trim();
    if (!message) { alert('Please enter a message to analyze'); return; }

    showLoading();

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        if (data.success) {
            displayPrediction(data);
            triggerFlash(data.is_spam);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to server. Make sure the backend is running.');
    } finally {
        hideLoading();
    }
});

function displayPrediction(data) {
    resultSection.style.display = 'block';

    const isSpam = data.is_spam;
    const confidencePercent = (data.confidence * 100).toFixed(1);
    const probabilityPercent = (data.probability * 100).toFixed(1);

    // Verdict banner
    predictionResult.innerHTML = `
        <i class="fas fa-${isSpam ? 'exclamation-triangle' : 'check-circle'}"></i>
        ${data.prediction}
    `;
    predictionResult.className = `prediction-result ${isSpam ? 'spam-background' : 'ham-background'}`;

    // Stats row
    confidencePct.textContent  = `${confidencePercent}%`;
    probabilityPct.textContent = `${probabilityPercent}%`;

    // Confidence bar
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceFill.className   = `progress-fill ${isSpam ? 'danger' : 'success'}`;
    confidenceValue.textContent = `${confidencePercent}%`;

    // Spam signal indicators
    const text = data.original_text || '';
    const matched = SPAM_SIGNALS.filter(s => s.pattern.test(text));

    if (matched.length > 0) {
        indicatorTags.innerHTML = matched.map(s =>
            `<span class="indicator-tag ${isSpam ? '' : 'safe'}">${s.label}</span>`
        ).join('');
        spamIndicators.style.display = 'block';
    } else {
        spamIndicators.style.display = 'none';
    }

    // Cleaned text
    cleanedText.innerHTML = `
        <strong><i class="fas fa-language"></i> Processed Text</strong>
        ${escapeHtml(data.cleaned_text || data.original_text)}
    `;
}

// ── Full-screen background flash ──
function triggerFlash(isSpam) {
    document.body.classList.remove('spam-flash', 'ham-flash');
    void document.body.offsetWidth; // reflow
    document.body.classList.add(isSpam ? 'spam-flash' : 'ham-flash');
}

// ── File Upload ──
fileUploadArea.addEventListener('click', () => fileInput.click());

fileUploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    fileUploadArea.style.borderColor = 'var(--primary-color)';
});

fileUploadArea.addEventListener('dragleave', e => {
    e.preventDefault();
    fileUploadArea.style.borderColor = 'var(--border-bright)';
});

fileUploadArea.addEventListener('drop', e => {
    e.preventDefault();
    fileUploadArea.style.borderColor = 'var(--border-bright)';
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        updateFileButtons();
    }
});

fileInput.addEventListener('change', updateFileButtons);

function updateFileButtons() {
    const hasFile = fileInput.files.length > 0;
    processFileBtn.disabled    = !hasFile;
    regenerateFileBtn.disabled = !hasFile;
    fileUploadArea.querySelector('p').textContent = hasFile
        ? `Selected: ${fileInput.files[0].name}`
        : 'Drag and drop a file here or click to browse';
}

// ── Process File ──
processFileBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) return;
    showLoading();
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
        const response = await fetch(`${API_URL}/predict-file`, { method: 'POST', body: formData });
        const data = await response.json();
        if (data.success) displayBatchResults(data);
        else alert('Error: ' + data.error);
    } catch (e) {
        alert('Failed to process file');
    } finally {
        hideLoading();
    }
});

// ── Generate Report ──
regenerateFileBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) return;
    showLoading();
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
        const response = await fetch(`${API_URL}/regenerate-file`, { method: 'POST', body: formData });
        const data = await response.json();
        if (data.success) {
            const blob = new Blob([data.report], { type: 'text/plain' });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href = url; a.download = 'spam_detection_report.txt'; a.click();
            URL.revokeObjectURL(url);
            alert(`Report generated!\nTotal: ${data.stats.total_messages} | Spam: ${data.stats.spam_count} | Ham: ${data.stats.ham_count}`);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (e) {
        alert('Failed to generate report');
    } finally {
        hideLoading();
    }
});

function displayBatchResults(data) {
    batchResults.style.display = 'block';

    statsGrid.innerHTML = `
        <div class="stat-card">
            <i class="fas fa-envelope"></i>
            <div class="stat-value">${data.stats.total_messages}</div>
            <div class="stat-label">Total</div>
        </div>
        <div class="stat-card">
            <i class="fas fa-exclamation-triangle"></i>
            <div class="stat-value">${data.stats.spam_count}</div>
            <div class="stat-label">Spam</div>
        </div>
        <div class="stat-card">
            <i class="fas fa-check-circle"></i>
            <div class="stat-value">${data.stats.ham_count}</div>
            <div class="stat-label">Ham</div>
        </div>
        <div class="stat-card">
            <i class="fas fa-chart-pie"></i>
            <div class="stat-value">${data.stats.spam_percentage.toFixed(1)}%</div>
            <div class="stat-label">Spam Rate</div>
        </div>
    `;

    let tableHtml = `
        <table class="results-data-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Message</th>
                    <th>Result</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.results.forEach(r => {
        tableHtml += `
            <tr>
                <td>${r.index}</td>
                <td>${escapeHtml(r.text)}</td>
                <td><span class="${r.is_spam ? 'spam-badge' : 'ham-badge'}">${r.prediction}</span></td>
                <td>${(r.confidence * 100).toFixed(1)}%</td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    resultsTable.innerHTML = tableHtml;
}

// ── Helpers ──
function showLoading() { loadingOverlay.style.display = 'flex'; }
function hideLoading()  { loadingOverlay.style.display = 'none'; }

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// ── Flash animations ──
const style = document.createElement('style');
style.textContent = `
    @keyframes spamBlink {
        0%        { background-color: #0a0a0a; }
        12%       { background-color: rgba(239, 68, 68, 0.6); }
        24%       { background-color: #0a0a0a; }
        36%       { background-color: rgba(239, 68, 68, 0.6); }
        48%       { background-color: #0a0a0a; }
        60%       { background-color: rgba(239, 68, 68, 0.6); }
        100%      { background-color: #0a0a0a; }
    }
    @keyframes hamBlink {
        0%        { background-color: #0a0a0a; }
        12%       { background-color: rgba(34, 197, 94, 0.55); }
        24%       { background-color: #0a0a0a; }
        36%       { background-color: rgba(34, 197, 94, 0.55); }
        48%       { background-color: #0a0a0a; }
        60%       { background-color: rgba(34, 197, 94, 0.55); }
        100%      { background-color: #0a0a0a; }
    }
    body.spam-flash { animation: spamBlink 1.8s ease-in-out forwards; }
    body.ham-flash  { animation: hamBlink  1.8s ease-in-out forwards; }
`;
document.head.appendChild(style);
