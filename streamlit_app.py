"""
SMS Spam Detector — Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os
import torch
import pandas as pd
import re
import io

# ── Path setup ──
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(ROOT_DIR, 'src')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
sys.path.insert(0, SRC_DIR)

from config import Config
from model import SpamTransformerWithEmbeddings
from preprocess import DataPreprocessor
from predict import SpamPredictor

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

    .main-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        background: linear-gradient(135deg, #1a1f35 0%, #0f1117 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2a2f45;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8ef7, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p { color: #8892b0; font-size: 1.05rem; margin: 0; }

    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f35;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #2a2f45;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
        color: white !important;
    }

    .spam-card {
        background: linear-gradient(135deg, #2d1515, #1a0a0a);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
    }
    .ham-card {
        background: linear-gradient(135deg, #0f2d1a, #0a1a0f);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
    }

    .verdict-spam {
        font-size: 2.2rem;
        font-weight: 900;
        color: #ef4444;
        text-align: center;
        letter-spacing: 4px;
    }
    .verdict-ham {
        font-size: 2.2rem;
        font-weight: 900;
        color: #22c55e;
        text-align: center;
        letter-spacing: 4px;
    }

    .metric-box {
        background: #0f1117;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a2f45;
    }
    .metric-box .value { font-size: 1.8rem; font-weight: 800; color: #4f8ef7; }
    .metric-box .label { font-size: 0.8rem; color: #8892b0; margin-top: 4px; }

    .signal-tag {
        display: inline-block;
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }
    .signal-tag-safe {
        display: inline-block;
        background: rgba(34,197,94,0.1);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }

    .stat-card {
        background: #1a1f35;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2a2f45;
    }
    .stat-card .num { font-size: 2rem; font-weight: 800; }
    .stat-card .lbl { font-size: 0.8rem; color: #8892b0; }

    .progress-wrap {
        background: #0f1117;
        border-radius: 8px;
        height: 12px;
        margin: 8px 0;
        overflow: hidden;
    }
    .progress-fill-spam {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    .progress-fill-ham {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #22c55e, #16a34a);
    }

    .stTextArea textarea {
        background: #1a1f35 !important;
        border: 1px solid #2a2f45 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }

    .perf-card {
        background: #1a1f35;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #2a2f45;
    }
    .perf-card .perf-val {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8ef7, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .perf-card .perf-lbl { color: #8892b0; font-size: 0.9rem; margin-top: 4px; }

    .info-box {
        background: rgba(79,142,247,0.08);
        border: 1px solid rgba(79,142,247,0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #8892b0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SPAM SIGNALS
# ════════════════════════════════════════════════════════════
SPAM_SIGNALS = [
    (r'\bfree\b',           'FREE offer'),
    (r'\bwin(ner|ning)?\b', 'Winner claim'),
    (r'\bcash\b',           'Cash mention'),
    (r'\bprize\b',          'Prize'),
    (r'\bclaim\b',          'Claim now'),
    (r'\bclick\b',          'Click bait'),
    (r'\burgent\b',         'URGENT'),
    (r'\blimited\b',        'Limited time'),
    (r'\bguaranteed?\b',    'Guaranteed'),
    (r'\bcongrat',          'Congratulations'),
    (r'\bverif',            'Verification'),
    (r'\bpassword\b',       'Password'),
    (r'\baccount\b',        'Account'),
    (r'\$\d+|\d+\$|£\d+',  'Money amount'),
    (r'http|www\.|\.com',   'URL/Link'),
    (r'\bsms\b|\btxt\b',    'SMS/TXT'),
    (r'\bstop\b',           'STOP keyword'),
]

def detect_signals(text):
    return [label for pattern, label in SPAM_SIGNALS
            if re.search(pattern, text, re.IGNORECASE)]

# ════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    device = Config.get_device()
    preprocessor = DataPreprocessor(max_length=Config.MAX_SEQUENCE_LENGTH)
    vocab_path = os.path.join(MODEL_DIR, 'vocab.pkl')
    model_path = os.path.join(MODEL_DIR, 'spam_transformer.pth')

    if not os.path.exists(vocab_path):
        return None, "vocab.pkl not found. Please train the model first."
    if not os.path.exists(model_path):
        return None, "spam_transformer.pth not found. Please train the model first."

    preprocessor.load_vocabulary(vocab_path)
    model = SpamTransformerWithEmbeddings(
        vocab_size=len(preprocessor.word2idx),
        d_model=Config.MODEL_SIZE,
        num_heads=Config.ATTENTION_HEADS,
        num_encoder_layers=Config.ENCODER_LAYERS,
        num_decoder_layers=Config.DECODER_LAYERS,
        d_ff=Config.FEEDFORWARD_SIZE,
        dropout=Config.DROPOUT_RATE,
        memory_length=Config.MEMORY_LENGTH
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = float(checkpoint.get('best_threshold', 0.5))
    return SpamPredictor(model, preprocessor, device, Config, threshold=threshold), None

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ SMS Spam Detector</h1>
    <p>Powered by Transformer Architecture · Deep Learning · NLP</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading AI model..."):
    predictor, error = load_model()

if error:
    st.error(f"⚠️ {error}")
    st.stop()

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📩  Single Message", "📂  Batch Processing", "📊  Model Statistics"])

# ────────────────────────────────────────────────────────────
# TAB 1 — SINGLE MESSAGE
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter SMS Message")
    message = st.text_area(
        label="",
        placeholder='Type or paste your SMS message here...\nExample: "Congratulations! You\'ve won a FREE $1000 gift card. Click here to claim now!"',
        height=140,
        key="single_msg"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔍  Detect Spam", key="detect_btn")

    if predict_btn:
        if not message.strip():
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = predictor.predict_single(message)

            is_spam    = result['is_spam']
            confidence = result['confidence']
            probability= result['probability']
            signals    = detect_signals(message)

            card_class   = "spam-card"   if is_spam else "ham-card"
            verdict_cls  = "verdict-spam" if is_spam else "verdict-ham"
            icon         = "🚨" if is_spam else "✅"
            label        = "SPAM" if is_spam else "HAM"
            fill_cls     = "progress-fill-spam" if is_spam else "progress-fill-ham"

            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f'<div class="{verdict_cls}">{icon} {label}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-box"><div class="value">{confidence*100:.1f}%</div><div class="label">Confidence</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-box"><div class="value">{probability*100:.1f}%</div><div class="label">Spam Probability</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="color:#8892b0;font-size:0.85rem;margin-bottom:4px;">Confidence Score</div>
            <div class="progress-wrap">
                <div class="{fill_cls}" style="width:{confidence*100:.1f}%"></div>
            </div>""", unsafe_allow_html=True)

            if signals:
                st.markdown("<br>", unsafe_allow_html=True)
                tag_cls  = "signal-tag" if is_spam else "signal-tag-safe"
                tags_html = " ".join([f'<span class="{tag_cls}">{s}</span>' for s in signals])
                st.markdown(f'<div style="color:#8892b0;font-size:0.85rem;margin-bottom:6px;">🔍 Detected Signals</div><div>{tags_html}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="info-box"><strong>📝 Processed Text:</strong><br>{result["cleaned_text"]}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 2 — BATCH PROCESSING
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Upload File for Batch Analysis")
    st.markdown('<div class="info-box">Upload a <strong>.txt</strong> file with one SMS message per line.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"], key="batch_file")

    if uploaded_file:
        c1, c2 = st.columns(2)
        with c1:
            analyze_btn = st.button("📊  Analyze File", key="analyze_btn")
        with c2:
            report_btn = st.button("📥  Generate Report", key="report_btn")

        if analyze_btn or report_btn:
            with st.spinner("Processing messages..."):
                content = uploaded_file.read().decode("utf-8", errors="replace")
                lines   = [l.strip() for l in content.splitlines() if l.strip()]
                results = [predictor.predict_single(line) for line in lines]
                df      = pd.DataFrame(results)

            spam_count = int(df['is_spam'].sum())
            ham_count  = len(df) - spam_count
            total      = len(df)
            spam_pct   = spam_count / total * 100 if total else 0

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="num" style="color:#4f8ef7">{total}</div><div class="lbl">Total</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="num" style="color:#ef4444">{spam_count}</div><div class="lbl">Spam</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="num" style="color:#22c55e">{ham_count}</div><div class="lbl">Ham</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-card"><div class="num" style="color:#f59e0b">{spam_pct:.1f}%</div><div class="lbl">Spam Rate</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📋 Detailed Results")
            display_df = pd.DataFrame({
                "#":           range(1, len(df) + 1),
                "Message":     df['text'].apply(lambda x: x[:80] + ('...' if len(x) > 80 else '')),
                "Prediction":  df['prediction'],
                "Confidence":  df['confidence'].apply(lambda x: f"{x*100:.1f}%"),
                "Probability": df['probability'].apply(lambda x: f"{x*100:.1f}%"),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if report_btn:
                lines_out = [
                    "=" * 80,
                    "SMS SPAM DETECTION REPORT",
                    "=" * 80,
                    f"\nTotal   : {total}",
                    f"SPAM    : {spam_count} ({spam_pct:.2f}%)",
                    f"HAM     : {ham_count} ({100-spam_pct:.2f}%)",
                    "\nDETAILED RESULTS:",
                    "-" * 80,
                ]
                for i, row in df.iterrows():
                    lines_out += [
                        f"\nMessage #{i+1}:",
                        f"  Text       : {row['text'][:100]}",
                        f"  Prediction : {row['prediction']}",
                        f"  Confidence : {row['confidence']*100:.1f}%",
                        "-" * 40,
                    ]
                st.download_button(
                    label="⬇️  Download Report",
                    data="\n".join(lines_out),
                    file_name="spam_detection_report.txt",
                    mime="text/plain"
                )

# ────────────────────────────────────────────────────────────
# TAB 3 — MODEL STATISTICS
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Model Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["98.92%", "97.81%", "94.51%", "96.13%"],
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    ):
        with col:
            st.markdown(f'<div class="perf-card"><div class="perf-val">{val}</div><div class="perf-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏗️ Model Architecture")
        st.dataframe(pd.DataFrame([
            ("Input Embedding",     "GloVe 300d + fine-tuning"),
            ("Positional Encoding", "Sinusoidal functions"),
            ("Encoder Layers",      "3× Multi-Head Self-Attention"),
            ("Decoder Layers",      "3× Cross-Attention + Memory"),
            ("Attention Heads",     "8 heads, d_k = 32"),
            ("Model Dimension",     "d_model = 256"),
            ("Feedforward Size",    "d_ff = 512"),
            ("Classification Head", "256 → 128 → 64 → 1"),
            ("Output Activation",   "Sigmoid → probability"),
            ("Max Sequence Length", "150 tokens"),
        ], columns=["Component", "Details"]), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### ⚙️ Training Configuration")
        st.dataframe(pd.DataFrame([
            ("Optimizer",          "AdamW (lr=0.0005)"),
            ("Loss Function",      "Binary Cross Entropy"),
            ("Batch Size",         "32"),
            ("Max Epochs",         "50"),
            ("Early Stopping",     "Patience = 15"),
            ("Warmup Steps",       "20"),
            ("Dropout Rate",       "0.25"),
            ("Gradient Clipping",  "max norm = 1.0"),
            ("Threshold Tuning",   "Auto on validation set"),
            ("Dataset Split",      "80% / 10% / 10%"),
        ], columns=["Parameter", "Value"]), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Comparison with Baseline Methods")
    st.dataframe(pd.DataFrame([
        ["Naive Bayes",            "97.50%", "96.12%", "88.23%", "92.01%"],
        ["Logistic Regression",    "97.80%", "96.34%", "89.91%", "93.01%"],
        ["Support Vector Machine", "98.10%", "97.01%", "92.01%", "94.44%"],
        ["LSTM",                   "98.40%", "97.20%", "93.10%", "95.10%"],
        ["✨ Proposed Transformer", "98.92%", "97.81%", "94.51%", "96.13%"],
    ], columns=["Method", "Accuracy", "Precision", "Recall", "F1-Score"]),
    use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.85rem;padding:1rem;border-top:1px solid #2a2f45;">
    SMS Spam Detector · CVR College of Engineering ·
    S. Sathyasai [23B81A67A6] · T. Srikar Reddy [23B81A67B3]
</div>
""", unsafe_allow_html=True)
