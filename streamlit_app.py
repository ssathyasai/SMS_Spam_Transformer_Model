"""
SMS Spam Detector — Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import sys, os, torch, re
import pandas as pd

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
# CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #070b14; color: #e2e8f0; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

/* ── HEADER ── */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a0533 50%, #0d1b3e 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem;
    line-height: 1.1;
}
.hero p {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
    font-weight: 500;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 1.8rem;
    flex-wrap: wrap;
}
.hero-stat { text-align: center; }
.hero-stat .hs-val {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-stat .hs-lbl { font-size: 0.72rem; color: #475569; font-weight: 600; letter-spacing: 0.5px; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #1e2533;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #475569;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}

/* ── INPUT AREA ── */
.input-card {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.input-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.6rem;
}
.stTextArea textarea {
    background: #070b14 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(5,150,105,0.3) !important;
}

/* ── RESULT CARDS ── */
.result-wrap {
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 20px;
    pointer-events: none;
}
.result-spam {
    background: linear-gradient(135deg, #1c0808 0%, #0f0505 100%);
    border: 2px solid #dc2626;
    box-shadow: 0 0 40px rgba(220,38,38,0.15), inset 0 1px 0 rgba(220,38,38,0.1);
}
.result-ham {
    background: linear-gradient(135deg, #071810 0%, #040e08 100%);
    border: 2px solid #16a34a;
    box-shadow: 0 0 40px rgba(22,163,74,0.15), inset 0 1px 0 rgba(22,163,74,0.1);
}
.verdict {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 6px;
    margin-bottom: 0.3rem;
    line-height: 1;
}
.verdict-sub {
    text-align: center;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
}
.verdict-spam-color { color: #f87171; }
.verdict-ham-color  { color: #4ade80; }

/* ── METRIC BOXES ── */
.mbox {
    background: rgba(0,0,0,0.4);
    border-radius: 14px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
}
.mbox .mv { font-size: 1.9rem; font-weight: 800; }
.mbox .ml { font-size: 0.72rem; color: #64748b; margin-top: 3px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── PROGRESS BAR ── */
.pbar-wrap {
    background: rgba(0,0,0,0.4);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 6px 0 4px;
}
.pbar-spam { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #dc2626, #ef4444, #f87171); }
.pbar-ham  { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #16a34a, #22c55e, #4ade80); }
.pbar-label { font-size: 0.75rem; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── SIGNAL TAGS ── */
.tags-section { margin-top: 1rem; }
.tags-title { font-size: 0.75rem; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.5rem; }
.stag {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 3px;
    letter-spacing: 0.3px;
}
.stag-spam { background: rgba(220,38,38,0.12); color: #f87171; border: 1px solid rgba(220,38,38,0.25); }
.stag-ham  { background: rgba(22,163,74,0.1);  color: #4ade80; border: 1px solid rgba(22,163,74,0.2); }

/* ── CLEANED TEXT ── */
.cleaned-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.5;
}
.cleaned-box strong { color: #94a3b8; display: block; margin-bottom: 4px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── STAT CARDS ── */
.scard {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.scard::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 16px 16px;
}
.scard-blue::after  { background: linear-gradient(90deg, #6366f1, #818cf8); }
.scard-red::after   { background: linear-gradient(90deg, #dc2626, #ef4444); }
.scard-green::after { background: linear-gradient(90deg, #16a34a, #22c55e); }
.scard-amber::after { background: linear-gradient(90deg, #d97706, #f59e0b); }
.scard .snum { font-size: 2.4rem; font-weight: 900; line-height: 1; }
.scard .slbl { font-size: 0.72rem; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

/* ── PERF CARDS ── */
.pcard {
    background: linear-gradient(135deg, #0d1b3e, #1a0533);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.pcard .pval {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pcard .plbl { font-size: 0.78rem; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

/* ── INFO BOX ── */
.ibox {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    color: #64748b;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── SECTION TITLES ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e2533, transparent);
}

/* ── FILE UPLOADER ── */
.stFileUploader > div {
    background: #0d1117 !important;
    border: 2px dashed #1e2533 !important;
    border-radius: 16px !important;
}

/* ── DATAFRAME ── */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #1e2533; }

/* ── HIDE DEFAULTS ── */
#MainMenu, footer, header { visibility: hidden; }
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
    (r'http|www\.|\.com',   'URL / Link'),
    (r'\bsms\b|\btxt\b',    'SMS / TXT'),
    (r'\bstop\b',           'STOP keyword'),
]

def detect_signals(text):
    return [lbl for pat, lbl in SPAM_SIGNALS if re.search(pat, text, re.IGNORECASE)]

# ════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    device      = Config.get_device()
    preprocessor = DataPreprocessor(max_length=Config.MAX_SEQUENCE_LENGTH)
    vocab_path  = os.path.join(MODEL_DIR, 'vocab.pkl')
    model_path  = os.path.join(MODEL_DIR, 'spam_transformer.pth')

    if not os.path.exists(vocab_path):
        return None, "vocab.pkl not found — please train the model first."
    if not os.path.exists(model_path):
        return None, "spam_transformer.pth not found — please train the model first."

    preprocessor.load_vocabulary(vocab_path)
    model = SpamTransformerWithEmbeddings(
        vocab_size          = len(preprocessor.word2idx),
        d_model             = Config.MODEL_SIZE,
        num_heads           = Config.ATTENTION_HEADS,
        num_encoder_layers  = Config.ENCODER_LAYERS,
        num_decoder_layers  = Config.DECODER_LAYERS,
        d_ff                = Config.FEEDFORWARD_SIZE,
        dropout             = Config.DROPOUT_RATE,
        memory_length       = Config.MEMORY_LENGTH,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    threshold = float(ckpt.get('best_threshold', 0.5))
    return SpamPredictor(model, preprocessor, device, Config, threshold=threshold), None

# ════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ AI-Powered Protection</div>
    <h1>SMS Spam Detector</h1>
    <p>Transformer Architecture · Deep Learning · Natural Language Processing</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hs-val">98.92%</div><div class="hs-lbl">Accuracy</div></div>
        <div class="hero-stat"><div class="hs-val">97.81%</div><div class="hs-lbl">Precision</div></div>
        <div class="hero-stat"><div class="hs-val">96.13%</div><div class="hs-lbl">F1-Score</div></div>
        <div class="hero-stat"><div class="hs-val">5,572</div><div class="hs-lbl">Trained On</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ──
with st.spinner("⚡ Loading AI model..."):
    predictor, error = load_model()

if error:
    st.error(f"⚠️ {error}")
    st.stop()

st.success("✅ Model loaded and ready!", icon="🤖")

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📩   Single Message", "📂   Batch Processing", "📊   Model Statistics"])

# ────────────────────────────────────────────────────────────
# TAB 1 — SINGLE MESSAGE
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">✏️ Enter SMS Message</div>', unsafe_allow_html=True)

    message = st.text_area(
        label="",
        placeholder='Paste your SMS message here...\nExample: "Congratulations! You\'ve won a FREE $1000 gift card. Click here to claim now!"',
        height=130,
        key="single_msg"
    )

    c_l, c_m, c_r = st.columns([1, 2, 1])
    with c_m:
        predict_btn = st.button("🔍   Analyze Message", key="detect_btn")

    if predict_btn:
        if not message.strip():
            st.warning("⚠️ Please enter a message first.")
        else:
            with st.spinner("🧠 Analyzing..."):
                result  = predictor.predict_single(message)
            is_spam     = result['is_spam']
            confidence  = result['confidence']
            probability = result['probability']
            signals     = detect_signals(message)

            wrap_cls    = "result-spam" if is_spam else "result-ham"
            vcolor      = "verdict-spam-color" if is_spam else "verdict-ham-color"
            icon        = "🚨" if is_spam else "✅"
            label       = "SPAM" if is_spam else "HAM"
            vsub        = "This message appears to be spam" if is_spam else "This message appears to be safe"
            bar_cls     = "pbar-spam" if is_spam else "pbar-ham"
            mv_color    = "#f87171" if is_spam else "#4ade80"
            tag_cls     = "stag-spam" if is_spam else "stag-ham"

            st.markdown(f'<div class="result-wrap {wrap_cls}">', unsafe_allow_html=True)

            # Verdict
            st.markdown(f"""
            <div class="verdict {vcolor}">{icon} {label}</div>
            <div class="verdict-sub {vcolor}">{vsub}</div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="mbox"><div class="mv" style="color:{mv_color}">{confidence*100:.1f}%</div><div class="ml">Confidence</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="mbox"><div class="mv" style="color:{mv_color}">{probability*100:.1f}%</div><div class="ml">Spam Probability</div></div>', unsafe_allow_html=True)
            with m3:
                risk = "High" if probability > 0.8 else "Medium" if probability > 0.5 else "Low"
                risk_color = "#f87171" if probability > 0.8 else "#fbbf24" if probability > 0.5 else "#4ade80"
                st.markdown(f'<div class="mbox"><div class="mv" style="color:{risk_color}">{risk}</div><div class="ml">Risk Level</div></div>', unsafe_allow_html=True)

            # Progress bar
            st.markdown(f"""
            <div style="margin-top:1.2rem;">
                <div class="pbar-label">Confidence Score — {confidence*100:.1f}%</div>
                <div class="pbar-wrap">
                    <div class="{bar_cls}" style="width:{confidence*100:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Signals
            if signals:
                tags_html = "".join([f'<span class="stag {tag_cls}">{s}</span>' for s in signals])
                st.markdown(f"""
                <div class="tags-section">
                    <div class="tags-title">🔍 Detected Spam Signals</div>
                    <div>{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

            # Cleaned text
            st.markdown(f"""
            <div class="cleaned-box">
                <strong>📝 Processed Text</strong>
                {result['cleaned_text']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 2 — BATCH PROCESSING
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📂 Upload File for Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="ibox">Upload a <strong>.txt</strong> file with one SMS message per line. The system will classify every message and show a full breakdown.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["txt"], key="batch_file")

    if uploaded_file:
        b1, b2 = st.columns(2)
        with b1:
            analyze_btn = st.button("📊   Analyze File", key="analyze_btn")
        with b2:
            report_btn  = st.button("📥   Generate Report", key="report_btn")

        if analyze_btn or report_btn:
            with st.spinner("🔄 Processing all messages..."):
                content = uploaded_file.read().decode("utf-8", errors="replace")
                lines   = [l.strip() for l in content.splitlines() if l.strip()]
                results = [predictor.predict_single(l) for l in lines]
                df      = pd.DataFrame(results)

            spam_count = int(df['is_spam'].sum())
            ham_count  = len(df) - spam_count
            total      = len(df)
            spam_pct   = spam_count / total * 100 if total else 0

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Summary</div>', unsafe_allow_html=True)

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f'<div class="scard scard-blue"><div class="snum" style="color:#818cf8">{total}</div><div class="slbl">Total Messages</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="scard scard-red"><div class="snum" style="color:#f87171">{spam_count}</div><div class="slbl">Spam Detected</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="scard scard-green"><div class="snum" style="color:#4ade80">{ham_count}</div><div class="slbl">Safe Messages</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="scard scard-amber"><div class="snum" style="color:#fbbf24">{spam_pct:.1f}%</div><div class="slbl">Spam Rate</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Detailed Results</div>', unsafe_allow_html=True)

            display_df = pd.DataFrame({
                "#":           range(1, len(df) + 1),
                "Message":     df['text'].apply(lambda x: x[:90] + ('...' if len(x) > 90 else '')),
                "Prediction":  df['prediction'],
                "Confidence":  df['confidence'].apply(lambda x: f"{x*100:.1f}%"),
                "Probability": df['probability'].apply(lambda x: f"{x*100:.1f}%"),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if report_btn:
                out = [
                    "=" * 80,
                    "         SMS SPAM DETECTION REPORT",
                    "=" * 80,
                    "",
                    f"  Total Messages  : {total}",
                    f"  SPAM Detected   : {spam_count}  ({spam_pct:.2f}%)",
                    f"  Safe Messages   : {ham_count}  ({100-spam_pct:.2f}%)",
                    "",
                    "DETAILED RESULTS:",
                    "-" * 80,
                ]
                for i, row in df.iterrows():
                    out += [
                        f"\n  Message #{i+1}",
                        f"  Text       : {row['text'][:100]}",
                        f"  Prediction : {row['prediction']}",
                        f"  Confidence : {row['confidence']*100:.1f}%",
                        "  " + "-" * 38,
                    ]
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="⬇️   Download Full Report",
                    data="\n".join(out),
                    file_name="spam_detection_report.txt",
                    mime="text/plain"
                )

# ────────────────────────────────────────────────────────────
# TAB 3 — MODEL STATISTICS
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🏆 Performance Metrics</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    for col, val, lbl in zip([p1,p2,p3,p4],
                              ["98.92%","97.81%","94.51%","96.13%"],
                              ["Accuracy","Precision","Recall","F1-Score"]):
        with col:
            st.markdown(f'<div class="pcard"><div class="pval">{val}</div><div class="plbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🏗️ Model Architecture</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ("Input Embedding",     "GloVe 300d + fine-tuning"),
            ("Positional Encoding", "Sinusoidal functions"),
            ("Encoder Layers",      "3× Multi-Head Self-Attention"),
            ("Decoder Layers",      "3× Cross-Attention + Memory"),
            ("Attention Heads",     "8 heads · d_k = 32"),
            ("Model Dimension",     "d_model = 256"),
            ("Feedforward Size",    "d_ff = 512"),
            ("Classification Head", "256 → 128 → 64 → 1"),
            ("Output Activation",   "Sigmoid → probability"),
            ("Max Sequence Length", "150 tokens"),
        ], columns=["Component", "Details"]), use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">⚙️ Training Configuration</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ("Optimizer",         "AdamW  (lr = 0.0005)"),
            ("Loss Function",     "Binary Cross Entropy"),
            ("Batch Size",        "32"),
            ("Max Epochs",        "50"),
            ("Early Stopping",    "Patience = 15"),
            ("Warmup Steps",      "20"),
            ("Dropout Rate",      "0.25"),
            ("Gradient Clipping", "max norm = 1.0"),
            ("Threshold Tuning",  "Auto on validation set"),
            ("Dataset Split",     "80% / 10% / 10%"),
        ], columns=["Parameter", "Value"]), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Comparison with Baseline Methods</div>', unsafe_allow_html=True)
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
<div style="text-align:center;padding:1.5rem;border-top:1px solid #1e2533;color:#334155;font-size:0.82rem;font-weight:500;">
    🛡️ SMS Spam Detector &nbsp;·&nbsp; CVR College of Engineering &nbsp;·&nbsp;
    S. Sathyasai [23B81A67A6] &nbsp;·&nbsp; T. Srikar Reddy [23B81A67B3]
</div>
""", unsafe_allow_html=True)
