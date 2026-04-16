import streamlit as st
import joblib
import numpy as np
import os
import time

st.set_page_config(page_title="TrueTextAI", page_icon="🔍", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], section.main {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #eee8f8 !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer { visibility: hidden !important; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 5rem !important; max-width: 660px !important; }

[data-testid="stTextArea"] label { font-size:13px !important; font-weight:600 !important; color:#3a2d6b !important; }
[data-testid="stTextArea"] textarea {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important; line-height: 1.75 !important;
    background: #ffffff !important; border: 2px solid #c9b8f0 !important;
    border-radius: 14px !important; color: #1a1035 !important; padding: 16px !important;
}
[data-testid="stTextArea"] textarea:focus { border-color:#7b4fd4 !important; box-shadow:0 0 0 4px rgba(123,79,212,0.15) !important; }
[data-testid="stTextArea"] textarea::placeholder { color:#a898cc !important; font-size:14px !important; }

[data-testid="stFileUploader"] section { background:#ffffff !important; border:2px dashed #c9b8f0 !important; border-radius:14px !important; }
[data-testid="stFileUploader"] label { font-size:13px !important; font-weight:600 !important; color:#3a2d6b !important; }

[data-testid="stTabs"] button { font-family:'Space Grotesk',sans-serif !important; font-size:14px !important; font-weight:500 !important; color:#7b6ea8 !important; padding:10px 20px !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#7b4fd4 !important; font-weight:700 !important; border-bottom:3px solid #7b4fd4 !important; }

[data-testid="stButton"] > button[kind="primary"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 17px !important; font-weight: 700 !important;
    background: #7b4fd4 !important; color: #ffffff !important;
    border: none !important; border-radius: 14px !important;
    padding: 16px 32px !important; height: auto !important;
    width: 100% !important; letter-spacing: 0.3px !important;
    box-shadow: 0 4px 18px rgba(123,79,212,0.4) !important;
    transition: all 0.2s ease !important; margin-top: 4px !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { background:#6338b8 !important; box-shadow:0 6px 24px rgba(123,79,212,0.55) !important; transform:translateY(-2px) !important; }
[data-testid="stButton"] > button[kind="primary"]:disabled { background:#c9b8f0 !important; color:#f0ecff !important; box-shadow:none !important; transform:none !important; }

[data-testid="stButton"] > button:not([kind="primary"]) {
    font-family:'Space Grotesk',sans-serif !important; font-size:13px !important; font-weight:500 !important;
    background:#ffffff !important; color:#7b6ea8 !important;
    border:1.5px solid #c9b8f0 !important; border-radius:8px !important; padding:6px 16px !important; height:auto !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover { color:#c04040 !important; border-color:#e08080 !important; background:#fff5f5 !important; }

[data-testid="stCaptionContainer"] p { font-family:'Space Grotesk',sans-serif !important; color:#8878b8 !important; font-size:12px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists("truetextai_model.pkl"):
        return None
    return joblib.load("truetextai_model.pkl")

model = load_model()

def predict(text):
    score     = model.decision_function([text])[0]
    ai_prob   = 1 / (1 + np.exp(-score))
    human_pct = round((1 - ai_prob) * 100)
    ai_pct    = round(ai_prob * 100)
    verdict   = "AI-Generated" if model.predict([text])[0] == 1 else "Human"
    top       = max(human_pct, ai_pct)
    confidence = "Very High" if top >= 90 else "High" if top >= 75 else "Moderate" if top >= 60 else "Low"
    return dict(verdict=verdict, human_pct=human_pct, ai_pct=ai_pct, confidence=confidence)


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:10px 0 32px;">
    <div style="display:inline-flex; align-items:center; gap:14px;
                background:#7b4fd4; border-radius:20px; padding:16px 32px;
                box-shadow:0 8px 28px rgba(123,79,212,0.4);">
        <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
            <circle cx="15" cy="15" r="11" stroke="white" stroke-width="2.5"/>
            <path d="M10 15h10M15 10v10" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
        <span style="font-family:'Space Grotesk',sans-serif; font-size:28px;
                     font-weight:800; color:#ffffff; letter-spacing:-0.5px;">
            TrueTextAI
        </span>
    </div>
    <p style="font-family:'Space Grotesk',sans-serif; font-size:15px;
              color:#5a4a8a; margin-top:16px; font-weight:400; line-height:1.6;">
        Instantly detect whether text was written by a <strong style="color:#3a2d6b;">Human</strong>
        or generated by <strong style="color:#3a2d6b;">AI</strong>
    </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown("""
    <div style="background:#fff3cd; border:2px solid #ffc107; border-radius:14px;
                padding:16px 20px; color:#856404; font-family:'Space Grotesk',sans-serif; font-size:14px;">
        ⚠️ <strong>Model not found.</strong> Run all cells in
        <code style="background:#ffe8a0; padding:2px 6px; border-radius:4px;">TrueTextAI.ipynb</code>
        first to generate <code style="background:#ffe8a0; padding:2px 6px; border-radius:4px;">truetextai_model.pkl</code>,
        then restart this app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════
# INPUT CARD
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#ffffff; border-radius:20px; padding:24px 28px 8px;
            border:1.5px solid #ddd0f5; box-shadow:0 4px 20px rgba(123,79,212,0.08);">
    <div style="font-family:'Space Grotesk',sans-serif; font-size:12px; font-weight:700;
                color:#7b4fd4; letter-spacing:2px; text-transform:uppercase; margin-bottom:16px;">
        Enter Text to Analyze
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  📝  Paste Text  ", "  📄  Upload .txt  "])

input_text = ""

with tab1:
    typed = st.text_area(
        "Your text",
        placeholder="Paste your text here… (minimum 50 characters required to analyze)",
        height=210,
        label_visibility="collapsed",
    )
    if typed.strip():
        input_text = typed.strip()

with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"], label_visibility="collapsed")
    if uploaded:
        content = uploaded.read().decode("utf-8", errors="ignore").strip()
        st.caption(f"✓  {uploaded.name}  ·  {len(content)} characters loaded")
        st.text_area("Preview", value=content[:500] + ("…" if len(content) > 500 else ""),
                     height=120, disabled=True, label_visibility="collapsed")
        input_text = content

char_count = len(input_text)

# Status message
if char_count == 0:
    st.caption("Type or paste text above — minimum 50 characters needed")
elif char_count < 50:
    st.markdown(
        f"<p style='font-family:Space Grotesk,sans-serif;font-size:13px;color:#c04040;margin:6px 0 4px;'>"
        f"⚠  {char_count} characters — need {50 - char_count} more</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<p style='font-family:Space Grotesk,sans-serif;font-size:13px;color:#2e7d32;margin:6px 0 4px;'>"
        f"✓  {char_count} characters — ready to analyze</p>",
        unsafe_allow_html=True
    )

# ── ANALYZE BUTTON — big and unmissable ──────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
analyze = st.button(
    "🔍   Analyze Text",
    type="primary",
    use_container_width=True,
    disabled=char_count < 50,
)
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════════════════════
if analyze and input_text:
    with st.spinner("Analyzing your text…"):
        time.sleep(0.5)
        r = predict(input_text)

    is_human = r["verdict"] == "Human"
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Verdict banner
    if is_human:
        bg, border, text, dot = "#edfaf0", "#6bcb8b", "#1a5c30", "#28a745"
        icon, lbl = "✅", "Human Written"
        desc = "This text shows patterns consistent with natural human writing."
    else:
        bg, border, text, dot = "#fdf0f0", "#f0a0a0", "#7a1a1a", "#dc3545"
        icon, lbl = "🤖", "AI Generated"
        desc = "This text shows patterns consistent with AI-generated content."

    st.markdown(f"""
    <div style="background:{bg}; border:2.5px solid {border}; border-radius:18px;
                padding:22px 26px; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:{dot}; flex-shrink:0;"></div>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:24px;
                         font-weight:800; color:{text}; letter-spacing:-0.3px;">
                {icon}&nbsp; {lbl}
            </span>
        </div>
        <p style="font-family:'Space Grotesk',sans-serif; font-size:14px;
                  color:{text}; opacity:0.8; margin:0; font-weight:400; padding-left:26px;">
            {desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Three score chips
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:16px;">
        <div style="background:#ffffff; border:2.5px solid #6bcb8b; border-radius:16px; padding:20px 12px; text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:34px; font-weight:800; color:#1a5c30; line-height:1;">{r['human_pct']}%</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:11px; font-weight:700; color:#4a9e6a; text-transform:uppercase; letter-spacing:1px; margin-top:6px;">Human</div>
        </div>
        <div style="background:#ffffff; border:2.5px solid #f0a0a0; border-radius:16px; padding:20px 12px; text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:34px; font-weight:800; color:#7a1a1a; line-height:1;">{r['ai_pct']}%</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:11px; font-weight:700; color:#b05050; text-transform:uppercase; letter-spacing:1px; margin-top:6px;">AI Generated</div>
        </div>
        <div style="background:#ffffff; border:2.5px solid #c9b8f0; border-radius:16px; padding:20px 12px; text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:800; color:#3c3489; line-height:1.2;">{r['confidence']}</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:11px; font-weight:700; color:#7b6ea8; text-transform:uppercase; letter-spacing:1px; margin-top:6px;">Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown(f"""
    <div style="background:#ffffff; border:1.5px solid #ddd0f5; border-radius:16px; padding:22px 24px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:12px; font-weight:700;
                    color:#7b4fd4; text-transform:uppercase; letter-spacing:2px; margin-bottom:18px;">
            Probability Breakdown
        </div>
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:600; color:#1a1035;">Human Written</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:#1a5c30;">{r['human_pct']}%</span>
            </div>
            <div style="height:12px; background:#e8f5e9; border-radius:100px; overflow:hidden;">
                <div style="height:100%; width:{r['human_pct']}%; border-radius:100px; background:linear-gradient(90deg,#56c47a,#28a745);"></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:600; color:#1a1035;">AI Generated</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:#7a1a1a;">{r['ai_pct']}%</span>
            </div>
            <div style="height:12px; background:#fdecea; border-radius:100px; overflow:hidden;">
                <div style="height:100%; width:{r['ai_pct']}%; border-radius:100px; background:linear-gradient(90deg,#f07070,#dc3545);"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Save history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "snippet": input_text[:55] + "…",
        "verdict": r["verdict"],
        "human":   r["human_pct"],
        "ai":      r["ai_pct"],
        "conf":    r["confidence"],
    })
    st.session_state.history = st.session_state.history[:15]


# ══════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════
if st.session_state.get("history"):
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_h, col_c = st.columns([5, 1])
    with col_h:
        st.markdown("""
        <div style="font-family:'Space Grotesk',sans-serif; font-size:12px; font-weight:700;
                    color:#7b4fd4; text-transform:uppercase; letter-spacing:2px; padding:6px 0 4px;">
            Recent Analyses
        </div>""", unsafe_allow_html=True)
    with col_c:
        if st.button("Clear"):
            st.session_state.history = []
            st.rerun()

    st.markdown("<div style='background:#ffffff; border:1.5px solid #ddd0f5; border-radius:16px; padding:8px 20px;'>", unsafe_allow_html=True)
    for item in st.session_state.history:
        is_h    = item["verdict"] == "Human"
        dot_col = "#28a745" if is_h else "#dc3545"
        lbl_col = "#1a5c30" if is_h else "#7a1a1a"
        lbl_bg  = "#edfaf0" if is_h else "#fdf0f0"
        lbl_bd  = "#6bcb8b" if is_h else "#f0a0a0"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; padding:11px 0;
                    border-bottom:1px solid #f0eafc;">
            <div style="width:8px; height:8px; border-radius:50%; background:{dot_col}; flex-shrink:0;"></div>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#3a2d6b;
                         flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                {item['snippet']}
            </span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:11px; font-weight:700;
                         background:{lbl_bg}; color:{lbl_col}; border:1.5px solid {lbl_bd};
                         border-radius:20px; padding:3px 12px; flex-shrink:0; white-space:nowrap;">
                {item['verdict']}
            </span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:11px; color:#8878b8;
                         flex-shrink:0; white-space:nowrap;">
                H:{item['human']}% · A:{item['ai']}%
            </span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)