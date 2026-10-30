"""
Page 6: Responsible AI & Ethics Transparency.
Ethical governance, model limitations, privacy safeguards, and clinical disclaimer.
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STYLES_PATH

st.set_page_config(page_title="Responsible AI – AI Learning Assistant", page_icon="🛡️", layout="wide")

if os.path.exists(STYLES_PATH):
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 24px;">
    <span class="feature-pill">Ethical AI Governance</span>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">Responsible AI, Privacy & Model Boundaries</h1>
    <p style="color: #94A3B8; margin-top: 6px; font-size: 1rem;">
        Principles, architectural transparency, and safety protocols governing the AI Learning Assistant.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Critical Mental Health & Scope Disclaimer
st.markdown("""
<div class="disclaimer-box" style="padding: 20px; font-size: 0.95rem; border-left: 6px solid #EF4444; background: rgba(239, 68, 68, 0.1); color: #FCA5A5;">
    <h3 style="margin: 0 0 8px 0; color: #F87171;">⚠️ Explicit Non-Clinical Disclaimer</h3>
    <b>The AI Learning Assistant is designed exclusively for educational and academic pedagogical support.</b><br>
    This platform detects situational sentiment and academic emotion expressed through written text descriptions of coursework difficulties 
    (e.g., encountering a bug in code or struggling with a calculus proof). 
    <br><br>
    <b>It DOES NOT and CANNOT diagnose, evaluate, or treat mental health conditions, depression, anxiety disorders, or psychological states.</b> 
    Students experiencing severe mental distress or crisis should contact professional campus counseling services or certified healthcare providers.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Responsible AI Core Pillars
st.markdown("### 🏛️ The Four Pillars of Responsible AI in Our Platform")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #38BDF8; margin-top: 0;">1. Privacy & Student Data Protection</h4>
        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
            • <b>Zero Personal Identifiable Information (PII):</b> The platform never requests, collects, or logs student names, roll numbers, IP addresses, or student email accounts.<br>
            • <b>Local & Ephemeral Storage:</b> Interaction logs are stored exclusively in user-controlled local CSV files without external tracking trackers or persistent cookies.<br>
            • <b>User-Controlled History:</b> Students and educators have instant one-click capability to export or permanently erase interaction records.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #818CF8; margin-top: 0;">2. Transparency & Confidence Calibration</h4>
        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
            • <b>Distinguishing Confidence from Certainty:</b> High model confidence (e.g. 94%) indicates strong alignment with training patterns, not absolute metaphysical truth about a student's inner state.<br>
            • <b>Explainable Decision Logic:</b> Every prediction is accompanied by a human-readable explanation showing which model contributed to the consensus.<br>
            • <b>Secondary Emotion Recognition:</b> We avoid rigid one-dimensional pigeonholing by explicitly displaying secondary emotional signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #10B981; margin-top: 0;">3. Safe & Constructive Generative AI</h4>
        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
            • <b>Strict Pedagogical Guardrails:</b> Prompts sent to Google Gemini enforce positive, empathetic reinforcement and prevent demotivating or judgmental responses.<br>
            • <b>De-escalation Protocols:</b> When high frustration is detected, the AI prioritizes emotional de-escalation (taking a break, stepping back) before technical deep-dives.<br>
            • <b>Non-Dogmatic Content:</b> Educational recommendations are framed as study strategies to experiment with, never as guaranteed correct answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #F59E0B; margin-top: 0;">4. Documented Technical Limitations</h4>
        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
            • <b>Domain Specificity:</b> Models are trained on academic and STEM learning texts and may misinterpret conversational small talk or idioms from other domains.<br>
            • <b>Sarcasm and Cultural Nuance:</b> Text-based models may struggle with subtle ironic phrasing or culturally specific expressions of frustration.<br>
            • <b>Input Length Boundaries:</b> Extremely short inputs (1-2 words) carry high epistemic uncertainty, reflected in lower confidence scores.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 3. Model Roles in the System Architecture
st.markdown("### 🧩 Roles of Individual Technologies")

st.markdown("""
| Technology Component | Architectural Role | Why It Was Chosen |
| :--- | :--- | :--- |
| **PyTorch BiLSTM** | Recurrent Sequence Classification | Low latency (~12ms CPU), minimal memory footprint, captures forward/backward word sequences. |
| **BERT / Transformer** | Contextual Sequence Representation | Dynamic self-attention captures long-range dependencies and nuanced semantic relationships. |
| **Rule-Based Engine** | Linguistic Post-Processor | Interprets contrastive conjunctions (e.g., 'but', 'however') and extracts compound mixed emotions. |
| **Google Gemini AI** | Generative Pedagogical Mentor | Translates raw emotion labels into empathetic, 6-part actionable study strategies and exercises. |
| **Plotly & Pandas** | Telemetry & Visual Analytics | Real-time interactive charting without page reloads. |
| **Streamlit SaaS UI** | Responsive Application Layer | Accessible across laptops, tablets, and mobile browsers with zero complex client build chains. |
""")

st.markdown("""
<div class="app-footer">
    AI Learning Assistant • Responsible AI & Ethics Protocol
</div>
""", unsafe_allow_html=True)
