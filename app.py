"""
AI Learning Assistant – Emotion Detection & Personalized Learning Support Platform.
Main Streamlit Application Entrypoint.
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.config import STYLES_PATH, EMOTIONS, EMOTION_META
from models.model_loader import load_all_models
from utils.storage import init_history_storage

# Page Configuration
st.set_page_config(
    page_title="AI Learning Assistant – Emotion & Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def inject_custom_css():
    if os.path.exists(STYLES_PATH):
        with open(STYLES_PATH, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_custom_css()
init_history_storage()

# Preload Models once into Session State
if "models_loaded" not in st.session_state:
    with st.spinner("Initializing Deep Learning Models (BiLSTM & BERT)..."):
        load_all_models()
        st.session_state.models_loaded = True

# Initialize Session States
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

if "gemini_enabled" not in st.session_state:
    st.session_state.gemini_enabled = True

# Sidebar Controls & Branding
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <span style="font-size: 2.2rem;">🎓</span>
        <div>
            <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; background: linear-gradient(135deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Learning Assistant</h2>
            <p style="margin: 0; font-size: 0.78rem; color: #94A3B8;">Emotion Detection & Support</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Generative AI Settings")
    st.session_state.gemini_enabled = st.toggle("Enable Gemini Live AI", value=st.session_state.gemini_enabled)
    
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Google Gemini API key. If empty, the intelligent offline pedagogical generator will be used."
    )
    if api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key_input
        st.success("API Key updated for this session!")

    st.markdown("---")
    st.markdown("### 🧠 AI Engine Status")
    st.markdown("""
    <div style="font-size: 0.82rem; line-height: 1.8; color: #94A3B8;">
        <div>🟢 <b>BiLSTM:</b> PyTorch (Trained & Active)</div>
        <div>🟢 <b>BERT:</b> Contextual Transformer (Active)</div>
        <div>🟢 <b>Ensemble:</b> Rule-Enhanced Hybrid</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷️ Target Emotion Classes")
    badges_html = " ".join([
        f"<span class='emotion-badge badge-{e.lower()}' style='font-size: 0.75rem; padding: 3px 8px; margin: 2px;'>{EMOTION_META[e]['emoji']} {e}</span>"
        for e in EMOTIONS
    ])
    st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

# Main app entry renders Home landing content directly if launched via app.py
st.markdown("""
<div class="hero-container">
    <span class="feature-pill">Academic AI Innovation Platform</span>
    <h1 class="hero-title">Understand Emotions.<br>Transform Learning.</h1>
    <p class="hero-subtitle">
        An AI-powered academic learning companion that understands student emotional challenges through deep learning text classification and delivers personalized pedagogical guidance via Google Gemini.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Navigation Action Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#38BDF8;">🎓 Learning Assistant</h3>
        <p style="color:#94A3B8; font-size:0.9rem;">Analyze student statements, detect mixed emotions, and generate 6-part personalized study plans.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Assistant ➔", key="nav_asst", use_container_width=True):
        st.switch_page("pages/2_🎓_Learning_Assistant.py")

with col2:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#818CF8;">📊 Analytics Dashboard</h3>
        <p style="color:#94A3B8; font-size:0.9rem;">Inspect real-time emotional trends, distribution pie charts, and temporal interaction patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Analytics ➔", key="nav_analytics", use_container_width=True):
        st.switch_page("pages/3_📊_Analytics_Dashboard.py")

with col3:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#10B981;">⚖️ Model Comparison</h3>
        <p style="color:#94A3B8; font-size:0.9rem;">Benchmark BiLSTM vs BERT architectures with real Accuracy, Precision, Recall, F1, and Confusion Matrices.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Compare Models ➔", key="nav_models", use_container_width=True):
        st.switch_page("pages/4_⚖️_Model_Comparison.py")

# System Workflow Section
st.markdown("---")
st.markdown("### 🔄 End-to-End Processing Pipeline")
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 24px 0;">
    <div class="step-card">
        <div class="step-number">1</div>
        <h4 style="margin: 0; color: #FFFFFF;">Student Input</h4>
        <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 8px;">Free-text description of academic challenge</p>
    </div>
    <div class="step-card">
        <div class="step-number">2</div>
        <h4 style="margin: 0; color: #FFFFFF;">Text Processing</h4>
        <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 8px;">Tokenization, vocab mapping & padding</p>
    </div>
    <div class="step-card">
        <div class="step-number">3</div>
        <h4 style="margin: 0; color: #FFFFFF;">Dual Deep Learning</h4>
        <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 8px;">BiLSTM & BERT parallel inference</p>
    </div>
    <div class="step-card">
        <div class="step-number">4</div>
        <h4 style="margin: 0; color: #FFFFFF;">Rule Enhancement</h4>
        <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 8px;">Mixed emotion & connector analysis</p>
    </div>
    <div class="step-card">
        <div class="step-number">5</div>
        <h4 style="margin: 0; color: #FFFFFF;">Gemini Support</h4>
        <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 8px;">6-part personalized study guidance</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
    AI Learning Assistant • Academic Project Platform • Built with PyTorch, Transformers, Google Gemini & Streamlit
</div>
""", unsafe_allow_html=True)
