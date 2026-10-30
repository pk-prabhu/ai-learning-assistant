"""
Page 1: Home / Landing Page.
Comprehensive architectural showcase, platform overview, features, and workflow.
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STYLES_PATH, EMOTIONS, EMOTION_META

# Page Configuration
st.set_page_config(page_title="Home – AI Learning Assistant", page_icon="🏠", layout="wide")

if os.path.exists(STYLES_PATH):
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <span class="feature-pill">Deep Learning & Generative AI</span>
    <h1 class="hero-title">Understand Emotions.<br>Transform Learning.</h1>
    <p class="hero-subtitle">
        An AI-powered academic companion that understands student emotional challenges through deep learning text classification and delivers personalized pedagogical guidance via Google Gemini.
    </p>
</div>
""", unsafe_allow_html=True)

# Action Buttons
c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 2])
with c_btn1:
    if st.button("🚀 Start Learning Assistant", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🎓_Learning_Assistant.py")
with c_btn2:
    if st.button("⚖️ Compare AI Models", use_container_width=True):
        st.switch_page("pages/4_⚖️_Model_Comparison.py")

st.markdown("<br>", unsafe_allow_html=True)

# 1. Interactive Workflow
st.markdown("### 🔄 How It Works: End-to-End Workflow")
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0;">
    <div class="step-card">
        <div class="step-number">1</div>
        <h4 style="margin:0; color:#FFFFFF;">Student Input</h4>
        <p style="color:#94A3B8; font-size:0.85rem; margin-top:8px;">Learner describes their academic obstacle in natural language.</p>
    </div>
    <div class="step-card">
        <div class="step-number">2</div>
        <h4 style="margin:0; color:#FFFFFF;">Preprocessing</h4>
        <p style="color:#94A3B8; font-size:0.85rem; margin-top:8px;">Normalization, vocabulary indexing, sequence tensor generation.</p>
    </div>
    <div class="step-card">
        <div class="step-number">3</div>
        <h4 style="margin:0; color:#FFFFFF;">BiLSTM & BERT</h4>
        <p style="color:#94A3B8; font-size:0.85rem; margin-top:8px;">Dual deep learning models compute contextual probabilities.</p>
    </div>
    <div class="step-card">
        <div class="step-number">4</div>
        <h4 style="margin:0; color:#FFFFFF;">Rule Enhancement</h4>
        <p style="color:#94A3B8; font-size:0.85rem; margin-top:8px;">Linguistic clause parsing detects mixed and secondary emotions.</p>
    </div>
    <div class="step-card">
        <div class="step-number">5</div>
        <h4 style="margin:0; color:#FFFFFF;">Gemini Guidance</h4>
        <p style="color:#94A3B8; font-size:0.85rem; margin-top:8px;">Emotion-tailored 6-part structured pedagogical support.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 2. About the Platform & Primary Scenarios
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#38BDF8; margin-top:0;">👨‍🏫 Scenario 1: Academic Support & Educators</h3>
        <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
            Educators and Teaching Assistants receive hundreds of student queries daily. This platform automatically triages student submissions, 
            identifying students who are <b>FRUSTRATED</b> (needing immediate debugging support) or <b>CONFUSED</b> (needing foundational conceptual clarity), 
            allowing educators to intervene empathetically and track course-wide sentiment trends.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#818CF8; margin-top:0;">🧑‍🎓 Scenario 2: Autonomous Self-Study Learners</h3>
        <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
            Self-paced students learning complex topics (like Python recursion, machine learning, or differential calculus) often hit emotional walls alone. 
            The assistant acts as an empathetic 24/7 mentor, translating feelings of frustration into concrete, manageable micro-actions and evidence-based study strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 3. Target Emotions Grid
st.markdown("### 🎯 The 5 Academic Emotional Dimensions")
cols = st.columns(5)
for i, emo in enumerate(EMOTIONS):
    meta = EMOTION_META[emo]
    with cols[i]:
        st.markdown(f"""
        <div class="glass-card" style="padding:16px; text-align:center; height:100%;">
            <div style="font-size:2rem; margin-bottom:8px;">{meta['emoji']}</div>
            <h4 style="margin:0; color:{meta['color']};">{meta['label']}</h4>
            <p style="font-size:0.75rem; color:#94A3B8; margin-top:8px; line-height:1.4;">{meta['description']}</p>
        </div>
        """, unsafe_allow_html=True)

# 4. Technology Stack & Key Highlights
st.markdown("---")
st.markdown("### 🛠️ Built with Production-Grade Technologies")

tcol1, tcol2, tcol3, tcol4 = st.columns(4)
with tcol1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val" style="color:#38BDF8;">PyTorch</div>
        <div class="metric-lbl">BiLSTM Neural Network</div>
    </div>
    """, unsafe_allow_html=True)

with tcol2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val" style="color:#818CF8;">BERT</div>
        <div class="metric-lbl">Transformer Representation</div>
    </div>
    """, unsafe_allow_html=True)

with tcol3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val" style="color:#10B981;">Gemini AI</div>
        <div class="metric-lbl">Pedagogical LLM Engine</div>
    </div>
    """, unsafe_allow_html=True)

with tcol4:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-val" style="color:#F59E0B;">Plotly</div>
        <div class="metric-lbl">Interactive Analytics</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
    AI Learning Assistant • College Academic Evaluation Project • End-to-End Machine Learning Platform
</div>
""", unsafe_allow_html=True)
