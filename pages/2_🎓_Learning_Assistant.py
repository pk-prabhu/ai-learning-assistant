"""
Page 2: Student Learning Assistant.
Main interactive workbench for emotion classification, model comparison, and Gemini guidance.
"""
import streamlit as st
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STYLES_PATH, EMOTIONS, EMOTION_META, SUBJECTS
from services.emotion_detection import analyze_student_emotion
from services.gemini_service import generate_personalized_guidance
from utils.storage import save_interaction
from utils.metrics import plot_emotion_probabilities

st.set_page_config(page_title="Learning Assistant – Emotion Detection", page_icon="🎓", layout="wide")

if os.path.exists(STYLES_PATH):
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 24px;">
    <span class="feature-pill">Live Interactive Assistant</span>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">Student Emotion Detection & Learning Support</h1>
    <p style="color: #94A3B8; margin-top: 6px; font-size: 1rem;">
        Describe what you are studying and what is giving you trouble. Our dual deep learning models will analyze your emotional state and generate a tailored study plan.
    </p>
</div>
""", unsafe_allow_html=True)

# Sample Inputs for Rapid Academic Testing
SAMPLE_QUERIES = {
    "Select an example test input...": "",
    "Boredom: Studying for hours with no interest": "I have been studying Python for hours, but I am losing interest.",
    "Confidence: Machine learning mastery": "I understand the concept of machine learning and can implement it.",
    "Confusion: Python recursion roadblocks": "I am unable to understand recursion in Python.",
    "Curiosity: Inquisitive about neural networks": "I want to learn more about how neural networks work.",
    "Frustration: Programming bugs and repeated failures": "I have tried solving this programming problem many times, but nothing works.",
    "Mixed Emotion: Confused about neural nets, yet excited": "I am confused about neural networks, but I am excited to understand them."
}

# Session State for Input Area
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Input Form Area
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_subj, col_sample = st.columns([1, 1])
    with col_subj:
        selected_subject = st.selectbox("Academic Subject / Domain (Optional)", SUBJECTS, index=0)
    with col_sample:
        selected_sample = st.selectbox("Load Benchmark Sample Input", list(SAMPLE_QUERIES.keys()), index=0)
        if selected_sample != "Select an example test input..." and SAMPLE_QUERIES[selected_sample]:
            st.session_state.input_text = SAMPLE_QUERIES[selected_sample]

    user_text = st.text_area(
        "Student Expression / Academic Challenge",
        value=st.session_state.input_text,
        height=140,
        placeholder="Describe your learning difficulty, study experience, or academic challenge in your own words..."
    )
    
    c_sub1, c_sub2, c_sub3 = st.columns([1.5, 1, 3])
    with c_sub1:
        analyze_clicked = st.button("🔍 Analyze Emotion & Generate Plan", type="primary", use_container_width=True)
    with c_sub2:
        if st.button("🗑️ Clear Input", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Process and Render Results
if analyze_clicked:
    if not user_text.strip():
        st.warning("⚠️ Please enter a text description of your academic difficulty before analyzing.")
    else:
        with st.spinner("🧠 Processing text with BiLSTM, BERT, and Google Gemini..."):
            start_time = time.time()
            
            # 1. Run Emotion Detection Pipeline
            analysis = analyze_student_emotion(user_text)
            
            # 2. Call Gemini Service
            gemini_key = st.session_state.get("gemini_api_key", None)
            gemini_enabled = st.session_state.get("gemini_enabled", True)
            
            guidance = generate_personalized_guidance(
                student_text=user_text,
                primary_emotion=analysis["primary_emotion"],
                secondary_emotion=analysis["secondary_emotion"],
                confidence=analysis["confidence"],
                subject=selected_subject,
                api_key_override=gemini_key,
                gemini_enabled=gemini_enabled
            )
            
            elapsed = time.time() - start_time
            
            # 3. Store in CSV History
            save_interaction(user_text, selected_subject, analysis, guidance)

        st.markdown("---")
        
        # Section A: Primary Results Header
        primary_emo = analysis["primary_emotion"]
        sec_emo = analysis["secondary_emotion"]
        p_meta = EMOTION_META[primary_emo]
        
        st.markdown(f"""
        <div class="glass-card" style="border-left: 6px solid {p_meta['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Primary Detected Emotion</span>
                    <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px;">
                        <span style="font-size: 2.2rem;">{p_meta['emoji']}</span>
                        <h2 style="margin: 0; color: {p_meta['color']}; font-size: 2rem;">{primary_emo}</h2>
                        <span class="emotion-badge badge-{primary_emo.lower()}">Confidence: {analysis['confidence']*100:.1f}%</span>
                    </div>
                </div>
                <div>
                    {f'''
                    <div style="text-align: right;">
                        <span style="font-size: 0.82rem; color: #94A3B8; text-transform: uppercase;">Secondary Emotion</span>
                        <div style="margin-top: 4px;">
                            <span class="emotion-badge badge-{sec_emo.lower()}">{EMOTION_META[sec_emo]['emoji']} {sec_emo} ({analysis['secondary_confidence']*100:.1f}%)</span>
                        </div>
                    </div>
                    ''' if sec_emo else '<span style="color:#64748B; font-size:0.85rem;">Single Emotion Dominant</span>'}
                </div>
            </div>
            <p style="color: #CBD5E1; font-size: 0.95rem; margin-top: 14px; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                <b>System Interpretation:</b> {analysis['explanation']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Section B: Model Probabilities & Comparison Columns
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            st.markdown("#### 📊 Emotion Probability Distribution")
            fig = plot_emotion_probabilities(analysis["probabilities"])
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown("#### 🤖 Model Inference Breakdown")
            b_pred = analysis["bilstm_prediction"]
            t_pred = analysis["bert_prediction"]
            
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div class="model-box bilstm">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#38BDF8;">BiLSTM (PyTorch)</b>
                        <span style="font-size:0.8rem; color:#94A3B8;">Tokens: {len(b_pred.get('tokens', []))}</span>
                    </div>
                    <div style="margin-top:6px; font-size:1.1rem; color:#FFFFFF;">
                        {EMOTION_META[b_pred['primary_emotion']]['emoji']} <b>{b_pred['primary_emotion']}</b>
                        <span style="font-size:0.85rem; color:#94A3B8;">({b_pred['confidence']*100:.1f}%)</span>
                    </div>
                </div>
                <div class="model-box bert">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#818CF8;">BERT (Transformer)</b>
                        <span style="font-size:0.8rem; color:#94A3B8;">Mode: Contextual</span>
                    </div>
                    <div style="margin-top:6px; font-size:1.1rem; color:#FFFFFF;">
                        {EMOTION_META[t_pred['primary_emotion']]['emoji']} <b>{t_pred['primary_emotion']}</b>
                        <span style="font-size:0.85rem; color:#94A3B8;">({t_pred['confidence']*100:.1f}%)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Section C: 6-Part Personalized Learning Support (Gemini)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-header">
            <span>✨ Personalized Learning Plan</span>
            <span style="font-size: 0.8rem; font-weight: normal; color: #94A3B8; margin-left: auto;">Source: {guidance.get('source', 'AI Engine')}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="recommendation-card">
            <!-- 1. Understanding -->
            <div style="margin-bottom: 20px;">
                <h4 style="margin: 0 0 6px 0; color: #38BDF8; font-size: 1.05rem;">1. Understanding Your Challenge</h4>
                <p style="color: #E2E8F0; margin: 0; font-size: 0.95rem; line-height: 1.6;">{guidance.get('understanding', '')}</p>
            </div>
            
            <!-- 2. Supportive Message -->
            <div style="margin-bottom: 20px; background: rgba(56, 189, 248, 0.08); padding: 12px 16px; border-radius: 8px; border-left: 3px solid #38BDF8;">
                <h4 style="margin: 0 0 4px 0; color: #38BDF8; font-size: 0.95rem;">2. Emotion-Aware Supportive Message</h4>
                <p style="color: #CBD5E1; margin: 0; font-size: 0.92rem; line-height: 1.5;">{guidance.get('supportive_message', '')}</p>
            </div>

            <!-- 3. Learning Tips -->
            <div style="margin-bottom: 20px;">
                <h4 style="margin: 0 0 8px 0; color: #818CF8; font-size: 1.05rem;">3. Personalized Learning Tips</h4>
                <ul style="color: #E2E8F0; margin: 0; padding-left: 20px; font-size: 0.93rem; line-height: 1.7;">
                    {''.join([f"<li>{tip}</li>" for tip in guidance.get('learning_tips', [])])}
                </ul>
            </div>

            <!-- 4. Next Actions -->
            <div style="margin-bottom: 20px;">
                <h4 style="margin: 0 0 8px 0; color: #10B981; font-size: 1.05rem;">4. Step-by-Step Next Actions</h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    {''.join([f"<div style='background:rgba(30,41,59,0.8); padding:8px 12px; border-radius:6px; color:#F1F5F9; font-size:0.9rem;'>{step}</div>" for step in guidance.get('next_actions', [])])}
                </div>
            </div>

            <!-- 5. Study Approach -->
            <div style="margin-bottom: 20px;">
                <h4 style="margin: 0 0 6px 0; color: #F59E0B; font-size: 1.05rem;">5. Recommended Study Methodology</h4>
                <p style="color: #E2E8F0; margin: 0; font-size: 0.93rem; line-height: 1.6;">{guidance.get('study_approach', '')}</p>
            </div>

            <!-- 6. Encouraging Conclusion -->
            <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 14px;">
                <h4 style="margin: 0 0 6px 0; color: #F8FAFC; font-size: 1rem;">6. Mentor's Closing Note</h4>
                <p style="color: #94A3B8; font-style: italic; margin: 0; font-size: 0.92rem; line-height: 1.5;">"{guidance.get('encouraging_conclusion', '')}"</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#64748B; margin-top:8px;'>Inference completed in {elapsed:.2f} seconds. Interaction logged to history.</div>", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box" style="margin-top: 40px;">
    ℹ️ <b>Responsible AI Notice:</b> Emotion predictions represent contextual estimates derived from student text expressions and do not constitute clinical or psychological diagnoses.
</div>
""", unsafe_allow_html=True)
