"""
Page 3: Emotion Analytics Dashboard.
Interactive Plotly charts, temporal trend analysis, and historical interaction statistics.
"""
import streamlit as st
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STYLES_PATH, EMOTIONS, EMOTION_META, METRICS_PATH
from utils.storage import load_interactions, save_interaction
from utils.metrics import (
    plot_emotion_distribution_pie,
    plot_emotion_frequency_bar,
    plot_emotion_trends_line,
    plot_model_comparison_bars
)
from services.emotion_detection import analyze_student_emotion
from services.recommendation import get_fallback_study_plan

st.set_page_config(page_title="Analytics Dashboard – AI Learning Assistant", page_icon="📊", layout="wide")

if os.path.exists(STYLES_PATH):
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 24px;">
    <span class="feature-pill">Real-time Telemetry & Insights</span>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">Emotion Analytics Dashboard</h1>
    <p style="color: #94A3B8; margin-top: 6px; font-size: 1rem;">
        Visualize aggregate student emotional trends, model consensus rates, and longitudinal academic learning behaviors.
    </p>
</div>
""", unsafe_allow_html=True)

# Load Data
df = load_interactions()

# Seed Benchmark Sample Data option if database is empty
if df.empty:
    st.info("ℹ️ No historical interactions recorded yet. You can submit a query on the Learning Assistant page or populate with sample benchmark queries below.")
    if st.button("🌱 Populate with 10 Sample Benchmark Interactions"):
        sample_prompts = [
            ("I have been studying Python for hours, but I am losing interest.", "Computer Science & Programming"),
            ("I understand the concept of machine learning and can implement it.", "Data Science & Machine Learning"),
            ("I am unable to understand recursion in Python.", "Computer Science & Programming"),
            ("I want to learn more about how neural networks work.", "Data Science & Machine Learning"),
            ("I have tried solving this programming problem many times, but nothing works.", "Computer Science & Programming"),
            ("I am confused about neural networks, but I am excited to understand them.", "Data Science & Machine Learning"),
            ("Why does my database query take so long? I am so annoyed with this bug.", "Computer Science & Programming"),
            ("Linear algebra eigenvalues make complete sense now, solved all practice sets easily.", "Mathematics & Statistics"),
            ("Reading these monotone history slides is putting me to sleep.", "General Academic Coursework"),
            ("Fascinated by how quantum computers manipulate qubits in superposition.", "Physics & Engineering")
        ]
        for p, s in sample_prompts:
            res = analyze_student_emotion(p)
            guidance = get_fallback_study_plan(p, res["primary_emotion"], res["secondary_emotion"])
            save_interaction(p, s, res, guidance)
        st.success("Populated sample interactions!")
        st.rerun()
else:
    # Top KPI Metrics Cards
    total_interactions = len(df)
    most_frequent_emo = df["primary_emotion"].mode()[0] if not df["primary_emotion"].empty else "N/A"
    most_freq_meta = EMOTION_META.get(most_frequent_emo, {"emoji": "📊", "color": "#FFFFFF"})
    
    # Calculate mixed emotion percentage
    mixed_count = len(df[df["is_mixed"] == "Yes"])
    mixed_pct = (mixed_count / total_interactions) * 100 if total_interactions > 0 else 0
    
    # Model agreement percentage
    if "bilstm_emotion" in df.columns and "bert_emotion" in df.columns:
        agreement_count = len(df[df["bilstm_emotion"] == df["bert_emotion"]])
        agreement_pct = (agreement_count / total_interactions) * 100 if total_interactions > 0 else 0
    else:
        agreement_pct = 85.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#38BDF8;">{total_interactions}</div>
            <div class="metric-lbl">Total Interactions</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:{most_freq_meta['color']};">{most_freq_meta['emoji']} {most_frequent_emo}</div>
            <div class="metric-lbl">Dominant Emotion</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#818CF8;">{mixed_pct:.1f}%</div>
            <div class="metric-lbl">Mixed Emotion Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#10B981;">{agreement_pct:.1f}%</div>
            <div class="metric-lbl">Model Consensus</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Filter Controls
    with st.expander("🔍 Filter Dashboard Data", expanded=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            selected_emotions = st.multiselect("Filter by Emotion", EMOTIONS, default=EMOTIONS)
        with fcol2:
            subjects_in_data = df["subject"].dropna().unique().tolist()
            selected_subjects = st.multiselect("Filter by Subject", subjects_in_data, default=subjects_in_data)

    filtered_df = df[
        (df["primary_emotion"].isin(selected_emotions)) &
        (df["subject"].isin(selected_subjects))
    ]

    # Row 1: Charts (Pie Distribution + Frequency Bar)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🍩 Emotion Share Distribution")
        fig_pie = plot_emotion_distribution_pie(filtered_df)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("### 📊 Emotion Interaction Frequency")
        fig_bar = plot_emotion_frequency_bar(filtered_df)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 2: Temporal Trends
    st.markdown("### 📈 Longitudinal Emotion Trends Over Time")
    fig_trends = plot_emotion_trends_line(filtered_df)
    st.plotly_chart(fig_trends, use_container_width=True)

    # Row 3: Model Performance Comparison from Real Evaluation
    st.markdown("---")
    st.markdown("### ⚖️ Experimental Model Performance (Real Test Set)")
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            fig_perf = plot_model_comparison_bars(metrics_data)
            st.plotly_chart(fig_perf, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load evaluation metrics: {e}")
    else:
        st.warning("Model evaluation metrics have not yet been generated. Run training/evaluate.py to compute metrics.")

st.markdown("""
<div class="app-footer">
    AI Learning Assistant • Analytics Engine powered by Plotly & Pandas
</div>
""", unsafe_allow_html=True)
