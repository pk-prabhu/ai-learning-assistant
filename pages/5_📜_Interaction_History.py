"""
Page 5: Interaction History & Audit Trail.
Searchable, filterable audit log of student interactions with CSV export.
"""
import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import STYLES_PATH, EMOTIONS, EMOTION_META
from utils.storage import load_interactions, clear_all_history

st.set_page_config(page_title="History – AI Learning Assistant", page_icon="📜", layout="wide")

if os.path.exists(STYLES_PATH):
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 24px;">
    <span class="feature-pill">Interaction Ledger</span>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">Student Interaction History</h1>
    <p style="color: #94A3B8; margin-top: 6px; font-size: 1rem;">
        Review previous emotion classifications, inspect model predictions, and export anonymized audit records.
    </p>
</div>
""", unsafe_allow_html=True)

df = load_interactions()

if df.empty:
    st.info("ℹ️ No interaction records currently saved. Once you submit queries on the Learning Assistant page, they will appear here.")
else:
    # Filter Controls
    fcol1, fcol2, fcol3 = st.columns([2, 1.5, 1.5])
    with fcol1:
        search_query = st.text_input("🔍 Search Student Text", placeholder="Search by keywords...")
    with fcol2:
        selected_emotions = st.multiselect("Filter by Emotion", EMOTIONS, default=EMOTIONS)
    with fcol3:
        sort_order = st.selectbox("Sort Order", ["Newest First", "Oldest First"])

    # Apply filters
    filtered_df = df.copy()
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["student_text"].str.contains(search_query.strip(), case=False, na=False)]
    
    if selected_emotions:
        filtered_df = filtered_df[filtered_df["primary_emotion"].isin(selected_emotions)]

    if sort_order == "Newest First":
        filtered_df = filtered_df.iloc[::-1].reset_index(drop=True)

    # Action Toolbar: CSV Export & Clear History
    act_col1, act_col2 = st.columns([1, 1])
    with act_col1:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export History as CSV",
            data=csv_data,
            file_name=f"student_interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with act_col2:
        if st.button("🗑️ Clear Entire History", use_container_width=True, type="secondary"):
            if clear_all_history():
                st.success("Interaction history successfully cleared!")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} total interactions**")

    # Display Records Table
    display_cols = [
        "timestamp", "student_text", "subject", "primary_emotion",
        "confidence", "secondary_emotion", "bilstm_emotion", "bert_emotion", "gemini_source"
    ]
    available_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "student_text": st.column_config.TextColumn("Student Text", width="large"),
            "subject": st.column_config.TextColumn("Subject", width="medium"),
            "primary_emotion": st.column_config.TextColumn("Primary Emotion", width="small"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "secondary_emotion": st.column_config.TextColumn("Secondary Emotion", width="small"),
            "bilstm_emotion": st.column_config.TextColumn("BiLSTM", width="small"),
            "bert_emotion": st.column_config.TextColumn("BERT", width="small"),
            "gemini_source": st.column_config.TextColumn("AI Plan Source", width="medium")
        }
    )

st.markdown("""
<div class="app-footer">
    AI Learning Assistant • Privacy-First Data Ledger
</div>
""", unsafe_allow_html=True)
