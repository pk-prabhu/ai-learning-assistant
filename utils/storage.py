"""
Persistent Interaction History Storage Layer.
Thread-safe CSV reading, writing, filtering, and export.
"""
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from utils.config import HISTORY_PATH

CSV_HEADERS = [
    "timestamp",
    "student_text",
    "subject",
    "primary_emotion",
    "confidence",
    "secondary_emotion",
    "secondary_confidence",
    "bilstm_emotion",
    "bert_emotion",
    "is_mixed",
    "gemini_source",
    "study_approach"
]

def init_history_storage():
    """Initializes CSV storage file with headers if it does not exist."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def save_interaction(
    student_text: str,
    subject: str,
    analysis_res: Dict[str, Any],
    gemini_res: Dict[str, Any]
) -> bool:
    """Appends an interaction record to the history CSV."""
    init_history_storage()
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = [
            now_str,
            student_text.strip().replace("\n", " "),
            subject,
            analysis_res.get("primary_emotion", ""),
            f"{analysis_res.get('confidence', 0.0):.4f}",
            analysis_res.get("secondary_emotion") or "None",
            f"{analysis_res.get('secondary_confidence', 0.0):.4f}",
            analysis_res.get("bilstm_prediction", {}).get("primary_emotion", ""),
            analysis_res.get("bert_prediction", {}).get("primary_emotion", ""),
            "Yes" if analysis_res.get("is_mixed") else "No",
            gemini_res.get("source", "Standard"),
            gemini_res.get("study_approach", "")
        ]
        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(record)
        return True
    except Exception as e:
        print(f"Error saving interaction to CSV: {e}")
        return False

def load_interactions() -> pd.DataFrame:
    """Reads all historical interactions from CSV."""
    init_history_storage()
    try:
        df = pd.read_csv(HISTORY_PATH)
        return df
    except Exception as e:
        print(f"Error reading interactions: {e}")
        return pd.DataFrame(columns=CSV_HEADERS)

def clear_all_history() -> bool:
    """Deletes all interaction records from CSV."""
    try:
        with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False
