"""
Global configuration, constants, and theme settings for AI Learning Assistant.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset" / "academic_emotions.csv"
HISTORY_PATH = DATA_DIR / "interaction_history" / "interactions.csv"
MODELS_DIR = BASE_DIR / "models"
TRAINED_DIR = MODELS_DIR / "trained"
ASSETS_DIR = BASE_DIR / "assets"
STYLES_PATH = ASSETS_DIR / "styles.css"

# Emotion Categories & Metadata
EMOTIONS = ["BORED", "CONFIDENT", "CONFUSED", "CURIOUS", "FRUSTRATED"]

EMOTION_META = {
    "BORED": {
        "label": "Bored",
        "emoji": "🥱",
        "color": "#94A3B8",        # Slate grey
        "badge_bg": "#1E293B",
        "description": "Lack of engagement, disinterest, or mental fatigue with repetitive or unchallenging material.",
        "support_focus": "Re-engagement, interactive challenge, real-world relevance, breaking study sessions."
    },
    "CONFIDENT": {
        "label": "Confident",
        "emoji": "🌟",
        "color": "#10B981",        # Emerald green
        "badge_bg": "#064E3B",
        "description": "High self-efficacy, conceptual mastery, and readiness to tackle advanced topics.",
        "support_focus": "Advanced applications, peer teaching, project-based mastery, stretch problems."
    },
    "CONFUSED": {
        "label": "Confused",
        "emoji": "🧩",
        "color": "#F59E0B",        # Amber orange
        "badge_bg": "#78350F",
        "description": "Conceptual ambiguity, logical roadblocks, or uncertainty about key principles.",
        "support_focus": "Foundational clarity, step-by-step mental models, analogies, visual explanations."
    },
    "CURIOUS": {
        "label": "Curious",
        "emoji": "💡",
        "color": "#6366F1",        # Indigo violet
        "badge_bg": "#312E81",
        "description": "Intrinsic drive to explore deeper mechanisms, ask 'why', and connect broader ideas.",
        "support_focus": "Deep-dive resources, research papers, experimental projects, architectural insights."
    },
    "FRUSTRATED": {
        "label": "Frustrated",
        "emoji": "🔥",
        "color": "#EF4444",        # Crimson red
        "badge_bg": "#7F1D1D",
        "description": "High stress from repetitive bugs, unmet expectations, or complex bottlenecks.",
        "support_focus": "De-escalation, targeted debugging protocols, psychological reassurance, micro-wins."
    }
}

# Model Defaults
BILSTM_MODEL_PATH = TRAINED_DIR / "bilstm_model.pt"
VOCAB_PATH = TRAINED_DIR / "vocab.json"
METRICS_PATH = TRAINED_DIR / "metrics.json"

# Gemini Defaults
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Subjects for context
SUBJECTS = [
    "Computer Science & Programming",
    "Data Science & Machine Learning",
    "Mathematics & Statistics",
    "Physics & Engineering",
    "Web & Full Stack Development",
    "General Academic Coursework"
]
