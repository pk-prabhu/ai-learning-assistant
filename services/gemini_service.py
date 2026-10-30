"""
Google Gemini API Integration Service.
Provides structured, emotion-aware pedagogical recommendations with comprehensive fallbacks.
"""
import os
import json
import re
import requests
from typing import Dict, Any, Optional
from services.recommendation import get_fallback_study_plan

def get_gemini_api_key() -> Optional[str]:
    """Retrieves Gemini API Key from environment or Streamlit secrets."""
    # 1. Check OS Environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    
    # 2. Check Streamlit Secrets if running under Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass
        
    return None

def generate_personalized_guidance(
    student_text: str,
    primary_emotion: str,
    secondary_emotion: Optional[str],
    confidence: float,
    subject: str = "General",
    api_key_override: Optional[str] = None,
    gemini_enabled: bool = True
) -> Dict[str, Any]:
    """
    Calls Google Gemini API with engineered pedagogical prompt.
    Returns structured 6-part learning plan. Gracefully falls back if API key is absent or offline.
    """
    if not gemini_enabled:
        fallback = get_fallback_study_plan(student_text, primary_emotion, secondary_emotion)
        fallback["source"] = "Offline Pedagogical Engine (Gemini Disabled by Toggle)"
        return fallback

    api_key = api_key_override.strip() if (api_key_override and api_key_override.strip()) else get_gemini_api_key()
    
    if not api_key:
        fallback = get_fallback_study_plan(student_text, primary_emotion, secondary_emotion)
        fallback["source"] = "Offline Pedagogical Engine (No Gemini API Key Configured)"
        fallback["notice"] = "To activate live Gemini generative responses, provide a GEMINI_API_KEY in the sidebar or environment variables."
        return fallback

    # System Prompt with Strict JSON Output Format
    system_instruction = (
        "You are an empathetic, world-class academic mentor, senior computer science educator, and learning psychologist. "
        "A student has expressed an academic difficulty, and our AI deep learning pipeline has classified their emotion. "
        "Your task is to provide an encouraging, technically insightful, highly actionable personalized learning plan tailored "
        "to both their academic challenge and emotional state."
    )

    user_prompt = f"""
Student Input Text: "{student_text}"
Academic Subject / Context: {subject}
Detected Primary Emotion: {primary_emotion} (Confidence: {confidence*100:.1f}%)
Detected Secondary Emotion: {secondary_emotion if secondary_emotion else "None"}

Please generate a 6-part personalized learning response. You MUST return your response as a valid JSON object with the following EXACT keys:
{{
  "understanding": "Clear, compassionate reflection of the student's specific difficulty in 1-2 sentences.",
  "supportive_message": "An emotion-aware supportive and psychologically reassuring statement tailored to {primary_emotion}.",
  "learning_tips": [
    "Tip 1: Practical, high-impact learning tip directly addressing their topic.",
    "Tip 2: Specific conceptual analogy or mental model.",
    "Tip 3: Resource or hands-on practice recommendation."
  ],
  "next_actions": [
    "Step 1: First concrete immediate action to take in the next 10 minutes.",
    "Step 2: Second step to trace, debug, or verify.",
    "Step 3: Third step to consolidate understanding."
  ],
  "study_approach": "Recommended study methodology (e.g. Feynman Technique, Pomodoro sprints, Rubber Duck debugging, or Prototype-first).",
  "encouraging_conclusion": "Brief, inspiring closing message empowering the student."
}}

Important: Return ONLY the raw JSON object. Do not enclose in markdown code fences or backticks.
"""

    # Call Gemini REST API (compatible with gemini-1.5-flash and gemini-2.0-flash)
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_instruction}\n\n{user_prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 800
        }
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        if response.status_code == 200:
            result_json = response.json()
            candidates = result_json.get("candidates", [])
            if candidates and "content" in candidates[0]:
                content_text = candidates[0]["content"]["parts"][0]["text"]
                # Parse JSON
                parsed = clean_and_parse_json(content_text)
                if parsed:
                    parsed["source"] = "Google Gemini AI (Live Model: gemini-1.5-flash)"
                    return parsed
        else:
            print(f"Gemini API returned status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception calling Gemini API: {e}")

    # Fallback if API call failed
    fallback = get_fallback_study_plan(student_text, primary_emotion, secondary_emotion)
    fallback["source"] = "Offline Pedagogical Engine (Fallback due to API Timeout/Error)"
    return fallback

def clean_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Strips markdown code fences and parses JSON safely."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text.strip())
    except Exception:
        # Match outermost brackets
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None
