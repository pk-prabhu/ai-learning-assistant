"""
Hybrid Emotion Detection Pipeline.
Combines BiLSTM, BERT, and Rule-Based Contextual Enhancement for Multi-Emotion Analysis.
"""
import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from models.model_loader import get_bilstm_predictor, get_bert_predictor
from utils.config import EMOTIONS, EMOTION_META
from training.preprocessing import clean_text

# Contrastive and concessive connectors for mixed emotion parsing
CONTRAST_CONNECTORS = ["but", "however", "although", "yet", "though", "while", "even though", "still", "nonetheless"]

# Lexical emotion markers for contextual reinforcement
EMOTION_LEXICON = {
    "BORED": ["bored", "boring", "losing interest", "dry", "repetitive", "monotonous", "monotone", "sleep", "tedious", "snooze", "uninterested", "unmotivated", "boilerplate", "chore", "dull", "uninspiring"],
    "CONFIDENT": ["understand", "mastered", "solved", "easy", "comfortable", "crushed", "flawlessly", "crystal clear", "proud", "optimal", "ready", "empowered", "solid grasp", "aced", "effortlessly", "flawless", "deeply"],
    "CONFUSED": ["unable to understand", "confused", "lost", "don't understand", "doesn't make sense", "puzzled", "baffled", "bewildered", "recursion", "cannot figure", "why does", "what is the difference", "unclear", "mixing them up"],
    "CURIOUS": ["want to learn", "curious", "explore", "wonder", "eager", "fascinated", "excited to understand", "excited", "keen", "experiment", "intrigued", "how does", "what makes", "dive deeper", "investigate"],
    "FRUSTRATED": ["tried solving", "nothing works", "going crazy", "crashes", "segmentation fault", "sick and tired", "tearing my hair", "nightmare", "angry", "broken", "furious", "fed up", "brick wall", "infuriating", "at my wits", "giving up", "failed", "deadlock"]
}

def analyze_student_emotion(text: str) -> Dict[str, Any]:
    """
    Main entrypoint for end-to-end emotion detection.
    Executes BiLSTM, BERT, ensembling, and rule-based secondary emotion parsing.
    """
    if not text or not text.strip():
        return {
            "error": "Input text cannot be empty.",
            "primary_emotion": "CONFUSED",
            "confidence": 0.0,
            "secondary_emotion": None,
            "probabilities": {e: 0.2 for e in EMOTIONS},
            "explanation": "No text provided for analysis."
        }

    raw_text = text.strip()
    cleaned = clean_text(raw_text)

    # 1. Run Deep Learning Models
    bilstm = get_bilstm_predictor()
    bert = get_bert_predictor()

    bilstm_res = bilstm.predict(raw_text)
    bert_res = bert.predict(raw_text)

    # 2. Ensemble Probabilities (Weighting: 0.45 BiLSTM + 0.55 BERT)
    w_bilstm = 0.45
    w_bert = 0.55
    
    ensemble_probs = {}
    for emo in EMOTIONS:
        p_bilstm = bilstm_res["probabilities"].get(emo, 0.0)
        p_bert = bert_res["probabilities"].get(emo, 0.0)
        ensemble_probs[emo] = (w_bilstm * p_bilstm) + (w_bert * p_bert)

    # Normalize ensemble probabilities
    total_p = sum(ensemble_probs.values())
    if total_p > 0:
        ensemble_probs = {k: v / total_p for k, v in ensemble_probs.items()}

    # 3. Rule-Based Contextual Enhancement & Mixed-Emotion Analysis
    primary_emo, secondary_emo, explanation, is_mixed = apply_contextual_rules(
        raw_text, cleaned, ensemble_probs, bilstm_res, bert_res
    )

    primary_conf = float(ensemble_probs[primary_emo])
    secondary_conf = float(ensemble_probs[secondary_emo]) if secondary_emo else 0.0

    return {
        "text": raw_text,
        "primary_emotion": primary_emo,
        "confidence": primary_conf,
        "secondary_emotion": secondary_emo,
        "secondary_confidence": secondary_conf,
        "probabilities": ensemble_probs,
        "is_mixed": is_mixed,
        "explanation": explanation,
        "bilstm_prediction": bilstm_res,
        "bert_prediction": bert_res,
        "models_in_agreement": (bilstm_res["primary_emotion"] == bert_res["primary_emotion"])
    }

def find_best_emotion_cue(clause: str) -> Optional[str]:
    """Finds the most specific (longest matching) emotional cue in a clause."""
    best_match_len = 0
    best_emo = None
    for emo, kws in EMOTION_LEXICON.items():
        for kw in kws:
            if kw in clause and len(kw) > best_match_len:
                best_match_len = len(kw)
                best_emo = emo
    return best_emo

def apply_contextual_rules(
    raw_text: str,
    cleaned: str,
    probs: Dict[str, float],
    bilstm_res: Dict[str, Any],
    bert_res: Dict[str, Any]
) -> Tuple[str, Optional[str], str, bool]:
    """
    Applies documented linguistic rules to interpret clauses, detect contrastive mixed emotions,
    and formulate a human-interpretable explanation.
    """
    sorted_by_prob = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_emo, top_prob = sorted_by_prob[0]
    runner_up_emo, runner_up_prob = sorted_by_prob[1]

    # Check for contrastive clauses (e.g. "I am confused about recursion, BUT I am excited to understand it")
    text_lower = raw_text.lower()
    has_contrast = any(f" {conn} " in text_lower for conn in CONTRAST_CONNECTORS)
    
    detected_mixed = False
    secondary_emotion = None
    rule_reasons = []

    if has_contrast:
        # Split at the first contrast connector
        split_pattern = r"\b(?:" + "|".join(CONTRAST_CONNECTORS) + r")\b"
        parts = re.split(split_pattern, text_lower, maxsplit=1)
        
        if len(parts) == 2:
            first_clause, second_clause = parts[0].strip(), parts[1].strip()
            
            # Identify dominant cues in each clause based on longest specific phrase
            cue_first = find_best_emotion_cue(first_clause)
            cue_second = find_best_emotion_cue(second_clause)
            
            if cue_first and cue_second and cue_first != cue_second:
                detected_mixed = True
                primary_emotion = cue_first
                secondary_emotion = cue_second
                rule_reasons.append(
                    f"Contrastive connector identified compound state: initial clause expressed '{primary_emotion}' "
                    f"while the subsequent clause expressed '{secondary_emotion}'."
                )
                # Boost confidence of primary slightly to reflect clear compound phrasing
                probs[primary_emotion] = max(probs[primary_emotion], 0.52)
                probs[secondary_emotion] = max(probs[secondary_emotion], 0.35)
                # Re-normalize
                tot = sum(probs.values())
                for k in probs: probs[k] /= tot
                top_emo = primary_emotion

    if not detected_mixed:
        primary_emotion = top_emo
        # If second highest score has substantial weight (> 20%), flag as secondary
        if runner_up_prob >= 0.20:
            secondary_emotion = runner_up_emo
            detected_mixed = True
            rule_reasons.append(
                f"Secondary emotion '{secondary_emotion}' detected with substantial probability ({runner_up_prob*100:.1f}%)."
            )
        else:
            secondary_emotion = None

    # Construct synthesized explanation
    b_emo = bilstm_res["primary_emotion"]
    t_emo = bert_res["primary_emotion"]
    
    if b_emo == t_emo:
        model_consensus = f"Both BiLSTM and BERT independently reached consensus on '{primary_emotion}'."
    else:
        model_consensus = f"BiLSTM indicated '{b_emo}' while BERT indicated '{t_emo}', successfully resolved via weighted ensembling."

    explanation_parts = [model_consensus]
    if rule_reasons:
        explanation_parts.extend(rule_reasons)
    else:
        explanation_parts.append(
            f"The primary classification is driven by contextual cues aligned with {EMOTION_META[primary_emotion]['description']}."
        )

    explanation = " ".join(explanation_parts)
    return primary_emotion, secondary_emotion, explanation, detected_mixed
