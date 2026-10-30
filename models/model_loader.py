"""
Singleton Model Loader and Caching Layer for Streamlit.
Optimizes memory usage and cold-start latency.
"""
import os
import torch
from pathlib import Path
from typing import Tuple, Optional
from models.bilstm_model import BiLSTMClassifier, BiLSTMPredictor
from models.bert_model import BERTPredictor
from training.preprocessing import Vocabulary
from utils.config import BILSTM_MODEL_PATH, VOCAB_PATH

_cached_bilstm_predictor: Optional[BiLSTMPredictor] = None
_cached_bert_predictor: Optional[BERTPredictor] = None

def get_bilstm_predictor() -> BiLSTMPredictor:
    """Returns cached BiLSTM predictor instance."""
    global _cached_bilstm_predictor
    if _cached_bilstm_predictor is not None:
        return _cached_bilstm_predictor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(VOCAB_PATH) and os.path.exists(BILSTM_MODEL_PATH):
        try:
            vocab = Vocabulary.load(str(VOCAB_PATH))
            checkpoint = torch.load(str(BILSTM_MODEL_PATH), map_location=device)
            
            model = BiLSTMClassifier(
                vocab_size=len(vocab),
                embed_dim=checkpoint.get("embed_dim", 128),
                hidden_dim=checkpoint.get("hidden_dim", 128),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=5,
                pad_idx=vocab.pad_idx
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            _cached_bilstm_predictor = BiLSTMPredictor(model, vocab, device=device)
            print("Successfully loaded trained BiLSTM model from checkpoint.")
            return _cached_bilstm_predictor
        except Exception as e:
            print(f"Error loading trained BiLSTM ({e}). Building untrained fallback.")
    
    # Fallback initialization if checkpoint not yet trained
    vocab = Vocabulary()
    vocab.build_vocab(["sample", "academic", "text", "python", "neural", "networks", "recursion"])
    model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
    _cached_bilstm_predictor = BiLSTMPredictor(model, vocab, device="cpu")
    return _cached_bilstm_predictor

def get_bert_predictor() -> BERTPredictor:
    """Returns cached BERT predictor instance."""
    global _cached_bert_predictor
    if _cached_bert_predictor is not None:
        return _cached_bert_predictor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _cached_bert_predictor = BERTPredictor(device=device)
    return _cached_bert_predictor

def load_all_models() -> Tuple[BiLSTMPredictor, BERTPredictor]:
    """Preloads both models for the application."""
    bilstm = get_bilstm_predictor()
    bert = get_bert_predictor()
    return bilstm, bert
