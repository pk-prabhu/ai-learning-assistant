"""
PyTorch Bidirectional LSTM (BiLSTM) Neural Network for Academic Emotion Classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from training.preprocessing import Vocabulary, clean_text, EMOTION_TO_LABEL, LABEL_TO_EMOTION

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional Long Short-Term Memory neural network for sequence classification.
    
    Architecture:
    Input -> Embedding -> Spatial Dropout -> BiLSTM (Forward + Backward)
          -> Max/Average Pooling -> Dense (Linear) -> ReLU -> Dropout -> Linear -> Softmax
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # BiLSTM outputs hidden_dim * 2 (forward + backward directions)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)                 # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        lstm_out, (hn, cn) = self.bilstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Max-pooling across sequence dimension for robust feature extraction
        pooled = torch.max(lstm_out, dim=1)[0]       # [batch_size, hidden_dim * 2]
        
        x_dense = self.fc1(pooled)                   # [batch_size, 64]
        x_dense = self.relu(x_dense)
        x_dense = self.dropout(x_dense)
        logits = self.fc2(x_dense)                   # [batch_size, num_classes]
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability distribution over classes."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs

class BiLSTMPredictor:
    """Wrapper class for preprocessing, inference, and probability extraction."""
    def __init__(self, model: BiLSTMClassifier, vocab: Vocabulary, max_len: int = 32, device: str = "cpu"):
        self.model = model.to(device)
        self.vocab = vocab
        self.max_len = max_len
        self.device = device
        self.model.eval()

    def predict(self, text: str) -> Dict[str, any]:
        """Classifies a single input string and returns structured probabilities."""
        seq = self.vocab.encode(text, self.max_len)
        tensor_x = torch.tensor([seq], dtype=torch.long, device=self.device)
        
        probs = self.model.predict_probabilities(tensor_x)[0].cpu().numpy()
        
        prob_dict = {LABEL_TO_EMOTION[i]: float(probs[i]) for i in range(len(probs))}
        sorted_emotions = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        
        primary_emotion, primary_conf = sorted_emotions[0]
        secondary_emotion, secondary_conf = sorted_emotions[1]
        
        return {
            "model_name": "BiLSTM (Bidirectional LSTM)",
            "primary_emotion": primary_emotion,
            "confidence": primary_conf,
            "secondary_emotion": secondary_emotion,
            "secondary_confidence": secondary_conf,
            "probabilities": prob_dict,
            "tokens": clean_text(text).split()
        }
