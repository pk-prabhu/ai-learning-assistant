"""
Text preprocessing and tokenization utilities for the BiLSTM and classical pipelines.
"""
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
import torch
from torch.utils.data import Dataset

EMOTION_TO_LABEL = {
    "BORED": 0,
    "CONFIDENT": 1,
    "CONFUSED": 2,
    "CURIOUS": 3,
    "FRUSTRATED": 4
}

LABEL_TO_EMOTION = {v: k for k, v in EMOTION_TO_LABEL.items()}

def clean_text(text: str) -> str:
    """Standardizes text: lowercase, strip punctuation, remove excessive whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\?!\.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Basic whitespace and punctuation tokenizer."""
    cleaned = clean_text(text)
    # Split tokens while preserving words
    tokens = re.findall(r"\b\w+\b", cleaned)
    return tokens

class Vocabulary:
    """Manages word-to-index mapping with <PAD> and <UNK> support."""
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.pad_idx = 0
        self.unk_idx = 1

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """Constructs vocabulary from a list of raw text strings."""
        counter = Counter()
        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)
        
        idx = len(self.word2idx)
        for word, count in counter.most_common():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text: str, max_len: int = 32) -> List[int]:
        """Encodes text to a fixed-length list of token IDs with padding/truncation."""
        tokens = tokenize(text)
        token_ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [self.pad_idx] * (max_len - len(token_ids))
        return token_ids

    def save(self, filepath: str):
        """Persists vocabulary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "pad_idx": self.pad_idx,
                "unk_idx": self.unk_idx
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Loads vocabulary from a JSON file."""
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.pad_idx = data["pad_idx"]
        vocab.unk_idx = data["unk_idx"]
        return vocab

    def __len__(self):
        return len(self.word2idx)

class TextDataset(Dataset):
    """PyTorch Dataset for academic emotion classification."""
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        seq = self.vocab.encode(text, self.max_len)
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)
