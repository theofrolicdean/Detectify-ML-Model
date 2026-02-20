import os
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec
import torch
import torch.nn as nn


class IndoTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self._preprocess(text) for text in X]
    
    def _preprocess(self, text):
        text = str(text)
        text = text.replace('-', ' ')
        text = re.sub(r'[\r\xa0\t]', '', text)
        text = re.sub(r"http\S+|www\S+", '', text)
        text = re.sub(r'\b\w*\.com\w*\b', '', text)
        text = re.sub(r'\[.*?\]|\(.*?\d\}|\{.*?\}', '', text)
        text = re.sub(r'\b(\w+)/(\w+)\b', r'\1 atau \2', text)
        text = re.sub(r'@[A-Za-z0-9]+|#[A-Za-z0-9]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        text = text.strip(' ')
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text


class _StubBitGen:
    """Dummy BitGenerator that absorbs __setstate__ calls from numpy 2.x pickles."""
    def __init__(self, *a, **kw):
        pass
    def __setstate__(self, state):
        pass  # silently ignore incompatible state
    def __getstate__(self):
        return {}


def _load_doc2vec_compat(path):
    """Load Doc2Vec model with numpy 2.x → 1.26.x compatibility.

    Models pickled under numpy 2.x store BitGenerator/RandomState in a format
    that numpy 1.26.x cannot restore.  We patch gensim's unpickle to use stub
    objects that silently ignore the incompatible state.
    """
    import pickle
    import gensim.utils

    class _NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy.random._pickle" and name == "__bit_generator_ctor":
                return lambda *a, **kw: _StubBitGen()
            if module == "numpy.random._pickle" and name == "__randomstate_ctor":
                return lambda *a, **kw: np.random.RandomState(42)
            if module == "numpy.random._mt19937" and name == "MT19937":
                return _StubBitGen
            return super().find_class(module, name)

    _orig_unpickle = gensim.utils.unpickle

    def _compat_unpickle(fname, **kwargs):
        with open(fname, "rb") as f:
            return _NumpyCompatUnpickler(f).load()

    gensim.utils.unpickle = _compat_unpickle
    try:
        return Doc2Vec.load(path)
    finally:
        gensim.utils.unpickle = _orig_unpickle


class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path=None, model=None):
        self.model_path = model_path
        self.model = model
        
    def fit(self, X, y=None):
        return self
    
    def _load_doc2vec(self, path):
        return _load_doc2vec_compat(path)

    def transform(self, X):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = self._load_doc2vec(self.model_path)
            else:
                raise ValueError(f"Doc2Vec model not loaded and model_path {self.model_path} not found.")
        
        features = []
        for text in X:
            words = simple_preprocess(text)
            vec = self.model.infer_vector(words, epochs=20)
            features.append(vec)
        return np.array(features)


# ─── PyTorch Model Architectures ───────────────────────────────────────────────

class BiLSTM(nn.Module):
    """Bidirectional LSTM — maps to bi_lstm.pth"""
    def __init__(self, input_size=1, hidden_dim=50, num_layers=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


class LSTMModel(nn.Module):
    """Unidirectional LSTM — maps to lstm.pth"""
    def __init__(self, input_size=1, hidden_dim=50, num_layers=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


class GRUModel(nn.Module):
    """Unidirectional GRU — maps to gru.pth"""
    def __init__(self, input_size=1, hidden_dim=50, num_layers=4, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


class BiGRUModel(nn.Module):
    """Bidirectional GRU — maps to bi_gru.pth"""
    def __init__(self, input_size=1, hidden_dim=50, num_layers=4, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


# ─── Model type → class mapping ───────────────────────────────────────────────

MODEL_REGISTRY = {
    "bi_lstm": BiLSTM,
    "lstm":    LSTMModel,
    "gru":     GRUModel,
    "bi_gru":  BiGRUModel,
}


# ─── Unified Classifier Wrapper ───────────────────────────────────────────────

class RNNClassifierWrapper(BaseEstimator):
    """Sklearn-compatible wrapper that loads any of the 4 PyTorch RNN models."""

    def __init__(self, model_path=None, model=None, model_type="bi_lstm", device="cpu"):
        self.model_path = model_path
        self.model = model
        self.model_type = model_type
        self.device = device
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(-1)
            preds = self.model(X_tensor)
            return (preds > 0.5).long().cpu().numpy().flatten()

    def predict_proba(self, X):
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(-1)
            preds = self.model(X_tensor).cpu().numpy()
            return np.hstack([1 - preds, preds])

    def _check_model(self):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                model_cls = MODEL_REGISTRY.get(self.model_type)
                if model_cls is None:
                    raise ValueError(
                        f"Unknown model_type '{self.model_type}'. "
                        f"Choose from: {list(MODEL_REGISTRY.keys())}"
                    )
                self.model = model_cls().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Model not loaded and model_path {self.model_path} not found.")


# ─── Backward-compatible alias ─────────────────────────────────────────────────

BiLSTMClassifierWrapper = RNNClassifierWrapper
