import os
import re
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec
import mlflow.pyfunc

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

# ─── Doc2Vec Compatibility ────────────────────────────────────────────────────

class _StubBitGen:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state): pass
    def __getstate__(self): return {}

def _load_doc2vec_compat(path):
    import gensim.utils
    class _NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy.random._pickle" and name == "__bit_generator_ctor": return lambda *a, **kw: _StubBitGen()
            if module == "numpy.random._pickle" and name == "__randomstate_ctor": return lambda *a, **kw: np.random.RandomState(42)
            if module == "numpy.random._mt19937" and name == "MT19937": return _StubBitGen
            return super().find_class(module, name)
    _orig_unpickle = gensim.utils.unpickle
    def _compat_unpickle(fname, **kwargs):
        with open(fname, "rb") as f: return _NumpyCompatUnpickler(f).load()
    gensim.utils.unpickle = _compat_unpickle
    try: return Doc2Vec.load(path)
    finally: gensim.utils.unpickle = _orig_unpickle

class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path=None, model=None):
        self.model_path = model_path
        self.model = model
    def fit(self, X, y=None): return self
    def transform(self, X):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path): self.model = _load_doc2vec_compat(self.model_path)
            else: raise ValueError(f"Doc2Vec model_path {self.model_path} not found.")
        features = []
        for text in X:
            words = simple_preprocess(text)
            vec = self.model.infer_vector(words, epochs=20)
            features.append(vec)
        return np.array(features)

class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_dim=50, num_layers=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

class RNNClassifierWrapper(BaseEstimator):
    def __init__(self, model_path=None, model=None, model_type="bi_lstm", device="cpu"):
        self.model_path = model_path
        self.model = model
        self.model_type = model_type
        self.device = device
    def fit(self, X, y=None): return self
    def predict(self, X):
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device).unsqueeze(-1)
            preds = self.model(X_tensor)
            return (preds > 0.5).long().cpu().numpy().flatten()
    def _check_model(self):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = BiLSTM().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint else checkpoint["model_state_dict"])
            else: raise ValueError(f"Model path {self.model_path} not found.")

class IndoTextPipelineWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for MLflow that allows loading the entire .pkl pipeline.
    It handles path resolution for internal components (Doc2Vec, PyTorch).
    """
    def load_context(self, context):
        with open(context.artifacts["pipeline_pkl"], "rb") as f:
            self.pipeline = pickle.load(f)
        
        # Patch paths to point to MLflow artifacts
        if "doc2vec_model" in context.artifacts:
            self.pipeline.named_steps["vectorizer"].model_path = context.artifacts["doc2vec_model"]
            self.pipeline.named_steps["vectorizer"].model = None
            
        if "pytorch_model" in context.artifacts:
            self.pipeline.named_steps["classifier"].model_path = context.artifacts["pytorch_model"]
            self.pipeline.named_steps["classifier"].model = None

    def predict(self, context, model_input):
        if isinstance(model_input, list):
            return self.pipeline.predict(model_input)
        # Handle pandas df
        return self.pipeline.predict(model_input.iloc[:, 0].tolist())

# Alias for backward compatibility if needed
BiLSTMClassifierWrapper = RNNClassifierWrapper
