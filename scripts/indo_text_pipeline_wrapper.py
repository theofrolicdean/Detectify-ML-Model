import os
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec
import tensorflow as tf

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

class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path=None, model=None):
        self.model_path = model_path
        self.model = model
        
    def fit(self, X, y=None):
        if self.model is None and self.model_path:
            self.model = Doc2Vec.load(self.model_path)
        return self
    
    def transform(self, X):
        if self.model is None:
            raise ValueError("Doc2Vec model not loaded. Provide model or model_path.")
        
        features = []
        for text in X:
            words = simple_preprocess(text)
            vec = self.model.infer_vector(words, epochs=20)
            features.append(vec)
        return np.array(features)

class BiLSTMClassifierWrapper(BaseEstimator):
    def __init__(self, model_path=None, model=None):
        self.model_path = model_path
        self.model = model
        
    def fit(self, X, y=None):
        if self.model is None and self.model_path:
            self.model = tf.keras.models.load_model(self.model_path)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Bi-LSTM model not loaded. Provide model or model_path.")
        
        # Reshape for Bi-LSTM: (n_samples, n_features, 1)
        X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
        preds = self.model.predict(X_reshaped, verbose=0)
        return (preds > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Bi-LSTM model not loaded. Provide model or model_path.")
        
        X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
        preds = self.model.predict(X_reshaped, verbose=0)
        return np.hstack([1-preds, preds])
