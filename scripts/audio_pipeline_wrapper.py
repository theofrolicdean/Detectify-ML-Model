import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import soundfile as sf

class AudioPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_duration=4, target_sample_rate=16000):
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        self.num_samples = max_duration * target_sample_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: list of file paths or list of waveforms
        """
        transformed = []
        for item in X:
            if isinstance(item, str):
                data, sample_rate = sf.read(item)
                waveform = torch.from_numpy(data.astype(np.float32))
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T # soundfile is (samples, channels)
            else:
                waveform = torch.tensor(item)
                sample_rate = self.target_sample_rate

            # 1. Resample
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # 2. Mono Conversion
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 3. Pad or Trim
            waveform = self._pad_or_trim(waveform)
            transformed.append(waveform)
        
        return torch.stack(transformed)

    def _pad_or_trim(self, waveform):
        channels, length = waveform.shape
        if length > self.num_samples:
            start = (length - self.num_samples) // 2
            waveform = waveform[:, start : start + self.num_samples]
        elif length < self.num_samples:
            pad_amount = self.num_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_layer = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), 
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.spec_layer(x)
        x = self.to_db(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class AudioCNNClassifier(BaseEstimator):
    def __init__(self, model_path=None, model=None, device="cpu"):
        self.model_path = model_path
        self.model = model
        self.device = device
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def _check_model(self):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = SimpleAudioCNN().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Model not loaded and model_path {self.model_path} not found.")
