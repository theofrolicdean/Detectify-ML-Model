import os
import time
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import mlflow
import dagshub
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "notebook/audio_detection/saved_models/tuning_2_2.pth"
TEST_LIST = "notebook/audio_detection/dataset/libri+gen/test_list.txt"
BATCH_SIZE = 1

class DeepfakeDataset(Dataset):
    def __init__(self, list_file, max_duration=4, target_sample_rate=16000, is_train=False):
        self.data = []
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        self.num_samples = max_duration * target_sample_rate
        self.is_train = is_train

        if not os.path.exists(list_file):
            raise FileNotFoundError(f"File {list_file} tidak ditemukan")

        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.data.append((parts[0], int(parts[1])))
    
    def __len__(self):
        return len(self.data)

    def _pad_or_trim(self, waveform):
        channels, length = waveform.shape
        if length > self.num_samples:
            start = (length - self.num_samples) // 2
            waveform = waveform[:, start : start + self.num_samples]
        elif length < self.num_samples:
            pad_amount = self.num_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        # Adjust path if needed (if it's absolute in the txt file but relative to repo)
        if not os.path.exists(audio_path):
             # Try relative to the notebook folder
             audio_path = os.path.join("notebook/audio_detection", audio_path)

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception:
            return torch.zeros(1, self.num_samples), torch.tensor(label, dtype=torch.long)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = self._pad_or_trim(waveform)
        return waveform, torch.tensor(label, dtype=torch.long)

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

def evaluate():
    if os.environ.get("OFFLINE_DEBUG") == "1":
        print("Offline mode enabled. Skipping DagsHub and MLflow.")
        run_eval_logic()
        return

    try:
        dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    except Exception as e:
        print(f"Warning: DagsHub initialization failed: {e}. Running locally.")
    
    try:
        with mlflow.start_run(run_name="Audio_Evaluation"):
            run_eval_logic()
    except Exception as e:
        print(f"Warning: MLflow start_run failed: {e}. Running evaluation without logging.")
        run_eval_logic()

def run_eval_logic():
    model = SimpleAudioCNN().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    test_ds = DeepfakeDataset(TEST_LIST)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            start_time = time.time()
            outputs = model(inputs)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    if not all_preds:
        print("No samples found to evaluate.")
        return

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p95_latency = np.percentile(latencies, 95)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"P50 Latency: {p50_latency:.2f}ms")
    print(f"P90 Latency: {p90_latency:.2f}ms")
    print(f"P95 Latency: {p95_latency:.2f}ms")

    try:
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "p50_latency_ms": p50_latency,
            "p90_latency_ms": p90_latency,
            "p95_latency_ms": p95_latency
        })
    except Exception:
        pass

if __name__ == "__main__":
    evaluate()
