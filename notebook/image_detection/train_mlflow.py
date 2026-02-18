import os
import time
import json
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

EXPERIMENT_NAME = "Image_AI_vs_Human"
MODEL_PATH = "notebook/image_detection/saved_models/final_model.pth"
DATA_DIR = "notebook/image_detection/processed_dataset/split_data"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def evaluate():
    all_preds, all_labels, all_probs, latencies = [], [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            outputs = model(images)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            latencies.append((end - start) * 1000)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    latencies = np.array(latencies)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "P50_latency_ms": np.percentile(latencies, 50),
        "P90_latency_ms": np.percentile(latencies, 90),
        "P95_latency_ms": np.percentile(latencies, 95),
        "throughput_samples_per_sec": len(latencies) / (latencies.sum() / 1000)
    }

with mlflow.start_run() as run:
    metrics = evaluate()

    mlflow.log_param("architecture", "ResNet18")
    mlflow.log_param("batch_size", BATCH_SIZE)

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.pytorch.log_model(model, "model")

    print("Run ID:", run.info.run_id)
