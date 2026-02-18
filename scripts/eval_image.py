import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import mlflow
import dagshub
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "notebook/image_detection/saved_models/final_model.pth"
DATA_DIR = "notebook/image_detection/processed_dataset" 
BATCH_SIZE = 1

# Image model architecture (using EfficientNet as seen in notebooks)
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, pretrained=False):
        super().__init__()
        self.base_model = timm.create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=0)
        num_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model.forward_features(x)
        out = self.classifier(features)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out

def create_model():
    return EfficientNetV2()

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Subdirectories (real -> 0, fake -> 1)
        mapping = {"real": 0, "fake": 1}
        for label_name, label_val in mapping.items():
            label_dir = os.path.join(data_dir, label_name)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    self.images.append(os.path.join(label_dir, img_name))
                    self.labels.append(label_val)
        
        # Limit to 1000 samples for quick evaluation
        if len(self.images) > 1000:
            import random
            combined = list(zip(self.images, self.labels))
            random.seed(42)
            subset = random.sample(combined, 1000)
            self.images, self.labels = zip(*subset)
            self.images = list(self.images)
            self.labels = list(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

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
        with mlflow.start_run(run_name="Image_Evaluation"):
            run_eval_logic()
    except Exception as e:
        print(f"Warning: MLflow start_run failed: {e}. Running evaluation without logging.")
        run_eval_logic()

def run_eval_logic():
    model = create_model().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"TestData directory {DATA_DIR} not found. Skipping image evaluation.")
        return

    test_ds = ImageDataset(DATA_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            
            start_time = time.time()
            outputs = model(inputs)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

            pred = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(labels.numpy())

    if not all_preds:
        print("No image samples found to evaluate.")
        return

    # Metrics
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
