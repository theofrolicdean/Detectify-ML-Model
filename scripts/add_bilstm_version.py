"""
Script: add_bilstm_version.py
Purpose: Add bi_lstm.pth as a new model version to the existing
         registered model 'detectify-indo-text-bi-lstm' on DagsHub.

Fixed: Use `name` instead of deprecated `artifact_path` for MLflow 3.x+
"""

import os
import sys
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import dagshub

# ─── BiLSTM Architecture (must match training architecture) ─────────────────

class BiLSTM(nn.Module):
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
        out = out[:, -1, :]   # Take output from last timestep
        out = self.fc(out)
        return self.sigmoid(out)

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    model_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models\bi_lstm.pth"
    registered_model_name = "detectify-indo-text-bi-lstm"
    experiment_name = "Indonesian_Text_Detection"

    # 1. Verify file exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)
    print(f"[OK] Found model file: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    # 2. Load the PyTorch model
    print("[INFO] Loading BiLSTM model weights...")
    device = torch.device("cpu")
    model = BiLSTM()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[OK] Model loaded and set to eval mode.")

    # 3. Connect to DagsHub
    print("[INFO] Connecting to DagsHub...")
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    print(f"[OK] Tracking URI: {mlflow.get_tracking_uri()}")

    # 4. Set experiment
    mlflow.set_experiment(experiment_name)

    # 5. Log model and register as new version
    print(f"[INFO] Starting MLflow run to add new version to '{registered_model_name}'...")
    with mlflow.start_run(run_name="Add_BiLSTM_v2_fixed") as run:
        print(f"[INFO] Run ID: {run.info.run_id}")

        # Log params
        mlflow.log_params({
            "model_type": "BiLSTM",
            "hidden_dim": 50,
            "num_layers": 4,
            "dropout": 0.2,
            "input_size": 1,
            "source_file": "text_detection_indo/saved_models/bi_lstm.pth"
        })

        # Use `name` instead of deprecated `artifact_path` (MLflow 3.x)
        print(f"[INFO] Uploading model to DagsHub as new version of '{registered_model_name}'...")
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="BiLSTM_Model",          # <-- use 'name' instead of 'artifact_path'
            registered_model_name=registered_model_name,
        )
        print(f"[OK] Model logged! Model URI: {model_info.model_uri}")

        # Also log the raw .pth file as a separate artifact (shows up in run artifacts)
        print("[INFO] Also logging raw .pth file as artifact...")
        mlflow.log_artifact(model_path, artifact_path="raw_model")
        print("[OK] Raw .pth file logged.")

    print(f"\n[SUCCESS] New version of '{registered_model_name}' registered on DagsHub!")
    print(f"  Run ID    : {run.info.run_id}")
    print(f"  Model URI : {model_info.model_uri}")
    print(f"\n  View run  : https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow/#/experiments/4/runs/{run.info.run_id}")

if __name__ == "__main__":
    main()
