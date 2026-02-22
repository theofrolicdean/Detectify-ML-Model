"""
Script: register_bilstm_final.py
Register ALL BiLSTM pipeline artifacts as a new version of
detectify-indo-text-bi-lstm on DagsHub.

Artifacts nested under BiLSTM_Model/ (visible in registered model version):
  BiLSTM_Model/                    ← PyTorch model (MLmodel + data/model.pth)
  BiLSTM_Model/bi_lstm.pth         ← raw .pth file (explicit)
  BiLSTM_Model/artifacts/          ← doc2Vec.d2v
  BiLSTM_Model/metrics_data/       ← y_val_bi_lstm.npy, y_bi_lstm_pred.npy
  BiLSTM_Model/code/               ← indo_text_pipeline_wrapper.py
"""

import os, sys, torch, torch.nn as nn, mlflow, mlflow.pytorch, dagshub

# ── BiLSTM (must match training architecture) ──────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_dim=50, num_layers=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_dim,
            num_layers=num_layers, dropout=dropout,
            bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT         = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer"
SAVED_MODELS = os.path.join(ROOT, r"notebook\text_detection_indo\saved_models")
OUTPUT_DIR   = os.path.join(ROOT, r"notebook\text_detection_indo\output")
MLRUNS_D2V   = os.path.join(ROOT, r"mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts")
SCRIPTS_DIR  = os.path.join(ROOT, "scripts")

# All files to upload: {display_name: (local_path, artifact_path_in_dagshub)}
ARTIFACTS = [
    ("doc2Vec.d2v",                 os.path.join(MLRUNS_D2V,   "doc2Vec.d2v"),                  "BiLSTM_Model/artifacts"),
    ("y_val_bi_lstm.npy",           os.path.join(OUTPUT_DIR,   "y_val_bi_lstm.npy"),             "BiLSTM_Model/metrics_data"),
    ("y_bi_lstm_pred.npy",          os.path.join(OUTPUT_DIR,   "y_bi_lstm_pred.npy"),            "BiLSTM_Model/metrics_data"),
    ("bi_lstm.pth (raw)",           os.path.join(SAVED_MODELS, "bi_lstm.pth"),                   "BiLSTM_Model"),
    ("indo_text_pipeline_wrapper",  os.path.join(SCRIPTS_DIR,  "indo_text_pipeline_wrapper.py"), "BiLSTM_Model/code"),
]

BI_LSTM_PTH      = os.path.join(SAVED_MODELS, "bi_lstm.pth")
REGISTERED_MODEL = "detectify-indo-text-bi-lstm"

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print(f"  Registering: {REGISTERED_MODEL} (new version)")
    print("=" * 62)

    # 1. Verify all files
    print("\n[1] Verifying files...")
    all_ok = True
    total_kb = 0
    for label, path, dest in ARTIFACTS:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0
        total_kb += size_kb
        status = "OK" if exists else "MISSING"
        print(f"    [{status}] {label:<30} {size_kb:>8.1f} KB")
        if not exists:
            all_ok = False
    print(f"\n    Total artifact upload: {total_kb/1024:.1f} MB")
    if not all_ok:
        print("\n[ERROR] Missing files. Aborting.")
        sys.exit(1)

    # 2. Load BiLSTM model weights
    print(f"\n[2] Loading bi_lstm.pth ({os.path.getsize(BI_LSTM_PTH)/1024:.0f} KB)...")
    device = torch.device("cpu")
    model = BiLSTM()
    ckpt = torch.load(BI_LSTM_PTH, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("    [OK] Model loaded.")

    # 3. Connect to DagsHub
    print("\n[3] Connecting to DagsHub...")
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    mlflow.set_experiment("Indonesian_Text_Detection")
    print(f"    [OK] {mlflow.get_tracking_uri()}")

    # 4. Log everything in a single run
    print(f"\n[4] Starting MLflow run...")
    with mlflow.start_run(run_name="BiLSTM_v8_full_pipeline") as run:
        print(f"    Run ID: {run.info.run_id}")

        mlflow.log_params({
            "model_type": "BiLSTM", "hidden_dim": 50,
            "num_layers": 4, "dropout": 0.2, "input_size": 1,
            "vectorizer": "Doc2Vec",
        })

        # 4a: Register the PyTorch model → creates the new model version
        print(f"\n    [4a] Uploading bi_lstm.pth as PyTorch model (~800 KB)...")
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="BiLSTM_Model",                    # MLflow 3.x: `name` not `artifact_path`
            registered_model_name=REGISTERED_MODEL,
        )
        print(f"         [OK] Model URI: {model_info.model_uri}")

        # 4b-4f: Log all additional artifacts
        step = ord('b')
        for label, path, artifact_path in ARTIFACTS:
            size_kb = os.path.getsize(path) / 1024
            print(f"\n    [4{chr(step)}] Uploading {label} ({size_kb:.0f} KB) → {artifact_path}/")
            mlflow.log_artifact(path, artifact_path=artifact_path)
            print(f"         [OK] Done.")
            step += 1

    # 5. Get the created version number
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

    print(f"\n{'='*62}")
    print(f"  [SUCCESS] {REGISTERED_MODEL} v{latest.version} on DagsHub!")
    print(f"\n  Artifact structure (inside version):")
    print(f"    BiLSTM_Model/                    bi_lstm.pth (raw)  ← .pth IS HERE!")
    print(f"    BiLSTM_Model/data/model.pth      bi_lstm.pth (MLflow format)")
    print(f"    BiLSTM_Model/artifacts/           doc2Vec.d2v")
    print(f"    BiLSTM_Model/metrics_data/        y_val_bi_lstm.npy")
    print(f"    BiLSTM_Model/metrics_data/        y_bi_lstm_pred.npy")
    print(f"    BiLSTM_Model/code/                indo_text_pipeline_wrapper.py")
    print(f"\n  Run ID   : {run.info.run_id}")
    print(f"  Model URI: {model_info.model_uri}")
    print(f"  Version  : v{latest.version}")
    print(f"  View     : https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow/#/experiments/4/runs/{run.info.run_id}")
    print(f"{'='*62}")

if __name__ == "__main__":
    main()
