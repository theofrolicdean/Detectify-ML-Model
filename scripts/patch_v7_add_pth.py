"""
Script: patch_v7_add_pth.py
Resumes the v7 run (dedb3df1b66b4a4f9b8f65a561312f82) and adds
bi_lstm.pth as an explicit raw artifact inside BiLSTM_Model/
so it is visible in v7's artifact listing on DagsHub.
"""

import mlflow
import dagshub

V7_RUN_ID = "dedb3df1b66b4a4f9b8f65a561312f82"
BI_LSTM_PTH = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models\bi_lstm.pth"

print("[1] Connecting to DagsHub...")
dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
print(f"    [OK] {mlflow.get_tracking_uri()}")

print(f"\n[2] Resuming v7 run ({V7_RUN_ID})...")
with mlflow.start_run(run_id=V7_RUN_ID):
    print(f"    Adding bi_lstm.pth as raw artifact under BiLSTM_Model/...")
    mlflow.log_artifact(BI_LSTM_PTH, artifact_path="BiLSTM_Model")
    print(f"    [OK] bi_lstm.pth added to v7!")

print(f"\n[SUCCESS] v7 artifacts now include:")
print(f"  BiLSTM_Model/bi_lstm.pth          ← raw .pth file (explicit)")
print(f"  BiLSTM_Model/data/model.pth        ← same model (MLflow format)")
print(f"  BiLSTM_Model/artifacts/doc2Vec.d2v")
print(f"  BiLSTM_Model/metrics_data/y_val_bi_lstm.npy")
print(f"  BiLSTM_Model/metrics_data/y_bi_lstm_pred.npy")
print(f"\n  View: https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow/#/experiments/4/runs/{V7_RUN_ID}")
