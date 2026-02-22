"""
Script: add_bilstm_full_pipeline.py
Purpose: Add the FULL IndoText BiLSTM pipeline as a new version to the
         registered model 'detectify-indo-text-bi-lstm' on DagsHub.

Includes:
  - Pipeline pickle  (indo_text_pipeline_bi_lstm.pkl)
  - PyTorch model    (bi_lstm.pth)
  - Doc2Vec model    (doc2Vec.d2v)
  - Output arrays    (y_bi_lstm_pred.npy, y_val_bi_lstm.npy)

Note: doc2Vec.d2v.wv.vectors.npy and doc2Vec.d2v.syn1neg.npy are ~1GB each.
      They are logged separately (not baked into the pyfunc model) to avoid
      extremely long upload times. They live as standalone run artifacts.

Fixed for MLflow 3.x: use `name` instead of deprecated `artifact_path`.
"""

import os
import sys
import mlflow
import mlflow.pyfunc
import dagshub

# Add scripts dir so IndoTextPipelineWrapper can be imported
sys.path.insert(0, r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts")
from indo_text_pipeline_wrapper import IndoTextPipelineWrapper  # noqa: E402

# ─── Paths ───────────────────────────────────────────────────────────────────

SAVED_MODELS = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models"
OUTPUT_DIR   = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\output"
MLRUNS_D2V   = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts"
SCRIPTS_DIR  = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts"

PIPELINE_PKL = os.path.join(SAVED_MODELS, "indo_text_pipeline_bi_lstm.pkl")
MODEL_PTH    = os.path.join(SAVED_MODELS, "bi_lstm.pth")
DOC2VEC      = os.path.join(MLRUNS_D2V,   "doc2Vec.d2v")
DOC2VEC_VEC  = os.path.join(MLRUNS_D2V,   "doc2Vec.d2v.wv.vectors.npy")
DOC2VEC_SYN  = os.path.join(MLRUNS_D2V,   "doc2Vec.d2v.syn1neg.npy")

OUTPUT_FILES = [
    os.path.join(OUTPUT_DIR, "y_bi_lstm_pred.npy"),
    os.path.join(OUTPUT_DIR, "y_val_bi_lstm.npy"),
]

REGISTERED_MODEL_NAME = "detectify-indo-text-bi-lstm"
EXPERIMENT_NAME       = "Indonesian_Text_Detection"
WRAPPER_SCRIPT        = os.path.join(SCRIPTS_DIR, "indo_text_pipeline_wrapper.py")


def to_uri(path: str) -> str:
    """Convert a Windows path to a file:/// URI for MLflow artifacts dict."""
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def check_required_files():
    required = {
        "Pipeline PKL": PIPELINE_PKL,
        "BiLSTM .pth":  MODEL_PTH,
        "Doc2Vec .d2v": DOC2VEC,
        "Wrapper script": WRAPPER_SCRIPT,
    }
    all_ok = True
    for label, path in required.items():
        exists = os.path.exists(path)
        size = f"{os.path.getsize(path) / 1024:.1f} KB" if exists else "MISSING"
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {label}: {path} ({size})")
        if not exists:
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print(" Indo BiLSTM Full Pipeline → DagsHub")
    print("=" * 60)

    # 1. Check files
    print("\n[STEP 1] Checking required files...")
    if not check_required_files():
        print("\n[ERROR] Some required files are missing. Aborting.")
        sys.exit(1)

    # 2. Connect to DagsHub
    print("\n[STEP 2] Connecting to DagsHub...")
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")

    # 3. Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 4. Build artifacts dict for the pyfunc wrapper
    # Note: doc2vec .npy sidecars (~1GB each) are logged separately below,
    #       not baked into the pyfunc model, to keep upload manageable.
    artifacts = {
        "pipeline_pkl":  to_uri(PIPELINE_PKL),
        "pytorch_model": to_uri(MODEL_PTH),
        "doc2vec_model": to_uri(DOC2VEC),
    }

    # 5. Run
    print(f"\n[STEP 3] Starting MLflow run...")
    with mlflow.start_run(run_name="Indo_BiLSTM_Full_Pipeline_v2") as run:
        print(f"  Run ID: {run.info.run_id}")

        # Log params
        mlflow.log_params({
            "model_type":   "BiLSTM",
            "hidden_dim":   50,
            "num_layers":   4,
            "dropout":      0.2,
            "vectorizer":   "Doc2Vec",
            "pipeline_pkl": "indo_text_pipeline_bi_lstm.pkl",
        })

        # ── 5a. Log pyfunc model (pipeline wrapper) ──
        print("\n[STEP 4] Uploading pyfunc pipeline (pkl + pth + doc2Vec)...")
        print("         This uploads ~10 MB — should take ~1-3 min...")
        model_info = mlflow.pyfunc.log_model(
            name="Indo_Text_Pipeline",        # MLflow 3.x: use `name` not `artifact_path`
            python_model=IndoTextPipelineWrapper(),
            artifacts=artifacts,
            registered_model_name=REGISTERED_MODEL_NAME,
            code_paths=[WRAPPER_SCRIPT],
        )
        print(f"  [OK] Model URI: {model_info.model_uri}")

        # ── 5b. Log .npy output files (small, <1MB each) ──
        print("\n[STEP 5] Logging output .npy files (predictions & ground truth)...")
        for npy_path in OUTPUT_FILES:
            if os.path.exists(npy_path):
                mlflow.log_artifact(npy_path, artifact_path="metrics_data")
                print(f"  [OK] {os.path.basename(npy_path)} logged.")
            else:
                print(f"  [SKIP] {npy_path} not found.")

        # ── 5c. Log doc2Vec sidecar .npy files as separate run artifacts ──
        #       These are large (~1GB each). Comment this block out if too slow.
        print("\n[STEP 6] Logging doc2Vec sidecar .npy files (LARGE ~1GB each)...")
        print("         This may take several minutes depending on your upload speed.")
        for d2v_sidecar in [DOC2VEC_VEC, DOC2VEC_SYN]:
            if os.path.exists(d2v_sidecar):
                size_mb = os.path.getsize(d2v_sidecar) / 1024 / 1024
                print(f"  Uploading {os.path.basename(d2v_sidecar)} ({size_mb:.0f} MB)...")
                mlflow.log_artifact(d2v_sidecar, artifact_path="doc2vec_sidecars")
                print(f"  [OK] {os.path.basename(d2v_sidecar)} logged.")
            else:
                print(f"  [SKIP] {d2v_sidecar} not found.")

    # 6. Done
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Full pipeline registered as new version of")
    print(f"          '{REGISTERED_MODEL_NAME}' on DagsHub!")
    print(f"  Run ID    : {run.info.run_id}")
    print(f"  Model URI : {model_info.model_uri}")
    print(f"  View run  : https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow/#/experiments/4/runs/{run.info.run_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
