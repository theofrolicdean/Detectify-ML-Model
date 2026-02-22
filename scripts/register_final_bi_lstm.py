import os
import sys
import mlflow
import mlflow.pyfunc
import dagshub
from dotenv import load_dotenv
import torch

# Add scripts directory to path to import the wrapper
sys.path.append(r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts")
from indo_text_pipeline_wrapper import IndoTextPipelineWrapper

load_dotenv()

def register_final():
    model_name = "detectify-indo-detection-text-bi-lstm"
    print(f"--- Starting Final Registration for Model: {model_name} ---")
    
    # 1. Initialize DagsHub
    print("Connecting to DagsHub...")
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    mlflow.set_experiment("Indonesian_Text_Detection")
    
    # 2. Define Paths
    base_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models"
    doc2vec_base = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts"
    wrapper_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts\indo_text_pipeline_wrapper.py"
    
    paths = {
        "pipeline_pkl": os.path.join(base_path, "indo_text_pipeline_bi_lstm.pkl"),
        "pytorch_model": os.path.join(base_path, "bi_lstm.pth"),
        "doc2vec_model": os.path.join(doc2vec_base, "doc2Vec.d2v"),
        "doc2vec_syn": os.path.join(doc2vec_base, "doc2Vec.d2v.syn1neg.npy"),
        "doc2vec_vec": os.path.join(doc2vec_base, "doc2Vec.d2v.wv.vectors.npy")
    }

    # 3. Verify Files and convert to URIs
    artifacts = {}
    print("Verifying files...")
    for key, path in paths.items():
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"  [OK] {key}: {path} ({size_gb:.2f} GB)")
            artifacts[key] = "file:///" + path.replace("\\", "/")
        else:
            print(f"  [ERROR] {key} NOT FOUND at {path}")
            return

    if not os.path.exists(wrapper_path):
        print(f"  [ERROR] Wrapper script NOT FOUND at {wrapper_path}")
        return
    print(f"  [OK] Wrapper script: {wrapper_path}")

    # 4. Log and Register
    print(f"\nLogging model to MLflow and registering as '{model_name}'...")
    print("!!! WARNING: This includes ~2.1GB of data. The upload WILL take time.")
    print("!!! The process may appear 'stuck' while the 2GB of .npy files are being transferred.")
    
    with mlflow.start_run(run_name="Final_BiLSTM_Full_Package") as run:
        print(f"Run ID: {run.info.run_id}")
        
        # We log it as a pyfunc model so it includes the custom logic and all artifacts
        mlflow.pyfunc.log_model(
            artifact_path="BiLSTM_Complete_Service",
            python_model=IndoTextPipelineWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name,
            code_paths=[wrapper_path]
        )
        
        print("\n" + "="*50)
        print(f"SUCCESS: Model '{model_name}' is being registered.")
        print(f"All artifacts (.pkl, .pth, .py, .d2v, .npy) are included.")
        print("="*50)

if __name__ == "__main__":
    register_final()
