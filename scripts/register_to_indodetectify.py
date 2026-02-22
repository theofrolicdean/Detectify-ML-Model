import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pyfunc
import dagshub
import sys
from dotenv import load_dotenv

# Add scripts directory to path to import the wrapper
sys.path.append(r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts")
from indo_text_pipeline_wrapper import IndoTextPipelineWrapper

load_dotenv()

def register_to_indodetectify():
    print("Initializing DagsHub connection...")
    # This will use the token if configured or prompt for it
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Paths
    base_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models"
    output_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\output"
    
    pipeline_pkl = os.path.join(base_path, "indo_text_pipeline_bi_lstm.pkl")
    model_pth = os.path.join(base_path, "bi_lstm.pth")
    
    # For the vectorizer, the files seem to be missing in saved_models but present in mlruns.
    # We will use the most recent ones found.
    doc2vec_base = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts"
    doc2vec_model = os.path.join(doc2vec_base, "doc2Vec.d2v")
    doc2vec_vec = os.path.join(doc2vec_base, "doc2Vec.d2v.wv.vectors.npy")
    doc2vec_syn = os.path.join(doc2vec_base, "doc2Vec.d2v.syn1neg.npy")

    # Check if essential files exist
    essential_files = [pipeline_pkl, model_pth, doc2vec_model]
    for f in essential_files:
        if not os.path.exists(f):
            print(f"Error: Essential file not found: {f}")
            # Try to look for doc2vec in base_path just in case
            if "doc2Vec" in f and not os.path.exists(f):
                alt_f = os.path.join(base_path, os.path.basename(f))
                if os.path.exists(alt_f):
                    print(f"Found alternative at {alt_f}")
                    # Update paths if found
                    if f == doc2vec_model: doc2vec_model = alt_f
                    continue
            return

    # Helpers for Windows URI conversion
    def to_uri(path):
        return "file:///" + os.path.abspath(path).replace("\\", "/")

    # Define artifacts for the custom wrapper
    artifacts = {
        "pipeline_pkl": to_uri(pipeline_pkl),
        "pytorch_model": to_uri(model_pth),
        "doc2vec_model": to_uri(doc2vec_model),
        "doc2vec_vec": to_uri(doc2vec_vec),
        "doc2vec_syn": to_uri(doc2vec_syn)
    }

    mlflow.set_experiment("Indonesian_Text_Detection")
    model_name = "Indodetectify-indo-text-bi-lstm"

    print(f"Starting MLflow run to log model to DagsHub as {model_name}...")
    with mlflow.start_run(run_name="Register_Indodetectify_BiLSTM_V2") as run:
        print(f"Run ID: {run.info.run_id}")
        
        print("Logging model using pyfunc flavors... this may take a few minutes...")
        # Log using pyfunc flavor
        mlflow.pyfunc.log_model(
            artifact_path="Indo_Text_Pipeline",
            python_model=IndoTextPipelineWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name,
            code_paths=[r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts\indo_text_pipeline_wrapper.py"]
        )
        print("Model logged successfully to MLflow tracking store.")
        
        # Log additional artifacts from output folder (metrics, etc)
        if os.path.exists(output_path):
            print("Logging additional artifacts from output directory...")
            for file in os.listdir(output_path):
                if file.endswith(".npy") and "bi_lstm" in file.lower():
                    file_path = os.path.join(output_path, file)
                    print(f"Logging output artifact: {file_path}")
                    mlflow.log_artifact(file_path, artifact_path="Indo_Text_Pipeline/metrics_data")
                    
        print(f"Successfully registered {model_name} to DagsHub!")

if __name__ == "__main__":
    register_to_indodetectify()
