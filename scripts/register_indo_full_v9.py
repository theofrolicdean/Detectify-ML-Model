import os
import sys
import mlflow
import mlflow.pyfunc
import dagshub
from dotenv import load_dotenv

# Add scripts directory to path to import the wrapper
sys.path.append(r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts")
from indo_text_pipeline_wrapper import IndoTextPipelineWrapper

load_dotenv()

def register_full_pipeline():
    print("Initializing DagsHub connection...")
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Paths
    base_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models"
    doc2vec_base = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts"
    wrapper_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts\indo_text_pipeline_wrapper.py"
    
    pipeline_pkl = os.path.join(base_path, "indo_text_pipeline_bi_lstm.pkl")
    model_pth = os.path.join(base_path, "bi_lstm.pth")
    doc2vec_model = os.path.join(doc2vec_base, "doc2Vec.d2v")
    doc2vec_syn = os.path.join(doc2vec_base, "doc2Vec.d2v.syn1neg.npy")
    doc2vec_vec = os.path.join(doc2vec_base, "doc2Vec.d2v.wv.vectors.npy")

    # Helpers for Windows URI conversion
    def to_uri(path):
        return "file:///" + os.path.abspath(path).replace("\\", "/")

    # Define artifacts for the custom wrapper
    artifacts = {
        "pipeline_pkl": to_uri(pipeline_pkl),
        "pytorch_model": to_uri(model_pth),
        "doc2vec_model": to_uri(doc2vec_model),
        "doc2vec_syn": to_uri(doc2vec_syn),
        "doc2vec_vec": to_uri(doc2vec_vec)
    }

    # Verify files exist
    for k, v in artifacts.items():
        path = v.replace("file:///", "")
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            return

    mlflow.set_experiment("Indonesian_Text_Detection")
    model_name = "detectify-indo-text-bi-lstm"

    print(f"Starting MLflow run to log full pipeline to DagsHub as {model_name}...")
    print("Note: This includes 2GB of .npy files and may take a long time to upload.")
    
    with mlflow.start_run(run_name="Full_Indo_Text_BiLSTM_v9") as run:
        print(f"Run ID: {run.info.run_id}")
        
        # Log using pyfunc flavor
        mlflow.pyfunc.log_model(
            artifact_path="BiLSTM_Pipeline_Full",
            python_model=IndoTextPipelineWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name,
            code_paths=[wrapper_path]
        )
        
        print(f"Successfully registered {model_name} version 1 (after clean) to DagsHub!")

if __name__ == "__main__":
    register_full_pipeline()
