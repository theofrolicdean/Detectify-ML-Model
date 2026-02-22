import os
import sys
import mlflow
import mlflow.pyfunc
import dagshub
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Add scripts directory to path to import the wrapper
sys.path.append(r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts")
from indo_text_pipeline_wrapper import IndoTextPipelineWrapper

load_dotenv()

def upload_artifact_with_bar(run_id, local_path, artifact_path, token):
    """
    Uploads a file to DagsHub MLflow artifact store using requests and tqdm for a real progress bar.
    """
    file_size = os.path.getsize(local_path)
    filename = os.path.basename(local_path)
    
    # DagsHub MLflow Artifact Upload URL
    url = f"https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow/api/2.0/mlflow/artifacts/upload"
    params = {
        "run_id": run_id,
        "path": f"{artifact_path}/{filename}"
    }
    
    headers = {
        "Authorization": f"Bearer {token}"
    }

    print(f"\nUploading {filename} ({file_size / (1024**2):.2f} MB)...")
    
    with open(local_path, "rb") as f:
        # We use a simple chunked upload to show progress
        # Unfortunately, the standard mlflow upload is a single POST/PUT
        # So we'll use a generator to track progress with requests-toolbelt or simple loop
        
        # To keep it simple and robust, we'll use requests and tqdm
        class ProgressReader:
            def __init__(self, f, pbar):
                self.f = f
                self.pbar = pbar
            def read(self, size=-1):
                chunk = self.f.read(size)
                self.pbar.update(len(chunk))
                return chunk
            def __len__(self):
                return file_size

        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, leave=True) as pbar:
            reader = ProgressReader(f, pbar)
            response = requests.post(url, params=params, headers=headers, data=reader)
            
    if response.status_code != 200:
        print(f"Error uploading {filename}: {response.text}")
        sys.exit(1)
    print(f"Successfully uploaded {filename}")

def register_final_with_real_bar():
    model_name = "detectify-indo-detection-text-bi-lstm"
    repo_owner = "theofrolicdean"
    repo_name = "Detectify-ML-Model"
    
    print(f"--- Registration with LIVE PROGRESS BAR: {model_name} ---")
    
    # 1. Initialize DagsHub
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("Indonesian_Text_Detection")
    
    # Get token for manual upload
    # We'll try to get it from environment first, then from dagshub.auth
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        import dagshub.auth
        token = dagshub.auth.get_token()
    
    base_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models"
    doc2vec_base = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts"
    wrapper_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts\indo_text_pipeline_wrapper.py"
    
    # Files to register
    small_artifacts = {
        "pipeline_pkl": os.path.join(base_path, "indo_text_pipeline_bi_lstm.pkl"),
        "pytorch_model": os.path.join(base_path, "bi_lstm.pth"),
        "doc2vec_model": os.path.join(doc2vec_base, "doc2Vec.d2v"),
    }
    
    large_files = [
        os.path.join(doc2vec_base, "doc2Vec.d2v.syn1neg.npy"),
        os.path.join(doc2vec_base, "doc2Vec.d2v.wv.vectors.npy"),
    ]

    with mlflow.start_run(run_name="Full_Package_With_Bar_v11") as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        
        # 2. Log Model Structure first
        print("Logging model structure...")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="BiLSTM_Complete",
            python_model=IndoTextPipelineWrapper(),
            artifacts=small_artifacts,
            code_paths=[wrapper_path]
        )
        
        # 3. Manual Upload of 2.1GB with TQDM Progress Bar
        print("\n" + "="*50)
        print("STARTING LARGE ARTIFACT UPLOAD (2.1GB)")
        print("="*50)
        
        for local_path in large_files:
            # Artifact path should match where the model expects it
            # Model artifacts are stored in 'BiLSTM_Complete/artifacts/'
            upload_artifact_with_bar(run_id, local_path, "BiLSTM_Complete/artifacts", token)

        # 4. Register
        print("\nFinalizing registration...")
        mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
        
        print("\n" + "="*60)
        print(f"SUCCESS! Registered model '{model_name}'")
        print(f"Run View: https://dagshub.com/{repo_owner}/{repo_name}.mlflow/#/experiments/4/runs/{run_id}")
        print("="*60)

if __name__ == "__main__":
    register_final_with_real_bar()
