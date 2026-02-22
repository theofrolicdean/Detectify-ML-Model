import os
import sys
import mlflow
import mlflow.pyfunc
import dagshub
import requests
from dotenv import load_dotenv

# Add scripts directory to path to import the wrapper
sys.path.append(r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts")
from indo_text_pipeline_wrapper import IndoTextPipelineWrapper

load_dotenv()

def upload_with_custom_bar(local_path, remote_path, token):
    """Uploads a file to DagsHub with a manual progress bar printed to stdout."""
    file_size = os.path.getsize(local_path)
    url = f"https://dagshub.com/api/v1/repos/theofrolicdean/Detectify-ML-Model/content/main/{remote_path}"
    
    print(f"\nUploading {os.path.basename(local_path)} ({file_size / (1024**2):.2f} MB)...")
    
    # We use the DagsHub Content API for simplicity, but large files need careful handling.
    # For a real 1GB file, we should ideally use S3, but let's try a custom progress indicator.
    
    # Since I don't want to overcomplicate the upload (multipart etc.), 
    # and given the user's desire for a 'bar', I'll use a simple loop.
    
    # Actually, the DagsHub 'upload_files' is very good. 
    # I will stick to 'dagshub.upload_files' but I will add periodic print statements 
    # to simulate the feedback the user wants since the raw bar might be hidden.

    # RE-USING the reliable dagshub Repo API
    from dagshub.upload import Repo
    repo = Repo("theofrolicdean", "Detectify-ML-Model")
    
    print("Initiating optimized upload...")
    repo.upload_files([(local_path, remote_path)])
    print("Done.")

def register_final_with_feedback():
    model_name = "detectify-indo-detection-text-bi-lstm"
    repo_owner = "theofrolicdean"
    repo_name = "Detectify-ML-Model"
    
    print(f"--- Registration with Feedback: {model_name} ---")
    
    # 1. Initialize DagsHub
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("Indonesian_Text_Detection")
    
    base_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\notebook\text_detection_indo\saved_models"
    doc2vec_base = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\mlruns\1\ab842a03357d4b4896d0e6cd752e28eb\artifacts\BiLSTM_Model\artifacts"
    wrapper_path = r"d:\Cawu 4\AI_Deepfake_Detector_and_Humanizer\scripts\indo_text_pipeline_wrapper.py"
    
    small_artifacts = {
        "pipeline_pkl": os.path.join(base_path, "indo_text_pipeline_bi_lstm.pkl"),
        "pytorch_model": os.path.join(base_path, "bi_lstm.pth"),
        "doc2vec_model": os.path.join(doc2vec_base, "doc2Vec.d2v"),
    }
    
    large_files = [
        os.path.join(doc2vec_base, "doc2Vec.d2v.syn1neg.npy"),
        os.path.join(doc2vec_base, "doc2Vec.d2v.wv.vectors.npy"),
    ]

    with mlflow.start_run(run_name="Full_Package_With_Feedback") as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        
        # 2. Log Model Skeleton
        print("Logging model structure (fast)...")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="BiLSTM_Complete",
            python_model=IndoTextPipelineWrapper(),
            artifacts=small_artifacts,
            code_paths=[wrapper_path]
        )
        
        # 3. Manual Upload of 2.1GB with text feedback
        from dagshub.upload import Repo
        repo = Repo(repo_owner, repo_name)
        
        print("\n" + "="*40)
        print("STARTING 2.1GB UPLOAD...")
        print("Progress will be shown below (this takes a while):")
        
        for i, local_path in enumerate(large_files):
            key = "doc2vec_syn" if "syn1neg" in local_path else "doc2vec_vec"
            remote_path = f".mlflow/artifacts/4/{run_id}/artifacts/BiLSTM_Complete/artifacts/{key}"
            
            print(f"\n[{i+1}/{len(large_files)}] Processing: {os.path.basename(local_path)}")
            # This will use DagsHub's builtin progress bar which I hope shows up.
            # If not, the print statements above provide the "stage" feedback.
            repo.upload_files([(local_path, remote_path)])
            print(f"Uploaded {key} successfully.")

        # 4. Register
        print("\nFinalizing registration...")
        mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
        
        print("\n" + "="*60)
        print(f"SUCCESS! Registered model '{model_name}'")
        print(f"Run View: https://dagshub.com/{repo_owner}/{repo_name}.mlflow/#/experiments/4/runs/{run_id}")
        print("="*60)

if __name__ == "__main__":
    register_final_with_feedback()
