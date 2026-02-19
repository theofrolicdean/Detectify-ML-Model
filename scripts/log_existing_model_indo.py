import sys
import io
import os
import dagshub
import mlflow
import tensorflow as tf
from gensim.models import Doc2Vec

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow")

MODEL_DIR = "notebook/text_detection_indo/saved_models"
DOC2VEC_PATH = os.path.join(MODEL_DIR, "doc2Vec.d2v")
BILSTM_PATH_H5 = os.path.join(MODEL_DIR, "bi_lstm.h5")
BILSTM_PATH_KERAS = os.path.join(MODEL_DIR, "bi_lstm.keras")

def log_existing_models():
    print("Logging existing models to MLflow...")
    
    with mlflow.start_run(run_name="Text_Detection_Indo_Existing_Models"):
        print("Run started.")
        mlflow.log_param("status", "pre-trained")
        
        # Log Doc2Vec model
        if os.path.exists(DOC2VEC_PATH):
            print(f"Logging Doc2Vec model from {DOC2VEC_PATH}...")
            mlflow.log_artifact(DOC2VEC_PATH, artifact_path="saved_models")
            print("Doc2Vec main file logged.")
            # Log associated numpy files if they exist
            for f in os.listdir(MODEL_DIR):
                if f.startswith("doc2Vec.d2v") and f != "doc2Vec.d2v":
                    print(f"Logging associated file: {f}...")
                    mlflow.log_artifact(os.path.join(MODEL_DIR, f), artifact_path="saved_models")
                    print(f"Logged {f}.")
        else:
            print(f"Warning: Doc2Vec model not found at {DOC2VEC_PATH}")

        # Log Bi-LSTM model (Check for both .h5 and .keras)
        model_path = None
        if os.path.exists(BILSTM_PATH_KERAS):
            model_path = BILSTM_PATH_KERAS
        elif os.path.exists(BILSTM_PATH_H5):
            model_path = BILSTM_PATH_H5
            
        if model_path:
            print(f"Logging Bi-LSTM model from {model_path}...")
            mlflow.log_artifact(model_path, artifact_path="saved_models") 
            print("Bi-LSTM file logged.")
            
            # Try to load and log as keras model to get metadata if compatible
            try:
                print("Attempting to load model as Keras model for structural logging...")
                model = tf.keras.models.load_model(model_path)
                mlflow.keras.log_model(model, "model")
                print("Bi-LSTM model loaded and logged as MLflow Keras model.")
            except Exception as e:
                print(f"Could not load/log model structure (might be version mismatch), logged file artifact only. Error: {e}")
        else:
             print("Warning: Bi-LSTM model not found in saved_models")

        print("Finished logging artifacts.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    if not os.path.exists(MODEL_DIR):
         print(f"Error: Model directory {MODEL_DIR} does not exist.")
    else:
        log_existing_models()
