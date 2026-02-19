import sys
import io
import os
import joblib
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from indo_text_pipeline_wrapper import IndoTextPreprocessor, Doc2VecTransformer, BiLSTMClassifierWrapper

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def create_and_log_pipeline():
    print("Initializing DagsHub/MLflow...", flush=True)
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow")
    
    MODEL_DIR = "notebook/text_detection_indo/saved_models"
    DOC2VEC_PATH = os.path.join(MODEL_DIR, "doc2Vec.d2v")
    BILSTM_PATH_H5 = os.path.join(MODEL_DIR, "bi_lstm.h5")
    # Using H5 as it's what we have pre-trained currently
    
    PIPELINE_OUT = os.path.join(MODEL_DIR, "indo_text_pipeline.pkl")

    print("Assembling Unified Pipeline...", flush=True)
    
    # 1. Instantiate components
    # We pass paths so they load during 'fit' (or we could load them here)
    preprocessor = IndoTextPreprocessor()
    transformer = Doc2VecTransformer(model_path=DOC2VEC_PATH)
    classifier = BiLSTMClassifierWrapper(model_path=BILSTM_PATH_H5)
    
    # 2. Build Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', transformer),
        ('classifier', classifier)
    ])
    
    # 3. "Fit" to load models (fit here just loads them as defined in fit methods)
    print("Loading models into pipeline...", flush=True)
    pipeline.fit(["test text"]) 
    
    # 4. Save pipeline locally
    print(f"Saving pipeline to {PIPELINE_OUT}...", flush=True)
    joblib.dump(pipeline, PIPELINE_OUT)
    
    # 5. Log to MLflow
    print("Logging pipeline to MLflow and Model Registry...", flush=True)
    with mlflow.start_run(run_name="Indo_Text_Unified_Pipeline_v1"):
        # Log requirements/parameters
        mlflow.log_param("pipeline_type", "Indo_BiLSTM_Pipeline")
        
        # Log the pipeline object
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="Indo_Text_Pipeline_BILSTM"
        )
        
        # Log individual files as artifacts for redundancy
        mlflow.log_artifact(PIPELINE_OUT, artifact_path="pipeline")
        mlflow.log_artifact(DOC2VEC_PATH, artifact_path="raw_models")
        mlflow.log_artifact(BILSTM_PATH_H5, artifact_path="raw_models")
        
        # Log associated doc2vec files
        for f in os.listdir(MODEL_DIR):
            if f.startswith("doc2Vec.d2v") and f != "doc2Vec.d2v":
                mlflow.log_artifact(os.path.join(MODEL_DIR, f), artifact_path="raw_models")

        print("Successfully logged and registered Indo Text Unified Pipeline.", flush=True)

if __name__ == "__main__":
    try:
        if not os.path.exists("notebook/text_detection_indo/saved_models"):
            print("Error: saved_models directory not found.")
        else:
            create_and_log_pipeline()
    except Exception as e:
        print(f"Critical Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
