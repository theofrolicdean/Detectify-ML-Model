import os
import time
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
MODEL_DIR = "notebook/text_detection/saved_models"
DATA_PATH = "notebook/text_detection/datasets/processed_combined_ai_human.csv"

def evaluate():
    if os.environ.get("OFFLINE_DEBUG") == "1":
        print("Offline mode enabled. Skipping DagsHub and MLflow.")
        models_to_evaluate = [
            {"model": "xgb_pipeline.pkl", "name": "detectify-text-en-xgboost"},
            {"model": "log_reg_pipeline.pkl", "name": "detectify-text-en-logreg"}
        ]
        for m_info in models_to_evaluate:
            run_eval_logic(m_info)
        return

    try:
        dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    except Exception as e:
        print(f"Warning: DagsHub initialization failed: {e}. Running locally.")
    
    # We'll evaluate both if possible, or just the main one
    models_to_evaluate = [
        {"model": "xgb_pipeline.pkl", "name": "detectify-text-en-xgboost"},
        {"model": "log_reg_pipeline.pkl", "name": "detectify-text-en-logreg"}
    ]

    for m_info in models_to_evaluate:
        try:
            with mlflow.start_run(run_name=m_info["name"]):
                run_eval_logic(m_info)
        except Exception as e:
            print(f"Warning: MLflow start_run failed for {m_info['name']}: {e}. Running locally.")
            run_eval_logic(m_info)

def run_eval_logic(m_info):
    m_path = os.path.join(MODEL_DIR, m_info["model"])

    if not os.path.exists(m_path):
        print(f"Skipping {m_info['name']}, file not found.")
        return

    # Load unified pipeline
    pipeline = joblib.load(m_path)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    # Use a small subset for quick evaluation if the file is massive
    df = df.sample(n=min(100, len(df)), random_state=42)
    
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    latencies = []
    all_preds = []

    for text in X:
        start_time = time.time()
        # Direct prediction with pipeline
        pred = pipeline.predict([text])[0]
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        all_preds.append(pred)

    if not all_preds:
        print(f"No samples found for {m_info['name']}.")
        return

    # Metrics
    acc = accuracy_score(y, all_preds)
    prec = precision_score(y, all_preds, zero_division=0)
    rec = recall_score(y, all_preds, zero_division=0)
    f1 = f1_score(y, all_preds, zero_division=0)

    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p95_latency = np.percentile(latencies, 95)

    print(f"Model: {m_info['name']} - Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"P50 Latency: {p50_latency:.2f}ms")
    print(f"P90 Latency: {p90_latency:.2f}ms")
    print(f"P95 Latency: {p95_latency:.2f}ms")

    try:
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "p50_latency_ms": p50_latency,
            "p90_latency_ms": p90_latency,
            "p95_latency_ms": p95_latency
        })
        
        # Model Logging & Registration
        if os.environ.get("OFFLINE_DEBUG") != "1":
            print(f"Registering {m_info['name']} (Pipeline) to DagsHub Model Registry...")
            # Log the whole pipeline object
            mlflow.sklearn.log_model(
                sk_model=pipeline, 
                artifact_path="model",
                registered_model_name=m_info["name"]
            )
            print(f"Successfully registered {m_info['name']}.")
    except Exception as e:
        if os.environ.get("OFFLINE_DEBUG") != "1":
            print(f"Warning: MLflow logging/registration failed for {m_info['name']}: {e}")

if __name__ == "__main__":
    evaluate()
