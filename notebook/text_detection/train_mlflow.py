import time
import json
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
EXPERIMENT_NAME = "Text_AI_vs_Human"
MODEL_PATH = "notebook/text_detection/saved_models/log_reg_model.pkl"
VECTORIZER_PATH = "notebook/text_detection/saved_models/tfidf_vectorizer.pkl"
DATA_PATH = "notebook/text_detection/datasets/processed_combined_ai_human.csv"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)
X_text = df["text"]
y = df["label"]
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

X = vectorizer.transform(X_text)

def evaluate():
    latencies = []

    start = time.time()
    y_probs = model.predict_proba(X)[:, 1]
    y_preds = model.predict(X)
    end = time.time()

    total_latency = (end - start) * 1000
    latencies.append(total_latency)

    latencies = np.array(latencies)

    return {
        "accuracy": accuracy_score(y, y_preds),
        "precision": precision_score(y, y_preds),
        "recall": recall_score(y, y_preds),
        "f1_score": f1_score(y, y_preds),
        "roc_auc": roc_auc_score(y, y_probs),
        "P50_latency_ms": np.percentile(latencies, 50),
        "P90_latency_ms": np.percentile(latencies, 90),
        "P95_latency_ms": np.percentile(latencies, 95),
        "throughput_samples_per_sec": len(y) / (total_latency / 1000)
    }

with mlflow.start_run():
    metrics = evaluate()
    mlflow.log_param("model_type", "LogisticRegression")
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlflow.sklearn.log_model(model, "model")
    print("Text model logged successfully.")
