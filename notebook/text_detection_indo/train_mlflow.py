import os
import time
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.keras
import mlflow.sklearn
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.models import load_model
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "Text_Indo_AI_vs_Human"

DATA_PATH = "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv"
BILSTM_PATH = "notebook/text_detection_indo/saved_models/bi_lstm.h5"
DOC2VEC_PATH = "notebook/text_detection_indo/saved_models/doc2Vec.d2v"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

X_text = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

def compute_metrics(y_true, y_pred, y_prob, latencies):
    latencies = np.array(latencies)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "P50_latency_ms": np.percentile(latencies, 50),
        "P90_latency_ms": np.percentile(latencies, 90),
        "P95_latency_ms": np.percentile(latencies, 95),
        "throughput_samples_per_sec": len(y_true) / (latencies.sum() / 1000)
    }

def evaluate_bilstm():
    print("Evaluating BiLSTM...")
    model = load_model(BILSTM_PATH)
    start = time.time()
    y_probs = model.predict(X_val)
    end = time.time()
    total_latency = (end - start) * 1000
    latencies = [total_latency]
    y_preds = (y_probs > 0.5).astype(int).flatten()
    metrics = compute_metrics(y_val, y_preds, y_probs.flatten(), latencies)
    return model, metrics

def evaluate_doc2vec():
    print("Evaluating Doc2Vec + Logistic Regression...")
    doc2vec_model = Doc2Vec.load(DOC2VEC_PATH)
    # Convert text to vectors
    X_train_vec = [doc2vec_model.infer_vector(text.split()) for text in X_train]
    X_val_vec = [doc2vec_model.infer_vector(text.split()) for text in X_val]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    start = time.time()
    y_probs = clf.predict_proba(X_val_vec)[:, 1]
    y_preds = clf.predict(X_val_vec)
    end = time.time()
    total_latency = (end - start) * 1000
    latencies = [total_latency]
    metrics = compute_metrics(y_val, y_preds, y_probs, latencies)
    return clf, metrics

with mlflow.start_run(run_name="BiLSTM_Model"):
    model, metrics = evaluate_bilstm()

    mlflow.log_param("model_type", "BiLSTM")

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.keras.log_model(model, "model")

    print("BiLSTM logged successfully.")

with mlflow.start_run(run_name="Doc2Vec_LogReg_Model"):
    model, metrics = evaluate_doc2vec()

    mlflow.log_param("model_type", "Doc2Vec + LogisticRegression")

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.sklearn.log_model(model, "model")

    print("Doc2Vec model logged successfully.")
