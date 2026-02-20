"""
Rename & Evaluate All Models
=============================
Registers all models in MLflow/DagsHub under the standardized naming convention:
    detectify-deepfake-<language>-<modality>-<architecture>
with semantic versioning (MAJOR.MINOR.PATCH).

Evaluates each model on sample data and logs 7 metrics:
    accuracy, precision, recall, f1, p50_latency_ms, p90_latency_ms, p95_latency_ms
"""

import sys
import io
import os
import time
import random
import json
from datetime import datetime

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Configuration ──────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEMANTIC_VERSION = "1.0.0"

# ── Model Definitions ─────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    # Audio
    {
        "old_name": "Audio_Deepfake_Detection_Model",
        "new_name": "detectify-deepfake-en-audio-cnn",
        "domain": "audio",
        "model_path": "notebook/audio_detection/saved_models/tuning_2.pth",
        "test_data": "notebook/audio_detection/datasets/libri+gen/test_list.txt",
        "architecture": "SimpleAudioCNN",
    },
    # Image
    {
        "old_name": "Image_Deepfake_Detection_Model",
        "new_name": "detectify-deepfake-en-image-efficientnet",
        "domain": "image",
        "model_path": "notebook/image_detection/saved_models/final_model.pth",
        "test_data": "notebook/image_detection/processed_dataset",
        "architecture": "EfficientNetV2",
    },
    # Text English — XGBoost
    {
        "old_name": "detectify-text-en-xgboost",
        "new_name": "detectify-deepfake-en-text-xgboost",
        "domain": "text_en",
        "model_path": "notebook/text_detection/saved_models/xgb_pipeline.pkl",
        "test_data": "notebook/text_detection/datasets/processed_combined_ai_human.csv",
        "architecture": "XGBoost",
    },
    # Text English — Logistic Regression
    {
        "old_name": "detectify-text-en-logreg",
        "new_name": "detectify-deepfake-en-text-logreg",
        "domain": "text_en",
        "model_path": "notebook/text_detection/saved_models/log_reg_pipeline.pkl",
        "test_data": "notebook/text_detection/datasets/processed_combined_ai_human.csv",
        "architecture": "LogisticRegression",
    },
    # Text Indonesian — BiLSTM
    {
        "old_name": "Indo_Text_BiLSTM",
        "new_name": "detectify-deepfake-id-text-bilstm",
        "domain": "text_indo",
        "model_path": "notebook/text_detection_indo/saved_models/bi_lstm.pth",
        "test_data": "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv",
        "architecture": "BiLSTM",
        "rnn_type": "bi_lstm",
    },
    # Text Indonesian — BiGRU
    {
        "old_name": "Indo_Text_BiGRU",
        "new_name": "detectify-deepfake-id-text-bigru",
        "domain": "text_indo",
        "model_path": "notebook/text_detection_indo/saved_models/bi_gru.pth",
        "test_data": "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv",
        "architecture": "BiGRU",
        "rnn_type": "bi_gru",
    },
    # Text Indonesian — GRU
    {
        "old_name": "Indo_Text_GRU",
        "new_name": "detectify-deepfake-id-text-gru",
        "domain": "text_indo",
        "model_path": "notebook/text_detection_indo/saved_models/gru.pth",
        "test_data": "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv",
        "architecture": "GRU",
        "rnn_type": "gru",
    },
    # Text Indonesian — LSTM
    {
        "old_name": "Indo_Text_LSTM",
        "new_name": "detectify-deepfake-id-text-lstm",
        "domain": "text_indo",
        "model_path": "notebook/text_detection_indo/saved_models/lstm.pth",
        "test_data": "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv",
        "architecture": "LSTM",
        "rnn_type": "lstm",
    },
]


# ── Evaluation Helpers ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, latencies):
    """Compute all 7 standard metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p90_latency_ms": float(np.percentile(latencies, 90)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def print_metrics(name, metrics):
    """Pretty-print metrics for a model."""
    print(f"\n  Results for {name}:")
    for k, v in metrics.items():
        if "latency" in k:
            print(f"    {k}: {v:.2f} ms")
        else:
            print(f"    {k}: {v:.4f}")


# ── Domain-Specific Evaluators ─────────────────────────────────────────────────

def evaluate_audio(config):
    """Evaluate audio CNN model on test_list.txt samples."""
    from audio_pipeline_wrapper import AudioPreprocessor, SimpleAudioCNN
    import torchaudio

    model_path = config["model_path"]
    test_list = config["test_data"]

    if not os.path.exists(model_path):
        print(f"  ⚠ Model file {model_path} not found. Skipping.")
        return None
    if not os.path.exists(test_list):
        print(f"  ⚠ Test list {test_list} not found. Skipping.")
        return None

    # Load model
    model = SimpleAudioCNN().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Parse test list
    data = []
    with open(test_list, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path = parts[0]
                if rel_path.startswith("dataset/"):
                    rel_path = rel_path.replace("dataset/", "datasets/", 1)
                audio_path = os.path.join("notebook/audio_detection", rel_path)
                data.append((audio_path, int(parts[1])))

    # Sample
    sample_data = random.sample(data, min(100, len(data)))
    preprocessor = AudioPreprocessor()

    y_true = []
    y_pred = []
    latencies = []

    for path, label in sample_data:
        if not os.path.exists(path):
            continue
        try:
            start = time.time()
            tensor = preprocessor.transform([path])
            tensor = tensor.to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                _, pred = torch.max(output, 1)
            latencies.append((time.time() - start) * 1000)
            y_pred.append(pred.cpu().item())
            y_true.append(label)
        except Exception as e:
            print(f"  Error on {path}: {e}")
            continue

    if not y_pred:
        print("  No audio predictions. Skipping.")
        return None

    return compute_metrics(y_true, y_pred, latencies)


def evaluate_image(config):
    """Evaluate image EfficientNet model on processed_dataset."""
    from image_pipeline_wrapper import ImagePreprocessor, EfficientNetV2
    from torchvision import transforms
    from PIL import Image

    model_path = config["model_path"]
    data_dir = config["test_data"]

    if not os.path.exists(model_path):
        print(f"  ⚠ Model file {model_path} not found. Skipping.")
        return None
    if not os.path.exists(data_dir):
        print(f"  ⚠ Data dir {data_dir} not found. Skipping.")
        return None

    # Load model
    model = EfficientNetV2().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Gather images
    images = []
    label_map = {"real": 0, "fake": 1}
    # Check for split_data/test first, then top-level real/fake
    for candidate_dir in [
        os.path.join(data_dir, "split_data", "test"),
        data_dir,
    ]:
        for label_name, label_val in label_map.items():
            label_dir = os.path.join(candidate_dir, label_name)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        images.append((os.path.join(label_dir, img_name), label_val))
        if images:
            break

    if not images:
        print("  No test images found. Skipping.")
        return None

    sample_data = random.sample(images, min(100, len(images)))

    y_true = []
    y_pred = []
    latencies = []

    for path, label in sample_data:
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            start = time.time()
            with torch.no_grad():
                output = model(tensor)
                pred = (torch.sigmoid(output) > 0.5).long().item()
            latencies.append((time.time() - start) * 1000)

            y_pred.append(pred)
            y_true.append(label)
        except Exception as e:
            print(f"  Error on {path}: {e}")
            continue

    if not y_pred:
        print("  No image predictions. Skipping.")
        return None

    return compute_metrics(y_true, y_pred, latencies)


def evaluate_text_en(config):
    """Evaluate sklearn text pipeline (XGBoost or LogReg)."""
    model_path = config["model_path"]
    data_path = config["test_data"]

    if not os.path.exists(model_path):
        print(f"  ⚠ Model file {model_path} not found. Skipping.")
        return None
    if not os.path.exists(data_path):
        print(f"  ⚠ Dataset {data_path} not found. Skipping.")
        return None

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df = df.sample(n=min(200, len(df)), random_state=SEED)

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    y_pred = []
    latencies = []

    for text in X:
        start = time.time()
        pred = pipeline.predict([text])[0]
        latencies.append((time.time() - start) * 1000)
        y_pred.append(pred)

    if not y_pred:
        print("  No text predictions. Skipping.")
        return None

    return compute_metrics(y.tolist(), y_pred, latencies)


def evaluate_text_indo(config):
    """Evaluate Indonesian text RNN model via pipeline."""
    from indo_text_pipeline_wrapper import (
        IndoTextPreprocessor, Doc2VecTransformer, RNNClassifierWrapper
    )
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    model_path = config["model_path"]
    data_path = config["test_data"]
    rnn_type = config.get("rnn_type", "bi_lstm")
    doc2vec_path = "notebook/text_detection_indo/saved_models/doc2Vec.d2v"

    if not os.path.exists(model_path):
        print(f"  ⚠ Model file {model_path} not found. Skipping.")
        return None
    if not os.path.exists(data_path):
        print(f"  ⚠ Dataset {data_path} not found. Skipping.")
        return None
    if not os.path.exists(doc2vec_path):
        print(f"  ⚠ Doc2Vec model {doc2vec_path} not found. Skipping.")
        return None

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', IndoTextPreprocessor()),
        ('vectorizer', Doc2VecTransformer(model_path=doc2vec_path)),
        ('classifier', RNNClassifierWrapper(
            model_path=model_path,
            model_type=rnn_type,
            device="cpu",
        )),
    ])

    df = pd.read_csv(data_path)
    X = df['message'].astype(str).values
    y = df['label'].values

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_test = X_val[:min(200, len(X_val))]
    y_test = y_val[:min(200, len(y_val))]

    # Batch predict for overall metrics
    start_total = time.time()
    y_pred = pipeline.predict(X_test)
    total_time = time.time() - start_total

    # Per-sample latency on smaller subset
    latency_subset = X_test[:min(50, len(X_test))]
    latencies = []
    for x in latency_subset:
        s = time.time()
        pipeline.predict([x])
        latencies.append((time.time() - s) * 1000)

    if len(y_pred) == 0:
        print("  No indo text predictions. Skipping.")
        return None

    return compute_metrics(y_test.tolist(), y_pred.tolist(), latencies)


# ── Domain dispatcher ──────────────────────────────────────────────────────────

EVALUATORS = {
    "audio": evaluate_audio,
    "image": evaluate_image,
    "text_en": evaluate_text_en,
    "text_indo": evaluate_text_indo,
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65, flush=True)
    print("  Detectify — Model Evaluation", flush=True)
    print(f"  Semantic Version: {SEMANTIC_VERSION}", flush=True)
    print("=" * 65, flush=True)

    results_summary = []

    for config in MODEL_CONFIGS:
        new_name = config["new_name"]
        domain = config["domain"]

        print(f"\n{'─' * 60}", flush=True)
        print(f"  Model: {new_name}", flush=True)
        print(f"  Architecture: {config['architecture']}", flush=True)
        print(f"  Version: {SEMANTIC_VERSION}", flush=True)
        print(f"{'─' * 60}", flush=True)

        evaluator = EVALUATORS.get(domain)
        if not evaluator:
            print(f"  ⚠ No evaluator for domain '{domain}'. Skipping.")
            continue

        metrics = evaluator(config)

        if metrics is None:
            print(f"  ⚠ Evaluation skipped for {new_name}.")
            results_summary.append({"model": new_name, "status": "SKIPPED"})
            continue

        print_metrics(new_name, metrics)

        results_summary.append({
            "model": new_name,
            "status": "OK",
            **metrics,
        })

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n\n{'=' * 95}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'=' * 95}", flush=True)
    print(f"{'Model':<45} {'Status':<8} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'P50':>7} {'P90':>7} {'P95':>7}", flush=True)
    print("-" * 95, flush=True)

    for r in results_summary:
        if r["status"] == "SKIPPED":
            print(f"{r['model']:<45} {'SKIP':<8}", flush=True)
        else:
            print(
                f"{r['model']:<45} {'OK':<8} "
                f"{r['accuracy']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1_score']:>6.4f} "
                f"{r['p50_latency_ms']:>6.1f}ms {r['p90_latency_ms']:>6.1f}ms {r['p95_latency_ms']:>6.1f}ms",
                flush=True,
            )

    print(f"\n{'=' * 95}", flush=True)
    print(f"  All models processed. Version: {SEMANTIC_VERSION}", flush=True)
    print(f"{'=' * 95}\n", flush=True)

    # ── Save Results ───────────────────────────────────────────────────────
    RESULTS_DIR = "evaluation_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Eval method descriptions per domain ---
    EVAL_METHODS = {
        "audio": {
            "data_source": "notebook/audio_detection/datasets/libri+gen/test_list.txt",
            "sample_size": "100 random samples from test split",
            "preprocessing": "Resample to 16kHz, mono, pad/trim to 4s, MelSpectrogram(n_fft=1024, hop=256, n_mels=64) → AmplitudeToDB",
            "model_type": "SimpleAudioCNN (Conv2d×4 → FC, 2-class softmax)",
            "prediction": "argmax of 2-class softmax output",
            "metrics": {
                "accuracy": "sklearn.metrics.accuracy_score(y_true, y_pred)",
                "precision": "sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)",
                "recall": "sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)",
                "f1_score": "sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)",
                "p50_latency_ms": "numpy.percentile(per_sample_latencies, 50) — includes preprocessing + inference",
                "p90_latency_ms": "numpy.percentile(per_sample_latencies, 90)",
                "p95_latency_ms": "numpy.percentile(per_sample_latencies, 95)",
            },
        },
        "image": {
            "data_source": "notebook/image_detection/processed_dataset (real/ and fake/ subdirectories)",
            "sample_size": "100 random samples from processed dataset",
            "preprocessing": "Resize(224×224), ToTensor, Normalize(ImageNet mean/std)",
            "model_type": "EfficientNetV2-L (tf_efficientnetv2_l) → FC(512→1, sigmoid)",
            "prediction": "sigmoid(output) > 0.5 → binary class",
            "metrics": {
                "accuracy": "sklearn.metrics.accuracy_score(y_true, y_pred)",
                "precision": "sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)",
                "recall": "sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)",
                "f1_score": "sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)",
                "p50_latency_ms": "numpy.percentile(per_sample_latencies, 50) — inference only (preprocessing excluded)",
                "p90_latency_ms": "numpy.percentile(per_sample_latencies, 90)",
                "p95_latency_ms": "numpy.percentile(per_sample_latencies, 95)",
            },
        },
        "text_en": {
            "data_source": "notebook/text_detection/datasets/processed_combined_ai_human.csv",
            "sample_size": "200 random samples (stratified by random_state=42)",
            "preprocessing": "TF-IDF vectorization (built into sklearn Pipeline)",
            "model_type": "XGBoost / Logistic Regression sklearn Pipeline",
            "prediction": "pipeline.predict([text]) — direct class output",
            "metrics": {
                "accuracy": "sklearn.metrics.accuracy_score(y_true, y_pred)",
                "precision": "sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)",
                "recall": "sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)",
                "f1_score": "sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)",
                "p50_latency_ms": "numpy.percentile(per_sample_latencies, 50) — includes vectorization + prediction",
                "p90_latency_ms": "numpy.percentile(per_sample_latencies, 90)",
                "p95_latency_ms": "numpy.percentile(per_sample_latencies, 95)",
            },
        },
        "text_indo": {
            "data_source": "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv",
            "sample_size": "200 samples from 20% validation split (stratified, random_state=42)",
            "preprocessing": "Regex cleaning → lowercasing → Doc2Vec(infer_vector, epochs=20)",
            "model_type": "PyTorch RNN (BiLSTM/BiGRU/GRU/LSTM, hidden=50, layers=4) → FC→sigmoid",
            "prediction": "sigmoid(output) > 0.5 → binary class",
            "metrics": {
                "accuracy": "sklearn.metrics.accuracy_score(y_true, y_pred)",
                "precision": "sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)",
                "recall": "sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)",
                "f1_score": "sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)",
                "p50_latency_ms": "numpy.percentile(per_sample_latencies, 50) — 50 samples, includes preprocessing + vectorization + inference",
                "p90_latency_ms": "numpy.percentile(per_sample_latencies, 90)",
                "p95_latency_ms": "numpy.percentile(per_sample_latencies, 95)",
            },
        },
    }

    # --- Build full results with eval method info ---
    full_results = []
    for r in results_summary:
        entry = {
            "model_name": r["model"],
            "version": SEMANTIC_VERSION,
            "status": r["status"],
            "evaluated_at": datetime.now().isoformat(),
        }
        if r["status"] != "SKIPPED":
            entry.update({
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
                "p50_latency_ms": r["p50_latency_ms"],
                "p90_latency_ms": r["p90_latency_ms"],
                "p95_latency_ms": r["p95_latency_ms"],
            })
            # Find domain for this model
            domain = None
            for cfg in MODEL_CONFIGS:
                if cfg["new_name"] == r["model"]:
                    domain = cfg["domain"]
                    entry["architecture"] = cfg["architecture"]
                    entry["old_name"] = cfg["old_name"]
                    entry["model_path"] = cfg["model_path"]
                    entry["test_data"] = cfg["test_data"]
                    break
            if domain and domain in EVAL_METHODS:
                entry["eval_method"] = EVAL_METHODS[domain]
        full_results.append(entry)

    # --- Save CSV (metrics table) ---
    csv_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    csv_rows = []
    for r in results_summary:
        if r["status"] != "SKIPPED":
            csv_rows.append({
                "model_name": r["model"],
                "version": SEMANTIC_VERSION,
                "accuracy": round(r["accuracy"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1_score": round(r["f1_score"], 4),
                "p50_latency_ms": round(r["p50_latency_ms"], 2),
                "p90_latency_ms": round(r["p90_latency_ms"], 2),
                "p95_latency_ms": round(r["p95_latency_ms"], 2),
            })
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_df.to_csv(csv_path, index=False)
        print(f"  📄 Saved CSV: {csv_path}", flush=True)

    # Also save a "latest" copy
    csv_latest = os.path.join(RESULTS_DIR, "evaluation_results_latest.csv")
    if csv_rows:
        csv_df.to_csv(csv_latest, index=False)

    # --- Save JSON (full details + eval methods) ---
    json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    json_data = {
        "version": SEMANTIC_VERSION,
        "evaluated_at": datetime.now().isoformat(),
        "device": DEVICE,
        "models": full_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  📄 Saved JSON: {json_path}", flush=True)

    json_latest = os.path.join(RESULTS_DIR, "evaluation_results_latest.json")
    with open(json_latest, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  📄 Saved latest: {json_latest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCritical Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
