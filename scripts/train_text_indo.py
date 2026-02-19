import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import dagshub
import mlflow
import mlflow.keras
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tqdm import tqdm

import sys
import io

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/theofrolicdean/Detectify-ML-Model.mlflow")

# Constants
SEED = 42
INPUT_DIM = 1000
EPOCHS = 20  # For Doc2Vec
DEEP_EPOCHS = 6 # For Bi-LSTM
DEEP_BATCH_SIZE = 15
UNIT = 50
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-4
DATA_PATH = "notebook/text_detection_indo/datasets/combined_ai_human_indonesia.csv"
MODEL_DIR = "notebook/text_detection_indo/saved_models"
DOC2VEC_PATH = os.path.join(MODEL_DIR, "doc2Vec.d2v")
BILSTM_PATH = os.path.join(MODEL_DIR, "bi_lstm.h5")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Set seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)

def text_preprocessing(text):
    text = str(text)
    text = text.replace('-', ' ')
    text = re.sub(r'[\r\xa0\t]', '', text)
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\b\w*\.com\w*\b', '', text)
    text = re.sub(r'\[.*?\]|\(.*?\d\}|\{.*?\}', '', text)
    text = re.sub(r'\b(\w+)/(\w+)\b', r'\1 atau \2', text)
    text = re.sub(r'@[A-Za-z0-9]+|#[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    text = text.strip(' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def data_preprocessing(df):
    df["label"] = df["label"].apply(lambda x: 1.0 if x == 1 else 0.0) # Assuming 1 is AI, 0 is Human based on notebook analysis or mapping
    # Actually notebook says: "ai" if label == 1.0 else "human". Then maps "ai":1, "human":0.
    # So original label 1.0 -> ai -> 1. Original label 0.0 -> human -> 0.
    # So we can just keep it as is if it's already 1/0, or map 'ai'/'human' to 1/0.
    # Let's handle the raw CSV which has 1/0 or ai/human.
    
    # Reloading logic from notebook to be safe:
    # df["label"] = df["label"].apply(convert_label) -> ai/human
    # X, y = df["message"], df["label"].map({"ai":1, "human":0}).values
    
    # Let's simplify.
    df["message"] = df["message"].apply(text_preprocessing)
    return df

def construct_tagged_document(X, y):
    tagged_data = [TaggedDocument(words=simple_preprocess(d), tags=[y[i], i]) 
                   for i, d in enumerate(X)]
    return tagged_data

def train_doc2vec(train_tagged, val_tagged):
    data_tagged = train_tagged + val_tagged
    
    if os.path.exists(DOC2VEC_PATH):
        print(f"Loading Doc2Vec model from {DOC2VEC_PATH}")
        d2v = Doc2Vec.load(DOC2VEC_PATH)
    else:
        print("Training Doc2Vec model...")
        cores = os.cpu_count()
        d2v = Doc2Vec(
            dm=1,
            vector_size=INPUT_DIM,
            negative=1,
            hs=0,
            min_count=0,
            workers=cores
        )
        d2v.build_vocab(data_tagged)
        
        for epoch in tqdm(range(EPOCHS), desc="Training Doc2Vec"):
            d2v.train(
                data_tagged,
                total_examples=d2v.corpus_count,
                epochs=1
            )
            d2v.alpha -= 0.002
            d2v.min_alpha = d2v.alpha
        
        d2v.save(DOC2VEC_PATH)
        print(f"Doc2Vec model saved to {DOC2VEC_PATH}")
        
        # Log to MLflow
        mlflow.log_artifact(DOC2VEC_PATH, artifact_path="models")

    return d2v

def d2v_vector(d2v, data_tagged, desc="Infer vectors"):
    labels = []
    features = []

    for doc in tqdm(data_tagged, desc=desc):
        labels.append(doc.tags[0])
        vec = d2v.infer_vector(doc.words, epochs=20)
        features.append(vec)

    features = np.array(features)
    return features, labels

def transform_data(features, labels):
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    labels = np.array(labels).reshape((-1,1))
    return features, labels

class MLflowLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)

def build_and_train_model(X_train, y_train, X_val, y_val):
    print("Building Bi-LSTM model...")
    model = Sequential(name="BI_LSTM")
    model.add(Bidirectional(LSTM(UNIT, return_sequences=True), input_shape=(INPUT_DIM, 1)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(UNIT, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(UNIT, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(UNIT)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
    
    model.summary()
    
    checkpoint = ModelCheckpoint(BILSTM_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
    
    print("Training Bi-LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=DEEP_EPOCHS,
        batch_size=DEEP_BATCH_SIZE,
        verbose=1,
        callbacks=[checkpoint, MLflowLogger()]
    )
    
    return model, history

def main():
    print("Starting Text Detection Indo Training Pipeline...")
    
    with mlflow.start_run(run_name="Text_Detection_Indo_BiLSTM"):
        # Log parameters
        mlflow.log_params({
            "seed": SEED,
            "input_dim": INPUT_DIM,
            "doc2vec_epochs": EPOCHS,
            "deep_epochs": DEEP_EPOCHS,
            "batch_size": DEEP_BATCH_SIZE,
            "unit": UNIT,
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": LEARNING_RATE
        })
        
        # 1. Load Data
        print(f"Loading data from {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
            
        df = pd.read_csv(DATA_PATH)
        
        # 2. Preprocessing
        print("Preprocessing data...")
        # Filtering based on length as done in notebook (optional but good for consistency)
        df = df[df["message"].str.len() > 20]
        df = df[df["message"].str.len() < 2000]
        df = df.reset_index(drop=True)
        
        df = data_preprocessing(df)
        
        # Convert label: notebook logic: "ai" if label==1 else "human". Then map "ai":1, "human":0.
        # This implies label 1 in CSV is AI, label 0 is Human.
        # Let's just ensure we map properly without string conversion overhead if possible,
        # but sticking to notebook logic ensures correctness.
        df["label_str"] = df["label"].apply(lambda x: "ai" if x == 1 else "human")
        
        # Sample and split
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        X = df["message"]
        y = df["label_str"].map({"ai": 1, "human": 0}).values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        
        # 3. Vectorization (Doc2Vec)
        print("Vectorizing data with Doc2Vec...")
        # We need to construct tagged documents using indexes relative to the split arrays
        # But construct_tagged_document uses `enumerate(X)` so it uses 0..N index.
        # Logic in notebook: 
        # train_tagged = construct_tagged_document(X_train, y_train)
        # val_tagged = construct_tagged_document(X_val, y_val)
        
        # Important: X_train is a Series, so simple_preprocess(d) works on value.
        train_tagged = construct_tagged_document(X_train.tolist(), y_train)
        val_tagged = construct_tagged_document(X_val.tolist(), y_val)
        
        d2v = train_doc2vec(train_tagged, val_tagged)
        
        X_train_vec, y_train_vec = d2v_vector(d2v, train_tagged, desc="Inferring Train Vectors")
        X_val_vec, y_val_vec = d2v_vector(d2v, val_tagged, desc="Inferring Val Vectors")
        
        X_train_reshaped, y_train_reshaped = transform_data(X_train_vec, y_train) # y_train is already values
        X_val_reshaped, y_val_reshaped = transform_data(X_val_vec, y_val)
        
        # 4. Train Model
        model, history = build_and_train_model(X_train_reshaped, y_train_reshaped, X_val_reshaped, y_val_reshaped)
        
        # 5. Log Model
        print("Logging model to MLflow...")
        mlflow.keras.log_model(model, "model")
        
        # Also log the specific h5 file as artifact
        if os.path.exists(BILSTM_PATH):
             mlflow.log_artifact(BILSTM_PATH, artifact_path="saved_models")
             
        print("Training complete. Models saved and logged.")

if __name__ == "__main__":
    main()
