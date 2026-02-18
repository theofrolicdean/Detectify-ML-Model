import os
import joblib
from sklearn.pipeline import Pipeline

MODEL_DIR = "notebook/text_detection/saved_models"

def convert():
    configs = [
        {
            "clf": "log_reg_model.pkl",
            "vec": "tfidf_vectorizer.pkl",
            "out": "log_reg_pipeline.pkl"
        },
        {
            "clf": "xgb_clf_model.pkl",
            "vec": "xgb_tfidf_vectorizer.pkl",
            "out": "xgb_pipeline.pkl"
        }
    ]

    for cfg in configs:
        clf_path = os.path.join(MODEL_DIR, cfg["clf"])
        vec_path = os.path.join(MODEL_DIR, cfg["vec"])
        out_path = os.path.join(MODEL_DIR, cfg["out"])

        if os.path.exists(clf_path) and os.path.exists(vec_path):
            print(f"Converting {cfg['clf']} and {cfg['vec']} to {cfg['out']}...")
            clf = joblib.load(clf_path)
            vec = joblib.load(vec_path)

            pipe = Pipeline([
                ('tfidf', vec),
                ('clf', clf)
            ])

            joblib.dump(pipe, out_path)
            print(f"Successfully saved pipeline to {out_path}")
        else:
            print(f"Skipping {cfg['out']}, input files not found.")

if __name__ == "__main__":
    convert()
