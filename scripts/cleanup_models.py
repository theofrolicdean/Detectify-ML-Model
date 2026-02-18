import mlflow
import dagshub
import os
import sys

def cleanup(model_name):
    dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
    client = mlflow.tracking.MlflowClient()
    
    try:
        print(f"Attempting to delete model: {model_name}")
        client.delete_registered_model(name=model_name)
        print(f"Successfully deleted {model_name}")
    except Exception as e:
        print(f"Note: Could not delete {model_name} (it may not exist yet or error: {e})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cleanup(sys.argv[1])
    else:
        print("Please provide a model name to delete.")
