import subprocess
import time
import sys
import webbrowser
import os
import mlflow

MLFLOW_PORT = "5000"
MLFLOW_HOST = "127.0.0.1"
VENV = "ML_CAWU_4_PROJECT"

def start_mlflow():
    print("Starting MLflow server...")

    mlflow_process = subprocess.Popen([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", MLFLOW_HOST,
        "--port", MLFLOW_PORT
    ])

    time.sleep(5)

    webbrowser.open(f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
    return mlflow_process


def run_script(script_path):
    print(f"Running: {script_path}")
    result = subprocess.run(["python -m", script_path])

    if result.returncode != 0:
        print(f"Error running {script_path}")
        sys.exit(1)

def run_image():
    run_script("notebook/image_detection/train_mlflow.py")

def run_text():
    run_script("notebook/text_detection/train_mlflow.py")

def run_text_indo():
    run_script("notebook/text_detection_indo/train_mlflow.py")

if __name__ == "__main__":
    print("AI-Human Detection MLOps Pipeline")
    print("1 - Run Image Model")
    print("2 - Run Text Model")
    print("3 - Run Indonesian Text Model")
    print("4 - Run All Models")

    choice = input("Select option: ")
    mlflow_process = start_mlflow()
    if choice == "1":
        run_image()
    elif choice == "2":
        run_text()
    elif choice == "3":
        run_text_indo()
    elif choice == "4":
        run_image()
        run_text()
        run_text_indo()
    else:
        print("Invalid choice.")
    print("Pipeline finished.")
