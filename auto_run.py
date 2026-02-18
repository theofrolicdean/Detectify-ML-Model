import subprocess
import os
import sys

def run_script(script_path):
    print(f"--- Running {script_path} ---")
    try:
        # Use sys.executable to ensure we use the same environment
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stdout)
        print(e.stderr)

def main():
    scripts = [
        "scripts/eval_audio.py",
        "scripts/eval_image.py",
        "scripts/eval_text.py"
    ]

    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Script {script} not found, skipping.")

    print("All evaluations complete. Check DagsHub for results.")

if __name__ == "__main__":
    main()
