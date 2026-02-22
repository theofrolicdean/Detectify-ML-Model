"""
Script: delete_all_versions.py
Delete ALL versions of detectify-indo-text-bi-lstm from DagsHub MLflow registry.
"""

import mlflow
import dagshub

REGISTERED_MODEL = "detectify-indo-text-bi-lstm"

print(f"[1] Connecting to DagsHub...")
dagshub.init(repo_owner='theofrolicdean', repo_name='Detectify-ML-Model', mlflow=True)
print(f"    [OK] {mlflow.get_tracking_uri()}")

client = mlflow.tracking.MlflowClient()

print(f"\n[2] Fetching all versions of '{REGISTERED_MODEL}'...")
versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
versions_sorted = sorted(versions, key=lambda v: int(v.version))

if not versions_sorted:
    print("    No versions found.")
else:
    print(f"    Found {len(versions_sorted)} version(s): {[v.version for v in versions_sorted]}")
    print(f"\n[3] Deleting all versions...")
    for v in versions_sorted:
        try:
            client.delete_model_version(name=REGISTERED_MODEL, version=v.version)
            print(f"    [OK] Deleted v{v.version}")
        except Exception as e:
            print(f"    [SKIP] v{v.version} — {e}")

print(f"\n[SUCCESS] All versions of '{REGISTERED_MODEL}' deleted.")
