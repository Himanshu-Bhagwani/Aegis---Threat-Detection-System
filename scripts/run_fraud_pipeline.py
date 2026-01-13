# scripts
#!/usr/bin/env python3
# scripts/run_fraud_pipeline.py
import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))  # cd to project root
import subprocess, sys

PY = sys.executable
SCRIPT = "src/fraud/aegis_fraud_pipeline.py"

# Example args: change as needed
args = [
    PY, SCRIPT,
    "--raw-dir", "data/raw/fraud",
    "--out-dir", "data/processed/fraud",
    # Uncomment to inject synthetic frauds:
    # "--generate-synthetic", "--n-synth", "3000",
    # Uncomment to use SMOTE during training:
    # "--smote",
    "--train"
]

print("Running:", " ".join(args))
subprocess.check_call(args)
