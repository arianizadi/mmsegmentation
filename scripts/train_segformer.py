#!/usr/bin/env python
import subprocess
import os
from datetime import datetime

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"segformer_{timestamp}.log")

cmd = [
    "python", "tools/train.py",
    "configs/segformer/segformer_mit-b5_1xb1-160k_railsem19-576x1024.py"
]

with open(log_file, "w") as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    process.wait()

print(f"Training completed. Log saved to {log_file}")