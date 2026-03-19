#!/usr/bin/env python
"""
3-stage OCRNet-HR48 training pipeline: Mapillary → Cityscapes → RailSem19

Distributed training on 3x Quadro RTX 6000 (GPUs 4, 5, 6).
Matches HRNet paper protocol: SyncBN, effective batch 48, PolyLR power=0.9.

Stage 1: Train on Mapillary (66 classes, up to 500k iters)
Stage 2: Fine-tune on Cityscapes (19 classes, up to 120k iters) from best Stage 1 ckpt
Stage 3: Fine-tune on RailSem19 (19 classes, up to 120k iters) from best Stage 2 ckpt

All stages use EarlyStoppingHook so they stop automatically at convergence.
"""
import subprocess
import os
import glob
from datetime import datetime

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# GPU and distributed training settings
GPUS = "4,5,6"
NUM_GPUS = 3
PORT = 29500

STAGE1_CONFIG = "configs/ocrnet/ocrnet_hr48_stage1_mapillary-512x1024.py"
STAGE2_CONFIG = "configs/ocrnet/ocrnet_hr48_stage2_cityscapes-512x1024.py"
STAGE3_CONFIG = "configs/ocrnet/ocrnet_hr48_stage3_railsem19-576x1024.py"


def run_stage(config, load_from=None, stage_name="stage"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ocrnet_{stage_name}_{timestamp}.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS

    cfg_options = []
    if load_from:
        cfg_options += ["--cfg-options", f"load_from={load_from}"]

    cmd = [
        "bash", "tools/dist_train.sh",
        config,
        str(NUM_GPUS),
    ] + cfg_options

    print(f"[{stage_name}] Starting distributed training on GPUs {GPUS}:")
    print(f"  {' '.join(cmd)}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        process.wait()
    print(f"[{stage_name}] Done (exit code {process.returncode}). Log: {log_file}")
    return log_file


def find_best_checkpoint(config_name):
    """Find the best_mIoU checkpoint from the most recent work_dirs run."""
    pattern = f"work_dirs/{config_name}/*/best_mIoU_iter_*.pth"
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No best_mIoU checkpoint found for {config_name}. "
            "Check work_dirs/ and set load_from manually."
        )
    return ckpts[-1]


# ── Stage 1: Mapillary ──────────────────────────────────────────────────────
run_stage(config=STAGE1_CONFIG, stage_name="stage1_mapillary")

stage1_ckpt = find_best_checkpoint("ocrnet_hr48_stage1_mapillary-512x1024")
print(f"Stage 1 best checkpoint: {stage1_ckpt}")

# ── Stage 2: Cityscapes from Mapillary ──────────────────────────────────────
run_stage(config=STAGE2_CONFIG, load_from=stage1_ckpt, stage_name="stage2_cityscapes")

stage2_ckpt = find_best_checkpoint("ocrnet_hr48_stage2_cityscapes-512x1024")
print(f"Stage 2 best checkpoint: {stage2_ckpt}")

# ── Stage 3: RailSem19 from Cityscapes ──────────────────────────────────────
run_stage(config=STAGE3_CONFIG, load_from=stage2_ckpt, stage_name="stage3_railsem19")

stage3_ckpt = find_best_checkpoint("ocrnet_hr48_stage3_railsem19-576x1024")
print(f"\nAll 3 stages complete.")
print(f"Final RS19 checkpoint: {stage3_ckpt}")
