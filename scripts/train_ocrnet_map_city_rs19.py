#!/usr/bin/env python
"""
3-stage OCRNet-HR48 training pipeline: Mapillary -> Cityscapes -> RailSem19

Distributed training on 3x Quadro RTX 6000 (GPUs 4, 5, 6).
Matches HRNet paper protocol: SyncBN, effective batch 48, PolyLR power=0.9.

Stage 1: Train on Mapillary (66 classes, up to 500k iters)
Stage 2: Fine-tune on Cityscapes (19 classes, up to 120k iters) from best Stage 1 ckpt
Stage 3: Fine-tune on RailSem19 (19 classes, up to 120k iters) from best Stage 2 ckpt

All stages use EarlyStoppingHook so they stop automatically at convergence.

Usage:
    python scripts/train_ocrnet_map_city_rs19.py              # start from Stage 1
    python scripts/train_ocrnet_map_city_rs19.py --start 2    # start from Stage 2
    python scripts/train_ocrnet_map_city_rs19.py --start 3    # start from Stage 3
"""
import argparse
import subprocess
import os
import glob
from datetime import datetime

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# GPU and distributed training settings
GPUS = "4,5,6"
NUM_GPUS = 3

STAGE1_CONFIG = "configs/ocrnet/ocrnet_hr48_stage1_mapillary-512x1024.py"
STAGE2_CONFIG = "configs/ocrnet/ocrnet_hr48_stage2_cityscapes-512x1024.py"
STAGE3_CONFIG = "configs/ocrnet/ocrnet_hr48_stage3_railsem19-576x1024.py"

STAGE1_WORKDIR = "ocrnet_hr48_stage1_mapillary-512x1024"
STAGE2_WORKDIR = "ocrnet_hr48_stage2_cityscapes-512x1024"
STAGE3_WORKDIR = "ocrnet_hr48_stage3_railsem19-576x1024"


def find_best_checkpoint(config_name):
    """Find the best_mIoU checkpoint from work_dirs.
    Handles both flat layout (work_dirs/<config>/best_mIoU_iter_*.pth)
    and timestamped layout (work_dirs/<config>/<timestamp>/best_mIoU_iter_*.pth).
    Returns the checkpoint with the highest iteration number (most recent best).
    """
    patterns = [
        "work_dirs/{}/best_mIoU_iter_*.pth".format(config_name),
        "work_dirs/{}/*/best_mIoU_iter_*.pth".format(config_name),
    ]
    ckpts = []
    for pattern in patterns:
        ckpts.extend(glob.glob(pattern))
    ckpts = sorted(set(ckpts))
    if not ckpts:
        raise FileNotFoundError(
            "No best_mIoU checkpoint found in work_dirs/{}.\n"
            "Either training has not been run yet, or early stopping fired before "
            "any checkpoint was saved.".format(config_name)
        )
    best = ckpts[-1]
    print("  Found checkpoint: {}".format(best))
    return best


def run_stage(config, load_from=None, stage_name="stage"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, "ocrnet_{}_{}.log".format(stage_name, timestamp))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS

    cfg_options = []
    if load_from:
        cfg_options += ["--cfg-options", "load_from={}".format(load_from)]

    cmd = ["bash", "tools/dist_train.sh", config, str(NUM_GPUS)] + cfg_options

    print("[{}] Starting distributed training on GPUs {}:".format(stage_name, GPUS))
    print("  {}".format(" ".join(cmd)))
    if load_from:
        print("  Loading from: {}".format(load_from))
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        process.wait()
    print("[{}] Done (exit code {}). Log: {}".format(stage_name, process.returncode, log_file))
    return log_file


def main():
    parser = argparse.ArgumentParser(description="3-stage OCRNet training pipeline")
    parser.add_argument(
        "--start", type=int, choices=[1, 2, 3], default=1,
        metavar="STAGE",
        help="Stage to start from (1=Mapillary, 2=Cityscapes, 3=RailSem19). "
             "When starting from Stage 2 or 3, the best checkpoint from the "
             "previous stage is found automatically. Default: 1"
    )
    args = parser.parse_args()

    # ── Stage 1: Mapillary ───────────────────────────────────────────────────
    if args.start <= 1:
        run_stage(config=STAGE1_CONFIG, stage_name="stage1_mapillary")

    print("\n[pipeline] Locating Stage 1 best checkpoint...")
    stage1_ckpt = find_best_checkpoint(STAGE1_WORKDIR)

    # ── Stage 2: Cityscapes from Mapillary ───────────────────────────────────
    if args.start <= 2:
        run_stage(config=STAGE2_CONFIG, load_from=stage1_ckpt, stage_name="stage2_cityscapes")

    print("\n[pipeline] Locating Stage 2 best checkpoint...")
    stage2_ckpt = find_best_checkpoint(STAGE2_WORKDIR)

    # ── Stage 3: RailSem19 from Cityscapes ───────────────────────────────────
    if args.start <= 3:
        run_stage(config=STAGE3_CONFIG, load_from=stage2_ckpt, stage_name="stage3_railsem19")

    print("\n[pipeline] Locating Stage 3 best checkpoint...")
    stage3_ckpt = find_best_checkpoint(STAGE3_WORKDIR)

    print("\nAll 3 stages complete.")
    print("Final RS19 checkpoint: {}".format(stage3_ckpt))


if __name__ == "__main__":
    main()
