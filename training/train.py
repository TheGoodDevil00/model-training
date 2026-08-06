#!/usr/bin/env python3
"""
train.py
========
Train a YOLO11n human-detector on the VisDrone-derived YOLO dataset produced
by `prepare_dataset.py`.

Typical usage
-------------
    # First convert the dataset:
    python training/prepare_dataset.py

    # Then train (uses the GPU if available, otherwise CPU):
    python training/train.py --epochs 100 --batch 16 --imgsz 640

    # Sweep the chosen inference input size for speed (smaller = faster on RPi):
    python training/train.py --imgsz 416 --epochs 100 --batch 24

The model weights are saved to:
    runs/detect/train/weights/best.pt   (highest validation mAP)
    runs/detect/train/weights/last.pt   (last epoch)
"""

import argparse

import torch
from ultralytics import YOLO

DATASET_YAML = "datasets/usable/yolo-human/data.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11n human detector (drone view)")
    p.add_argument("--data", type=str, default=DATASET_YAML, help="Path to data.yaml")
    p.add_argument("--weights", type=str, default="yolo11n.pt",
                   help="Pretrained weights to start from (yolo11n.pt) or 'yolo11n.yaml' for from-scratch")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Training input size (px, divisible by 32)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (auto-lowered if OOM)")
    p.add_argument("--device", type=str, default=None,
                   help="0,1,.. for GPU, 'cpu' for CPU (default: auto GPU)")
    p.add_argument("--patience", type=int, default=30,
                   help="Stop early if no val mAP improvement for N epochs")
    p.add_argument("--cache", choices=["True", "False", "ram", "disk"], default="ram",
                   help="Cache images in RAM for faster training")
    p.add_argument("--project", type=str, default="runs/detect", help="Output project folder")
    p.add_argument("--name", type=str, default="train", help="Experiment name")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-select device: GPU if available, else CPU.
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}  (cuda available: {torch.cuda.is_available()})")

    model = YOLO(args.weights)

    # amp is auto-enabled on CUDA; disable it if you hit numeric issues on CPU.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        cache=args.cache,
        project=args.project,
        name=args.name,
        # Useful defaults for a single-class human detector:
        workers=4,
        # ladder-tuning trick: train at 640 then fine-tune at the target size
        # --- you can lower 'conf' later at inference instead of re-training.
        val=True,
        plots=True,
    )

    print("\nTraining finished. Best weights:")
    print(f"  {args.project}/{args.name}/weights/best.pt")
    print("\nNext step:  python training/inference.py --model runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    main()
