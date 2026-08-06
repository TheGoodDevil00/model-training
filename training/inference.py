#!/usr/bin/env python3
"""
inference.py
============
Run the trained human-detector on a video file (or webcam) and write an
output video with the detected humans drawn as bounding boxes. Also prints
the live FPS so you can validate the Raspberry Pi throughput target.

Usage
-----
    # On a video file:
    python training/inference.py \
        --model runs/detect/train/weights/best.pt \
        --input path/to/drone.mp4 \
        --output output_annotated.mp4

    # On the webcam (live):
    python training/inference.py --model runs/detect/train/weights/best.pt --input 0

    # Fast / quantized export (e.g. NCNN) for Raspberry Pi:
    python training/inference.py \
        --model runs/detect/train/weights/best_ncnn_model \
        --input path/to/drone.mp4 --output output.mp4 --imgsz 416
"""

import argparse
import time

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Detect humans in a video with a YOLO model")
    p.add_argument("--model", type=str, required=True,
                   help="Model path (.pt, or exported NCNN/ONNX/OpenVINO folder)")
    p.add_argument("--input", type=str, required=True,
                   help="Input video path, or 0 for webcam")
    p.add_argument("--output", type=str, default="output_annotated.mp4",
                   help="Output video path")
    p.add_argument("--imgsz", type=int, default=416,
                   help="Inference input size for the CNN (smaller = faster)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--max-dist", type=int, default=3600,
                   help="Max side (px) to resize output frames to keep the viewer fast")
    return p.parse_args()


def main():
    args = parse_args()

    # Import lazily so a missing ultralytics install doesn't break arg parsing.
    from ultralytics import YOLO

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.input}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[inference] source {args.input}  {w}x{h} @ {fps_in:.1f} fps")

    # Scale down the writer only if the input is very large (keeps encode cheap).
    out_w, out_h = w, h
    if max(w, h) > args.max_dist:
        scale = args.max_dist / max(w, h)
        out_w, out_h = int(w * scale), int(h * scale)

    writer = cv2.VideoWriter(args.output,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open writer for: {args.output}")

    frame_idx = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Run detection. verbose=False keeps FPS readout clean.
            results = model(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]

            # Draw boxes (+ tiny label) on the original frame.
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls] if isinstance(model.names, dict) else str(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Write (scaled) output frame.
            out_frame = frame if (out_w, out_h) == (w, h) else cv2.resize(frame, (out_w, out_h))
            writer.write(out_frame)

            frame_idx += 1
            # FPS readout every 30 frames.
            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t0
                fps = frame_idx / elapsed
                print(f"[inference] frame {frame_idx}  avg {fps:.1f} fps, "
                      f"{len(results.boxes)} detection(s) in last frame", flush=True)
    except KeyboardInterrupt:
        print("\n[inference] interrupted by user")
    finally:
        cap.release()
        writer.release()
        elapsed = time.perf_counter() - t0
        if frame_idx:
            print(f"\n[inference] done. {frame_idx} frames in {elapsed:.1f}s "
                  f"= {frame_idx / elapsed:.1f} avg fps")
        print(f"[inference] annotated video written to: {args.output}")


if __name__ == "__main__":
    main()
