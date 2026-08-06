# Human Detector (drone view) — YOLO11n

## Objective
Train a tiny YOLO11n model that detects **humans** in drone footage and
returns bounding-box frames from a video. Target: **>= 15 FPS** on a
**Raspberry Pi 5 (4 GB)**.

## Data used
VisDrone2019-DET (the `datasets/usable/...` folders). Images are 960x540.
Raw annotation format per line (comma separated):

    x, y, width, height, score, category, truncation, occlusion

- `(x, y)` is the **top-left** corner in pixels (absolute, NOT normalized).
- Relevant human classes we keep and merge into a single `human` class:
  - `1` = pedestrian
  - `2` = people
- Everything else is discarded (cars, vans, trucks, bicycles, ignored regions...)
  so the model only learns humans.

---

## Pipeline (in this folder)

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `prepare_dataset.py` | VisDrone txt -> YOLO labels + `data.yaml` |
| 2 | `train.py` | Train YOLO11n |
| 3 | `export.py` | Export / quantize for the Pi (NCNN / ONNX int8) |
| 4 | `inference.py` | Run video -> annotated bounding-box video |

---

## 1. Prepare the dataset
```bash
source venv/bin/activate
python training/prepare_dataset.py
```
Produces `datasets/usable/yolo-human/{train,val}/...` with normalized
center-xywh YOLO labels (single class: `0` = human), plus `data.yaml`.

Notes:
- By default every frame is kept (including frames with no humans) so the
  model learns true negatives (doesn't fire on cars/buildings).
- `--only-with-humans` keeps only frames that contain at least one human
  (smaller, but weaker negatives).
- `--copy` copies the images instead of symlinking (symlink is the default).

---

## 2. Train
```bash
python training/train.py --epochs 100 --batch 16 --imgsz 640
```
Defaults: starts from `yolo11n.pt` (COCO-pretrained -> much faster
convergence and better results than training from scratch). Device auto-selects
CUDA if the dev machine has a GPU (`torch.cuda.is_available()` was True).

Other knobs:
- `--epochs` (start at 100; more = better but slower).
- `--batch` raise on strong GPU (16/32/64), lower if CPU OOM.
- `--imgsz`: train at 640 for accuracy. For an edge model you can train at
  416 to match the deployment size (a "ladder" trick: train 640, then
  fine-tune a few epochs at 416 before exporting).
- `--patience` early-stops if val mAP stops improving.

Output: `runs/detect/train/weights/best.pt` (best val mAP) and `last.pt`.

> Do NOT run training inside this session — it is a long GPU/CPU job. Run it
> on the dev machine with the command above.

---

## 3. Validate on a video (dev machine)
```bash
python training/inference.py \
    --model runs/detect/train/weights/best.pt \
    --input path/to/drone.mp4 --output out.mp4
```

---

## 4. Optimize for Raspberry Pi 5 (>= 15 FPS)

### Why NCNN
Raspberry Pi 5 has an ARM **Cortex-A76** CPU (no GPU acceleration for CUDA/
TensorRT, and OpenVINO is x86-only). **NCNN** is a lightweight CNN runtime
heavily optimised for ARM NEON, so it is the best fit. A YOLO11n exported to
NCNN at `imgsz=416` usually hits ~15-25 FPS on a Pi 5.

### Export (on the dev machine, once)
```bash
# fp16 (fast + accurate) — recommended first
python training/export.py --model runs/detect/train/weights/best.pt \
    --format ncnn --imgsz 416 --half

# int8 (smallest / sometimes a bit faster) — needs calibration
python training/export.py --model runs/detect/train/weights/best.pt \
    --format ncnn --imgsz 416 --int8 --data datasets/usable/yolo-human/data.yaml

# Alternative: ONNX int8 (run with onnxruntime on the Pi)
python training/export.py --model runs/detect/train/weights/best.pt \
    --format onnx --imgsz 416 --int8 --data datasets/usable/yolo-human/data.yaml
```
`export.py` calls `model.export()` and ultralytics auto-installs `ncnn` /
`onnx` / `onnxslim` on first use. The exported folder lands next to the model
(e.g. `runs/detect/train/weights/best_ncnn_model`).

Copy that folder (and `training/inference.py`) to the Pi.

### On the Raspberry Pi
```bash
# Python >=3.11 on Raspberry Pi OS (Bookworm). Install:
pip install ultralytics ncnn onnxruntime opencv-python-headless

export OMP_NUM_THREADS=4
export NCNN_THREADS=4          # or threads count = number of cores you allow

python inference.py \
    --model best_ncnn_model \
    --input drone.mp4 --output out.mp4 --imgsz 416
```

### Tuning knobs to reach / sustain 15+ FPS
1. **Input size is the biggest lever.** `--imgsz 416` vs 640 ~= 2.4x fewer
   pixels. Drop to 352 if you need more headroom (slightly less recall on
   far-away people).
2. **int8 vs fp16 NCNN.** int8 is smaller and often a touch faster; try both
   and pick whichever gives >= 15 FPS with the accuracy you need.
3. **Confidence / NMS.** Raise `--conf` (e.g. 0.4) and use `--iou 0.5` to
   cut false positives and reduce post-processing load.
4. **Run frames from the top** and reuse the input tensor to avoid re-allocation;
   the inference script already reuses the model. Consider processing every
   other frame if pure fps is not attainable (cheap temporal trick).
5. **Keep the Pi cool.** Enable all 4 cores (add a heatsink / fan); without
   active cooling the CPU throttles and FPS drops. Optionally set
   `arm_boost=1` in `/boot/firmware/config.txt`.
6. **`predict()` overhead.** In production, call the NCNN `Net` directly and
   pass raw frames; avoid per-frame Python list allocations. The provided
   `inference.py` is a clear baseline you can then harden.
7. **Page/pinning.** On the 4 GB Pi keep the OS lean (headless boot, no
   desktop) so RAM stays free.

### Realistic expectations
| Input size | NCNN quant | Est. FPS on Pi 5 |
|------------|-----------|------------------|
| 640        | fp16/int8 | ~6-10            |
| 416        | fp16      | ~15-20           |
| 416        | int8      | ~18-25           |
| 352        | int8      | ~22-30           |

---

## Class mapping recap
| VisDrone cat | meaning      | kept as |
|-------------|--------------|---------|
| 1           | pedestrian   | human   |
| 2           | people       | human   |
| 0,3-11      | everything else | dropped |

---

## 5. Laptop eval suite (mimics the Pi 5)

Before moving to hardware, run the eval suite in `training/eval/` — it
recreates Pi 5 compute constraints (CPU-only, thread-count parity, a
laptop->Pi speed correction factor, RAM alert) so you can verify the 15+ FPS
target and the accuracy/speed trade-off on your laptop.

```bash
# Throughput vs input size & format (the 15+ FPS check):
python training/eval/evaluate.py speed \
    --models runs/detect/train/weights/best.pt \
    --video sample_drone.mp4 --imgsz 640,416,352 --threads 4

# Accuracy (mAP) at the deployment size:
python training/eval/evaluate.py accuracy \
    --model runs/detect/train/weights/best.pt --imgsz 416 --threads 4
```

See `training/eval/README.md` for the full methodology and limits.

