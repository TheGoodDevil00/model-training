# Laptop Eval Suite — RPi-5 "simulator"

Evaluate your trained human detector **on the laptop** while reproducing the
Raspberry Pi 5's compute constraints, so you can sanity-check the **15+ FPS**
target and the accuracy/speed trade-off before flashing anything to the Pi.

## What it simulates (Raspberry Pi 5: 4 x Cortex-A76 @ ~2.4 GHz, 4 GB)
- **CPU-only inference** — forces Torch to CPU (`CUDA_VISIBLE_DEVICES=-1`),
  matching a board with no CUDA/TensorRT.
- **Thread-count parity** — locks Torch / OpenCV / NCNN to the Pi's core
  count (`--threads`, default 4). Your laptop likely has more cores (e.g.
  12 here), so limiting to 4 reproduces the Pi's parallelism.
- **Estimated Pi 5 FPS** — `raw_laptop_CPU_fps * --rpi-factor` (default 0.55)
  to account for the Cortex-A76's lower per-core throughput vs a modern
  laptop core. Tune the factor from your own Pi measurements once you have
  them.
- **RAM alert** — warns if the process RSS exceeds the Pi 5's `--mem-gb`
  (default 4.0) limit.

> **Limitation / honesty note:** this cannot reproduce the Pi's exact clock,
> cache, memory bandwidth, or thermal throttling. Use the estimated Pi 5 FPS
> as a *planning guide*; validate on real hardware for the final decision.
> Also, the NCNN path only runs here if `ncnn` is installed in the venv
> (ultralytics auto-installs it at export time).

## Speed benchmark (FPS / latency)
Compare formats (`.pt` CPU vs NCNN vs ONNX) and input sizes, and see which
combination clears 15 FPS:
```bash
source venv/bin/activate
python training/eval/evaluate.py speed \
    --models runs/detect/train/weights/best.pt \
             runs/detect/train/weights/best_ncnn_model \
    --video /path/to/drone.mp4 \
    --imgsz 640,416,352 \
    --threads 4
```
Outputs a table + a JSON report (`--out eval_speed.json`) and a verdict on
whether the fastest config meets the 15 FPS target.

## Accuracy benchmark (mAP)
```bash
python training/eval/evaluate.py accuracy \
    --model runs/detect/train/weights/best.pt \
    --imgsz 416 --threads 4
```
Runs Ultralytics `val()` on `datasets/usable/yolo-human/data.yaml` and prints
`mAP50`, `mAP50-95`, precision, recall.

## Recommended workflow
1. Train (see `../notes.md`).
2. `speed` benchmark with the plain `.pt` on CPU + 4 threads and sweep
   `--imgsz 640,416,352` → pick sizes that give est. Pi 5 FPS >= 15.
3. `accuracy` benchmark the candidate sizes → pick the smallest size whose
   mAP you still accept (typically 416: good speed, decent recall on people).
4. Export NCNN (fp16 / int8) and `speed` benchmark it too → confirm the
   quantized model is still fast (and `accuracy` it to check for drift).
5. Deploy the winner to the Pi and re-measure with `../inference.py` (its
   built-in FPS readout) to calibrate `--rpi-factor` for future runs.
