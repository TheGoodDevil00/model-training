#!/usr/bin/env python3
"""
tui.py
======
Unified Terminal User Interface (TUI) for the Rescue Swarm Human Detector pipeline.

Consolidates all dataset preparation, model training, edge export, performance evaluation,
and video inference workflows into a single interactive terminal menu with real-time
hardware telemetry monitoring (CPU, RAM, GPU VRAM, and Temperatures).

Usage:
    python training/tui.py
"""

import sys
import os
import subprocess
import shutil
import threading
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def get_python_exe():
    """Detects project venv python, falling back to sys.executable."""
    if os.name == "nt":
        venv_py = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:
        venv_py = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


# Auto re-exec into project venv if available and not already in use
_target_py = get_python_exe()
if _target_py != sys.executable and Path(_target_py).exists() and not os.environ.get("SWARM_VENV_REEXEC"):
    os.environ["SWARM_VENV_REEXEC"] = "1"
    try:
        os.execv(_target_py, [_target_py] + sys.argv)
    except Exception:
        pass


DATASET_DIR = PROJECT_ROOT / "datasets" / "usable" / "yolo-human"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
BASE_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"

# ANSI Terminal Color Helpers
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


# --------------------------------------------------------------------------- #
# System Telemetry & Hardware Stats
# --------------------------------------------------------------------------- #

def get_system_stats():
    """Queries current CPU, RAM, GPU, VRAM, and thermal sensors."""
    stats = {}

    # 1. RAM Usage
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                mem_info[parts[0].strip()] = int(parts[1].split()[0])
        total_mb = mem_info.get("MemTotal", 0) // 1024
        avail_mb = mem_info.get("MemAvailable", 0) // 1024
        used_mb = total_mb - avail_mb
        pct = (used_mb / total_mb * 100) if total_mb else 0
        stats["ram"] = f"{used_mb / 1024:.1f} / {total_mb / 1024:.1f} GB ({pct:.0f}%)"
    except Exception:
        stats["ram"] = "N/A"

    # 2. CPU Load
    try:
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        cpu_pct = min(100.0, (load1 / cpu_count) * 100)
        stats["cpu"] = f"{cpu_pct:.0f}% (Load: {load1:.2f})"
    except Exception:
        stats["cpu"] = "N/A"

    # 3. CPU Temperature
    try:
        cpu_temps = []
        for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                t = int(zone.read_text().strip()) / 1000.0
                if 20 <= t <= 110:
                    cpu_temps.append(t)
            except Exception:
                pass
        if cpu_temps:
            stats["cpu_temp"] = f"{max(cpu_temps):.1f}°C"
        else:
            stats["cpu_temp"] = "N/A"
    except Exception:
        stats["cpu_temp"] = "N/A"

    # 4. NVIDIA GPU Stats (nvidia-smi)
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                g_temp, g_used, g_total, g_util = parts[:4]
                stats["gpu_temp"] = f"{g_temp}°C"
                stats["vram"] = f"{int(g_used) / 1024:.2f} / {int(g_total) / 1024:.2f} GB"
                stats["gpu_util"] = f"{g_util}%"
    except Exception:
        pass

    return stats


class TelemetryThread(threading.Thread):
    """Background daemon updating the terminal window title bar with live stats."""
    def __init__(self, interval=3.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            stats = get_system_stats()
            vram_str = f" | VRAM: {stats['vram']}" if "vram" in stats else ""
            gpu_str = f" | GPU: {stats['gpu_temp']}" if "gpu_temp" in stats else ""
            title = f"Swarm Pipeline | CPU: {stats.get('cpu', 'N/A')} ({stats.get('cpu_temp', 'N/A')}) | RAM: {stats.get('ram', 'N/A')}{gpu_str}{vram_str}"
            try:
                sys.stdout.write(f"\033]0;{title}\007")
                sys.stdout.flush()
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        try:
            sys.stdout.write("\033]0;\007")
            sys.stdout.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# UI Displays & Headers
# --------------------------------------------------------------------------- #

def print_banner():
    """Prints status header, workspace state, and live hardware monitoring telemetry."""
    os.system("clear" if os.name != "nt" else "cls")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}      RESCUE SWARM: YOLO11n Human Detector - Unified Pipeline TUI       {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

    # Workspace status indicators
    dataset_ok = DATASET_DIR.exists() and (DATASET_DIR / "data.yaml").exists()
    dataset_status = f"{GREEN}Ready ({DATASET_DIR.relative_to(PROJECT_ROOT)}){RESET}" if dataset_ok else f"{YELLOW}Not found (run Step 1){RESET}"

    model_ok = DEFAULT_WEIGHTS.exists()
    model_status = f"{GREEN}Found ({DEFAULT_WEIGHTS.relative_to(PROJECT_ROOT)}){RESET}" if model_ok else f"{YELLOW}Not trained yet{RESET}"

    exports = list((PROJECT_ROOT / "runs").glob("**/*_ncnn_model")) + list((PROJECT_ROOT / "runs").glob("**/*.onnx"))
    export_status = f"{GREEN}{len(exports)} exported model(s) found{RESET}" if exports else f"{DIM}None{RESET}"

    py_in_use = get_python_exe()
    stats = get_system_stats()

    print(f"  {BOLD}Python Env:{RESET}     {CYAN}{py_in_use}{RESET}")
    print(f"  {BOLD}Dataset Status:{RESET} {dataset_status}")
    print(f"  {BOLD}Best Weights:{RESET}   {model_status}")
    print(f"  {BOLD}Edge Exports:{RESET}   {export_status}")
    print(f"{DIM}  ----------------------------------------------------------------------{RESET}")
    print(f"  {BOLD}System & Hardware Telemetry Monitoring:{RESET}")
    print(f"    • {BOLD}CPU Usage:{RESET}    {stats.get('cpu', 'N/A'):<18} |  {BOLD}CPU Temp:{RESET} {stats.get('cpu_temp', 'N/A')}")
    print(f"    • {BOLD}RAM Memory:{RESET}   {stats.get('ram', 'N/A')}")
    if "vram" in stats:
        print(f"    • {BOLD}GPU (RTX 3050):{RESET} Util {stats.get('gpu_util', 'N/A'):<4} | VRAM {stats.get('vram', 'N/A'):<16} | Temp {stats.get('gpu_temp', 'N/A')}")
    print(f"{DIM}------------------------------------------------------------------------{RESET}\n")


def run_command(cmd_list):
    """Executes a command array with live terminal output and background telemetry updates."""
    cmd_str = " ".join(cmd_list)
    print(f"\n{BOLD}{GREEN}Executing:{RESET} {cmd_str}\n")

    # Log initial hardware snapshot
    start_stats = get_system_stats()
    gpu_snapshot = f" | GPU Temp: {start_stats['gpu_temp']} | VRAM: {start_stats['vram']}" if "vram" in start_stats else ""
    print(f"{DIM}[Hardware Start Snapshot] RAM: {start_stats.get('ram', 'N/A')} | CPU: {start_stats.get('cpu', 'N/A')} ({start_stats.get('cpu_temp', 'N/A')}){gpu_snapshot}{RESET}")
    print(f"{DIM}------------------------ [Output Stream Start] ------------------------{RESET}\n")

    # Launch background titlebar telemetry logger
    monitor = TelemetryThread(interval=2.5)
    monitor.start()

    try:
        res = subprocess.run(cmd_list, cwd=PROJECT_ROOT)
        monitor.stop()
        print(f"\n{DIM}------------------------- [Output Stream End] -------------------------{RESET}")
        
        end_stats = get_system_stats()
        end_gpu = f" | GPU Temp: {end_stats['gpu_temp']} | VRAM: {end_stats['vram']}" if "vram" in end_stats else ""
        print(f"{DIM}[Hardware Finish Snapshot] RAM: {end_stats.get('ram', 'N/A')} | CPU Temp: {end_stats.get('cpu_temp', 'N/A')}{end_gpu}{RESET}")

        if res.returncode == 0:
            print(f"\n{BOLD}{GREEN}Command completed successfully!{RESET}")
        else:
            print(f"\n{BOLD}{RED}Command failed with exit code {res.returncode}.{RESET}")
    except KeyboardInterrupt:
        monitor.stop()
        print(f"\n{BOLD}{YELLOW}Process interrupted by user.{RESET}")
    except Exception as e:
        monitor.stop()
        print(f"\n{BOLD}{RED}Error launching command: {e}{RESET}")

    input(f"\n{DIM}Press Enter to return to the main menu...{RESET}")


def prompt_string(label, default=""):
    prompt = f"  {label} [{default}]: " if default else f"  {label}: "
    val = input(prompt).strip()
    return val if val else default


def prompt_int(label, default):
    val = prompt_string(label, str(default))
    try:
        return int(val)
    except ValueError:
        print(f"{RED}Invalid integer. Using default {default}{RESET}")
        return default


def prompt_float(label, default):
    val = prompt_string(label, str(default))
    try:
        return float(val)
    except ValueError:
        print(f"{RED}Invalid float. Using default {default}{RESET}")
        return default


# --------------------------------------------------------------------------- #
# Menu Handlers
# --------------------------------------------------------------------------- #

def menu_prepare_dataset():
    print_banner()
    print(f"{BOLD}Step 1: Prepare VisDrone Dataset{RESET}\n")
    print("  1. Default (Symlink images, keep negative frames)")
    print("  2. Only frames with humans (--only-with-humans)")
    print("  3. Copy images instead of symlinking (--copy)")
    print("  4. Custom paths and options")
    print("  0. Back")

    choice = input("\nSelect option [1-4]: ").strip()
    if choice == "0" or not choice:
        return

    cmd = [get_python_exe(), "training/prepare_dataset.py"]

    if choice == "2":
        cmd.append("--only-with-humans")
    elif choice == "3":
        cmd.append("--copy")
    elif choice == "4":
        train_path = prompt_string("Train split dir", "datasets/usable/VisDrone2019-DET-train")
        val_path = prompt_string("Val split dir", "datasets/usable/VisDrone2019-DET-val")
        out_path = prompt_string("Output dir", "datasets/usable/yolo-human")
        cmd.extend(["--train", train_path, "--val", val_path, "--out", out_path])
        if input("  Copy images instead of symlink? (y/N): ").lower().startswith("y"):
            cmd.append("--copy")
        if input("  Only keep images with humans? (y/N): ").lower().startswith("y"):
            cmd.append("--only-with-humans")

    run_command(cmd)


def menu_train():
    print_banner()
    print(f"{BOLD}Step 2: Train Model (YOLO11n){RESET}\n")
    print("  1. Stage 1 Fine-Tune (Frozen backbone: 10 layers, 100 epochs, batch 16)")
    print("  2. Stage 2 Fine-Tune (Unfreeze backbone, cosine LR, best weights, batch 16)")
    print("  3. Quick Pipeline Smoke Test (3 epochs, batch 16, 0.2 fraction)")
    print("  4. Custom Training Run")
    print("  0. Back")

    choice = input("\nSelect option [1-4]: ").strip()
    if choice == "0" or not choice:
        return

    cmd = [get_python_exe(), "training/train.py"]

    if choice == "1":
        cmd.extend(["--epochs", "100", "--batch", "16", "--imgsz", "512", "--freeze", "10", "--single-cls"])
    elif choice == "2":
        weights = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolo11n.pt"
        weights_in = prompt_string("Pretrained weights path", weights)
        cmd.extend(["--freeze", "0", "--epochs", "40", "--batch", "16", "--imgsz", "512", "--single-cls", "--cos-lr", "--weights", weights_in])
    elif choice == "3":
        cmd.extend(["--epochs", "3", "--batch", "16", "--fraction", "0.2", "--profile"])
    elif choice == "4":
        epochs = prompt_int("Epochs", 100)
        imgsz = prompt_int("Input resolution (px)", 512)
        batch = prompt_int("Batch size", 16)
        freeze = prompt_int("Frozen layers (10 = backbone, 0 = none)", 10)
        weights = prompt_string("Weights checkpoint", "yolo11n.pt")
        cmd.extend(["--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch),
                    "--freeze", str(freeze), "--weights", weights])
        if input("  Enable Cosine LR schedule? (Y/n): ").lower() != "n":
            cmd.append("--cos-lr")

    run_command(cmd)


def menu_export():
    print_banner()
    print(f"{BOLD}Step 3: Edge Export & Quantization (Raspberry Pi 5 Target){RESET}\n")
    print("  1. NCNN FP16 (Fast & accurate for ARM NEON - Recommended)")
    print("  2. NCNN INT8 (Calibrated ARM NEON quantization)")
    print("  3. ONNX INT8 (Alternative ONNX runtime)")
    print("  4. Custom Export")
    print("  0. Back")

    choice = input("\nSelect option [1-4]: ").strip()
    if choice == "0" or not choice:
        return

    weights = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "runs/detect/train/weights/best.pt"
    model_path = prompt_string("Model checkpoint to export", weights)

    cmd = [get_python_exe(), "training/export.py", "--model", model_path]

    if choice == "1":
        cmd.extend(["--format", "ncnn", "--imgsz", "416", "--half"])
    elif choice == "2":
        data_path = prompt_string("Calibration data.yaml", "datasets/usable/yolo-human/data.yaml")
        cmd.extend(["--format", "ncnn", "--imgsz", "416", "--int8", "--data", data_path])
    elif choice == "3":
        data_path = prompt_string("Calibration data.yaml", "datasets/usable/yolo-human/data.yaml")
        cmd.extend(["--format", "onnx", "--imgsz", "416", "--int8", "--data", data_path])
    elif choice == "4":
        fmt = prompt_string("Format (ncnn / onnx)", "ncnn")
        imgsz = prompt_int("Input resolution (px)", 416)
        cmd.extend(["--format", fmt, "--imgsz", str(imgsz)])
        quant = prompt_string("Quantization (half / int8 / none)", "half").lower()
        if quant == "half":
            cmd.append("--half")
        elif quant == "int8":
            data_path = prompt_string("Calibration data.yaml", "datasets/usable/yolo-human/data.yaml")
            cmd.extend(["--int8", "--data", data_path])

    run_command(cmd)


def menu_evaluate():
    print_banner()
    print(f"{BOLD}Step 4: RPi 5 Hardware Simulation & Evaluation{RESET}\n")
    print("  1. Speed Benchmark (FPS / Latency sweep across resolutions)")
    print("  2. Accuracy Benchmark (mAP50 / mAP50-95 on validation set)")
    print("  0. Back")

    choice = input("\nSelect option [1-2]: ").strip()
    if choice == "0" or not choice:
        return

    cmd = [get_python_exe(), "training/eval/evaluate.py"]

    if choice == "1":
        cmd.append("speed")
        default_model = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "runs/detect/train/weights/best.pt"
        models_input = prompt_string("Models to benchmark (space-separated)", default_model)
        video_input = prompt_string("Sample drone video path", "sample_drone.mp4")
        resolutions = prompt_string("Resolution sweep (comma-separated)", "640,416,352")
        threads = prompt_int("Simulated CPU threads", 4)
        
        cmd.extend(["--models"] + models_input.split())
        cmd.extend(["--video", video_input, "--imgsz", resolutions, "--threads", str(threads)])
    elif choice == "2":
        cmd.append("accuracy")
        default_model = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "runs/detect/train/weights/best.pt"
        model_input = prompt_string("Model path", default_model)
        imgsz = prompt_int("Validation resolution (px)", 416)
        threads = prompt_int("Simulated CPU threads", 4)
        cmd.extend(["--model", model_input, "--imgsz", str(imgsz), "--threads", str(threads)])

    run_command(cmd)


def menu_inference():
    print_banner()
    print(f"{BOLD}Step 5: Run Video / Webcam Inference{RESET}\n")
    
    default_model = str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolo11n.pt"
    # Find exported models if present
    exports = list((PROJECT_ROOT / "runs").glob("**/*_ncnn_model"))
    if exports:
        default_model = str(exports[0].relative_to(PROJECT_ROOT))

    model_path = prompt_string("Model path (.pt or exported folder)", default_model)
    input_src = prompt_string("Input video path or 0 for webcam", "0")
    output_path = prompt_string("Output video path", "output_annotated.mp4")
    imgsz = prompt_int("Inference input resolution (px)", 416)
    conf = prompt_float("Confidence threshold", 0.35)

    cmd = [get_python_exe(), "training/inference.py",
           "--model", model_path,
           "--input", input_src,
           "--output", output_path,
           "--imgsz", str(imgsz),
           "--conf", str(conf)]

    run_command(cmd)


# --------------------------------------------------------------------------- #
# Main Loop
# --------------------------------------------------------------------------- #

def main():
    while True:
        print_banner()
        print(f"{BOLD}Main Menu — Select Workflow:{RESET}\n")
        print(f"  {CYAN}1.{RESET} 📁 Prepare Dataset         (VisDrone -> YOLO single-class format)")
        print(f"  {CYAN}2.{RESET} 🏋️ Train Model             (YOLO11n fine-tuning & smoke test)")
        print(f"  {CYAN}3.{RESET} 📦 Export & Quantize       (NCNN / ONNX int8/fp16 for RPi 5)")
        print(f"  {CYAN}4.{RESET} 📊 Evaluate Performance    (RPi 5 hardware speed & accuracy simulation)")
        print(f"  {CYAN}5.{RESET} 🎥 Video / Webcam Inference(Real-time detection & FPS monitoring)")
        print(f"  {CYAN}0.{RESET} 🚪 Exit")

        choice = input(f"\n{BOLD}Select an option [0-5]: {RESET}").strip()

        if choice == "1":
            menu_prepare_dataset()
        elif choice == "2":
            menu_train()
        elif choice == "3":
            menu_export()
        elif choice == "4":
            menu_evaluate()
        elif choice == "5":
            menu_inference()
        elif choice == "0" or choice.lower() in ("q", "quit", "exit"):
            print(f"\n{GREEN}Exiting Rescue Swarm TUI. Goodbye!{RESET}\n")
            break


if __name__ == "__main__":
    main()
