import time
import torch
import cv2
import os
import glob
import numpy as np
from multioutput_model import MultiOutputGenerator
import torch_tensorrt  
import json
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIGURATION ===
MODEL_ID = 0
# === Resolve repo root (= parent of scripts/) ===
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent  
merged_model_path = DATA_ROOT / f'Models/MultiOutput/{MODEL_ID}/merged_multi_generator_CMtoMM_CC.pth.tar'
frame_dir = DATA_ROOT / 'Validation/CM'
input_shape = (1, 1, 256, 256)  # Batch, Channels, Height, Width

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
multi_gen = MultiOutputGenerator().to(device)
ckpt = torch.load(merged_model_path, map_location=device)
multi_gen.load_state_dict(ckpt['state_dict'])
multi_gen.eval()
print(" Loaded merged multi-output generator.")

# === COMPILE WITH TORCH-TENSORRT ===
dummy_input = torch.randn(*input_shape).to(device)

trt_model = torch_tensorrt.compile(
    multi_gen,
    inputs=[torch_tensorrt.Input(dummy_input.shape)],
    enabled_precisions={torch.half},  # Use FP16 internally
    device=torch_tensorrt.Device("cuda:0")
)

print("Compiled model with Torch-TensorRT (FP16).")

# === WARM-UP ===
for _ in range(20):
    with torch.no_grad():
        _ = trt_model(dummy_input)

# === INFERENCE ON REAL FRAMES ===
png_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
inference_times = []

print(f" Found {len(png_paths)} frames in: {frame_dir}")

for path in png_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f" Skipped unreadable image: {path}")
        continue

    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256))

    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = trt_model(img_tensor)
    torch.cuda.synchronize()
    end = time.time()

    inference_times.append((end - start) * 1000)

avg_time_ms = sum(inference_times) / len(inference_times)
print(f"\n Average inference time on real PNG frames: {avg_time_ms:.2f} ms")

# === SAVE TIMINGS ===
with open("inference_times.json", "w") as f:
    json.dump(inference_times, f)
print(f" Saved per-run inference times to 'inference_times.json'")

# === LOAD TIMINGS AGAIN (for plotting) ===
with open("inference_times.json", "r") as f:
    inference_times = json.load(f)

print(f" Loaded {len(inference_times)} inference time records.")

# === STATS ===
mean_time = sum(inference_times) / len(inference_times)
min_time = min(inference_times)
max_time = max(inference_times)
print(f"Mean: {mean_time:.2f} ms | Min: {min_time:.2f} ms | Max: {max_time:.2f} ms")

# === PLOT 1: Histogram ===
plt.figure(figsize=(8, 5))
plt.hist(inference_times, bins=20, edgecolor='black')
plt.title("Inference Time Distribution")
plt.xlabel("Inference time (ms)")
plt.ylabel("Frequency")
plt.axvline(mean_time, color='red', linestyle='--', label=f"Mean: {mean_time:.2f} ms")
plt.legend()
plt.tight_layout()
plt.savefig("inference_time_histogram.png")
print(" Saved histogram as 'inference_time_histogram.png'.")

# === PLOT 2: Boxplot ===
plt.figure(figsize=(6, 4))
plt.boxplot(inference_times, vert=False)
plt.title("Inference Time Boxplot")
plt.xlabel("Inference time (ms)")
plt.tight_layout()
plt.savefig("inference_time_boxplot.png")
print(" Saved boxplot as 'inference_time_boxplot.png'.")

# === PLOT 3: Time vs Run Index ===
plt.figure(figsize=(8, 4))
plt.plot(inference_times, marker='o', markersize=2, linestyle='-')
plt.title("Inference Time per Frame")
plt.xlabel("Frame index")
plt.ylabel("Inference time (ms)")
plt.axhline(mean_time, color='red', linestyle='--', label=f"Mean: {mean_time:.2f} ms")
plt.legend()
plt.tight_layout()
plt.savefig("inference_time_vs_run.png")
print(" Saved time-vs-run plot as 'inference_time_vs_run.png'.")
