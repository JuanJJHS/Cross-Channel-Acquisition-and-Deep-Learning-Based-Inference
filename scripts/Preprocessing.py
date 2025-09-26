# -*- coding: utf-8 -*-
"""
preprocess_raw_to_splits.py  (SEQUENCE-LEVEL SPLIT)
- Input:  ../RAW/S_*/{MM,CC,CM}/frame_*.png
- Output: ../Train/{MM,CC,CM}/frame_XXXXXX.png
          ../Validation/{MM,CC,CM}/frame_XXXXXX.png

Split:
  Train:Validation = TRAIN_PARTS:VAL_PARTS at the **session (S_*)** level.

Augmentations:
  - Validation: raw-resized only
  - Training: for each source index, emit TWO frames:
      [even] raw-resized
      [odd ] 2x2 tile+shuffle (same shuffle order reused across channels)

Notes:
  - All paths are relative to the repo root (= parent of scripts/).
  - Thresholds are applied to any channel present in gt_thresholds.
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ------------------------
# Config (edit if needed)
# ------------------------
SEED = 42
OUT_SIZE = (256, 256)  # (W, H)

# SEQUENCE-LEVEL RATIO => TRAIN_PARTS=5, VAL_PARTS=1
TRAIN_PARTS = 2
VAL_PARTS   = 1

# Background suppression thresholds
gt_thresholds: Dict[str, int] = {
    "MM": 0,
    "CC": 0,
}

# Limit channels considered (intersection with what's on disk)
CHANNELS_WANTED = ["MM", "CC", "CM"]

# ------------------------
# Helpers
# ------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent  # parent of scripts/

def sessions_in(raw_root: Path) -> List[str]:
    return sorted([d.name for d in raw_root.iterdir()
                   if d.is_dir() and re.match(r"^S.+", d.name)])

def channel_dirs(session_dir: Path) -> List[str]:
    return [c for c in CHANNELS_WANTED if (session_dir / c).is_dir()]

def list_frames(ch_dir: Path) -> List[str]:
    return sorted([f.name for f in ch_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"])

def ensure_out_dirs(root: Path, channels: List[str]) -> Tuple[Path, Path]:
    train_root = root / "Train"
    val_root   = root / "Validation"
    for base in (train_root, val_root):
        base.mkdir(parents=True, exist_ok=True)
        for c in channels:
            (base / c).mkdir(parents=True, exist_ok=True)
    return train_root, val_root

def tile_and_shuffle(image_pil: Image.Image, tiles_x: int = 2, tiles_y: int = 2,
                     shuffle_indices: List[int] | None = None) -> Tuple[Image.Image, List[int]]:
    w, h = image_pil.size
    tw, th = w // tiles_x, h // tiles_y
    tiles = [image_pil.crop((x, y, x + tw, y + th))
             for y in range(0, h, th)
             for x in range(0, w, tw)]
    if shuffle_indices is None:
        shuffle_indices = list(range(len(tiles)))
        random.shuffle(shuffle_indices)
    out = Image.new('L', (w, h))
    for idx, tile in enumerate(tiles):
        x = (shuffle_indices[idx] % tiles_x) * tw
        y = (shuffle_indices[idx] // tiles_x) * th
        out.paste(tile, (x, y))
    return out, shuffle_indices

def read_resize_gray(fp: Path, out_size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {fp}")
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_LANCZOS4)
    return img

def apply_threshold(img_gray: np.ndarray, thr: int) -> np.ndarray:
    if thr <= 0:
        return img_gray
    return np.where(img_gray < thr, 0, img_gray).astype(np.uint8)

# ------------------------
# Main
# ------------------------
def main():
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    root = repo_root()
    raw_root = root / "RAW"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing RAW/ at {raw_root}")

    sessions = sessions_in(raw_root)
    if not sessions:
        raise FileNotFoundError(f"No S_* sessions found under {raw_root}")

    # Decide SEQUENCE-LEVEL split
    total_parts = TRAIN_PARTS + VAL_PARTS
    val_fraction = VAL_PARTS / total_parts
    val_count = max(1, int(round(len(sessions) * val_fraction)))  # ensure ≥1 if possible
    sess_shuffled = sessions[:]
    random.shuffle(sess_shuffled)
    val_sessions = set(sess_shuffled[:val_count])
    train_sessions = [s for s in sessions if s not in val_sessions]

    print(f"Sessions found: {sessions}")
    print(f"Train:Validation (sequences) = {TRAIN_PARTS}:{VAL_PARTS}")
    print(f"Validation sessions ({len(val_sessions)}): {sorted(val_sessions)}")
    print(f"Training sessions   ({len(train_sessions)}): {train_sessions}")

    # Prepare outputs
    train_root, val_root = ensure_out_dirs(root, CHANNELS_WANTED)

    global_train_idx = 0
    global_val_idx   = 0

    # -------- VALIDATION SESSIONS (raw-resized only) --------
    for s in val_sessions:
        print(f"\n=== Validation session: {s} ===")
        s_dir = raw_root / s
        ch_present = channel_dirs(s_dir)
        if not ch_present:
            print(f"[SKIP] {s}: none of {CHANNELS_WANTED} present.")
            continue

        ch_files = {c: list_frames(s_dir / c) for c in ch_present}
        n_min = min(len(v) for v in ch_files.values())
        if n_min == 0:
            print(f"[SKIP] {s}: empty channel folder detected.")
            continue

        print(f"{s}: frames={n_min} → Validation={n_min}")
        for i in tqdm(range(n_min), desc=f"{s} (val)"):
            out_name = f"frame_{global_val_idx:06d}.png"
            for c in ch_present:
                src_fp = s_dir / c / ch_files[c][i]
                img = read_resize_gray(src_fp, OUT_SIZE)
                img = apply_threshold(img, gt_thresholds.get(c, 0))
                Image.fromarray(img).save(val_root / c / out_name)
            global_val_idx += 1

    # -------- TRAINING SESSIONS (raw + tile+shuffle) --------
    for s in train_sessions:
        print(f"\n=== Training session: {s} ===")
        s_dir = raw_root / s
        ch_present = channel_dirs(s_dir)
        if not ch_present:
            print(f"[SKIP] {s}: none of {CHANNELS_WANTED} present.")
            continue

        ch_files = {c: list_frames(s_dir / c) for c in ch_present}
        n_min = min(len(v) for v in ch_files.values())
        if n_min == 0:
            print(f"[SKIP] {s}: empty channel folder detected.")
            continue

        print(f"{s}: frames={n_min} → Train (x2 aug)={2*n_min}")
        for i in tqdm(range(n_min), desc=f"{s} (train)"):
            # [even] raw-resized
            even_name = f"frame_{global_train_idx:06d}.png"
            for c in ch_present:
                src_fp = s_dir / c / ch_files[c][i]
                img = read_resize_gray(src_fp, OUT_SIZE)
                img = apply_threshold(img, gt_thresholds.get(c, 0))
                Image.fromarray(img).save(train_root / c / even_name)
            global_train_idx += 1

            # [odd] tile+shuffle (shared order across channels)
            odd_name = f"frame_{global_train_idx:06d}.png"
            shuffle_indices = None
            for c in ch_present:
                src_fp = s_dir / c / ch_files[c][i]
                img = read_resize_gray(src_fp, OUT_SIZE)
                pil = Image.fromarray(img)
                pil, shuffle_indices = tile_and_shuffle(pil, 2, 2, shuffle_indices)
                if c in gt_thresholds:
                    arr = np.array(pil)
                    arr = apply_threshold(arr, gt_thresholds.get(c, 0))
                    pil = Image.fromarray(arr)
                pil.save(train_root / c / odd_name)
            global_train_idx += 1

    print("\nDone.")
    print(f"Train frames written (per channel): {global_train_idx}")
    print(f"Validation frames written (per channel): {global_val_idx}")

if __name__ == "__main__":
    main()
