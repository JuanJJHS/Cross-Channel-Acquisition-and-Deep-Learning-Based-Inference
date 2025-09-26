# -*- coding: utf-8 -*-
"""
Evaluate SSIM, PSNR, and GCS (generated_*.png).
Tinting applied to each subframe
Saves animated GIF and violin plots of the metrics.
"""

import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import config

from skimage.feature import graycomatrix, graycoprops
from skimage.color   import rgb2gray
from skimage.util    import img_as_ubyte

def rel_glcm_contrast(gt,
                      pred,
                      distances=(1,),
                      angles=(0,),
                      levels=256,
                      eps=1e-6):
    """
    GLCM‑contrast similarity in [0,1]:
        1 → perfect texture match
        0 → worst (relative‑error ≥ 1)

    pred, gt : np.ndarray  shape (H,W) or (H,W,1) or (H,W,3)
               values float 0‑1 or uint8 0‑255.
    """
    def _prep(im):
        # squeeze (H,W,1) → (H,W)
        im = np.squeeze(im)

        # RGB → gray
        if im.ndim == 3:
            im = rgb2gray(im)

        # skimage expects uint8 or uint16
        return img_as_ubyte(im)   # scales float 0‑1 → uint8

    p_u8 = _prep(pred)
    g_u8 = _prep(gt)

    c_pred = graycoprops(
        graycomatrix(p_u8, distances, angles,
                     levels=levels, symmetric=True, normed=True),
        'contrast')[0, 0]

    c_gt   = graycoprops(
        graycomatrix(g_u8, distances, angles,
                     levels=levels, symmetric=True, normed=True),
        'contrast')[0, 0]

    rel_err = abs(c_pred - c_gt)/(c_gt + 1e-6)   # 0 best … 1 worst+
    return max(0.0, 1.0 - rel_err)           # invert → 1 best … 0 worst

# === Tinting helper ===
def apply_tint(img, color):
    tinted = img.copy()
    if color == "red":
        tinted[..., 1:] = 0
    elif color == "green":
        tinted[..., 0] = 0
        tinted[..., 2] = 0
    elif color == "blue":
        tinted[..., :2] = 0
    else:
        raise ValueError(f"Unsupported tint color: {color}")
    return tinted

def process_generated_images(input_dir, output_gif_path,left_color, center_right_color,apply_gamma=False, gamma=0.8):
    image_paths = sorted(glob.glob(os.path.join(input_dir, "generated_*.png")))
    assert len(image_paths) > 0, f"No images found in {input_dir}"

    # Load first image to get dimensions
    sample = Image.open(image_paths[0])
    frame_width, frame_height = sample.size
    split_width = 2
    subframe_width = (frame_width - 2 * split_width) // 3

    # Storage
    ssim_center_vs_right = []
    psnr_center_vs_right = []
    rel_contrast_center_vs_right = []
    processed_frames = []

    for img_path in image_paths:
        frame = Image.open(img_path).convert("RGB")
        np_frame = np.array(frame)

        # Subframes
        left = np_frame[:, :subframe_width]
        center = np_frame[:, subframe_width + split_width : 2 * subframe_width + split_width]
        right = np_frame[:, 2 * subframe_width + 2 * split_width:]

        # === Metrics ===
        ssim_center_vs_right.append(ssim(center, right, channel_axis=-1))
        psnr_center_vs_right.append(psnr(center, right))
        rel_contrast_center_vs_right.append(rel_glcm_contrast(center, right))
      
        # === Optional gamma correction just for visualization
        if apply_gamma:
            def gamma_correct(img, gamma):
                return np.clip(np.power(img / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
            left = gamma_correct(left, gamma)
            center = gamma_correct(center, gamma)
            right = gamma_correct(right, gamma)

        # Apply tinting
        tinted_left = apply_tint(left, left_color)
        tinted_center = apply_tint(center, center_right_color)
        tinted_right = apply_tint(right, center_right_color)

        # Stack visuals with dividers
        divider = np.ones((frame_height, split_width, 3), dtype=np.uint8) * 255
        combined = np.hstack([
            tinted_left, divider,
            tinted_center, divider,
            tinted_right
        ])
        processed_frames.append(Image.fromarray(combined))

    # Save GIF
    processed_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=processed_frames[1:],
        loop=0,
        duration=50  # ms per frame
    )

    return ssim_center_vs_right, psnr_center_vs_right, rel_contrast_center_vs_right


def plot_metrics(ssim_values, psnr_values, rel_contrast_values, output_dir, center_color, inp, out):
    # Map simple color names to matplotlib tab colors
    tab_colors = {
        "red": "tab:red",
        "green": "tab:green",
        "blue": "tab:blue"
    }
    plot_color = tab_colors.get(center_color, "tab:gray")

    plt.figure(figsize=(10, 4))

    # SSIM
    plt.subplot(1, 3, 1)
    parts = plt.violinplot([ssim_values], showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(plot_color)
        pc.set_edgecolor(plot_color)
        pc.set_alpha(0.6)
    plt.xticks([1], [""])
    # plt.title("SSIM Distribution")
    plt.ylabel("SSIM")
    plt.grid(True, linestyle="--", alpha=0.4)

    # PSNR
    plt.subplot(1, 3, 2)
    parts = plt.violinplot([psnr_values], showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(plot_color)
        pc.set_edgecolor(plot_color)
        pc.set_alpha(0.6)
    plt.xticks([1], [""])
    # plt.title("PSNR Distribution")
    plt.ylabel("PSNR [dB]")
    plt.ylim(20, 50)
    plt.grid(True, linestyle="--", alpha=0.4)
    
    # Rel contrast GCS
    plt.subplot(1, 3, 3)
    parts = plt.violinplot([rel_contrast_values], showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(plot_color)
        pc.set_edgecolor(plot_color)
        pc.set_alpha(0.6)
    plt.xticks([1], [""])
    plt.ylabel("GCS")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violin_metrics_{inp}to{out}.png"))
    plt.show()

def process_and_plot_generated_images(input_dir, output_gif_path, output_csv_path, left_color, center_right_color, inp, out):
    ssim_vals, psnr_vals, rel_contrast_vals = process_generated_images(input_dir, output_gif_path, left_color, center_right_color,apply_gamma=False, gamma=0.8)
    print(f"Saved GIF to {output_gif_path}")

    plot_metrics(ssim_vals, psnr_vals, rel_contrast_vals, os.path.dirname(output_gif_path), center_right_color, inp, out)

    df = pd.DataFrame({"FrameIndex": range(1, len(ssim_vals)+1), "SSIM": ssim_vals, "PSNR": psnr_vals, "GCS": rel_contrast_vals})
    df.to_csv(output_csv_path, index=False)
    print(f"Saved metrics CSV to {output_csv_path}")

if __name__ == "__main__":
    inp = config.INP
    out = config.OUT
    left_color = config.LEFT_COLOR
    center_right_color = config.CENTER_RIGHT_COLOR

    input_dir = config.GENERATED_IMAGES_DIR_TEMPLATE.format(MODEL=config.MODEL, SETTINGS_ID=config.SETTINGS_ID, checkpoint="best")
    output_gif_path = os.path.join(input_dir, f"tinted_triplets_{inp}to{out}.gif")
    output_csv_path = os.path.join(input_dir, f"metrics_{inp}to{out}.csv")

    process_and_plot_generated_images(input_dir, output_gif_path, output_csv_path, left_color, center_right_color, inp, out)

    
    