# Cross-Channel-Acquisition-and-Deep-Learning-Based-Inference

U-Net / Pix2Pix pipeline for **reconstructing fluorescence channels** from **cross-channel frames** acquired in multicolor microscopy.  
The framework trains per-target models (U-Net baselines and Pix2Pix cGANs) to map a single cross-channel input (e.g., CM→MM or CM→CC) to missing channels in **real time**, with helpers to visualize triplets, concatenate GIFs, and compute metrics.

---

## Key ideas

- **Cross-channel input**: pick one acquired channel whose excitation/emission overlaps the targets so it carries *complementary* structural + spectral cues.
- **Per-channel models**: train one model per target channel for spectral specificity and stability.
- **Real-time deployment**: Torch-TensorRT acceleration (optional) for low-latency inference.
- **Visualization-first**: utilities to crop subframes, build 3-panel/5-panel GIFs with 2-px separators, tint outputs, and overlay predictions vs. GT.

---

## Workflow

  1) Install the requirements.txt
  2) Run Preprocessing.py
  3) Run run_models.py
