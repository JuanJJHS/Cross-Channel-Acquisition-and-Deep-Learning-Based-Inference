import os, glob
import torch
import torch.nn as nn
from pathlib import Path

from generator_model import Generator

class MultiOutputGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super().__init__()
        self.gen1 = Generator(in_channels, out_channels, features)
        self.gen2 = Generator(in_channels, out_channels, features)
    def forward(self, x):
        return self.gen1(x), self.gen2(x)

def load_ckpt_maybe(path, device):
    ckpt = torch.load(path, map_location=device)
    # support both {'state_dict': ...} and raw state_dict
    return ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

if __name__ == "__main__":
    MODEL_ID = 0

    # === Resolve repo root (= parent of scripts/) ===
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_ROOT  = SCRIPT_DIR.parent  

    # Preferred checkpoint names
    ckpt_CM_MM = DATA_ROOT / f"Models/CMtoMM/{MODEL_ID}/generator_best.pth.tar"
    ckpt_CM_CC = DATA_ROOT / f"Models/CMtoCC/{MODEL_ID}/generator_best.pth.tar"

    # Fallback to latest epoch file if "best" is missing
    if not ckpt_CM_MM.is_file():
        candidates = sorted((DATA_ROOT / f"Models/CMtoMM/{MODEL_ID}").glob("generator_epoch_*.pth.tar"))
        if candidates:
            ckpt_CM_MM = candidates[-1]
    if not ckpt_CM_CC.is_file():
        candidates = sorted((DATA_ROOT / f"Models/CMtoCC/{MODEL_ID}").glob("generator_epoch_*.pth.tar"))
        if candidates:
            ckpt_CM_CC = candidates[-1]

    # Final existence check
    if not ckpt_CM_MM.is_file():
        raise FileNotFoundError(f"Missing CM->MM checkpoint at {ckpt_CM_MM}")
    if not ckpt_CM_CC.is_file():
        raise FileNotFoundError(f"Missing CM->CC checkpoint at {ckpt_CM_CC}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init and load
    multi_gen = MultiOutputGenerator(in_channels=1, out_channels=1, features=16).to(device)
    multi_gen.gen1.load_state_dict(load_ckpt_maybe(str(ckpt_CM_MM), device))
    multi_gen.gen2.load_state_dict(load_ckpt_maybe(str(ckpt_CM_CC), device))
    print("✓ Loaded gen1 (CM→MM) and gen2 (CM→CC).")

    # Save merged model under repo root
    merged_dir = DATA_ROOT / f"Models/MultiOutput/{MODEL_ID}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merged_dir / "merged_multi_generator_CMtoMM_CC.pth.tar"
    torch.save({'state_dict': multi_gen.state_dict()}, str(merged_path))
    print(f"✓ Saved combined multi-output generator to '{merged_path}'")

    # Quick inference smoke test
    multi_gen.eval()
    dummy_input = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        out1, out2 = multi_gen(dummy_input)
    print(f"Outputs: {tuple(out1.shape)}, {tuple(out2.shape)}")
