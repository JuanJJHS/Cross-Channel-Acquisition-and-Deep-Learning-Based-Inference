import torch
from torch.utils.data import DataLoader
import config
from dataset import CTGTDataset
from generator_model import Generator
from utils import load_checkpoint
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_model(checkpoint_path, device="cuda"):
    print(f"Loading model from {checkpoint_path}...")
    model = Generator(in_channels=1, out_channels=1).to(device) #using 2 channels (gamma is the aditional)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(checkpoint_path, model, optimizer, device)
    model.eval()
    return model


def to_uint8(arr_or_tensor, max_data_range=3.0):
    if torch.is_tensor(arr_or_tensor):
        arr = arr_or_tensor.squeeze().cpu().detach().numpy()
    else:
        arr = np.asarray(arr_or_tensor)
    arr = np.clip(arr, 0, max_data_range) / max_data_range * 255.0
    return arr.round().astype(np.uint8)

def save_composite_image(inp, gt, pred, filename, out_dir):
    """
    inp, gt, pred are tensors   (1,H,W) or (B=1,1,H,W)
    """
    os.makedirs(out_dir, exist_ok=True)

    inp_img  = Image.fromarray(to_uint8(inp,  1))
    gt_img   = Image.fromarray(to_uint8(gt,   1))
    pred_img = Image.fromarray(to_uint8(pred, 1))

    w, h = inp_img.width, inp_img.height
    sep  = Image.new("L", (2, h), 255)                  # white bar

    comp = Image.new("L", (w*3 + 4, h))
    comp.paste(inp_img,  (0,          0))
    comp.paste(sep,      (w,          0))
    comp.paste(gt_img,   (w + 2,      0))
    comp.paste(sep,      (w*2 + 2,    0))
    comp.paste(pred_img, (w*2 + 4,    0))

    comp.save(os.path.join(out_dir, filename))
# ------------------------------------------------------------

def evaluate(checkpoint_path, test_input_dir, test_output_dir, output_dir):
    device = config.DEVICE
    model = load_model(checkpoint_path, device)

    test_dataset = CTGTDataset(
        input_dir  = test_input_dir,
        target_dir = test_output_dir,
        train      = False           # ← disables γ-jitter
    )
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = config.NUM_WORKERS
    )
    
    loop = tqdm(test_loader)
    
    with torch.no_grad():
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)         
            y_fake = model(x)
            save_composite_image(x, y, y_fake, f"generated_{idx:04d}.png", output_dir) 
            print(f"Saved composite image {idx}")

 
if __name__ == "__main__":
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR.format(DATE=config.DATE, MODEL=config.MODEL, SETTINGS_ID=config.SETTINGS_ID),"generator_best.pth.tar")

    print(f"Using checkpoint path: {checkpoint_path}")
    test_input_dir = config.VALIDATION_INPUT_DIR_TEMPLATE.format(DATE=config.DATE, INP=config.INP, TEST_FOLDER=config.TEST_FOLDER)
    test_output_dir = config.VALIDATION_OUTPUT_DIR_TEMPLATE.format(DATE=config.DATE, OUT=config.OUT, TEST_FOLDER=config.TEST_FOLDER)
    output_dir = config.GENERATED_IMAGES_DIR_TEMPLATE.format(MODEL=config.MODEL,SETTINGS_ID=config.SETTINGS_ID,TEST_FOLDER=config.TEST_FOLDER,checkpoint="best")
    os.makedirs(output_dir, exist_ok=True)

    evaluate(checkpoint_path, test_input_dir, test_output_dir, output_dir)
   