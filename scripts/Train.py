# -*- coding: utf-8 -*-
"""
GAN Training Script with Validation SSIM Evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

import numpy as np
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_checkpoint
from dataset import CTGTDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torchmetrics.image import StructuralSimilarityIndexMeasure

from Evaluation import evaluate
from Postprocessing import process_and_plot_generated_images

import config

#######################################################
def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
    torch.manual_seed(base + worker_id)
########################################################

# Initialize SSIM metric globally
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)

# Save Configuration
def save_config(save_dir):
    config_path = os.path.join(save_dir, "config_parameters.txt")
    with open(config_path, "w") as f:
        f.write("Training Configuration Parameters\n")
        f.write("=" * 40 + "\n")
        keys_to_save = ["LEARNING_RATE", "BATCH_SIZE", "NUM_WORKERS", "L1_LAMBDA", "L_ADV", "NUM_EPOCHS"]
        for key in keys_to_save:
            value = getattr(config, key, "Not Found")
            f.write(f"{key}: {value}\n")
    print(f"Configuration parameters saved to {config_path}")

# Plotting Training Losses
def plot_losses(log_file, save_dir):
    df = pd.read_csv(log_file)
    df = df.dropna()
    x_values = df["Epoch"] + df["Batch"] / df["Batch"].max()
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, df["D_Loss"], label="Discriminator Loss", linestyle='dashed')
    plt.plot(x_values, df["G_Loss"], label="Generator Loss")
    plt.xlabel("Epochs (with batch-level resolution)")
    plt.ylabel("Loss")
    plt.title("Training Loss Evolution")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/loss_plot.png")
    plt.close()

# Plot function for SSIM and Grad-NCC
def plot_validation_metrics(val_log_file, save_dir):
    df = pd.read_csv(val_log_file)
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["SSIM"], label="SSIM", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Validation SSIM Evolution")
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.savefig(f"{save_dir}/validation_metrics_plot.png")
    plt.close()    

# Training function
def train_fn(epoch, disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, save_dir): 
    log_file = os.path.join(save_dir, "training_losses.csv")

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 0:
            writer.writerow(["Epoch", "Batch", "D_Loss", "G_Loss"])

        loop = tqdm(loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_fake = disc(x, y_fake.detach())
                D_loss = (bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake))) / 2

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generator
            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake) 
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake))            
                L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
                G_loss = config.L_ADV*G_fake_loss + L1

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            writer.writerow([epoch, idx, D_loss.item(), G_loss.item()])
            loop.set_description(f"D Loss: {D_loss.item():.4f}, G Loss: {G_loss.item():.4f}")

# Validation function with SSIM and Grad-NCC
def validate_ssim(gen, loader, save_dir, epoch, ssim_metric):
    gen.eval()
    ssim_scores = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            preds = gen(x)

            ssim_value = ssim_metric(preds, y) #was preds
            ssim_scores.append(ssim_value.item())

    avg_ssim     = float(np.mean(ssim_scores))

    print(f"\nValidation Metrics at Epoch {epoch}: SSIM = {avg_ssim:.4f}")

    val_log_file = os.path.join(save_dir, "validation_metrics.csv")
    new_file = not os.path.exists(val_log_file)

    with open(val_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["Epoch", "SSIM"])
        writer.writerow([epoch, avg_ssim])

    gen.train()
    return avg_ssim

# Main training
def main():
   
    print("using device: ", config.DEVICE)
    print("batch size: ", config.BATCH_SIZE)
    print("num epochs: ", config.NUM_EPOCHS)
    
    disc = Discriminator(in_channels=2).to(config.DEVICE) # # of channels modified
    gen = Generator(in_channels=1, out_channels=1).to(config.DEVICE) # # of channels modified

    optim_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    save_dir = config.MODEL_SAVE_DIR.format(MODEL=config.MODEL, SETTINGS_ID=config.SETTINGS_ID)
    os.makedirs(save_dir, exist_ok=True)

    save_config(save_dir)

    # Load Train Dataset
    input_dir = config.TRAIN_INPUT_DIR_TEMPLATE.format(INP=config.INP)
    output_dir = config.TRAIN_OUTPUT_DIR_TEMPLATE.format(OUT=config.OUT)
    
    # --- TRAIN DATASET/LOADER ---
    train_dataset = CTGTDataset(
        input_dir           = input_dir,
        target_dir          = output_dir,
        train               = True,
        jitter_range        = (0.1, 1.0),   # ← your requested range
        jitter_mode         = "tile_only",       # RAW and TILE get jitter
        deterministic_pairs = True,         # same γ for even/odd pair
        base_seed           = seed,         # identical across models
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size     = config.BATCH_SIZE,
        shuffle        = True,              # keep True for good training
        num_workers    = config.NUM_WORKERS,
        pin_memory     = True,
        persistent_workers = (config.NUM_WORKERS > 0),
        worker_init_fn = worker_init_fn,
        generator      = loader_gen,        # reproducible shuffle
    )

    # Load Test Dataset (for validation)
    test_input_dir = config.VALIDATION_INPUT_DIR_TEMPLATE.format(INP=config.INP)
    test_output_dir = config.VALIDATION_OUTPUT_DIR_TEMPLATE.format(OUT=config.OUT)
    
    # --- VALIDATION/TEST (NO JITTER) ---
    test_dataset = CTGTDataset(
        input_dir           = test_input_dir,
        target_dir          = test_output_dir,
        train               = False,        # disables jitter
        jitter_mode         = "none",
        deterministic_pairs = True,
        base_seed           = seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size     = 4,
        shuffle        = False,             # val/test: fixed order
        num_workers    = config.NUM_WORKERS,
        pin_memory     = True,
        persistent_workers = (config.NUM_WORKERS > 0),
        worker_init_fn = worker_init_fn,
        generator      = loader_gen,        # harmless here
    )
    
    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    CHECKPOINT_FREQUENCY = config.CHECKPOINT_FREQUENCY 
    
    # Initialize early stopping parameters
    best_ssim = 0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"epoch: {epoch}")
        train_fn(epoch, disc, gen, train_loader, optim_disc, optim_gen, L1_LOSS, BCE, gen_scaler, disc_scaler, save_dir) 

        # Validation - SSIM Evaluation
        avg_ssim  = validate_ssim(gen, test_loader, save_dir, epoch, ssim_metric)
        
        # Plot Training Losses and Validation SSIM
        plot_losses(os.path.join(save_dir, "training_losses.csv"), save_dir)
        plot_validation_metrics(os.path.join(save_dir, "validation_metrics.csv"), save_dir)


        # Save checkpoint
        if ((epoch % CHECKPOINT_FREQUENCY == 0) or (epoch == (config.NUM_EPOCHS - 1))):
            save_checkpoint(gen, optim_gen, filename=os.path.join(save_dir, f"generator_epoch_{epoch}.pth.tar"))
            save_checkpoint(disc, optim_disc, filename=os.path.join(save_dir, f"discriminator_epoch_{epoch}.pth.tar"))
            print(f"Saved checkpoint at epoch {epoch}")

        # Early Stopping Logic
        if avg_ssim > best_ssim + config.MIN_DELTA:
            best_ssim = avg_ssim
            patience_counter = 0
            best_checkpoint_filename = os.path.join(save_dir, f"generator_best.pth.tar")
            save_checkpoint(gen, optim_gen, filename=best_checkpoint_filename)
            print(f"Saved new best model at epoch {epoch} with SSIM: {best_ssim:.4f}")

        else:
            patience_counter += 1
            print(f"No SSIM improvement. Patience {patience_counter}/{config.PATIENCE}")

        # Add SSIM floor to prevent early stop too early
        if patience_counter >= config.PATIENCE and best_ssim > config.MIN_THRESHOLD:
            print(f"Early stopping triggered at epoch {epoch}. Best SSIM: {best_ssim:.4f}")
            break
        elif patience_counter >= config.PATIENCE:
            print(f"Early stop skipped (SSIM too low: {best_ssim:.4f} < {config.MIN_THRESHOLD})")

            
    # Runs whether early stopped or finished all epochs
    best_checkpoint_filename = os.path.join(save_dir, "generator_best.pth.tar")

    if os.path.exists(best_checkpoint_filename):
        test_input_dir = config.VALIDATION_INPUT_DIR_TEMPLATE.format(INP=config.INP)
        test_output_dir = config.VALIDATION_OUTPUT_DIR_TEMPLATE.format(OUT=config.OUT)
        output_dir = config.GENERATED_IMAGES_DIR_TEMPLATE.format(
            MODEL=config.MODEL,
            SETTINGS_ID=config.SETTINGS_ID,
            checkpoint="best"
        )
        os.makedirs(output_dir, exist_ok=True)

        print("\n--- Running evaluation on best checkpoint ---")
        evaluate(best_checkpoint_filename, test_input_dir, test_output_dir, output_dir)
        print(f"\nEvaluation completed. Results saved in {output_dir}")
    else:
        print(f"\nWarning: Best checkpoint not found at {best_checkpoint_filename}")

    # === Process generated images ===
    process_and_plot_generated_images(
        input_dir=output_dir,
        output_gif_path=os.path.join(output_dir, f"tinted_triplets_{config.INP}to{config.OUT}.gif"),
        output_csv_path=os.path.join(output_dir, f"metrics_{config.INP}to{config.OUT}.csv"),
        left_color=config.LEFT_COLOR,
        center_right_color=config.CENTER_RIGHT_COLOR,
        inp=config.INP,
        out=config.OUT
    )
        
if __name__ == "__main__":
    ############################################################################
    seed = 1337
    seed_everything(seed)
    loader_gen = torch.Generator().manual_seed(seed)   # controls shuffle order
    ############################################################################
    main()
