import subprocess

configs = [
    {"INP": "CM", "OUT": "CC", "L1_LAMBDA": 100, "L_ADV": 0, "SETTINGS_ID": 0},
    {"INP": "CM", "OUT": "MM", "L1_LAMBDA": 100, "L_ADV": 0, "SETTINGS_ID": 0},
    {"INP": "CM", "OUT": "CC", "L1_LAMBDA": 100, "L_ADV": 0.05, "SETTINGS_ID": 1},
    {"INP": "CM", "OUT": "MM", "L1_LAMBDA": 100, "L_ADV": 0.05, "SETTINGS_ID": 1},
]

config_template = """
import torch
import os

def get_center_right_color(out_channel):
    color_map = {{
        "MM": "green",
        "CC": "red",
        "BB": "blue"
    }}
    return color_map.get(out_channel, "green")

# --- compute repo root as parent of scripts/ (this file lives in scripts/) ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 8

SETTINGS_ID = {SETTINGS_ID}
L_ADV = {L_ADV}
L1_LAMBDA = {L1_LAMBDA}
NUM_EPOCHS = 200

CHECKPOINT_FREQUENCY = 100
PATIENCE = 10
MIN_DELTA = 5e-4
MIN_THRESHOLD = 0.2

INP = "{INP}"
OUT = "{OUT}"
MODEL = f"{{INP}}to{{OUT}}"

LEFT_COLOR = "red"
CENTER_RIGHT_COLOR = get_center_right_color(OUT)

# --- ABSOLUTE paths built as simple strings (no Path ops, no templates left) ---
MODEL_SAVE_DIR = DATA_ROOT + f"/Models/{{MODEL}}/{{SETTINGS_ID}}"
TRAIN_INPUT_DIR_TEMPLATE = DATA_ROOT + f"/Train/{{INP}}"
TRAIN_OUTPUT_DIR_TEMPLATE = DATA_ROOT + f"/Train/{{OUT}}"
VALIDATION_INPUT_DIR_TEMPLATE = DATA_ROOT + f"/Validation/{{INP}}"
VALIDATION_OUTPUT_DIR_TEMPLATE = DATA_ROOT + f"/Validation/{{OUT}}"

CHECKPOINT_PATH_TEMPLATE = DATA_ROOT + f"/Models/{{MODEL}}/{{SETTINGS_ID}}/generator_epoch_{{{{checkpoint}}}}.pth.tar"
GENERATED_IMAGES_DIR_TEMPLATE = DATA_ROOT + f"/Generated_Frames/{{MODEL}}_{{SETTINGS_ID}}_{{{{checkpoint}}}}_epochs"
"""

for config in configs:
    print(f"Running config: {config}")
    with open("config.py", "w") as f:
        f.write(config_template.format(**config))

    subprocess.run(["python", "Train.py"])
