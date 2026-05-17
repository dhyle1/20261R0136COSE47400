#threshold search
import os
import json
import torch
import clip
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    classification_report
)

# =========================================================
# Config
# =========================================================

TEST_CSV = "./test_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters"

TRAINED_MODEL_PATH = "./model_best.pt"

MODEL_NAME = "ViT-B/16"
PROMPT_FILE = "./clip_prompt_base.json"

BATCH_SIZE = 8
SEED = 42

THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

# =========================================================
# Seed / Device
# =========================================================

torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# =========================================================
# Load Label Info
# =========================================================

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)

label2id = label_map["label2id"]
id2label = label_map["id2label"]

TARGET_GENRES = list(label2id.keys())
NUM_CLASSES = len(TARGET_GENRES)

print(f"Loaded {NUM_CLASSES} genres")

# =========================================================
# Load CLIP Model
# =========================================================

print("Loading CLIP model...")

model, preprocess = clip.load(MODEL_NAME, device=device)

model = model.float()

checkpoint = torch.load(
    TRAINED_MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Checkpoint loaded "
        f"(epoch={checkpoint.get('epoch', 'unknown')})"
    )

else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully")

# =========================================================
# Load Test CSV
# =========================================================

test_df = pd.read_csv(TEST_CSV)

print(f"Test samples: {len(test_df)}")

# =========================================================
# Load Prompt Templates
# =========================================================

print(f"Loading prompts from: {PROMPT_FILE}")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)

text_prompts = []

for genre in TARGET_GENRES:

    if genre in prompt_dict:
        text_prompts.append(prompt_dict[genre])

    else:
        default_prompt = f"a movie poster for a {genre} film"
        text_prompts.append(default_prompt)

        print(
            f"[WARNING] "
            f"No prompt for {genre}, using default"
        )

text_tokens = clip.tokenize(text_prompts).to(device)

# =========================================================
# Dataset
# =========================================================

class MoviePosterDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        df,
        preprocess,
        img_folder,
        target_genres
    ):
        self.df = df
        self.preprocess = preprocess
        self.img_folder = img_folder
        self.target_genres = target_genres

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.img_folder,
            row["img_filename"]
        )

        image = Image.open(img_path).convert("RGB")

        image_tensor = self.preprocess(image)

        label_tensor = torch.tensor(
            row[self.target_genres].values.astype(float),
            dtype=torch.float32
        )

        return image_tensor, label_tensor

# =========================================================
# DataLoader
# =========================================================

test_dataset = MoviePosterDataset(
    test_df,
    preprocess,
    IMG_FOLDER,
    TARGET_GENRES
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =========================================================
# Inference
# =========================================================

print("Starting inference...")

all_probs = []
all_true_labels = []

with torch.no_grad():

    # Encode text prompts
    text_features = model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    for batch_imgs, batch_labels in tqdm(test_loader):

        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        # Encode images
        image_features = model.encode_image(batch_imgs)

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        # Similarity logits
        logit_scale = model.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.T

        # Sigmoid probabilities
        probs = torch.sigmoid(logits)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_true_labels.append(
            batch_labels.cpu().numpy()
        )

# =========================================================
# Merge Results
# =========================================================

all_probs = np.concatenate(all_probs, axis=0)

true_labels = np.concatenate(
    all_true_labels,
    axis=0
)

print("Inference completed")

# =========================================================
# Threshold Search
# =========================================================

print("\n==============================")
print("Threshold Search")
print("==============================")

best_threshold = None
best_macro_f1 = -1
best_micro_f1 = -1
best_predictions = None

for threshold in THRESHOLDS:

    predictions = (
        all_probs > threshold
    ).astype(int)

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        true_labels,
        predictions,
        average="micro",
        zero_division=0
    )

    print(
        f"Threshold = {threshold:.2f} | "
        f"Macro F1 = {macro_f1:.4f} | "
        f"Micro F1 = {micro_f1:.4f}"
    )

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1

        best_threshold = threshold
        best_predictions = predictions

# =========================================================
# Best Result
# =========================================================

print("\n==============================")
print("BEST RESULT")
print("==============================")

print(f"Best Threshold : {best_threshold}")
print(f"Best Macro F1 : {best_macro_f1:.4f}")
print(f"Best Micro F1 : {best_micro_f1:.4f}")

# =========================================================
# Classification Report
# =========================================================

print("\n==============================")
print("Classification Report")
print("==============================")

report = classification_report(
    true_labels,
    best_predictions,
    target_names=TARGET_GENRES,
    zero_division=0
)

print(report)
