#Macro F1: 0.1266
#Micro F1: 0.2546
import torch
import clip
import os
import json
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm




TEST_CSV = "./test_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters"

TRAINED_MODEL_PATH = "./model_best.pt"

MODEL_NAME = "ViT-B/16"
PROMPT_FILE = "./clip_prompt_base.json"
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")



with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)

label2id = label_map["label2id"]
id2label = label_map["id2label"]
TARGET_GENRES = list(label2id.keys())
NUM_CLASSES = len(TARGET_GENRES)

print(f"labels loaded: {NUM_CLASSES} genres")


model, preprocess = clip.load(MODEL_NAME, device=device)

model = model.float()

checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=device)

# 저장 방식이 checkpoint dict인 경우
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"checkpoint loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"best loss: {checkpoint.get('best_loss', checkpoint.get('avg_loss', 'unknown'))}")

else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Full fine-tuned CLIP model loaded")



test_df = pd.read_csv(TEST_CSV)

print(f"test set: {len(test_df)} samples")
print(f"loading custom prompts from: {PROMPT_FILE}")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)

text_prompt = []

for genre in TARGET_GENRES:
    if genre in prompt_dict:
        text_prompt.append(prompt_dict[genre])
    else:
        default_prompt = f"a movie poster for a {genre} film"
        text_prompt.append(default_prompt)
        print(f"warning: No custom prompt for [{genre}], using default")

text_tokens = clip.tokenize(text_prompt).to(device)



class MoviePosterTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, preprocess, img_folder, target_genres):
        self.df = df
        self.preprocess = preprocess
        self.img_folder = img_folder
        self.target_genres = target_genres

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = os.path.join(self.img_folder, row["img_filename"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)

        label_tensor = torch.tensor(
            row[self.target_genres].values.astype(float),
            dtype=torch.float32
        )

        return img_tensor, label_tensor


test_dataset = MoviePosterTestDataset(
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



print("inferencing...")

true_label_list = []
pred_label_list = []


#THRESHOLD = -2.5
THRESHOLD = 0.08

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for batch_imgs, batch_labels in tqdm(test_loader):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        image_features = model.encode_image(batch_imgs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        probs = torch.sigmoid(logits)
        preds = (probs.cpu().numpy() > THRESHOLD).astype(int)

        pred_label_list.append(preds)
        true_label_list.append(batch_labels.cpu().numpy())

pred_label = np.concatenate(pred_label_list, axis=0)
true_label = np.concatenate(true_label_list, axis=0)


# =========================================================
# Metrics
# =========================================================

macro_f1 = f1_score(true_label, pred_label, average="macro", zero_division=0)
micro_f1 = f1_score(true_label, pred_label, average="micro", zero_division=0)

print(f"Macro F1: {round(macro_f1, 4)}")
print(f"Micro F1: {round(micro_f1, 4)}")

print("\nClassification Report:")
print(
    classification_report(
        true_label,
        pred_label,
        target_names=TARGET_GENRES,
        zero_division=0
    )
)