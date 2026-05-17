import torch
import clip
import os
import json
import pandas as pd 
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm

VAL_CSV = 'train_annotations2.csv'
TEST_CSV = "./test_annotations.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters"
TRAINED_MODEL_PATH = "./trained_clip_lora/model_best.pt"
MODEL_NAME = "ViT-B/16"
PROMPT_FILE = "./clip_prompt_base.json"
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 5e-5
LORA_R = 16
LORA_ALPHA = 64
SEED = 42
torch.manual_seed(SEED)
device = "cuda"

print("Loading model...")
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
id2label = label_map["id2label"]
TARGET_GENRES = list(label2id.keys())

model, preprocess = clip.load("ViT-B/16")
for param in model.parameters():
    param.required_grad = False

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["attn"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
model.eval()
print(f"model loaded")

val_df = pd.read_csv(VAL_CSV, encoding='latin-1')

# filters out rows with missing poster files
val_df = val_df[
    val_df["img_filename"].apply(
        lambda filename: os.path.exists(os.path.join(IMG_FOLDER, filename))
    )
].reset_index(drop=True)

print(f'loading custom prompts from: {PROMPT_FILE}')
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)
print(f"val set: {len(val_df)} samples")
text_prompt = []
for genre in TARGET_GENRES:
    if genre in prompt_dict:
        text_prompt.append(prompt_dict[genre])
    else:
        default_prompt = f"a movie poster for a {genre} film"
        text_prompt.append(default_prompt)
        print(f'warning: No custiom prompt for [{genre}], using default')
text_tokens = clip.tokenize(text_prompt).to(device)

print(f"inferencing...")
ture_label = val_df[TARGET_GENRES].values
all_logits = []
with torch.no_grad():
    for index, row in tqdm(val_df.iterrows(), total=len(val_df)):
        img_path = os.path.join(IMG_FOLDER, row["img_filename"])
        img = Image.open(img_path).convert("RGB")
        img_input = preprocess(img).unsqueeze(0).to(device)
        img_feature = model.encode_image(img_input)
        text_feature = model.encode_text(text_tokens)
        
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feature @ text_feature.T 

        all_logits.append(logits[0].cpu().numpy())
all_logits = torch.tensor(np.array(all_logits))
all_probs = torch.sigmoid(all_logits).numpy() # apply sigmoid

# top3
top3_indices = np.argsort(all_probs, axis=1)[:, -3:]

top3_correct = 0

for i, true_row in enumerate(ture_label):
    true_indices = np.where(true_row == 1)[0]

    if any(idx in top3_indices[i] for idx in true_indices):
        top3_correct += 1

top3_accuracy = top3_correct / len(ture_label)

print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

# threshold search
threshold_candidates = np.arange(0.05, 0.55, 0.05)
best_macro = 0
best_preds = None
for threshold in threshold_candidates:
    preds = (all_probs > threshold).astype(int)
    macro = f1_score(ture_label, preds, average="macro")
    if macro > best_macro:
        best_macro = macro
        best_threshold = threshold
        best_preds = preds
print(f'best threshold: {best_threshold}')
print(f"best macro_f1: {round(best_macro, 4)}")
macro_f1 = f1_score(ture_label, best_preds, average="macro")
micro_f1 = f1_score(ture_label, best_preds, average="micro")
print(f"Final Maco_f1: {round(macro_f1, 4)}")
print(f"Final Mico_f1: {round(micro_f1, 4)}")
