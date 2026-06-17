"""
评估脚本：LoRA + BCE 模型
与 evaluation.py 评估逻辑完全一致，仅针对 BCE 训练模型。
"""
import torch
import clip
import os
import json
import pandas as pd
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm

TEST_CSV = "./test_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
GENRE_SPLIT = "./genre_split.json"
IMG_FOLDER = "./processed_posters"
PROMPT_FILE = "./clip_prompt_base.json"
MODEL_PATH = "./trained_clip_lora_bce/model_best_clip+lora_bce.pt"

LORA_R = 16
LORA_ALPHA = 64
SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model (LoRA + BCE)...")
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
TARGET_GENRES = list(label_map['label2id'].keys())

with open(GENRE_SPLIT, "r", encoding="utf-8") as f:
    genre_split = json.load(f)
train_genres = genre_split["train_genres"]
retained_genres = genre_split["retained_genres"]

train_indices = [TARGET_GENRES.index(g) for g in train_genres]
retained_indices = [TARGET_GENRES.index(g) for g in retained_genres]
print(f"train genres({len(train_genres)}): {train_genres}")
print(f"retained genres({len(retained_genres)}): {retained_genres}\n")

model, preprocess = clip.load("ViT-B/16", device=device)
for param in model.parameters():
    param.requires_grad = False
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["attn"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# BCE loss for evaluation (matches training loss)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


total_params, trainable_params = count_params(model)
print(f'total parameters:{total_params:,}')
print(f'nums of trainable parameters:{trainable_params:,}')
print(f'trainable parameter proportion:{trainable_params / total_params:.4}')

print(f'loading custom prompts from: {PROMPT_FILE}')
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)
text_prompt = [prompt_dict.get(g, f'a movie poster for a {g} film') for g in TARGET_GENRES]
text_tokens = clip.tokenize(text_prompt).to(device)
test_df = pd.read_csv(TEST_CSV)
print(f"test set: {len(test_df)} samples")
print(f'model starts reasoning...')

all_logits = []
all_label = []
total_loss = 0.0
with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = os.path.join(IMG_FOLDER, row['img_filename'])
        img_input = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        img_feature = model.encode_image(img_input).float()
        text_feature = model.encode_text(text_tokens).float()
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feature @ text_feature.T
        label = torch.tensor(row[TARGET_GENRES].values.astype(np.float32)).unsqueeze(0).to(device)
        loss = criterion(logits, label)
        total_loss += loss.item()

        all_logits.append(logits.cpu().numpy().squeeze())
        all_label.append(row[TARGET_GENRES].values.astype(np.float32))

all_logits = np.array(all_logits)
all_label = np.array(all_label)

FIXED_THRESHOLD = -4.0
preds = (all_logits > FIXED_THRESHOLD).astype(int)
print(f"fixed_threshold: {FIXED_THRESHOLD}")
macro_f1 = f1_score(all_label, preds, average="macro")
micro_f1 = f1_score(all_label, preds, average="micro")

train_f1_macro = f1_score(all_label[:, train_indices], preds[:, train_indices], average='macro')
train_f1_micro = f1_score(all_label[:, train_indices], preds[:, train_indices], average='micro')
retained_f1_macro = f1_score(all_label[:, retained_indices], preds[:, retained_indices], average='macro')
retained_f1_micro = f1_score(all_label[:, retained_indices], preds[:, retained_indices], average='micro')

overall_logit_mean = all_logits.mean()
retained_logit_mean = all_logits[:, retained_indices].mean()

correct_logit = np.mean([all_logits[i][all_label[i] == 1].mean() for i in range(len(all_logits)) if sum(all_label[i]) > 0])
wrong_logit = np.mean([all_logits[i][all_label[i] == 0].mean() for i in range(len(all_logits))])
logits_gap = correct_logit - wrong_logit

per_class_precision = precision_score(all_label, preds, average=None, zero_division=0)
per_class_recall = recall_score(all_label, preds, average=None, zero_division=0)
per_class_f1 = f1_score(all_label, preds, average=None, zero_division=0)
per_class_support = np.sum(all_label, axis=0)

per_genre_df = pd.DataFrame({
    'genre': TARGET_GENRES,
    'genres type': ['train_genres' if g in train_genres else 'retain_genres' for g in TARGET_GENRES],
    'num of sample': per_class_support,
    'precision': per_class_precision.round(4),
    'recall': per_class_recall.round(4),
    'F1': per_class_f1.round(4)
})
per_genre_df = per_genre_df.sort_values(by='num of sample', ascending=True).reset_index(drop=True)

train_group = per_genre_df[per_genre_df["genres type"] == "train_genres"]
retained_group = per_genre_df[per_genre_df["genres type"] == "retained_genres"]

print(f'model: lora_bce')
print(f'macro f1:{macro_f1:.4f}')
print(f'micro f1:{micro_f1:.4f}')
print(f'train f1 (macro/micro):{train_f1_macro:.4f} / {train_f1_micro:.4f}')
print(f'retained f1 (macro/micro):{retained_f1_macro:.4f} / {retained_f1_micro:.4f}')
print(f'average test loss (BCE):{total_loss / len(test_df):.4f}')
print(f'overall average logit:{overall_logit_mean:.4f}')
print(f'retain category average logit:{retained_logit_mean:.4f}')
print(f'correct logit:{correct_logit:.4f}')
print(f'wrong logits:{wrong_logit:.4f}')
print(f'correct-wrong logit gap:{logits_gap:.4f}')
print(f"{'genre':<15} {'genres type':<10} {'num of sample':<8} {'precision':<10} {'recall':<20} {'F1':<10}")
for _, row in per_genre_df.iterrows():
    mark = "! " if row["num of sample"] < 10 else "   "
    print(f"{mark}{row['genre']:<15} {row['genres type']:<15} {row['num of sample']:<10} {row['precision']:<10} {row['recall']:<20} {row['F1']:<10}")
print(f"train genre avg: precision={train_group['precision'].mean():.4f}, recall={train_group['recall'].mean():.4f}, F1(macro)={train_group['F1'].mean():.4f}, F1(micro)={train_f1_micro:.4f}")
print(f"retained genre avg: precision={retained_group['precision'].mean():.4f}, recall={retained_group['recall'].mean():.4f}, F1(macro)={retained_group['F1'].mean():.4f}, F1(micro)={retained_f1_micro:.4f}")
