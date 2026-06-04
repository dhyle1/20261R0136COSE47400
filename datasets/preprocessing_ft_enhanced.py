"""
Full FT 增强版：蒸馏 + neg_reg + 低学习率
三招对抗 FocalLoss 导致的编码器漂移和过预测
"""
import torch
import clip
import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def count_model_parameters(model):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params
    def format_params_size(num_params):
        bytes_ = num_params * 4
        if bytes_ < 1024:
            return f'{bytes_}B'
        elif bytes_ < 1024 * 1024:
            return f'{bytes_ / 1024:.4f}KB'
        elif bytes_ < 1024 * 1024 * 1024:
            return f'{bytes_ / (1024*1024):.4f}MB'
        else:
            return f'{bytes_ / (1024 * 1024 * 1024):.4f}GB'
    return{
        'total_params':total_params,
        'trainable_params':trainable_params,
        'nontrainable_params':non_trainable_params,
        'total_params_size':format_params_size(total_params),
        'nontrainable_params_size':format_params_size(non_trainable_params),
        'proportion_of trainable_params':f'{trainable_params / total_params:.4f}'
    }


print(f"run start — Full FT ENHANCED (distill + neg_reg + low LR)")
TRAIN_CSV = "./train_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters"
PROMPT_FILE = "./clip_prompt_base.json"
SAVE_MODLE_PATH = "./trained_clip_ft_enhanced"
GENRE_SPLIT = "./genre_split.json"
os.makedirs(SAVE_MODLE_PATH, exist_ok=True)
with open(GENRE_SPLIT, "r", encoding="utf-8") as f:
    genre_split = json.load(f)
train_genres = genre_split["train_genres"]
retained_genres = genre_split["retained_genres"]

print(f"train genres({len(train_genres)}): {train_genres}")
print(f"retained genres({len(retained_genres)}): {retained_genres}")

MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-5    # 降为原始的 1/5，减慢漂移
GAMMA = 2.0
NEG_REG = 0.2           # 负样本 BCE 保底
LAMBDA_DISTILL = 0.1    # 特征蒸馏强度（FT 比 LoRA 需要更大）
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cuda"

print(f"Hyperparams: LR={LEARNING_RATE}, gamma={GAMMA}, neg_reg={NEG_REG}, distill={LAMBDA_DISTILL}")
print(f"Using GPU : {torch.cuda.get_device_name(0)}")
with open("genre_label_map.json") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
id2label = label_map["id2label"]
TARGET_GENRES = list(label2id.keys())
NUM_CLASSES = len(TARGET_GENRES)
print(f'labels loaded:{NUM_CLASSES} genres')

# ================================================================
# Model: Full FT
# ================================================================
model, preprocess = clip.load("ViT-B/16", device=device)
model = model.float()
model.logit_scale.requires_grad = False
for name, param in model.named_parameters():
    if name != "logit_scale":
        param.requires_grad = True

param_stats = count_model_parameters(model)
for k, v in param_stats.items():
    print(f"{k}: {v}")
print("CLIP ViT-B/16 + full_fine-tuning configured")

# ================================================================
# Frozen CLIP for feature distillation
# ================================================================
frozen_clip, _ = clip.load("ViT-B/16", device=device)
frozen_clip.eval()
for p in frozen_clip.parameters():
    p.requires_grad = False
print(f"Frozen CLIP loaded for distillation (λ={LAMBDA_DISTILL})")

# ================================================================
# Data
# ================================================================
train_df = pd.read_csv(TRAIN_CSV)
print(f'loading custom prompts from: {PROMPT_FILE}')
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)
text_prompt = []
for genre in TARGET_GENRES:
    if genre in prompt_dict:
        text_prompt.append(prompt_dict[genre])
    else:
        default_prompt = f"a movie poster for a {genre} film"
        text_prompt.append(default_prompt)
        print(f'warning: No custom prompt for [{genre}], using default')
text_tokens = clip.tokenize(text_prompt).to(device)
print(f"custom prompts loaded successfully")

class MoviePosterDataset(torch.utils.data.Dataset):
    def __init__(self, df, preprocess, img_folder):
        self.df = df
        self.preprocess = preprocess
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.img_folder, row["img_filename"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)
        label_tensor = torch.tensor(row[TARGET_GENRES].values.astype(float), dtype=torch.float32)
        return img_tensor, label_tensor

train_dataset = MoviePosterDataset(train_df, preprocess, IMG_FOLDER)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ================================================================
# Class weights
# ================================================================
class_count = train_df[TARGET_GENRES].sum().values
class_count = torch.tensor(class_count).clamp(min=1)
class_weight_raw = torch.tensor(
    len(train_df) / (NUM_CLASSES * class_count), dtype=torch.float32
)
class_weight = torch.sqrt(class_weight_raw).clamp(min=1.0).to(device)
print(f"Class weight range: [{class_weight.min():.2f}, {class_weight.max():.2f}]")

# ================================================================
# FocalLoss + neg_reg
# ================================================================
class FocalLoss(torch.nn.Module):
    """Focal Loss with negative regularization.
    L = FL(alpha, gamma) + neg_reg * BCE_on_negatives
    """
    def __init__(self, alpha=None, gamma=2.0, neg_reg=0.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.neg_reg = neg_reg
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * 1.0
            loss = alpha_t * loss

        if self.neg_reg > 0:
            neg_mask = (1 - targets)
            loss = loss + self.neg_reg * bce_loss * neg_mask

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

loss_fn = FocalLoss(alpha=class_weight, gamma=GAMMA, neg_reg=NEG_REG, reduction='none')
print(f"Using FocalLoss (gamma={GAMMA}, neg_reg={NEG_REG})")

# ================================================================
# Training
# ================================================================
best_loss = float("inf")
best_model_path = os.path.join(SAVE_MODLE_PATH, "model_best_ft_enhanced.pt")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for batch_imgs, batch_labels in pbar:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        image_features = model.encode_image(batch_imgs).float()
        text_features = model.encode_text(text_tokens).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T

        loss = loss_fn(logits, batch_labels)

        mask = torch.ones_like(batch_labels)
        for genre in retained_genres:
            genre_idx = TARGET_GENRES.index(genre)
            mask[:, genre_idx] = 0.0
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-8)

        # Feature distillation
        if LAMBDA_DISTILL > 0:
            with torch.no_grad():
                frozen_feat = frozen_clip.encode_image(batch_imgs).float()
                frozen_feat = frozen_feat / frozen_feat.norm(dim=-1, keepdim=True)
            distill_loss = (1.0 - (image_features * frozen_feat).sum(dim=-1)).mean()
            loss = loss + LAMBDA_DISTILL * distill_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": round(loss.item(), 4)})

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'epoch {epoch + 1} finished. AVG_LOSS = {avg_loss:.4f}')

    epoch_save_path = os.path.join(SAVE_MODLE_PATH, f'model_epoch_{epoch + 1}.pt')
    torch.save(model.state_dict(), epoch_save_path)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(SAVE_MODLE_PATH, "model_best_ft_enhanced.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"best model updated (Loss: {best_loss:.4f})")
    print(f"Best model saved to: {best_model_path}")
