import torch
import clip
import os
import pandas as pd
import json
from PIL import Image
from peft import LoraConfig, get_peft_model
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
        

print(f"run start")
TRAIN_CSV = "./train_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters"
PROMPT_FILE = "./clip_prompt_base.json"
SAVE_MODLE_PATH = "./trained_clip_lora"

GENRE_SPLIT = "./genre_split.json"
with open(GENRE_SPLIT, "r", encoding="utf-8") as f:
    genre_split = json.load(f)
train_genres = genre_split["train_genres"]
retained_genres = genre_split["retained_genres"]

print(f"train genres({len(train_genres)}): {train_genres}")
print(f"retained genres({len(retained_genres)}): {retained_genres}")

MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 5e-5
LORA_R = 16
LORA_ALPHA = 64
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cuda"

print(f"Using GPU : {torch.cuda.get_device_name(0)}")
with open("genre_label_map.json") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
id2label = label_map["id2label"]
TARGET_GENRES = list(label2id.keys())
NUM_CLASSES = len(TARGET_GENRES)
print(f'labels loaded:{NUM_CLASSES} genres')

model, preprocess = clip.load("ViT-B/16")
for param in model.parameters():
    param.requires_grad = False
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["attn"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)

param_stats = count_model_parameters(model)

for k, v in param_stats.items():
    print(f"{k}: {v}")
model.print_trainable_parameters()
print("CLIP ViT-B/16 + LoRA configured")

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
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=0.01
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS
    )
class_count = train_df[TARGET_GENRES].sum().values
class_count = torch.tensor(class_count).clamp(min=1)
class_weight = torch.tensor(
    len(train_df) / (NUM_CLASSES * class_count),
    dtype=torch.float32
).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)
print(f"start training...")

best_loss = float("inf")
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
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fn(logits, batch_labels)

        mask = torch.ones_like(batch_labels)
        for genre in retained_genres:
            genre_idx = TARGET_GENRES.index(genre)
            mask[:, genre_idx] = 0.0

        loss = loss * mask

        loss = loss.sum() / (mask.sum() + 1e-8)   

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss":round(loss.item(), 4)})
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'epoch{epoch + 1} finished. AVG_LOSS = {avg_loss}')

    epoch_save_path = os.path.join(SAVE_MODLE_PATH, f'model_epoch_{epoch + 1}')
    model.save_pretrained(epoch_save_path)
    torch.save(model.state_dict(), os.path.join(SAVE_MODLE_PATH, f'model_epoch_{epoch + 1}.pt'))

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(SAVE_MODLE_PATH,"model_best_clip+lora")
        model.save_pretrained(best_model_path)
        torch.save(model.state_dict(), os.path.join(SAVE_MODLE_PATH, "model_best_clip+lora.pt"))
        print(f"best model updated(Loss : {round(best_loss, 4)})")
    print(f"Best model saved to: {best_model_path}")
