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

test_df = pd.read_csv(TEST_CSV)
print(f'loading custom prompts from: {PROMPT_FILE}')
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)
print(f"test set: {len(test_df)} samples")
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
ture_label = test_df[TARGET_GENRES].values
pred_label = []
with torch.no_grad():
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = os.path.join(IMG_FOLDER, row["img_filename"])
        img = Image.open(img_path).convert("RGB")
        img_input = preprocess(img).unsqueeze(0).to(device)
        img_feature = model.encode_image(img_input)
        text_feature = model.encode_text(text_tokens)
        
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feature @ text_feature.T 

        threshold = 0.15
        pred = (logits[0].cpu().numpy() > threshold).astype(int)
        pred_label.append(pred)
pred_label = np.array(pred_label)

macro_f1 = f1_score(ture_label, pred_label, average="macro")
micro_f1 = f1_score(ture_label, pred_label, average="micro")
print(f"Maco_f1: {round(macro_f1, 4)}")
print(f"Mico_f1: {round(micro_f1, 4)}")