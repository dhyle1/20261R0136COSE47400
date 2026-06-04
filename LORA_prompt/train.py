import torch
import clip
import os
import pandas as pd
import json
from PIL import Image
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn.functional as F 

# ==========================================
# 1. Multi-Label Focal Loss 정의 (추가된 부분)
# ==========================================
class MultiLabelFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='none'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha  # 클래스별 불균형 가중치 (BCE의 pos_weight 역할)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        # 수치적 안정성을 위해 F.binary_cross_entropy_with_logits 활용
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 예측 확률 p 및 p_t 계산
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Focal Loss 핵심 컴포넌트 적용
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        # 클래스 균형 가중치(alpha) 적용
        if self.alpha is not None:
            alpha_factor = targets * self.alpha + (1 - targets)
            loss = alpha_factor * loss

        # 제외할 장르(retained_genres) 마스킹 처리
        if mask is not None:
            loss = loss * mask
            
            return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ==========================================
# 2. 유틸리티 함수 정의
# ==========================================
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
            
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'nontrainable_params': non_trainable_params,
        'total_params_size': format_params_size(total_params),
        'nontrainable_params_size': format_params_size(non_trainable_params),
        'proportion_of trainable_params': f'{trainable_params / total_params:.4f}'
    }

# ==========================================
# 3. 환경 설정 및 데이터 로드
# ==========================================
print(f"run start")
TRAIN_CSV = "./train_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters/processed_posters"
PROMPT_FILE = "./genre_specialized_prompt.json"
SAVE_MODLE_PATH = "./trained_clip_lora"
GENRE_SPLIT = "./genre_split.json"

# 폴더 생성 생성 보장
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
LEARNING_RATE = 5e-5
LORA_R = 16
LORA_ALPHA = 64
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using GPU : {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
id2label = label_map["id2label"]
TARGET_GENRES = list(label2id.keys())
NUM_CLASSES = len(TARGET_GENRES)
print(f'labels loaded:{NUM_CLASSES} genres')

# ==========================================
# 4. CLIP 모델 및 LoRA 설정
# ==========================================
model, preprocess = clip.load(MODEL_NAME, device=device)
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

# ==========================================
# 5. 프롬프트 토큰화 및 데이터셋 설정
# ==========================================
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
        # 무한 루프 방지를 위해 최대 시도 횟수를 지정합니다.
        for _ in range(10):
            row = self.df.iloc[index]
            pure_filename = os.path.basename(row["img_filename"])
            img_path = os.path.join(self.img_folder, pure_filename)
            
            try:
                # 이미지를 열어보고 정상적이면 그대로 반환합니다.
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img)
                label_tensor = torch.tensor(row[TARGET_GENRES].values.astype(float), dtype=torch.float32)
                return img_tensor, label_tensor
                
            except FileNotFoundError:
                
                index = (index + 1) % len(self.df)
                
        # 만약 10번 연속으로 파일이 없다면 에러를 내어 데이터셋 상태를 점검하게 합니다.
        raise RuntimeError("Too many missing image files in a row. Please check your image folder.")
train_dataset = MoviePosterDataset(train_df, preprocess, IMG_FOLDER)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

# ==========================================
# 6. 최적화 및 Focal Loss 빌드
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 클래스별 가중치 계산 (가중치 밸런싱용)
class_count = train_df[TARGET_GENRES].sum().values
class_count = torch.tensor(class_count).clamp(min=1)
class_weight = torch.tensor(
    len(train_df) / (NUM_CLASSES * class_count),
    dtype=torch.float32
).to(device)

# 변경된 부분: MultiLabelFocalLoss 적용 (gamma 파라미터 2.0 세팅)
loss_fn = MultiLabelFocalLoss(alpha=class_weight, gamma=2.0, reduction='none')
print(f"start training with Focal Loss...")

# ==========================================
# 7. Training Loop (학습 루프)
# ==========================================
best_loss = float("inf")
best_model_path = os.path.join(SAVE_MODLE_PATH, "model_best_clip+lora")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    
    for batch_imgs, batch_labels in pbar:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()

        # CLIP Feature 추출 및 정규화(L2 Norm)
        image_features = model.encode_image(batch_imgs).float()
        text_features = model.encode_text(text_tokens).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Logits 계산 (Cosine Similarity * 100)
        logits = 100.0 * image_features @ text_features.T

        # 마스크 행렬 생성 (retained_genres에 해당하는 클래스는 손실 계산 제외)
        mask = torch.ones_like(batch_labels)
        for genre in retained_genres:
            genre_idx = TARGET_GENRES.index(genre)
            mask[:, genre_idx] = 0.0

        # 변경된 부분: Focal Loss 함수 내부에서 마스킹 및 평균 계산까지 처리 완료
        loss = loss_fn(logits, batch_labels, mask=mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": round(loss.item(), 4)})
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'epoch {epoch + 1} finished. AVG_LOSS = {avg_loss}')

    # Epoch 단위 체크포인트 저장
    epoch_save_path = os.path.join(SAVE_MODLE_PATH, f'model_epoch_{epoch + 1}')
    model.save_pretrained(epoch_save_path)
    torch.save(model.state_dict(), os.path.join(SAVE_MODLE_PATH, f'model_epoch_{epoch + 1}.pt'))

    # Best Model 갱신 및 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save_pretrained(best_model_path)
        torch.save(model.state_dict(), os.path.join(SAVE_MODLE_PATH, "model_best_clip+lora.pt"))
        print(f"best model updated (Loss : {round(best_loss, 4)})")
        
    print(f"Best model saved to: {best_model_path}")