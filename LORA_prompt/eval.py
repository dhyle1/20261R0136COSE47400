import torch
import clip
import os
import pandas as pd
import json
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# ==========================================
# 1. 환경 설정 및 경로 지정
# ==========================================
print("Evaluation process starting...")
TEST_CSV = "./test_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters/processed_posters"
PROMPT_FILE = "./genre_specialized_prompt.json"

# 저장했던 베스트 모델 폴더 경로 (LoRA 가중치가 들어있는 곳)
LOAD_MODEL_PATH = "./trained_clip_lora/model_best_clip+lora" 

MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

# 장르 맵 로드
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
TARGET_GENRES = list(label2id.keys())

# 테스트 CSV 로드 및 컬럼 동기화
test_df = pd.read_csv(TEST_CSV)
TARGET_GENRES = [genre for genre in TARGET_GENRES if genre in test_df.columns]
NUM_CLASSES = len(TARGET_GENRES)
print(f"Evaluation targets: {NUM_CLASSES} genres")


# ==========================================
# 2. 예외 처리가 포함된 Evaluation Dataset 정의
# ==========================================
class MoviePosterEvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, preprocess, img_folder):
        self.df = df
        self.preprocess = preprocess
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # 최대 10번 재시도하며 없는 파일 건너뛰기
        for _ in range(10):
            row = self.df.iloc[index]
            pure_filename = os.path.basename(row["img_filename"])
            img_path = os.path.join(self.img_folder, pure_filename)
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img)
                label_tensor = torch.tensor(row[TARGET_GENRES].values.astype(float), dtype=torch.float32)
                return img_tensor, label_tensor, True # 정상 로드 여부 플래그
                
            except FileNotFoundError:
                index = (index + 1) % len(self.df)
                
        # 연속으로 실패할 경우 더미 데이터 반환 (Dataloader 에러 방지용 가짜 플래그)
        return torch.zeros(3, 224, 224), torch.zeros(NUM_CLASSES), False

test_dataset = MoviePosterEvalDataset(test_df, preprocess=None, img_folder=IMG_FOLDER)


# ==========================================
# 3. CLIP 모델 로드 및 LoRA 가중치 병합
# ==========================================
base_model, preprocess = clip.load(MODEL_NAME, device=device)
test_dataset.preprocess = preprocess  # 전처리 함수 연결

print(f"Loading LoRA weights from: {LOAD_MODEL_PATH}")
model = PeftModel.from_pretrained(base_model, LOAD_MODEL_PATH)
model.to(device)
model.eval()
print("Model loaded and set to evaluation mode.")


# ==========================================
# 4. 텍스트 프롬프트 토큰화
# ==========================================
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)

text_prompt = []
for genre in TARGET_GENRES:
    if genre in prompt_dict:
        text_prompt.append(prompt_dict[genre])
    else:
        text_prompt.append(f"a movie poster for a {genre} film")

text_tokens = clip.tokenize(text_prompt).to(device)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)


# ==========================================
# 5. Inference & 예측 확률값 수집
# ==========================================
all_preds = []
all_targets = []

print("Starting Inference...")
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for batch_imgs, batch_labels, valid_flags in tqdm(test_loader, desc="Evaluating"):
        
        valid_idx = torch.where(valid_flags == True)[0]
        if len(valid_idx) == 0:
            continue
            
        batch_imgs = batch_imgs[valid_idx].to(device)
        batch_labels = batch_labels[valid_idx].to(device)

        image_features = model.encode_image(batch_imgs).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits = 100.0 * image_features @ text_features.T
        probs = torch.sigmoid(logits)
        
        preds_probs = probs.cpu().numpy()
        targets = batch_labels.cpu().numpy()
        
        all_preds.append(preds_probs) 
        all_targets.append(targets)

# 수집된 배치를 하나의 거대한 행렬로 통합
if len(all_preds) > 0:
    all_preds_probs = np.vstack(all_preds) 
    all_targets = np.vstack(all_targets)
else:
    all_preds_probs = np.array([])
    all_targets = np.array([])


# ==========================================
# 6. 다양한 Threshold 탐색 및 성능 메트릭 집계 (Precision, Recall, F1 기본 평가)
# ==========================================
if len(all_preds_probs) == 0:
    print("\n" + "!"*50)
    print(" CRITICAL ERROR: No valid images were evaluated!")
    print("!"*50)
else:
    best_f1 = -1
    best_thresh = 0.5
    best_metrics = {}
    
    threshold_results = []

    # 0.05부터 0.95까지 0.05 간격으로 탐색
    threshold_candidates = np.arange(0.05, 0.96, 0.05)
    
    print("\nSearching for the best threshold...")
    for thresh in threshold_candidates:
        # 임계값 기준 이진화 (0 또는 1)
        current_preds = (all_preds_probs > thresh).astype(int)
        
        # 기본 Precision, Recall, F1-Score 계산 (각 클래스별 지표의 단순 평균)
        prec = precision_score(all_targets, current_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, current_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, current_preds, average='macro', zero_division=0)
        
        threshold_results.append({
            'Threshold': round(thresh, 2),
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        # F1-Score를 최대로 만드는 최적의 임계값 지점 저장
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {'Precision': prec, 'Recall': rec, 'F1-Score': f1}

    # --- 테이블 형태로 결과 출력 ---
    print("\n" + "="*55)
    print(f" {'Thresh':<8} | {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("="*55)
    for res in threshold_results:
        print(f" {res['Threshold']:<8} | {res['Precision']:<12.4f} {res['Recall']:<12.4f} {res['F1-Score']:<12.4f}")
    print("="*55)
    
    # --- 종합 요약 리포트 ---
    print("\n" + "*"*50)
    print("          👑 BEST THRESHOLD SUMMARY          ")
    print("*"*50)
    print(f" 🌟 Best Threshold : {best_thresh:.2f}")
    print(f"    -> Precision   : {best_metrics['Precision']:.4f}")
    print(f"    -> Recall      : {best_metrics['Recall']:.4f}")
    print(f"    -> F1-Score    : {best_metrics['F1-Score']:.4f}")
    print("*"*50)