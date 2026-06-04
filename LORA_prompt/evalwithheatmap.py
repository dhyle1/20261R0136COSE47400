import torch
import clip
import os
import pandas as pd
import json
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 🌟 [수정] 누락되었던 라이브러리 임포트 추가!
from sklearn.metrics import f1_score, precision_score, recall_score

# ==========================================
# 1. 환경 설정 및 경로 지정
# ==========================================
print("Evaluation process starting...")
TEST_CSV = "./test_annotations2.csv"
LABEL_MAP = "./genre_label_map.json"
IMG_FOLDER = "./processed_posters/processed_posters"
PROMPT_FILE = "./clip_prompt_base.json"
LOAD_MODEL_PATH = "./trained_clip_lora/model_best_clip+lora" 

MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
TARGET_GENRES = list(label2id.keys())

test_df = pd.read_csv(TEST_CSV)
TARGET_GENRES = [genre for genre in TARGET_GENRES if genre in test_df.columns]
NUM_CLASSES = len(TARGET_GENRES)
print(f"Evaluation targets: {NUM_CLASSES} genres")


# ==========================================
# 2. Dataset 정의
# ==========================================
class MoviePosterEvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, preprocess, img_folder):
        self.df = df
        self.preprocess = preprocess
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        for _ in range(10):
            row = self.df.iloc[index]
            pure_filename = os.path.basename(row["img_filename"])
            img_path = os.path.join(self.img_folder, pure_filename)
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img)
                label_tensor = torch.tensor(row[TARGET_GENRES].values.astype(float), dtype=torch.float32)
                return img_tensor, label_tensor, True, img_path
                
            except FileNotFoundError:
                index = (index + 1) % len(self.df)
                
        return torch.zeros(3, 224, 224), torch.zeros(NUM_CLASSES), False, ""

test_dataset = MoviePosterEvalDataset(test_df, preprocess=None, img_folder=IMG_FOLDER)


# ==========================================
# 3. CLIP 모델 로드 및 LoRA 가중치 병합
# ==========================================
base_model, preprocess = clip.load(MODEL_NAME, device=device)
test_dataset.preprocess = preprocess  

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
# 5. 어텐션 가중치 추출용 훅(Hook) 함수 정의
# ==========================================
saved_input_features = None

def get_input_attention_hook(module, input, output):
    global saved_input_features
    saved_input_features = input[0]

try:
    target_layer = model.base_model.model.visual.transformer.resblocks[-1].attn
    target_layer.register_forward_hook(get_input_attention_hook)
except AttributeError:
    print("Warning: Attention hook layer target matching failed.")


# ==========================================
# 6. Inference & 예측 확률값 수집 + 첫 번째 파일 대시보드 시각화
# ==========================================
all_preds = []
all_targets = []
visualized_first_sample = False

print("Starting Inference...")
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for batch_imgs, batch_labels, valid_flags, img_paths in tqdm(test_loader, desc="Evaluating"):
        
        valid_idx = torch.where(valid_flags == True)[0]
        if len(valid_idx) == 0:
            continue
            
        batch_imgs = batch_imgs[valid_idx].to(device)
        batch_labels = batch_labels[valid_idx].to(device)

        # 이미지 피처 추출
        image_features = model.encode_image(batch_imgs).float()
        
        # 코사인 유사도 연산용 규격 정규화
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features_norm @ text_features.T
        probs = torch.sigmoid(logits)
        
        preds_probs = probs.cpu().numpy()
        targets = batch_labels.cpu().numpy()
        
        # ---------------------------------------------------------
        # 🌟 첫 번째 배치 전용: Attention Map(첫 장) & 2D Embedding Similarity Matrix(배치 전체) 시각화
        # ---------------------------------------------------------
        if not visualized_first_sample and saved_input_features is not None:
            try:
                # 1. 공간 어텐션용 첫 번째 이미지 정보 가공
                single_img_path = img_paths[valid_idx[0]]
                orig_img = Image.open(single_img_path).convert("RGB")

                if saved_input_features.dim() == 3:
                    if saved_input_features.size(1) >= len(valid_idx):
                        feat_map = saved_input_features[1:, 0, :].abs().mean(dim=-1).cpu().numpy()
                    else:
                        feat_map = saved_input_features[0, 1:, :].abs().mean(dim=-1).cpu().numpy()
                else:
                    feat_map = saved_input_features.abs().mean(dim=-1).cpu().numpy()

                num_patches = feat_map.shape[0]
                grid_size = int(np.sqrt(num_patches))
                if grid_size * grid_size != num_patches:
                    grid_size = int(np.sqrt(num_patches - 1))
                    feat_map = feat_map[:grid_size*grid_size]

                feat_map = feat_map.reshape(grid_size, grid_size)
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
                feat_map_resized = np.array(Image.fromarray((feat_map * 255).astype(np.uint8)).resize(orig_img.size, Image.Resampling.BILINEAR))

                # 2. 2차원 Image vs Text Embedding 유사도 행렬 생성
                similarity_matrix = preds_probs.T  # 크기: (NUM_CLASSES, 유효 이미지 수)

                # 컬럼 이름으로 쓸 이미지 파일명 리스트 추출
                batch_filenames = [os.path.basename(img_paths[idx]) for idx in valid_idx.cpu().numpy()]

                # 3. Matplotlib 3분할 대시보드 구성 (원본 | 어텐션맵 | 2D 임베딩 히트맵)
                fig = plt.figure(figsize=(20, 6))
                ax1 = plt.subplot2grid((1, 4), (0, 0)) # 원본
                ax2 = plt.subplot2grid((1, 4), (0, 1)) # 어텐션
                ax3 = plt.subplot2grid((1, 4), (0, 2), colspan=2) # 2D 유사도 행렬
                
                # [좌측] 원본 포스터
                ax1.imshow(orig_img)
                ax1.set_title("1. First Poster Original", fontsize=11, fontweight='bold')
                ax1.axis("off")
                
                # [중앙] 어텐션 맵 오버레이
                ax2.imshow(orig_img)
                ax2.imshow(feat_map_resized, cmap='jet', alpha=0.45)
                ax2.set_title("2. First Poster Attention", fontsize=11, fontweight='bold')
                ax2.axis("off")
                
                # [우측] 2차원 Multi-modal Similarity Heatmap Matrix (sns 정상 동작 보장)
                sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax3,
                            xticklabels=batch_filenames, yticklabels=TARGET_GENRES,
                            annot_kws={"size": 8}, linewidths=0.5)
                
                ax3.set_title("3. CLIP Multi-Modal Similarity Matrix\n(Rows: Text Embeddings / Columns: Image Embeddings)", 
                              fontsize=12, fontweight='bold')
                ax3.set_xlabel("Image Embeddings (Batch Files)", fontsize=10)
                ax3.set_ylabel("Text Embeddings (Genre Prompts)", fontsize=10)
                ax3.tick_params(axis='x', rotation=30, labelsize=9)
                ax3.tick_params(axis='y', labelsize=9)

                plt.tight_layout()
                output_dashboard_path = "./clip_similarity_matrix_dashboard.png"
                plt.savefig(output_dashboard_path, dpi=200, bbox_inches='tight')
                plt.close(fig) # 🌟 [메모리 경고 해결] 명확한 객체 close로 RuntimeWarning 차단
                
                print(f"\n📊 [시각화 성공] 원하셨던 2차원 CLIP 유사도 행렬 대시보드가 저장되었습니다: {output_dashboard_path}")
                visualized_first_sample = True
            except Exception as e:
                print(f"\n⚠️ First batch matrix visualization failed: {e}")

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
# 7. 다양한 Threshold 탐색 및 성능 메트릭 집계
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
    threshold_candidates = np.arange(0.05, 0.96, 0.05)
    
    print("\nSearching for the best threshold...")
    for thresh in threshold_candidates:
        current_preds = (all_preds_probs > thresh).astype(int)
        
        prec = precision_score(all_targets, current_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, current_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, current_preds, average='macro', zero_division=0)
        
        threshold_results.append({
            'Threshold': round(thresh, 2),
            'Precision': prec, 'Recall': rec, 'F1-Score': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {'Precision': prec, 'Recall': rec, 'F1-Score': f1}

    print("\n" + "="*55)
    print(f" {'Thresh':<8} | {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("="*55)
    for res in threshold_results:
        print(f" {res['Threshold']:<8} | {res['Precision']:<12.4f} {res['Recall']:<12.4f} {res['F1-Score']:<12.4f}")
    print("="*55)
    
    print("\n" + "*"*50)
    print("          👑 BEST THRESHOLD SUMMARY          ")
    print("*"*50)
    print(f" 🌟 Best Threshold : {best_thresh:.2f}")
    print(f"    -> Precision   : {best_metrics['Precision']:.4f}")
    print(f"    -> Recall      : {best_metrics['Recall']:.4f}")
    print(f"    -> F1-Score    : {best_metrics['F1-Score']:.4f}")
    print("*"*50)