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

# 장르 분할 로드 (train vs retained)
GENRE_SPLIT = "./genre_split.json"
with open(GENRE_SPLIT, "r", encoding="utf-8") as f:
    genre_split = json.load(f)
train_genres = genre_split["train_genres"]
retained_genres = genre_split["retained_genres"]
train_indices = [TARGET_GENRES.index(g) for g in train_genres if g in TARGET_GENRES]
retained_indices = [TARGET_GENRES.index(g) for g in retained_genres if g in TARGET_GENRES]
print(f"train genres: {len(train_indices)}, retained genres: {len(retained_indices)}")


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


USE_LORA = True  # False → 裸 CLIP baseline, 不加载训练权重

# ==========================================
# 3. CLIP 모델 로드 및 LoRA 가중치 병합
# ==========================================
base_model, preprocess = clip.load(MODEL_NAME, device=device)
test_dataset.preprocess = preprocess  # 전처리 함수 연결

if USE_LORA:
    print(f"Loading LoRA weights from: {LOAD_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, LOAD_MODEL_PATH)
else:
    print("SKIPPING LoRA weights — using frozen CLIP baseline")
    model = base_model
model.to(device)
model.eval()
print("Model loaded and set to evaluation mode.")


# ==========================================
# 4. 텍스트 프롬프트 토큰화
# ==========================================
PROMPT_BLEND = 0.4  # logit-level blend: visual vs generic weight

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)

visual_prompts = []
generic_prompts = []
for genre in TARGET_GENRES:
    if genre in prompt_dict:
        visual_prompts.append(prompt_dict[genre])
    else:
        visual_prompts.append(f"a movie poster for a {genre} film")
    generic_prompts.append(f"a movie poster for a {genre} film")

visual_tokens = clip.tokenize(visual_prompts).to(device)
generic_tokens = clip.tokenize(generic_prompts).to(device)

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
all_img_feat = []  # save image features for blend sweep

print("Starting Inference...")
with torch.no_grad():
    # Text features for both prompt types (normalized, saved raw for sweep)
    vis_text_feat_raw = model.encode_text(visual_tokens).float()
    gen_text_feat_raw = model.encode_text(generic_tokens).float()
    vis_text_feat = vis_text_feat_raw / vis_text_feat_raw.norm(dim=-1, keepdim=True)
    gen_text_feat = gen_text_feat_raw / gen_text_feat_raw.norm(dim=-1, keepdim=True)
    # Keep both text features separate — blend in logit space instead of embedding space

    for batch_imgs, batch_labels, valid_flags in tqdm(test_loader, desc="Evaluating"):
        
        valid_idx = torch.where(valid_flags == True)[0]
        if len(valid_idx) == 0:
            continue
            
        batch_imgs = batch_imgs[valid_idx].to(device)
        batch_labels = batch_labels[valid_idx].to(device)

        image_features = model.encode_image(batch_imgs).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits_vis = 100.0 * image_features @ vis_text_feat.T
        logits_gen = 100.0 * image_features @ gen_text_feat.T
        logits = PROMPT_BLEND * logits_vis + (1 - PROMPT_BLEND) * logits_gen

        logits_np = logits.cpu().numpy()
        targets = batch_labels.cpu().numpy()
        img_feat_np = image_features.cpu().numpy()

        all_preds.append(logits_np)
        all_targets.append(targets)
        all_img_feat.append(img_feat_np)

# 수집된 배치를 하나의 거대한 행렬로 통합
if len(all_preds) > 0:
    all_logits = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
else:
    all_logits = np.array([])
    all_targets = np.array([])


# ==========================================
# 6. Logit-space Threshold Scan (负区间 + 大步长)
# ==========================================
if len(all_logits) == 0:

    print("\n" + "!"*50)
    print(" CRITICAL ERROR: No valid images were evaluated!")
    print("!"*50)
else:
    best_f1 = -1
    best_thresh = 0.0
    best_metrics = {}

    threshold_results = []

    # logit 从 -5.0 到 2.0, 步长 0.01  (对齐工作区 evaluation.py 精度)
    threshold_candidates = np.arange(-5.0, 2.0, 0.01)

    print("\nSearching for the best threshold (logit space)...")
    for thresh in threshold_candidates:
        # logit > threshold → 预测为正
        current_preds = (all_logits > thresh).astype(int)

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
    print(f" {'Thresh(logit)':<15} | {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("="*55)
    for res in threshold_results:
        print(f" {res['Threshold']:<15.1f} | {res['Precision']:<12.4f} {res['Recall']:<12.4f} {res['F1-Score']:<12.4f}")
    print("="*55)
    
    # --- 종합 요약 리포트 ---
    print("\n" + "*"*50)
    print("          👑 BEST THRESHOLD SUMMARY          ")
    print("*"*50)
    print(f" 🌟 Best Threshold (logit) : {best_thresh:.2f}")
    print(f"     ≈ prob {1/(1+np.exp(-best_thresh)):.4f}")
    print(f"    -> Precision   : {best_metrics['Precision']:.4f}")
    print(f"    -> Recall      : {best_metrics['Recall']:.4f}")
    print(f"    -> F1-Score    : {best_metrics['F1-Score']:.4f}")
    print("*"*50)

    # --- Per-Class (best threshold, logit space) ---
    best_preds = (all_logits > best_thresh).astype(int)
    tp = np.sum((best_preds == 1) & (all_targets == 1), axis=0).astype(np.float64)
    fp = np.sum((best_preds == 1) & (all_targets == 0), axis=0).astype(np.float64)
    fn = np.sum((best_preds == 0) & (all_targets == 1), axis=0).astype(np.float64)
    eps = 1e-8
    pc_p = tp / np.maximum(tp + fp, eps)
    pc_r = tp / np.maximum(tp + fn, eps)
    pc_f = 2 * pc_p * pc_r / np.maximum(pc_p + pc_r, eps)

    # 用列宽 12 给 genre 名, 后面加 type 标记 + 正样本数
    pos_count = all_targets.sum(axis=0).astype(int)
    print("\n" + "="*88)
    print(" {:<16} | {:<8} | {:>5} | {:<10} | {:<10} | {:<10}".format("Genre", "Type", "Pos", "Precision", "Recall", "F1-Score"))
    print("="*88)
    for i, genre in enumerate(TARGET_GENRES):
        gtype = "train" if i in train_indices else "retained"
        print(" {:<16} | {:<8} | {:>5} | {:<10.4f} | {:<10.4f} | {:<10.4f}".format(genre, gtype, pos_count[i], pc_p[i], pc_r[i], pc_f[i]))
    print("="*88)

    # --- 分组汇总: Train vs Retained ---
    train_f1_macro = f1_score(all_targets[:, train_indices], best_preds[:, train_indices], average='macro', zero_division=0)
    train_f1_micro = f1_score(all_targets[:, train_indices], best_preds[:, train_indices], average='micro', zero_division=0)
    retained_f1_macro = f1_score(all_targets[:, retained_indices], best_preds[:, retained_indices], average='macro', zero_division=0)
    retained_f1_micro = f1_score(all_targets[:, retained_indices], best_preds[:, retained_indices], average='micro', zero_division=0)

    print(f"\n  Train ({len(train_indices)} classes):    Macro F1 = {train_f1_macro:.4f}  |  Micro F1 = {train_f1_micro:.4f}")
    print(f"  Retained ({len(retained_indices)} classes): Macro F1 = {retained_f1_macro:.4f}  |  Micro F1 = {retained_f1_micro:.4f}")


# ==========================================
# 7. PROMPT_BLEND Sweep (no retrain needed — re-blend in logit space)
# ==========================================
if len(all_img_feat) > 0:
    all_img = np.vstack(all_img_feat)  # (N, 512), L2-normalized

    # Cache raw text features (normalized) on CPU
    vt = vis_text_feat.cpu().numpy()   # (23, 512)
    gt = gen_text_feat.cpu().numpy()

    # Pre-compute logits for both prompt types (N x 23)
    logits_vis_all = 100.0 * all_img @ vt.T
    logits_gen_all = 100.0 * all_img @ gt.T

    BLEND_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # ---- 7a. Global blend sweep ----
    print(f"\n{'='*85}")
    print(f" 7a. GLOBAL BLEND SWEEP: same blend for all 23 classes")
    print(f"{'='*85}")
    print(f" {'blend':<7} | {'BestThr':>8} | {'Train F1':>9} | {'Retained F1':>11} | {'Overall F1':>10}")
    print("-" * 65)

    best_overall = -1
    best_global_blend = PROMPT_BLEND

    for blend in BLEND_CANDIDATES:
        sweep_logits = blend * logits_vis_all + (1 - blend) * logits_gen_all

        best_f1 = -1
        for thresh in np.arange(-6.0, 4.5, 0.5):
            preds = (sweep_logits > thresh).astype(int)
            f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thresh

        train_f1 = f1_score(all_targets[:, train_indices],
                            (sweep_logits[:, train_indices] > best_thr).astype(int),
                            average='macro', zero_division=0)
        retained_f1 = f1_score(all_targets[:, retained_indices],
                               (sweep_logits[:, retained_indices] > best_thr).astype(int),
                               average='macro', zero_division=0)

        marker = " ← training" if abs(blend - PROMPT_BLEND) < 0.01 else ""
        print(f" {blend:<7.1f} | {best_thr:>8.1f} | {train_f1:>9.4f} | {retained_f1:>11.4f} | {best_f1:>10.4f}{marker}")

        if best_f1 > best_overall:
            best_overall = best_f1
            best_global_blend = blend

    print("-" * 65)
    print(f"  Best global blend = {best_global_blend:.1f}  (Overall Macro F1 = {best_overall:.4f})")

    # ---- 7b. Per-class optimal blend ----
    print(f"\n{'='*85}")
    print(f" 7b. PER-CLASS OPTIMAL BLEND: each class independently optimized")
    print(f"{'='*85}")
    print(f" {'Genre':<16} | {'Type':<8} | {'Pos':>5} | {'Best w':>7} | {'Best F1':>8} | {'F1@train':>8} | Gain")
    print("-" * 80)

    per_class_best_w = []
    per_class_best_f1 = []

    for i, genre in enumerate(TARGET_GENRES):
        best_cls_f1 = -1
        best_cls_w = PROMPT_BLEND

        for blend in BLEND_CANDIDATES:
            # Blend this class's logits from two prompt types
            logits_i = blend * logits_vis_all[:, i] + (1 - blend) * logits_gen_all[:, i]

            for thresh in np.arange(-6.0, 4.5, 0.5):
                preds_i = (logits_i > thresh).astype(int)
                f1_i = f1_score(all_targets[:, i], preds_i, average='binary', zero_division=0)
                if f1_i > best_cls_f1:
                    best_cls_f1 = f1_i
                    best_cls_w = blend

        # F1 at training blend (for comparison)
        logits_train = PROMPT_BLEND * logits_vis_all[:, i] + (1 - PROMPT_BLEND) * logits_gen_all[:, i]
        f1_train_blend = -1
        for thresh in np.arange(-6.0, 4.5, 0.5):
            preds_i = (logits_train > thresh).astype(int)
            f1_i = f1_score(all_targets[:, i], preds_i, average='binary', zero_division=0)
            if f1_i > f1_train_blend:
                f1_train_blend = f1_i

        gain = best_cls_f1 - f1_train_blend
        gain_str = f"+{gain:.4f}" if gain > 0.005 else ("─" if gain > -0.005 else f"{gain:.4f}")
        gtype = "train" if i in train_indices else "retained"
        pos = int(all_targets[:, i].sum())

        print(f" {genre:<16} | {gtype:<8} | {pos:>5} | {best_cls_w:>7.1f} | {best_cls_f1:>8.4f} | {f1_train_blend:>8.4f} | {gain_str}")

        per_class_best_w.append(best_cls_w)
        per_class_best_f1.append(best_cls_f1)

    # ---- 7c. Per-class blended logits → overall F1 ----
    print(f"\n{'='*85}")
    print(f" 7c. POTENTIAL: if each class uses its own optimal blend")
    print(f"{'='*85}")

    # Build per-class blended logits (logit-space blend)
    opt_logits = np.zeros_like(logits_vis_all)
    for i in range(len(TARGET_GENRES)):
        w = per_class_best_w[i]
        opt_logits[:, i] = w * logits_vis_all[:, i] + (1 - w) * logits_gen_all[:, i]

    best_opt_f1 = -1
    best_opt_thr = 0
    for thresh in np.arange(-6.0, 4.5, 0.5):
        preds = (opt_logits > thresh).astype(int)
        f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
        if f1 > best_opt_f1:
            best_opt_f1 = f1
            best_opt_thr = thresh

    opt_train_f1 = f1_score(all_targets[:, train_indices],
                            (opt_logits[:, train_indices] > best_opt_thr).astype(int),
                            average='macro', zero_division=0)
    opt_ret_f1 = f1_score(all_targets[:, retained_indices],
                          (opt_logits[:, retained_indices] > best_opt_thr).astype(int),
                          average='macro', zero_division=0)

    print(f"  Per-class blend → Overall F1 = {best_opt_f1:.4f}  (vs global best = {best_overall:.4f},  gain = {best_opt_f1 - best_overall:+.4f})")
    print(f"    Train F1 = {opt_train_f1:.4f},  Retained F1 = {opt_ret_f1:.4f}")
    print(f"  Training blend = {PROMPT_BLEND}")

    # ---- 7d. Summary: best w by group ----
    train_w = [per_class_best_w[i] for i in train_indices]
    retained_w = [per_class_best_w[i] for i in retained_indices]
    print(f"\n  Avg best blend:  Train = {np.mean(train_w):.2f}  |  Retained = {np.mean(retained_w):.2f}")
    print(f"  Suggested training blend: {np.mean(train_w):.2f} (train-only mean)")