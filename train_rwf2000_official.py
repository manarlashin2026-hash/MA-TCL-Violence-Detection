"""
MA-TCL on RWF-2000 -- OFFICIAL train/val split

Same model, loss, optimizer, schedule and hyper-parameters as
code_3_tcl_network_enhanced_bootstrap_overfitting_RWF2000_2000sample.py.
The ONLY differences:
  1. Training clips come from RWF-2000/train, evaluation clips from RWF-2000/val
     (the partition released by the dataset authors) instead of a random 80/20
     split of the merged folders.
  2. Random seeds are fixed (python / numpy / torch).
  3. USE_TCL switch: set to False to feed zeros instead of TCL features
     (video-only control run with an otherwise identical recipe).
  4. RESUMABLE: the full training state (model, optimizer, scheduler, AMP
     scaler, history, early-stopping counters) is written to
     <SAVE_DIR>/last_checkpoint.pth after EVERY epoch. If the script is
     started again and that file exists, training continues from the next
     epoch. To force a fresh run, delete or rename the results folder.
Nothing is copied or modified inside the dataset folder.
"""

import os
import shutil
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEModel
from torch.optim import AdamW
from tqdm import tqdm
import gc
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# SOURCE: Original RWF-2000 dataset
RWF2000_BASE = "./data/RWF-2000"

USE_TCL = True   # False -> video-only control (TCL features replaced by zeros)
SEED = 42
# Model settings
MODEL_CKPT = "MCG-NJU/videomae-base"
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2
EPOCHS = 40  # Reduced for subset
NUM_FRAMES = 16
SAVE_DIR = "./rwf2000_official_split_results" if USE_TCL else "./rwf2000_official_split_results_noTCL"

INITIAL_LR = 5e-5


os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# STEP 1: LOAD THE OFFICIAL PARTITION
# =============================================================================

def list_official_split(base, split):
    """Return (paths, labels) for RWF-2000/<split>/{Fight,NonFight}.
    Applies the same readability check used before (opens, > 10 frames)."""
    paths, labels, skipped = [], [], []
    for folder, label in (("Fight", 1), ("NonFight", 0)):
        d = os.path.join(base, split, folder)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing folder: {d}")
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                continue
            p = os.path.join(d, f)
            cap = cv2.VideoCapture(p)
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 10
            cap.release()
            if ok:
                paths.append(p); labels.append(label)
            else:
                skipped.append(p)
    return paths, labels, skipped

# =============================================================================
# STEP 2: TRAINING COMPONENTS (Same as before)
# =============================================================================

class TrainingLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'learning_rate': [],
            'epoch_time': [], 'best_epoch': 0, 'best_acc': 0.0, 'best_f1': 0.0
        }

    def log_epoch(self, epoch, metrics):
        for key in ['train_loss', 'train_acc', 'val_acc', 'val_f1', 
                    'val_precision', 'val_recall', 'learning_rate', 'epoch_time']:
            self.history[key].append(metrics.get(key, 0))
        
        if metrics['val_f1'] > self.history['best_f1']:
            self.history['best_acc'] = metrics['val_acc']
            self.history['best_f1'] = metrics['val_f1']
            self.history['best_epoch'] = epoch

    def save_history(self):
        with open(f"{self.save_dir}/training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_training_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('MA-TCL Training on RWF-2000 (official split)', fontsize=16, fontweight='bold')

        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.history['val_acc'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=self.history['best_acc'], color='r', linestyle='--')
        axes[0, 1].set_title(f'Val Accuracy (Best: {self.history["best_acc"]*100:.2f}%)')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(epochs, self.history['learning_rate'], 'purple', linewidth=2)
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, self.history['val_f1'], 'orange', linewidth=2)
        axes[1, 0].set_title('F1-Score')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, self.history['val_precision'], 'b-', label='Precision')
        axes[1, 1].plot(epochs, self.history['val_recall'], 'r-', label='Recall')
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        if self.history['train_acc'] and any(self.history['train_acc']):
            gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
            axes[1, 2].plot(epochs, gap, 'red', linewidth=2)
            axes[1, 2].axhline(y=0, color='green', linestyle='--')
            axes[1, 2].set_title('Train-Val Gap')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.pdf", dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary(self):
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Best Epoch: {self.history['best_epoch'] + 1}")
        print(f"Best Val Accuracy: {self.history['best_acc']*100:.2f}%")
        print(f"Best Val F1: {self.history['best_f1']:.4f}")
        if self.history['train_acc'] and any(self.history['train_acc']):
            gap = self.history['train_acc'][-1] - self.history['val_acc'][-1]
            print(f"Final Train-Val Gap: {gap*100:.2f}%")
        print("="*80)

class MotionAwareTCLExtractor:
    @staticmethod
    def compute_frame_motion(frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        motion_simple = diff.mean() / 255.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_flow = min(1.0, magnitude.mean() / 10.0)
        except:
            motion_flow = motion_simple
        return motion_simple * 0.4 + motion_flow * 0.6

    @staticmethod
    def detect_people_proximity_with_motion(frame, prev_frame=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        
        edges_fine = cv2.Canny(center_region, 30, 100)
        edges_coarse = cv2.Canny(center_region, 100, 200)
        edges = edges_fine * 0.6 + edges_coarse * 0.4
        edge_density = edges.sum() / (center_region.shape[0] * center_region.shape[1])
        
        texture_var = np.std(center_region)
        hist = cv2.calcHist([center_region], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10)) / 8.0
        
        proximity = min(1.0, max(0.0, edge_density * 0.008 + (texture_var / 255.0) * 0.4 + entropy * 0.3))
        motion = MotionAwareTCLExtractor.compute_frame_motion(prev_frame, frame) if prev_frame is not None else 0.0
        
        return proximity, motion

    @staticmethod
    def compute_motion_aware_tcl_signature(frames, proximity_threshold=0.35, motion_threshold=0.15):
        proximity_scores = []
        motion_scores = []
        prev_frame = None
        
        for frame in frames:
            prox, motion = MotionAwareTCLExtractor.detect_people_proximity_with_motion(frame, prev_frame)
            proximity_scores.append(prox)
            motion_scores.append(motion)
            prev_frame = frame
        
        prox_arr = np.array(proximity_scores)
        motion_arr = np.array(motion_scores)
        
        high_prox = (prox_arr > proximity_threshold).astype(int)
        max_streak = current_streak = 0
        streaks = []
        
        for val in high_prox:
            if val == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        
        duration = max_streak / len(frames) if frames else 0.0
        consistency = min(1.0, 1.0 / (prox_arr.std() + 0.1) / 10.0)
        avg_prox = prox_arr.mean()
        event_score = 1.0 / (len(streaks) if streaks else 1)
        
        high_prox_idx = np.where(prox_arr > proximity_threshold)[0]
        motion_contact = motion_arr[high_prox_idx].mean() if len(high_prox_idx) > 0 else 0.0
        motion_consist = min(1.0, 1.0 / (motion_arr.std() + 0.1) / 5.0)
        peak_motion = motion_arr.max()
        risk_score = ((prox_arr > proximity_threshold) & (motion_arr > motion_threshold)).sum() / len(frames)
        
        num_bins = 8
        bin_size = len(prox_arr) // num_bins
        binned = []
        for i in range(num_bins):
            start = i * bin_size
            end = start + bin_size if i < num_bins - 1 else len(prox_arr)
            if end > start:
                binned.append(prox_arr[start:end].mean() * 0.5 + motion_arr[start:end].mean() * 0.5)
            else:
                binned.append(0)
        
        return np.array([duration, consistency, avg_prox, event_score,
                        motion_contact, motion_consist, peak_motion, risk_score] + binned, dtype=np.float32)

class MotionAwareTCLDataset(Dataset):
    def __init__(self, video_paths, labels, processor, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.tcl_extractor = MotionAwareTCLExtractor()

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None, None
        
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if frames:
                    frames.append(frames[-1])
        
        cap.release()
        
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1] if frames else np.zeros((224,224,3), dtype=np.uint8))
        
        tcl_feats = self.tcl_extractor.compute_motion_aware_tcl_signature(frames)
        return frames, tcl_feats

    def __getitem__(self, idx):
        try:
            frames, tcl_feats = self.load_video(self.video_paths[idx])
        except:
            frames = None
        
        if frames is None:
            return {
                'pixel_values': torch.zeros(NUM_FRAMES, 3, 224, 224),
                'tcl_features': torch.zeros(16),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        
        try:
            inputs = self.processor(frames, return_tensors="pt")
            pixels = inputs['pixel_values'].squeeze(0)
        except:
            pixels = torch.zeros(NUM_FRAMES, 3, 224, 224)
            tcl_feats = np.zeros(16, dtype=np.float32)
        
        return {
            'pixel_values': pixels,
            'tcl_features': torch.from_numpy(tcl_feats),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MotionAwareTCLFusionModule(nn.Module):
    def __init__(self, video_dim=768, tcl_dim=16):
        super().__init__()
        self.tcl_encoder = nn.Sequential(
            nn.Linear(tcl_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 256)
        )
        self.attention = nn.MultiheadAttention(video_dim, 8, dropout=0.2, batch_first=True)
        self.risk_gate = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + 256, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(512, 256), nn.Dropout(0.5)
        )

    def forward(self, video_feats, tcl_feats):
        tcl_enc = self.tcl_encoder(tcl_feats)
        risk_w = self.risk_gate(tcl_enc)
        video_seq = video_feats.unsqueeze(1)
        tcl_query = tcl_enc.unsqueeze(1)
        # NOTE: with one pooled video token and one TCL token, the attention weight is 1 at
        # inference, so the output does not depend on the key (and hence not on this projection);
        # TCL reaches the classifier through risk_gate and the concatenation below (Sec. III-C).
        tcl_proj = nn.Linear(256, video_feats.size(-1)).to(video_feats.device)
        tcl_key = tcl_proj(tcl_query)
        attended, attn_w = self.attention(video_seq, tcl_key, video_seq)
        attended = attended.squeeze(1) * (1.0 + risk_w)
        combined = torch.cat([attended, tcl_enc], dim=1)
        return self.fusion(combined), attn_w

class MotionAwareTCLViolenceDetector(nn.Module):
    def __init__(self, backbone='MCG-NJU/videomae-base', num_classes=2):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(backbone)
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for i in range(len(self.backbone.encoder.layer) - 3):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = False
        self.tcl_fusion = MotionAwareTCLFusionModule(self.backbone.config.hidden_size, 16)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, pixels, tcl_feats, return_attention=False):
        video_out = self.backbone(pixels)
        video_feats = video_out.last_hidden_state.mean(dim=1)
        fused, attn = self.tcl_fusion(video_feats, tcl_feats)
        logits = self.classifier(fused)
        return (logits, attn) if return_attention else logits

def bootstrap_confidence_intervals(true_labels, predicted_labels, n_iterations=1000):
    y_true = np.asarray(true_labels)
    y_pred = np.asarray(predicted_labels)
    n = y_true.shape[0]
    rng = np.random.default_rng(42)
    
    acc_samples, prec_samples, rec_samples, f1_samples = [], [], [], []
    
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        acc_samples.append(accuracy_score(yt, yp))
        prec_samples.append(precision_score(yt, yp, average='macro', zero_division=0))
        rec_samples.append(recall_score(yt, yp, average='macro', zero_division=0))
        f1_samples.append(f1_score(yt, yp, average='macro', zero_division=0))
    
    def pack(samples):
        samples = np.asarray(samples)
        return {
            "mean": float(np.mean(samples)),
            "ci_lower": float(np.percentile(samples, 2.5)),
            "ci_upper": float(np.percentile(samples, 97.5))
        }
    
    return {
        "accuracy": pack(acc_samples),
        "precision": pack(prec_samples),
        "recall": pack(rec_samples),
        "f1_score": pack(f1_samples)
    }

def print_bootstrap_results(results):
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95% CI)")
    print("="*80)
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        m = results[metric]
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {m['mean']*100:.2f}%")
        print(f"  95% CI: [{m['ci_lower']*100:.2f}%, {m['ci_upper']*100:.2f}%]")
    print("="*80)

def plot_confusion_matrix(cm, save_dir, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonFight', 'Fight'],
                yticklabels=['NonFight', 'Fight'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.title(f'MA-TCL on {dataset_name}\nAccuracy: {acc*100:.2f}%', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{dataset_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("\n" + "="*80)
    print("MA-TCL ON RWF-2000 -- OFFICIAL TRAIN/VAL SPLIT")
    print(f"USE_TCL = {USE_TCL} | results -> {SAVE_DIR}")
    print("="*80)

    train_paths, train_labels, skip_tr = list_official_split(RWF2000_BASE, "train")
    val_paths, val_labels, skip_va = list_official_split(RWF2000_BASE, "val")
    print(f"Train: {len(train_paths)} (Fight {sum(train_labels)}, NonFight {len(train_labels)-sum(train_labels)}) | skipped {len(skip_tr)}")
    print(f"Val:   {len(val_paths)} (Fight {sum(val_labels)}, NonFight {len(val_labels)-sum(val_labels)}) | skipped {len(skip_va)}")
    with open(f"{SAVE_DIR}/split_info.json", 'w') as f:
        json.dump({"train": len(train_paths), "val": len(val_paths),
                   "skipped_train": skip_tr, "skipped_val": skip_va,
                   "use_tcl": USE_TCL, "seed": SEED}, f, indent=2)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    
    train_dataset = MotionAwareTCLDataset(train_paths, train_labels, processor, augment=True)
    val_dataset = MotionAwareTCLDataset(val_paths, val_labels, processor, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model
    print("\n🧠 Building model...")
    model = MotionAwareTCLViolenceDetector(MODEL_CKPT, 2).to(device)
    
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]).to(device), label_smoothing=0.2)
    scaler = torch.amp.GradScaler('cuda')
    
    logger = TrainingLogger(SAVE_DIR)
    
    print("Starting training...")
    print("-" * 80)

    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0
    finished = False
    val_true, val_preds = [], []

    LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
    BEST_CKPT = os.path.join(SAVE_DIR, "best_rwf2000_model.pth")

    if os.path.exists(LAST_CKPT):
        state = torch.load(LAST_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        scaler.load_state_dict(state['scaler_state'])
        logger.history = state['history']
        best_f1 = state['best_f1']
        patience_counter = state['patience_counter']
        start_epoch = state['epoch'] + 1
        finished = state.get('finished', False)
        val_true, val_preds = state['val_true'], state['val_preds']
        torch.manual_seed(SEED + start_epoch)   # new shuffle order after a resume
        print(f"[RESUME] Loaded {LAST_CKPT}")
        print(f"[RESUME] Completed epochs: {start_epoch} | best F1 so far: {best_f1:.4f} | patience: {patience_counter}/10")
        if finished:
            print("[RESUME] Training had already finished; regenerating the final report only.")
    elif os.path.exists(BEST_CKPT):
        # Leftover from an interrupted run that had no resume information.
        # Keep it under another name instead of overwriting it.
        backup = os.path.join(SAVE_DIR, f"best_rwf2000_model_interrupted_{int(time.time())}.pth")
        os.replace(BEST_CKPT, backup)
        print(f"[INFO] Previous interrupted run kept as: {backup}")
        print("[INFO] Starting from epoch 1 (that run saved weights only, no optimizer/history).")

    for epoch in range(start_epoch, EPOCHS):
        if finished:
            break
        epoch_start = time.time()
        
        # ==========================
        # TRAINING PHASE
        # ==========================
        model.train()
        total_loss = 0
        train_preds, train_true = [], []
        
        optimizer.zero_grad() # Initialize gradients

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            pixels = batch['pixel_values'].to(device)
            tcl_feats = batch['tcl_features'].to(device)
            if not USE_TCL:
                tcl_feats = torch.zeros_like(tcl_feats)
            labels_b = batch['labels'].to(device)

            with torch.amp.autocast('cuda'):
                logits = model(pixels, tcl_feats)
                loss = criterion(logits, labels_b)
                # Normalize loss for accumulation
                loss = loss / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            # Step optimizer only after accumulating gradients
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(epoch + i / len(train_loader))

            total_loss += loss.item() * ACCUMULATION_STEPS
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(labels_b.cpu().numpy())

        # Handle remaining gradients
        if (i + 1) % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)

        # ==========================
        # VALIDATION PHASE
        # ==========================
        model.eval()
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                pixels = batch['pixel_values'].to(device)
                tcl_feats = batch['tcl_features'].to(device)
                if not USE_TCL:
                    tcl_feats = torch.zeros_like(tcl_feats)
                labels_b = batch['labels'].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(pixels, tcl_feats)

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels_b.cpu().numpy())

        # Metrics
        val_acc = accuracy_score(val_true, val_preds)
        cm = confusion_matrix(val_true, val_preds)
        report = classification_report(val_true, val_preds, output_dict=True, zero_division=0)
        
        # Calculate Macro F1
        val_f1 = f1_score(val_true, val_preds, average='macro')
        
        epoch_time = time.time() - epoch_start

        # Log Metrics
        logger.log_epoch(epoch, {
            'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc,
            'val_f1': val_f1, 'val_precision': report['macro avg']['precision'],
            'val_recall': report['macro avg']['recall'],
            'learning_rate': scheduler.get_last_lr()[0], 'epoch_time': epoch_time
        })

        print(f"\nEPOCH {epoch+1}/{EPOCHS}")
        print(f"  Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")
        print(f"  Confusion: [{cm[0,0]},{cm[0,1]}] | [{cm[1,0]},{cm[1,1]}]")

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'accuracy': val_acc,
                'f1': val_f1,
                'cm': cm.tolist()
            }, os.path.join(SAVE_DIR, "best_rwf2000_model.pth"))
            print("  ⭐ NEW BEST MODEL SAVED")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/10)")

        # Early Stopping decision (applied after the state is saved below)
        stop_now = patience_counter >= 10

        # ---- save full state so the run can continue after any interruption ----
        tmp_path = LAST_CKPT + ".tmp"
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'history': logger.history,
            'best_f1': best_f1,
            'patience_counter': patience_counter,
            'epoch': epoch,
            'finished': bool(stop_now or epoch == EPOCHS - 1),
            'val_true': [int(v) for v in val_true],
            'val_preds': [int(v) for v in val_preds],
        }, tmp_path)
        os.replace(tmp_path, LAST_CKPT)      # atomic: never leaves a half-written file
        logger.save_history()                # history JSON is also kept up to date
        print(f"  💾 State saved (epoch {epoch+1} done) -> safe to resume")

        if stop_now:
            print("\n🛑 Early stopping triggered.")
            break

        # Clean Memory
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================
    # FINAL RESULTS & REPORTING
    # ==========================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON RWF-2000 (OFFICIAL VAL SET)")
    print("="*80)
    
    # Save History & Plots
    logger.save_history()
    logger.plot_training_curves()
    logger.print_summary()
    print(f"Last-epoch Val Accuracy: {accuracy_score(val_true, val_preds)*100:.2f}%")
    
    # Generate Bootstrap CIs
    print("Computing Bootstrap Confidence Intervals...")
    bootstrap_results = bootstrap_confidence_intervals(val_true, val_preds)
    print_bootstrap_results(bootstrap_results)
    
    # Save Bootstrap Results
    with open(f"{SAVE_DIR}/bootstrap_results.json", 'w') as f:
        json.dump(bootstrap_results, f, indent=2)
        
    # Plot Confusion Matrix
    plot_confusion_matrix(confusion_matrix(val_true, val_preds), SAVE_DIR, "RWF-2000")

    print("\n" + "="*80)
    print(f"✅ DONE! All results saved in: {SAVE_DIR}")
    print("Files created:")
    print(" 1. best_rwf2000_model.pth (Best weights)")
    print(" 2. training_history.json (Logs)")
    print(" 3. training_curves.pdf (Visuals)")
    print(" 4. confusion_matrix_RWF-2000.pdf (Error analysis)")
    print(" 5. bootstrap_results.json (Statistical proof)")
    print("="*80)

if __name__ == '__main__':
    main()