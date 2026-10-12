"""
MA-TCL (configuration 5, Table 3 and Table 4) on SC-Fight-300:
VideoMAE + 16-D motion-aware TCL descriptor + risk-gated fusion,
stratified 80/20 split (seed 42), bootstrap confidence intervals of the final-epoch model.

Run from the repository root; see README.md for the expected data layout.
"""
import os
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
FIGHT_DIR = "./data/SC-Fight-300/fight"
NOFIGHT_DIR = "./data/SC-Fight-300/noFight"

MODEL_CKPT = "MCG-NJU/videomae-base"

# RTX 5060 OPTIMIZATION SETTINGS
BATCH_SIZE = 2             # Low batch size to save VRAM
ACCUMULATION_STEPS = 2     # Accumulate gradients to simulate Batch Size 4
EPOCHS = 50
NUM_FRAMES = 16
SAVE_DIR = "./fight_model_ma_tcl_final"

INITIAL_LR = 5e-5

os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# TRAINING LOGGER
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
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['train_acc'].append(metrics.get('train_acc', 0))
        self.history['val_acc'].append(metrics['val_acc'])
        self.history['val_f1'].append(metrics['val_f1'])
        self.history['val_precision'].append(metrics.get('val_precision', 0))
        self.history['val_recall'].append(metrics.get('val_recall', 0))
        self.history['learning_rate'].append(metrics['learning_rate'])
        self.history['epoch_time'].append(metrics.get('epoch_time', 0))

        if metrics['val_f1'] > self.history['best_f1']:
            self.history['best_acc'] = metrics['val_acc']
            self.history['best_f1'] = metrics['val_f1']
            self.history['best_epoch'] = epoch

    def save_history(self):
        filepath = f"{self.save_dir}/training_history.json"
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved: {filepath}")

    def plot_training_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('MA-TCL Training History', fontsize=16, fontweight='bold')

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Val Accuracy
        axes[0, 1].plot(epochs, self.history['val_acc'], 'g-', linewidth=2)
        best_epoch = self.history['best_epoch'] + 1
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=self.history['best_acc'], color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title(f'Val Accuracy (Best: {self.history["best_acc"]*100:.2f}%)')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)

        # Learning Rate
        axes[0, 2].plot(epochs, self.history['learning_rate'], 'purple', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('LR Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

        # F1-Score
        axes[1, 0].plot(epochs, self.history['val_f1'], 'orange', linewidth=2)
        axes[1, 0].axhline(y=self.history['best_f1'], color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title(f'F1-Score (Best: {self.history["best_f1"]:.4f})')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)

        # Precision vs Recall
        axes[1, 1].plot(epochs, self.history['val_precision'], 'b-', linewidth=2,
                        label='Precision', marker='o', markersize=3)
        axes[1, 1].plot(epochs, self.history['val_recall'], 'r-', linewidth=2,
                        label='Recall', marker='s', markersize=3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Overfitting Analysis
        if self.history['train_acc'] and any(self.history['train_acc']):
            gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
            axes[1, 2].plot(epochs, gap, 'red', linewidth=2, label='Train-Val Gap')
            axes[1, 2].axhline(y=0, color='green', linestyle='--', alpha=0.7, label='No Gap')
            axes[1, 2].fill_between(epochs, 0, gap, where=[g > 0 for g in gap],
                                    alpha=0.3, color='red')
        else:
            axes[1, 2].text(0.5, 0.5, 'Train acc not logged',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy Gap')
        axes[1, 2].set_title('Overfitting Analysis')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary(self):
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Total Epochs: {len(self.history['train_loss'])}")
        print(f"Best Epoch: {self.history['best_epoch'] + 1}")
        print(f"Best Val Accuracy: {self.history['best_acc']*100:.2f}%")
        print(f"Best Val F1 (macro): {self.history['best_f1']:.4f}")

        if self.history['train_acc'] and any(self.history['train_acc']):
            final_gap = self.history['train_acc'][-1] - self.history['val_acc'][-1]
            print(f"Final Train-Val Gap: {final_gap*100:.2f}%")
            if final_gap < 0.03:
                print("Status: Excellent generalization (gap < 3%)")
            elif final_gap < 0.05:
                print("Status: Good generalization (gap < 5%)")
            else:
                print("Status: Mild overfitting detected")
        print("="*80)

# =============================================================================
# MOTION-AWARE TCL EXTRACTOR (RESTORED CLASS)
# =============================================================================
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

        # TCL features
        high_prox = (prox_arr > proximity_threshold).astype(int)
        max_streak = 0
        current_streak = 0
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
        
        # Temporal bins
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

# =============================================================================
# DATASET (OPTIMIZED FOR RAM)
# =============================================================================
class MotionAwareTCLDataset(Dataset):
    def __init__(self, video_paths, labels, processor, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.tcl_extractor = MotionAwareTCLExtractor()

    def __len__(self):
        return len(self.video_paths)

    def augment_frames(self, frames):
        if not self.augment or np.random.rand() > 0.7:
            return frames
        frames_arr = np.array(frames)
        if np.random.rand() > 0.4:
            factor = np.random.uniform(0.65, 1.35)
            frames_arr = np.clip(frames_arr * factor, 0, 255).astype(np.uint8)
        if np.random.rand() > 0.5:
            frames_arr = np.flip(frames_arr, axis=2)
        return [f for f in frames_arr]

    def load_video(self, path):
        # RAM FIX: Use CAP_PROP_POS_FRAMES to only read needed frames
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
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    cap.release()
                    return None, None
        cap.release()

        # Handle corrupt/short videos
        if len(frames) < NUM_FRAMES:
            while len(frames) < NUM_FRAMES:
                frames.append(frames[-1] if frames else np.zeros((224,224,3), dtype=np.uint8))

        if self.augment:
            frames = self.augment_frames(frames)

        tcl_feats = self.tcl_extractor.compute_motion_aware_tcl_signature(frames)
        return frames, tcl_feats

    def __getitem__(self, idx):
        try:
            frames, tcl_feats = self.load_video(self.video_paths[idx])
        except Exception as e:
            print(f"Error loading {self.video_paths[idx]}: {e}")
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

# =============================================================================
# MODEL
# =============================================================================
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

# =============================================================================
# UTILITIES
# =============================================================================
def setup_device():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. CUDA is required for this script.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda")

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()

def plot_confusion_matrix(cm, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NoFight', 'Fight'],
                yticklabels=['NoFight', 'Fight'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - MA-TCL', fontweight='bold')
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(1, -0.15, f'Accuracy: {acc*100:.2f}%',
             ha='center', va='top', transform=plt.gca().transAxes, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_dir}/confusion_matrix.pdf")

# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
def bootstrap_confidence_intervals(
    true_labels, predicted_labels, n_iterations=1000, confidence_level=0.95, random_state=42, average='macro'
):
    y_true = np.asarray(true_labels)
    y_pred = np.asarray(predicted_labels)
    n = y_true.shape[0]
    rng = np.random.default_rng(random_state)
    acc_samples, prec_samples, rec_samples, f1_samples = [], [], [], []

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        acc_samples.append(accuracy_score(yt, yp))
        prec_samples.append(precision_score(yt, yp, average=average, zero_division=0))
        rec_samples.append(recall_score(yt, yp, average=average, zero_division=0))
        f1_samples.append(f1_score(yt, yp, average=average, zero_division=0))

    alpha = 1.0 - confidence_level
    lo = (alpha / 2.0) * 100.0
    hi = (1.0 - alpha / 2.0) * 100.0

    def pack(samples):
        samples = np.asarray(samples, dtype=np.float64)
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "ci_lower": float(np.percentile(samples, lo)),
            "ci_upper": float(np.percentile(samples, hi)),
            "samples": samples.tolist()
        }

    return {
        "settings": {"n_iterations": int(n_iterations), "confidence_level": float(confidence_level)},
        "accuracy": pack(acc_samples), "precision": pack(prec_samples),
        "recall": pack(rec_samples), "f1_score": pack(f1_samples)
    }

def print_bootstrap_results(results):
    cl = results["settings"]["confidence_level"]
    ci_pct = int(round(cl * 100))
    print("\n" + "=" * 80)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({ci_pct}% CI)")
    print("=" * 80)
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        m = results[metric]
        print(f"\n{metric.upper()}:")
        print(f"  Point Estimate: {m['mean']:.4f}")
        print(f"  {ci_pct}% CI: [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")
    print("=" * 80)

def save_bootstrap_results(results, save_dir):
    path = os.path.join(save_dir, "bootstrap_confidence_intervals.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def plot_bootstrap_distributions(results, save_dir, bins=30):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Bootstrap Distribution of Performance Metrics', fontsize=14, fontweight='bold')
    items = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1-score")]
    for idx, (key, title) in enumerate(items):
        ax = axes[idx // 2, idx % 2]
        data = results[key]
        samples = np.asarray(data["samples"], dtype=np.float64)
        ax.hist(samples, bins=bins, alpha=0.75, edgecolor='black')
        ax.axvline(data["mean"], linestyle='--', linewidth=2, label=f"Mean: {data['mean']:.4f}")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bootstrap_distributions.png"), dpi=300)
    plt.close()

# =============================================================================
# MAIN
# =============================================================================
def main():
    device = setup_device()

    print("\n" + "="*80)
    print("MOTION-AWARE TCL NETWORK - FINAL VERSION")
    print("="*80)

    # Load data
    print("Loading dataset...")
    video_files = []
    labels = []

    for root, _, files in os.walk(FIGHT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
                labels.append(1)

    for root, _, files in os.walk(NOFIGHT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
                labels.append(0)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)

    print("\nComputing MA-TCL signatures...")
    train_dataset = MotionAwareTCLDataset(train_paths, train_labels, processor, augment=True)
    val_dataset = MotionAwareTCLDataset(val_paths, val_labels, processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    print("\nBuilding model...")
    model = MotionAwareTCLViolenceDetector(MODEL_CKPT, 2).to(device)

    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]).to(device), label_smoothing=0.2)
    scaler = torch.amp.GradScaler('cuda')

    logger = TrainingLogger(SAVE_DIR)

    print("\nStarting training...")
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        train_preds, train_true = [], []
        
        optimizer.zero_grad() # Reset gradients at start of epoch

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            pixels = batch['pixel_values'].to(device)
            tcl_feats = batch['tcl_features'].to(device)
            labels_b = batch['labels'].to(device)

            with torch.amp.autocast('cuda'):
                logits = model(pixels, tcl_feats)
                loss = criterion(logits, labels_b)
                # Normalize loss for gradient accumulation
                loss = loss / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            # Step optimizer only after ACCUMULATION_STEPS
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(epoch + i / len(train_loader))

            total_loss += loss.item() * ACCUMULATION_STEPS # Scale back up for logging
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(labels_b.cpu().numpy())

        # Handle remaining gradients if len(loader) is not divisible by steps
        if (i + 1) % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)

        # VAL
        model.eval()
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                pixels = batch['pixel_values'].to(device)
                tcl_feats = batch['tcl_features'].to(device)
                labels_b = batch['labels'].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(pixels, tcl_feats)

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels_b.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        cm = confusion_matrix(val_true, val_preds)
        report = classification_report(val_true, val_preds, output_dict=True, zero_division=0)
        val_f1 = (report['0']['f1-score'] + report['1']['f1-score']) / 2
        epoch_time = time.time() - epoch_start

        logger.log_epoch(epoch, {
            'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc,
            'val_f1': val_f1, 'val_precision': report['1']['precision'],
            'val_recall': report['1']['recall'],
            'learning_rate': scheduler.get_last_lr()[0], 'epoch_time': epoch_time
        })

        print(f"\nEPOCH {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'accuracy': val_acc,
                'f1': val_f1
            }, os.path.join(SAVE_DIR, "best_ma_tcl_model.pth"))
            print(f" NEW BEST SAVED.")
        else:
            patience_counter += 1
            print(f" No improvement ({patience_counter}/12)")

        if patience_counter >= 12 and epoch > 20:
            print("Early stopping.")
            break
        clean_cache()

    logger.save_history()
    logger.plot_training_curves()
    logger.print_summary()
    
    print("\nComputing Bootstrap Confidence Intervals...")
    bootstrap_results = bootstrap_confidence_intervals(val_true, val_preds)
    print_bootstrap_results(bootstrap_results)
    save_bootstrap_results(bootstrap_results, SAVE_DIR)
    plot_bootstrap_distributions(bootstrap_results, SAVE_DIR)
    plot_confusion_matrix(confusion_matrix(val_true, val_preds), SAVE_DIR)

    print(f"\nDone. Results saved in {SAVE_DIR}")

if __name__ == '__main__':
    main()