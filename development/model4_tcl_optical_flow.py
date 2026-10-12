"""
Development configuration 4 (Table 3): RGB VideoMAE + optical-flow VideoMAE + 12-D TCL,
cross-modal attention with adaptive gating. SC-Fight-300.

Run from the repository root; see README.md for the expected data layout.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEModel
from torch.optim import AdamW
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
FIGHT_DIR = "./data/SC-Fight-300/fight"
NOFIGHT_DIR = "./data/SC-Fight-300/noFight"

MODEL_CKPT = "MCG-NJU/videomae-base"
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2
EPOCHS = 50
NUM_FRAMES = 16
SAVE_DIR = "./fight_model_tcl_optical_flow"

INITIAL_LR = 5e-5
WARMUP_EPOCHS = 5

# Load your pretrained TCL model as initialization (optional but recommended)
PRETRAINED_TCL_PATH = "./fight_model_tcl/best_tcl_model.pth"  # Set to None to train from scratch

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# OPTICAL FLOW EXTRACTOR
# -----------------------------------------------------------------------------
class OpticalFlowExtractor:
    """
    Compute dense optical flow between consecutive frames
    Flow captures motion patterns that distinguish fights from hugs
    """
    
    @staticmethod
    def compute_flow_sequence(frames):
        """
        Compute optical flow for entire video
        Returns flow visualizations as RGB images
        """
        if len(frames) < 2:
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in frames]
        
        flow_images = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Convert flow to RGB representation
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Create HSV image
            hsv = np.zeros((gray1.shape[0], gray1.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
            hsv[..., 1] = 255  # Full saturation
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude
            
            # Convert to RGB
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            flow_images.append(flow_rgb)
        
        # Duplicate last flow to match frame count
        flow_images.append(flow_images[-1])
        
        return flow_images
    
    @staticmethod
    def compute_motion_statistics(frames):
        """
        Extract motion statistics for the entire video
        Useful for understanding motion patterns
        """
        if len(frames) < 2:
            return {
                'avg_magnitude': 0.0,
                'max_magnitude': 0.0,
                'motion_variance': 0.0
            }
        
        magnitudes = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(magnitude.mean())
        
        magnitudes = np.array(magnitudes)
        
        return {
            'avg_magnitude': magnitudes.mean(),
            'max_magnitude': magnitudes.max(),
            'motion_variance': magnitudes.std()
        }

# -----------------------------------------------------------------------------
# TCL EXTRACTOR (Same as before)
# -----------------------------------------------------------------------------
class EnhancedTemporalContactLengthExtractor:
    @staticmethod
    def detect_people_proximity_v2(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        center_h_start, center_h_end = h // 4, 3 * h // 4
        center_w_start, center_w_end = w // 4, 3 * w // 4
        center_region = gray[center_h_start:center_h_end, center_w_start:center_w_end]
        
        edges_fine = cv2.Canny(center_region, 30, 100)
        edges_coarse = cv2.Canny(center_region, 100, 200)
        edges_combined = (edges_fine * 0.6 + edges_coarse * 0.4)
        edge_density = edges_combined.sum() / (center_region.shape[0] * center_region.shape[1])
        
        texture_variance = np.std(center_region)
        
        hist = cv2.calcHist([center_region], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        entropy_norm = entropy / 8.0
        
        proximity = (
            edge_density * 0.008 +
            (texture_variance / 255.0) * 0.4 +
            entropy_norm * 0.3
        )
        
        return min(1.0, max(0.0, proximity))
    
    @staticmethod
    def compute_contact_signature_v2(frames, proximity_threshold=0.35):
        proximity_scores = []
        
        for frame in frames:
            prox = EnhancedTemporalContactLengthExtractor.detect_people_proximity_v2(frame)
            proximity_scores.append(prox)
        
        proximity_array = np.array(proximity_scores)
        high_prox_mask = (proximity_array > proximity_threshold).astype(int)
        
        max_streak = 0
        current_streak = 0
        streaks = []
        
        for val in high_prox_mask:
            if val == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        duration_score = max_streak / len(frames) if frames else 0.0
        consistency_score = 1.0 / (proximity_array.std() + 0.1)
        consistency_score = min(1.0, consistency_score / 10.0)
        avg_proximity = proximity_array.mean()
        num_contact_events = len(streaks) if streaks else 1
        event_score = 1.0 / num_contact_events
        
        num_bins = 8
        bin_size = len(proximity_array) // num_bins
        binned_pattern = []
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < num_bins - 1 else len(proximity_array)
            bin_mean = proximity_array[start_idx:end_idx].mean() if end_idx > start_idx else 0
            binned_pattern.append(bin_mean)
        
        feature_vector = np.array([
            duration_score,
            consistency_score,
            avg_proximity,
            event_score
        ] + binned_pattern, dtype=np.float32)
        
        return feature_vector

# -----------------------------------------------------------------------------
# DUAL-STREAM DATASET (RGB + Optical Flow)
# -----------------------------------------------------------------------------
class DualStreamDataset(Dataset):
    """
    Dataset that provides both RGB frames and optical flow
    """
    def __init__(self, video_paths, labels, processor, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.tcl_extractor = EnhancedTemporalContactLengthExtractor()
        self.flow_extractor = OpticalFlowExtractor()

    def __len__(self):
        return len(self.video_paths)

    def augment_frames(self, rgb_frames, flow_frames):
        """Augment both RGB and flow consistently"""
        if not self.augment or np.random.rand() > 0.6:
            return rgb_frames, flow_frames
        
        # Horizontal flip (both streams)
        if np.random.rand() > 0.5:
            rgb_frames = [cv2.flip(f, 1) for f in rgb_frames]
            flow_frames = [cv2.flip(f, 1) for f in flow_frames]
        
        # Brightness (RGB only)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.75, 1.25)
            rgb_frames = [np.clip(f * factor, 0, 255).astype(np.uint8) for f in rgb_frames]
        
        # Crop (both streams)
        if np.random.rand() > 0.5:
            h, w = rgb_frames[0].shape[:2]
            crop_ratio = np.random.uniform(0.90, 0.98)
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            top = np.random.randint(0, max(1, h - crop_h + 1))
            left = np.random.randint(0, max(1, w - crop_w + 1))
            
            rgb_frames = [cv2.resize(f[top:top+crop_h, left:left+crop_w], (w, h)) 
                         for f in rgb_frames]
            flow_frames = [cv2.resize(f[top:top+crop_h, left:left+crop_w], (w, h)) 
                          for f in flow_frames]
        
        return rgb_frames, flow_frames

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        if not cap.isOpened():
            return None, None, None
            
        try:
            while len(frames) < 300:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            return None, None, None

        # Sample frames
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        rgb_frames = [frames[i] for i in indices]
        
        # Compute optical flow
        flow_frames = self.flow_extractor.compute_flow_sequence(rgb_frames)
        
        # Extract TCL features
        tcl_features = self.tcl_extractor.compute_contact_signature_v2(rgb_frames)
        
        return rgb_frames, flow_frames, tcl_features

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        rgb_frames, flow_frames, tcl_features = self.load_video(video_path)
        
        if rgb_frames is None:
            return {
                'rgb_values': torch.zeros(NUM_FRAMES, 3, 224, 224),
                'flow_values': torch.zeros(NUM_FRAMES, 3, 224, 224),
                'tcl_features': torch.zeros(12),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        # Augment
        rgb_frames, flow_frames = self.augment_frames(rgb_frames, flow_frames)
        
        try:
            # Process RGB
            rgb_inputs = self.processor(rgb_frames, return_tensors="pt")
            rgb_values = rgb_inputs['pixel_values'].squeeze(0)
            
            # Process Flow
            flow_inputs = self.processor(flow_frames, return_tensors="pt")
            flow_values = flow_inputs['pixel_values'].squeeze(0)
            
        except Exception as e:
            print(f"Error processing video: {e}")
            rgb_values = torch.zeros(NUM_FRAMES, 3, 224, 224)
            flow_values = torch.zeros(NUM_FRAMES, 3, 224, 224)
            tcl_features = np.zeros(12, dtype=np.float32)
        
        return {
            'rgb_values': rgb_values,
            'flow_values': flow_values,
            'tcl_features': torch.from_numpy(tcl_features),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# MOTION-AWARE FUSION MODULE
# -----------------------------------------------------------------------------
class MotionAwareFusionModule(nn.Module):
    """
    Fuses RGB, Optical Flow, and TCL features with attention
    
    Motion-guided attention
    - TCL tells us WHEN contact occurs
    - Flow tells us HOW people move
    - RGB tells us WHAT is happening
    """
    def __init__(self, feature_dim=768, tcl_dim=12):
        super().__init__()
        
        # TCL encoder
        self.tcl_encoder = nn.Sequential(
            nn.Linear(tcl_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128)
        )
        
        # Cross-modal attention: Flow guides RGB
        self.flow_to_rgb_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Cross-modal attention: RGB guides Flow
        self.rgb_to_flow_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Gating mechanism (learns importance of each stream)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # 3 weights: RGB, Flow, TCL
            nn.Softmax(dim=-1)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2 + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
    def forward(self, rgb_features, flow_features, tcl_features):
        """
        Args:
            rgb_features: [batch, feature_dim]
            flow_features: [batch, feature_dim]
            tcl_features: [batch, 12]
        """
        # Encode TCL
        tcl_encoded = self.tcl_encoder(tcl_features)  # [batch, 128]
        
        # Cross-modal attention
        rgb_seq = rgb_features.unsqueeze(1)
        flow_seq = flow_features.unsqueeze(1)
        
        # Flow attends to RGB
        flow_attended, _ = self.flow_to_rgb_attention(flow_seq, rgb_seq, rgb_seq)
        flow_attended = flow_attended.squeeze(1)
        
        # RGB attends to Flow
        rgb_attended, _ = self.rgb_to_flow_attention(rgb_seq, flow_seq, flow_seq)
        rgb_attended = rgb_attended.squeeze(1)
        
        # Adaptive gating
        combined_for_gate = torch.cat([rgb_attended, flow_attended, tcl_encoded], dim=1)
        weights = self.gate(combined_for_gate)  # [batch, 3]
        
        # Weighted combination
        weighted_rgb = weights[:, 0:1] * rgb_attended
        weighted_flow = weights[:, 1:2] * flow_attended
        weighted_tcl = weights[:, 2:3] * tcl_encoded
        
        # Final fusion
        combined = torch.cat([weighted_rgb, weighted_flow, weighted_tcl], dim=1)
        fused = self.fusion(combined)
        
        return fused

# -----------------------------------------------------------------------------
# COMPLETE DUAL-STREAM MODEL
# -----------------------------------------------------------------------------
class TCLOpticalFlowNetwork(nn.Module):
    """
    Complete architecture: RGB + Optical Flow + TCL
    """
    def __init__(self, backbone='MCG-NJU/videomae-base', num_classes=2):
        super().__init__()
        
        # RGB stream encoder
        self.rgb_encoder = VideoMAEModel.from_pretrained(backbone)
        
        # Optical Flow stream encoder (shared architecture, different weights)
        self.flow_encoder = VideoMAEModel.from_pretrained(backbone)
        
        # Freeze early layers for efficiency
        for encoder in [self.rgb_encoder, self.flow_encoder]:
            for param in encoder.embeddings.parameters():
                param.requires_grad = False
            
            for i in range(len(encoder.encoder.layer) - 3):
                for param in encoder.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        feature_dim = self.rgb_encoder.config.hidden_size
        
        # fusion module
        self.fusion_module = MotionAwareFusionModule(
            feature_dim=feature_dim,
            tcl_dim=12
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, rgb_input, flow_input, tcl_features):
        """
        Args:
            rgb_input: [batch, num_frames, 3, H, W]
            flow_input: [batch, num_frames, 3, H, W]
            tcl_features: [batch, 12]
        """
        # Extract features from both streams
        rgb_output = self.rgb_encoder(rgb_input)
        flow_output = self.flow_encoder(flow_input)
        
        # Pool features
        rgb_features = rgb_output.last_hidden_state.mean(dim=1)
        flow_features = flow_output.last_hidden_state.mean(dim=1)
        
        # Fuse with motion-aware attention
        fused_features = self.fusion_module(rgb_features, flow_features, tcl_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
def setup_device():
    if not torch.cuda.is_available():
        raise RuntimeError("🚨 No GPU!")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda")

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# MAIN TRAINING
# -----------------------------------------------------------------------------
def main():
    device = setup_device()
    
    print("\n" + "="*80)
    print("TCL + OPTICAL FLOW DUAL-STREAM NETWORK")
    print("="*80)
    print()
    print("⏱️  Training time: 3-4 hours")
    print()
    print("="*80 + "\n")
    
    # Load data
    print("📂 Loading dataset...")
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

    print(f"📊 Total: {len(video_files)} (Fight: {sum(labels)}, NoFight: {len(labels)-sum(labels)})")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_files, labels, test_size=0.2, stratify=labels, random_state=42
    )

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    
    print("\n⚙️  Creating dual-stream datasets (computing optical flow)...")
    train_dataset = DualStreamDataset(train_paths, train_labels, processor, augment=True)
    val_dataset = DualStreamDataset(val_paths, val_labels, processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Create model
    print("\n🧠 Building TCL + Optical Flow Network...")
    model = TCLOpticalFlowNetwork(backbone=MODEL_CKPT, num_classes=2)
    
    # Optional: Load pretrained TCL weights for RGB stream
    if PRETRAINED_TCL_PATH and os.path.exists(PRETRAINED_TCL_PATH):
        print(f"📥 Loading pretrained TCL weights from: {PRETRAINED_TCL_PATH}")
        try:
            checkpoint = torch.load(PRETRAINED_TCL_PATH, map_location='cpu')
            if 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']
            
            # Load RGB encoder weights (partial loading)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                             if k in model_dict and 'rgb_encoder' in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print("✅ Loaded pretrained TCL weights for RGB stream")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
            print("   Training from scratch...")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} / {total_params:,} trainable")
    
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    class_weights = torch.tensor([1.3, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print("\n🔥 Starting training...")
    print("-" * 80)

    best_acc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            rgb_values = batch['rgb_values'].to(device)
            flow_values = batch['flow_values'].to(device)
            tcl_features = batch['tcl_features'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(rgb_values, flow_values, tcl_features)
                loss = criterion(logits, labels) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if step % 20 == 0:
                    clean_cache()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
        
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                rgb_values = batch['rgb_values'].to(device)
                flow_values = batch['flow_values'].to(device)
                tcl_features = batch['tcl_features'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(rgb_values, flow_values, tcl_features)
                
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_true, val_preds)
        cm = confusion_matrix(val_true, val_preds)
        report = classification_report(val_true, val_preds, output_dict=True, zero_division=0)
        
        balanced_f1 = (report['0']['f1-score'] + report['1']['f1-score']) / 2
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"  Loss: {avg_train_loss:.4f} | Acc: {acc*100:.2f}% | F1: {balanced_f1:.4f}")
        print(f"  NoFight: P={report['0']['precision']:.3f} R={report['0']['recall']:.3f} F1={report['0']['f1-score']:.3f}")
        print(f"  Fight:   P={report['1']['precision']:.3f} R={report['1']['recall']:.3f} F1={report['1']['f1-score']:.3f}")
        print(f"  Confusion: NoFight [{cm[0,0]:2d},{cm[0,1]:2d}] | Fight [{cm[1,0]:2d},{cm[1,1]:2d}]")
        
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'accuracy': acc,
                'f1': balanced_f1
            }, os.path.join(SAVE_DIR, "best_dual_stream_model.pth"))
            
            print(f"  💾 NEW BEST: {acc*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
        
        print("="*80)
        
        if patience_counter >= patience and epoch > 25:
            print("\n⏹️  Early stopping")
            break
        
        clean_cache()

    # FINAL RESULTS
    print("\n" + "="*80)
    print("🎯 FINAL RESULTS (TCL + Optical Flow):")
    print("="*80)
    print(classification_report(val_true, val_preds, target_names=['NoFight', 'Fight'], digits=4))
    print(f"\n🏆 Best Accuracy: {best_acc*100:.2f}%")
    print()
    print("   - TCL modeling (duration of proximity)")
    print("   - Optical flow (motion patterns)")
    print("   - Motion-aware fusion (adaptive attention)")
    print(f"\n💾 Model saved: {SAVE_DIR}/best_dual_stream_model.pth")
    print("="*80)

if __name__ == '__main__':
    main()