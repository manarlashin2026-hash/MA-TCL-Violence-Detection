"""
Development configuration 3 (Table 3): VideoMAE + proximity-only TCL descriptor (12-D)
with cross-modal attention fusion. SC-Fight-300.

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
EPOCHS = 60
NUM_FRAMES = 16  # VideoMAE standard
SAVE_DIR = "./fight_model_tcl"

INITIAL_LR = 8e-5

# -----------------------------------------------------------------------------
# TEMPORAL CONTACT LENGTH (TCL) EXTRACTOR
# -----------------------------------------------------------------------------
class TemporalContactLengthExtractor:
    """
    Explicitly measures contact duration
    
    This is what distinguishes fights from hugs:
    - Fight: Long, sustained contact (10+ seconds)
    - Hug: Brief contact (2-3 seconds)
    - Handshake: Very brief (1 second)
    """
    
    @staticmethod
    def detect_people_proximity(frame):
        """
        Detect if people are in close proximity
        Uses center-mass concentration as proxy
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Divide frame into grid
        cell_h, cell_w = h // 4, w // 4
        
        # Center cells (where people usually are)
        center_cells = [
            gray[cell_h:3*cell_h, cell_w:3*cell_w]
        ]
        
        # Edge detection (more edges = more object boundaries = people close)
        edges = cv2.Canny(center_cells[0], 50, 150)
        edge_density = edges.sum() / (center_cells[0].shape[0] * center_cells[0].shape[1])
        
        # Motion concentration
        # High value = activity concentrated in center (people interacting)
        center_variance = np.std(center_cells[0])
        
        # Proximity score (0-1)
        proximity = min(1.0, (edge_density * 0.01 + center_variance / 255.0))
        
        return proximity
    
    @staticmethod
    def compute_contact_signature(frames, proximity_threshold=0.3):
        """
        Contact Length Signature
        
        Returns:
            - duration_score: How long contact lasts
            - consistency_score: How sustained contact is
            - pattern: Temporal pattern of contact
        """
        proximity_scores = []
        
        for frame in frames:
            prox = TemporalContactLengthExtractor.detect_people_proximity(frame)
            proximity_scores.append(prox)
        
        proximity_array = np.array(proximity_scores)
        
        # 1. DURATION: Longest continuous streak of high proximity
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
        
        # Duration score: longest streak / total frames
        duration_score = max_streak / len(frames) if frames else 0.0
        
        # 2. CONSISTENCY: How stable is the contact?
        # Low variance = sustained contact (fight)
        # High variance = intermittent contact (hug)
        consistency_score = 1.0 / (proximity_array.std() + 0.1)
        consistency_score = min(1.0, consistency_score / 10.0)  # Normalize
        
        # 3. PATTERN: Contact distribution over time
        # Fights: High proximity throughout
        # Hugs: High proximity in middle, low at edges
        avg_proximity = proximity_array.mean()
        
        # 4. NUMBER OF CONTACT EVENTS
        num_contact_events = len(streaks)
        event_score = 1.0 / (num_contact_events + 1)  # Fewer events = longer single contact
        
        # Combined signature
        signature = {
            'duration': duration_score,
            'consistency': consistency_score,
            'avg_proximity': avg_proximity,
            'num_events': num_contact_events,
            'event_score': event_score,
            'pattern': proximity_array.tolist()
        }
        
        return signature
    
    @staticmethod
    def signature_to_vector(signature):
        """Convert signature to fixed-size feature vector"""
        # Temporal binning: divide timeline into 8 bins
        pattern = np.array(signature['pattern'])
        num_bins = 8
        bin_size = len(pattern) // num_bins
        
        binned_pattern = []
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < num_bins - 1 else len(pattern)
            bin_mean = pattern[start_idx:end_idx].mean() if end_idx > start_idx else 0
            binned_pattern.append(bin_mean)
        
        # Feature vector: [duration, consistency, avg_prox, event_score, 8 temporal bins]
        feature_vector = [
            signature['duration'],
            signature['consistency'],
            signature['avg_proximity'],
            signature['event_score']
        ] + binned_pattern
        
        return np.array(feature_vector, dtype=np.float32)

# -----------------------------------------------------------------------------
# TCL-AWARE DATASET
# -----------------------------------------------------------------------------
class TCLDataset(Dataset):
    """Dataset that extracts Temporal Contact Length features"""
    
    def __init__(self, video_paths, labels, processor, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.tcl_extractor = TemporalContactLengthExtractor()

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        if not cap.isOpened():
            return None, None
            
        try:
            while len(frames) < 400:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            return None, None

        # Sample frames uniformly
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        
        # Compute TCL signature
        tcl_signature = self.tcl_extractor.compute_contact_signature(sampled_frames)
        tcl_features = self.tcl_extractor.signature_to_vector(tcl_signature)
        
        return sampled_frames, tcl_features

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames, tcl_features = self.load_video(video_path)
        
        if frames is None:
            return {
                'pixel_values': torch.zeros(NUM_FRAMES, 3, 224, 224),
                'tcl_features': torch.zeros(12),  # 4 global + 8 temporal bins
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        try:
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        except:
            pixel_values = torch.zeros(NUM_FRAMES, 3, 224, 224)
            tcl_features = np.zeros(12, dtype=np.float32)
        
        return {
            'pixel_values': pixel_values,
            'tcl_features': torch.from_numpy(tcl_features),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# TCL FUSION MODULE
# -----------------------------------------------------------------------------
class TCLFusionModule(nn.Module):
    """
    Fuses appearance features with Temporal Contact Length features
    
    Design:
    - TCL features guide attention over video frames
    - Learns which frames are important based on contact patterns
    """
    def __init__(self, video_feature_dim=768, tcl_feature_dim=12):
        super().__init__()
        
        # TCL encoder (learns to interpret contact signatures)
        self.tcl_encoder = nn.Sequential(
            nn.Linear(tcl_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256)
        )
        
        # Cross-attention: TCL features guide video attention
        self.tcl_guided_attention = nn.MultiheadAttention(
            embed_dim=video_feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(video_feature_dim + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
    def forward(self, video_features, tcl_features):
        """
        Args:
            video_features: [batch, feature_dim] - from VideoMAE
            tcl_features: [batch, 12] - contact length signature
        
        Returns:
            fused: [batch, 256] - combined representation
        """
        # Encode TCL features
        tcl_encoded = self.tcl_encoder(tcl_features)  # [batch, 256]
        
        # Use TCL to create attention query
        video_seq = video_features.unsqueeze(1)  # [batch, 1, feature_dim]
        tcl_query = tcl_encoded.unsqueeze(1)  # [batch, 1, 256]
        
        # Expand dimensions for attention
        tcl_key = tcl_query.expand(-1, video_seq.size(1), -1)
        
        # Project to same dimension as video for attention
        tcl_proj = nn.Linear(256, video_features.size(-1)).to(video_features.device)
        tcl_key_proj = tcl_proj(tcl_key)
        
        # TCL-guided attention on video features
        attended_video, attention_weights = self.tcl_guided_attention(
            video_seq, tcl_key_proj, video_seq
        )
        attended_video = attended_video.squeeze(1)  # [batch, feature_dim]
        
        # Concatenate and fuse
        combined = torch.cat([attended_video, tcl_encoded], dim=1)
        fused = self.fusion(combined)
        
        return fused, attention_weights

# -----------------------------------------------------------------------------
# COMPLETE TCL NETWORK
# -----------------------------------------------------------------------------
class TCLViolenceDetector(nn.Module):
    """
    Complete Temporal Contact Length Network
    
    Components:
    1. Explicit temporal contact modeling (not implicit)
    2. TCL-guided attention mechanism
    3. Interpretable features (can visualize "when" contact occurs)
    """
    def __init__(self, backbone='MCG-NJU/videomae-base', num_classes=2):
        super().__init__()
        
        # Backbone: VideoMAE (appearance encoder)
        self.backbone = VideoMAEModel.from_pretrained(backbone)
        
        # Freeze early layers
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(len(self.backbone.encoder.layer) - 3):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = False
        
        video_feature_dim = self.backbone.config.hidden_size
        
        # TCL fusion module
        self.tcl_fusion = TCLFusionModule(
            video_feature_dim=video_feature_dim,
            tcl_feature_dim=12
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, pixel_values, tcl_features, return_attention=False):
        """
        Args:
            pixel_values: [batch, num_frames, 3, H, W]
            tcl_features: [batch, 12] - contact length signature
        
        Returns:
            logits: [batch, num_classes]
            attention_weights: (optional) for visualization
        """
        # Extract video features
        video_output = self.backbone(pixel_values)
        video_features = video_output.last_hidden_state.mean(dim=1)
        
        # Fuse with TCL features
        fused_features, attention_weights = self.tcl_fusion(video_features, tcl_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        if return_attention:
            return logits, attention_weights
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
# MAIN
# -----------------------------------------------------------------------------
def main():
    device = setup_device()
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("TEMPORAL CONTACT LENGTH (TCL) NETWORK")
    print("="*80)
    print()
    print()
    print()
    print("="*80 + "\n")
    
    print("🔍 Loading dataset...")
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
    print(f"📂 Train: {len(train_paths)}, Val: {len(val_paths)}")

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    
    print("\n⏱️  Computing Temporal Contact Length signatures...")
    train_dataset = TCLDataset(train_paths, train_labels, image_processor, augment=True)
    val_dataset = TCLDataset(val_paths, val_labels, image_processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Model
    print("\n🧠 Building TCL Network...")
    model = TCLViolenceDetector(backbone=MODEL_CKPT, num_classes=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}% trainable)")
    
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=12, T_mult=2, eta_min=1e-7
    )
    
    # Weighted loss with label smoothing
    class_weights = torch.tensor([1.5, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print("\n🔥 Starting TCL Training")
    print("-" * 80)

    best_acc = 0.0
    best_balanced_f1 = 0.0
    patience = 18
    patience_counter = 0

    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            tcl_features = batch['tcl_features'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(pixel_values, tcl_features)
                loss = criterion(logits, labels) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(epoch + step / len(train_loader))
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(device)
                tcl_features = batch['tcl_features'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(pixel_values, tcl_features)
                
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
        
        if balanced_f1 > best_balanced_f1:
            best_balanced_f1 = balanced_f1
            best_acc = acc
            patience_counter = 0
            
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'accuracy': acc,
                'f1': balanced_f1
            }, os.path.join(SAVE_DIR, "best_tcl_model.pth"))
            
            print(f"  💾 NEW BEST: Acc={acc*100:.2f}%, F1={balanced_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
        
        print("="*80)
        
        if patience_counter >= patience and epoch > 25:
            print("\n⏹️  Early stopping")
            break
        
        clean_cache()

    # FINAL
    print("\n" + "="*80)
    print("🎯 FINAL TCL NETWORK RESULTS:")
    print("="*80)
    print(classification_report(val_true, val_preds, target_names=['NoFight', 'Fight'], digits=4))
    print(f"\n🏆 Best Accuracy: {best_acc*100:.2f}%")
    print(f"🏆 Best Balanced F1: {best_balanced_f1:.4f}")
    print()
    print()
    print("="*80)

if __name__ == '__main__':
    main()