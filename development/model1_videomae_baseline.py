"""
Development configuration 1 (Table 3): VideoMAE fine-tuned on RGB frames
(last two transformer blocks unfrozen, LoRA adapters on the remaining blocks). SC-Fight-300.

Run from the repository root; see README.md for the expected data layout.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# AGGRESSIVE CONFIGURATION FOR DIFFICULT DATASETS
# -----------------------------------------------------------------------------
FIGHT_DIR = "./data/SC-Fight-300/fight"
NOFIGHT_DIR = "./data/SC-Fight-300/noFight"

MODEL_CKPT = "MCG-NJU/videomae-base"
BATCH_SIZE = 2  # Smaller for more gradient updates
ACCUMULATION_STEPS = 2  # Effective batch = 4
EPOCHS = 20  # More epochs
LEARNING_RATE = 3e-4  # Higher LR
NUM_FRAMES = 16
SAVE_DIR = "./fight_model_aggressive"

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
# AUGMENTED DATASET
# -----------------------------------------------------------------------------
class FightDatasetAugmented(Dataset):
    def __init__(self, video_paths, labels, processor, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.video_paths)

    def augment_frames(self, frames):
        """Apply random augmentations"""
        if not self.augment or np.random.rand() > 0.5:
            return frames
        
        aug_type = np.random.choice(['brightness', 'flip', 'crop'])
        
        if aug_type == 'brightness':
            # Random brightness adjustment
            factor = np.random.uniform(0.7, 1.3)
            frames = [np.clip(f * factor, 0, 255).astype(np.uint8) for f in frames]
        elif aug_type == 'flip':
            # Horizontal flip
            frames = [cv2.flip(f, 1) for f in frames]
        elif aug_type == 'crop':
            # Random crop and resize
            h, w = frames[0].shape[:2]
            crop_size = int(min(h, w) * 0.9)
            top = np.random.randint(0, h - crop_size + 1)
            left = np.random.randint(0, w - crop_size + 1)
            frames = [cv2.resize(f[top:top+crop_size, left:left+crop_size], (w, h)) 
                     for f in frames]
        
        return frames

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        if not cap.isOpened():
            return None
            
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
            return None

        # Sample frames
        if len(frames) >= NUM_FRAMES:
            # Random temporal offset during training
            if self.augment:
                max_start = len(frames) - NUM_FRAMES
                start_idx = np.random.randint(0, max(1, max_start))
                indices = np.linspace(start_idx, min(start_idx + NUM_FRAMES*2, len(frames)-1), 
                                    NUM_FRAMES, dtype=int)
            else:
                indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        else:
            indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
            
        sampled_frames = [frames[i] for i in indices]
        
        # Apply augmentation
        sampled_frames = self.augment_frames(sampled_frames)
        
        return sampled_frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video(video_path)
        
        if frames is None:
            pixel_values = torch.zeros(NUM_FRAMES, 3, 224, 224)
        else:
            try:
                inputs = self.processor(frames, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
            except:
                pixel_values = torch.zeros(NUM_FRAMES, 3, 224, 224)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    device = setup_device()
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    print("\n🔍 Loading dataset...")
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

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    
    train_dataset = FightDatasetAugmented(train_paths, train_labels, image_processor, augment=True)
    val_dataset = FightDatasetAugmented(val_paths, val_labels, image_processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Model with FULL fine-tuning of last layers
    print("\n🧠 Loading model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=2,
        label2id={'NoFight': 0, 'Fight': 1},
        id2label={0: 'NoFight', 1: 'Fight'},
        ignore_mismatched_sizes=True
    )

    # Unfreeze last 2 transformer blocks + classifier
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last layers
    for param in model.videomae.encoder.layer[-2:].parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Also add LoRA on top
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # Higher rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "value", "key", "dense"]  # More modules
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    # Aggressive optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    
    # Simple step decay scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n🔥 Aggressive Training: {EPOCHS} epochs, LR={LEARNING_RATE}")
    print("-" * 80)

    best_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
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
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=pixel_values)
                
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_true, val_preds)
        cm = confusion_matrix(val_true, val_preds)
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"  Loss: {avg_train_loss:.4f} | Acc: {acc*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Confusion: NoFight [{cm[0][0]:2d},{cm[0][1]:2d}] | Fight [{cm[1][0]:2d},{cm[1][1]:2d}]")
        
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
            print(f"  💾 NEW BEST: {acc*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
        
        print("="*80)
        
        if patience_counter >= patience and epoch > 10:
            print("\n⏹️  Early stopping triggered")
            break
        
        clean_cache()

    # FINAL
    print("\n" + "="*80)
    print("FINAL REPORT:")
    print(classification_report(val_true, val_preds, target_names=['NoFight', 'Fight']))
    print(f"\n🎯 Best Accuracy: {best_acc*100:.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()