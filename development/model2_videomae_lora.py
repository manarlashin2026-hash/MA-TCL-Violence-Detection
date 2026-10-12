"""
Development configuration 2 (Table 3): VideoMAE adapted with LoRA adapters only. SC-Fight-300.

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FIGHT_DIR = "./data/SC-Fight-300/fight"
NOFIGHT_DIR = "./data/SC-Fight-300/noFight"
MODEL_CKPT = "MCG-NJU/videomae-base"
BATCH_SIZE = 4
EPOCHS = 40
NUM_FRAMES = 16
SAVE_DIR = "./videomae_lora"

os.makedirs(SAVE_DIR, exist_ok=True)

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        while len(frames) < 300:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return None

        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        return [frames[i] for i in indices]

    def __getitem__(self, idx):
        frames = self.load_video(self.video_paths[idx])
        
        if frames is None:
            return {
                'pixel_values': torch.zeros(NUM_FRAMES, 3, 224, 224),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        
        inputs = self.processor(frames, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    device = torch.device('cuda')
    
    # Load data
    video_files = []
    labels = []
    
    for root, _, files in os.walk(FIGHT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                labels.append(1)
    
    for root, _, files in os.walk(NOFIGHT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                labels.append(0)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
    train_dataset = VideoDataset(train_paths, train_labels, processor)
    val_dataset = VideoDataset(val_paths, val_labels, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model with LoRA
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values=pixel_values)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}: Accuracy = {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(f"{SAVE_DIR}/best_lora")
    
    print(f"\nFinal LoRA Accuracy: {best_acc*100:.2f}%")
    print(classification_report(val_true, val_preds, target_names=['NoFight', 'Fight']))

if __name__ == '__main__':
    main()