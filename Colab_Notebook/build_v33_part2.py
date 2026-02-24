"""
Build Cow_Lameness_Analysis_v33.ipynb — Part 2 (Sections 4-9)
Dataset, Model (LoRA), Training Loop
"""
import json, os

def _split(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": _split(source),
            "outputs": [], "execution_count": None}

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split(source)}

cells = []

# ═══════════════════════════════════════════════════════════════
# SECTION 4: Dataset (Video Loading — No Caching)
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 4: Dataset & DataLoader (Video-Level)

**Change:** Unlike v32 (cached features), v33 reads video clips during training to allow backward pass through the VideoMAE backbone (LoRA).
"""))

cells.append(code("""# ============================================================
# SECTION 4: Dataset Definition
# ============================================================

class CowLamenessDatasetV33(Dataset):
    \"\"\"
    Dataset for end-to-end training (Video -> LoRA -> Head).
    Loads video clips on-the-fly.
    \"\"\"
    def __init__(self, video_paths, pose_features, labels, cfg, transform=None):
        self.video_paths = video_paths
        self.pose_features = pose_features
        self.labels = labels
        self.cfg = cfg
        self.transform = transform
        
        # Processor for VideoMAE
        from transformers import VideoMAEImageProcessor
        self.processor = VideoMAEImageProcessor.from_pretrained(cfg["VIDEOMAE_MODEL"])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vpath = self.video_paths[idx]
        label = self.labels[idx]
        pose = self.pose_features[idx] # Already normalized

        # 1. Load Video
        pixel_values = self._load_video(vpath)
        
        # 2. Convert to Tensor
        # pose is (16,) -> (T, 16) repeated or just (16,) depending on fusion?
        # We'll use late fusion: (16,) vector concatenated after temporal pooling?
        # OR repeated per frame? Let's repeat per clip to match sequence length.
        
        # For temporal transformer, we have T tokens.
        # VideoMAE gives (T, 768).
        # Pose is (16,). We can repeat it to (T, 16).
        return {
            "pixel_values": pixel_values,  # (C, T, H, W) for HF
            "pose_features": torch.tensor(pose, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

    def _load_video(self, path):
        \"\"\"Load video, sample clips, output (C, T, H, W).\"\"\"
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Sampling strategy: Uniformly sample NUM_CLIPS * CLIP_LENGTH frames?
        # No, VideoMAE expects a single clip of T frames.
        # But we want long-term temporal reasoning (8 clips).
        # We will shape input as (B, N_CLIPS, C, T, H, W) -> folded to (B*N, C, T, H, W)
        
        total_frames = len(frames)
        required_frames = self.cfg["NUM_CLIPS"] * self.cfg["CLIP_LENGTH"]
        
        if total_frames < required_frames:
            # Pad with last frame
            pad = [frames[-1]] * (required_frames - total_frames)
            frames.extend(pad)
        
        # Sub-sample or take consecutive?
        # Let's take evenly spaced clips to cover the whole gait cycle.
        indices = np.linspace(0, len(frames)-1, required_frames).astype(int)
        sampled_frames = [frames[i] for i in indices]
        
        # Process with HF ImageProcessor
        # Input to processor: list of numpy arrays or list of list of numpy arrays
        # If we pass a list of 128 frames, it might truncate/sample to 16. 
        # We must process each CLIP individually to ensure we get (NUM_CLIPS, 16, 3, 224, 224).
        
        # Reshape frames into (NUM_CLIPS, CLIP_LENGTH, H, W, 3)
        clips = []
        for i in range(self.cfg["NUM_CLIPS"]):
            start = i * self.cfg["CLIP_LENGTH"]
            end = start + self.cfg["CLIP_LENGTH"]
            clip_frames = sampled_frames[start:end]
            clips.append(clip_frames)
            
        # Process each clip
        # processor(images=clip_frames) returns pixel_values (1, 16, 3, 224, 224)
        # We process a batch of clips: list of list of frames
        inputs = self.processor(clips, return_tensors="pt")
        
        # Output pixel_values: (NUM_CLIPS, 16, 3, 224, 224)
        vid = inputs["pixel_values"]
        
        # No need to transpose! VideoMAE expects (B, T, C, H, W). 
        # vid is already (NUM_CLIPS, 16, 3, 224, 224).
        
        return vid

def collate_fn_v33(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch]) # (B, N, T, C, H, W)
    pose = torch.stack([x["pose_features"] for x in batch])        # (B, 16)
    labels = torch.stack([x["label"] for x in batch])              # (B,)
    return pixel_values, pose, labels
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 5: Model Definition (LoRA)
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 5: Hybrid Model with LoRA

**Architecture:**
1.  **Backbone:** `VideoMAEModel` (Pretrained) wrapped with **PEFT LoRA**.
2.  **Adapter:** 2-layer FFN (Gradient-enabled).
3.  **Fusion:** Concatenates VideoMAE [CLS] token with Pose features.
4.  **Head:** Transformer Encoder + Classifier.
"""))

cells.append(code("""# ============================================================
# SECTION 5: Model with LoRA
# ============================================================

class CowLamenessModelV33(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. VideoMAE Backbone
        self.backbone = VideoMAEModel.from_pretrained(cfg["VIDEOMAE_MODEL"])
        
        # 2. Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=cfg["LORA_R"],
            lora_alpha=cfg["LORA_ALPHA"],
            lora_dropout=cfg["LORA_DROPOUT"],
            target_modules=cfg["LORA_TARGET_MODULES"]
        )
        # Try to apply LoRA, with fallback debug info
        try:
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()
        except ValueError as e:
            print(f"❌ LoRA Error: {e}")
            print("🔍 Available modules in backbone:")
            for name, _ in self.backbone.named_modules():
                print(f"  - {name}")
            raise e
        
        # 3. Domain Adapter (Trainable)
        self.adapter = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )
        
        # 4. Temporal Transformer
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=cfg["NUM_HEADS"],
                                       dim_feedforward=512, dropout=cfg["DROPOUT"],
                                       batch_first=True),
            num_layers=cfg["NUM_LAYERS"]
        )
        
        # 5. Pose Projection
        self.pose_proj = nn.Sequential(
            nn.Linear(cfg["POSE_FEAT_DIM"], 64),
            nn.ReLU(),
            nn.Linear(64, 256) # Project to model dim
        )
        
        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values, pose_features):
        \"\"\"
        pixel_values: (B, N_CLIPS, T, C, H, W) from dataset/collate
        pose_features: (B, 16)
        \"\"\"
        B, N, T, C, H, W = pixel_values.shape
        
        # Fold batch & clips: (B*N, T, C, H, W) - VideoMAE expects (B, T, C, H, W)
        x = pixel_values.view(B*N, T, C, H, W)
        
        # backbone forward (VideoMAE)
        outputs = self.backbone(pixel_values=x)
        
        # VideoMAE (MAE style) does NOT have a CLS token at index 0 typically.
        # It outputs (B*N, 1568, 768). Use Mean Pooling.
        features = outputs.last_hidden_state.mean(dim=1) # (B*N, 768)
        
        # Domain Adapter
        features = self.adapter(features) # (B*N, 256)
        
        # Unfold: (B, N, 256)
        features = features.view(B, N, 256)
        
        # Pose Injection: Add pose embedding to *every* time step?
        # Or concat as extra token? Let's add to every step (residual style).
        pose_embed = self.pose_proj(pose_features).unsqueeze(1) # (B, 1, 256)
        features = features + pose_embed
        
        # Temporal Encoder
        # features is (B, N, 256)
        temp_out = self.temporal_encoder(features)
        
        # Global Pooling (Mean)
        x_pool = temp_out.mean(dim=1) # (B, 256)
        
        # Classifier
        logits = self.classifier(x_pool)
        return logits
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 6: Training Loop
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 6: Training Loop with Gradient Accumulation & AMP

Since we are training the backbone, memory usage is higher.
- Using `torch.cuda.amp` for Mixed Precision (fp16).
- Batch size is small (4), so we use standard optimization.
"""))

cells.append(code("""# ============================================================
# SECTION 6: Training Routines
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss = 0
    n_batches = 0
    scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, (vid, pose, label) in enumerate(loader):
        vid, pose, label = vid.to(device), pose.to(device), label.to(device).float()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(vid, pose).squeeze(-1)
            loss = criterion(logits, label)
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\\n⚠️ WARNING: NaN/Inf loss at batch {batch_idx}! Skipping batch.")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping BEFORE step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
        
        # Check for gradient explosion
        if grad_norm > 10.0:
            print(f"\\n⚠️ WARNING: Large gradient norm ({grad_norm:.2f}) at batch {batch_idx}")
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"\\rBatch {batch_idx}/{len(loader)} Loss: {loss.item():.4f} | GradNorm: {grad_norm:.3f}", end="")
    
    avg_loss = total_loss / max(n_batches, 1)
    print(f"\\rEpoch complete - Avg Loss: {avg_loss:.4f}                    ")
    return avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for vid, pose, label in loader:
            vid, pose, label = vid.to(device), pose.to(device), label.to(device).float()
            
            with torch.cuda.amp.autocast():
                logits = model(vid, pose).squeeze(-1)
                loss = criterion(logits, label)
                
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(label.cpu().numpy())
            total_loss += loss.item()
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    
    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }
    return metrics, all_probs, all_labels
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 7: Cross Validation
# ═══════════════════════════════════════════════════════════════
cells.append(code("""# ============================================================
# SECTION 7: 5-Fold CV Execution
# ============================================================

def run_cv_v33(data_df, pose_features, cfg, device):
    cv = StratifiedGroupKFold(n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=cfg["SEED"])
    
    video_paths = data_df["video_path"].values
    labels = data_df["label"].values
    groups = data_df["animal_id"].values
    
    global best_models 
    best_models = []
    
    global fold_results
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(video_paths, labels, groups)):
        print(f"\\n{'='*40}\\nFOLD {fold+1}/{cfg['CV_FOLDS']}\\n{'='*40}")
        
        # Standardize Pose Features (Fit on TRAIN only to avoid leakage)
        scaler = StandardScaler()
        # Scale train
        train_pose = scaler.fit_transform(pose_features[train_idx])
        # Scale val using train stats
        val_pose = scaler.transform(pose_features[val_idx]) # transform only!
        
        # Generators
        train_ds = CowLamenessDatasetV33(video_paths[train_idx], train_pose, labels[train_idx], cfg)
        val_ds = CowLamenessDatasetV33(video_paths[val_idx], val_pose, labels[val_idx], cfg)
        
        train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn_v33)
        val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn_v33)
        
        # Model Init
        model = CowLamenessModelV33(cfg).to(device)
        
        # Optimizer (LoRA needs higher LR, head regular)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg["LR_BACKBONE"]},
            {'params': model.adapter.parameters(), 'lr': cfg["LR_HEAD"]},
            {'params': model.temporal_encoder.parameters(), 'lr': cfg["LR_HEAD"]},
            {'params': model.classifier.parameters(), 'lr': cfg["LR_HEAD"]},
            {'params': model.pose_proj.parameters(), 'lr': cfg["LR_HEAD"]},
        ], weight_decay=cfg["WEIGHT_DECAY"])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Class weights for imbalanced data (642 healthy vs 525 lame)
        n_healthy = (labels[train_idx] == 0).sum()
        n_lame = (labels[train_idx] == 1).sum()
        pos_weight = torch.tensor([n_healthy / n_lame]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        best_acc = 0
        best_f1 = 0 
        patience = 0
        best_model_state = None
        
        # Metrics History
        history = {
            "train_loss": [], "val_loss": [],
            "val_acc": [], "val_auc": [], "val_f1": []
        }
        
        for epoch in range(cfg["EPOCHS"]):
            t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg)
            v_metrics, v_probs, v_labels = evaluate(model, val_loader, criterion, device)
            
            # Record History
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_metrics["loss"])
            history["val_acc"].append(v_metrics["accuracy"])
            history["val_auc"].append(v_metrics["auc"])
            history["val_f1"].append(v_metrics["f1"])
            
            scheduler.step(v_metrics["loss"])
            
            print(f"  Ep {epoch+1}: T_Loss={t_loss:.3f} | V_Loss={v_metrics['loss']:.3f} | Acc={v_metrics['accuracy']:.3f} | F1={v_metrics['f1']:.3f} | AUC={v_metrics['auc']:.3f}")
            
            # Save best model based on F1 (balanced metric for skewed data)
            if v_metrics["f1"] > best_f1:
                best_f1 = v_metrics["f1"]
                best_acc = v_metrics["accuracy"]
                best_auc = v_metrics["auc"]
                patience = 0
                import copy
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            else:
                patience += 1
                
            if patience >= cfg["PATIENCE"]:
                print(f"⏹ Early stopping at epoch {epoch+1} (best F1: {best_f1:.3f} at epoch {best_epoch})")
                break
        
        # Load best model and evaluate on validation set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            final_metrics, final_probs, final_labels = evaluate(model, val_loader, criterion, device)
        else:
            # Fallback if no best model saved
            final_metrics, final_probs, final_labels = v_metrics, all_probs, all_labels
        
        # Save fold results for visualization
        fold_results.append({
            "fold": fold + 1,
            "accuracy": final_metrics["accuracy"],
            "auc": final_metrics["auc"],
            "f1": final_metrics["f1"],
            "precision": final_metrics.get("precision", 0), 
            "recall": final_metrics.get("recall", 0),
            "best_epoch": best_epoch if best_model_state is not None else epoch + 1,
            "history": history,
            "fold_probs": final_probs,
            "fold_labels": final_labels
        })
        
        # Store best model for Part 3 saving
        best_models.append({
             "fold": fold + 1,
             "model": best_model_state # Store the dict
        })
        
        del model, optimizer
        torch.cuda.empty_cache()
        
    print(f"\\n🏆 Overall CV Accuracy: {np.mean([r['accuracy'] for r in fold_results]):.3f} ± {np.std([r['accuracy'] for r in fold_results]):.3f}")
    
    # Aggregate for global plots
    global all_probs, all_labels
    all_probs = np.concatenate([r["fold_probs"] for r in fold_results])
    all_labels = np.concatenate([r["fold_labels"] for r in fold_results])

run_cv_v33(data_df, pose_features, CFG, DEVICE)
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 8: Training Summary & Results Overview
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 8: Cross-Validation Training Summary

After completing 5-fold CV, this section provides:
- Overall performance summary (mean ± std across folds)
- Quick visualization of training progress
- Model checkpoint information
"""))

cells.append(code("""# ============================================================
# SECTION 8: Training Summary
# ============================================================

if 'fold_results' in globals() and len(fold_results) > 0:
    print("\\n" + "="*60)
    print("📊 CROSS-VALIDATION TRAINING SUMMARY")
    print("="*60)
    
    # Calculate overall statistics
    accuracies = [r.get("accuracy", 0) for r in fold_results]
    aucs = [r.get("auc", 0) for r in fold_results]
    f1s = [r.get("f1", 0) for r in fold_results]
    precisions = [r.get("precision", 0) for r in fold_results]
    recalls = [r.get("recall", 0) for r in fold_results]
    
    print(f"\\nOverall Performance (Mean ± Std across {len(fold_results)} folds):")
    print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  F1-Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    print(f"\\nPer-Fold Summary:")
    for i, r in enumerate(fold_results):
        print(f"  Fold {r.get('fold', i+1)}: Acc={r.get('accuracy', 0):.3f}, "
              f"F1={r.get('f1', 0):.3f}, AUC={r.get('auc', 0):.3f}, "
              f"Best Epoch={r.get('best_epoch', 0)}")
    
    print(f"\\n✅ Training completed successfully!")
    print(f"📁 Results saved to: {CFG['RESULTS_DIR']}")
    print("="*60)
else:
    print("⚠️ No training results found. Please run Section 7 (CV Execution) first.")
"""))

# Save part 2
notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU"},
    "cells": cells
}

SCRIPT_DIR = r"c:\Users\HP\Desktop\Clone Repos\CowLameness\Colab_Notebook"
out_path = SCRIPT_DIR + "\\_v33_part2.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Part 2 saved: {out_path} ({len(cells)} cells)")
