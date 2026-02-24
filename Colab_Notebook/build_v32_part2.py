"""
Build Cow_Lameness_Analysis_v32.ipynb — Part 2 (Sections 4-8)
VideoMAE (Partial FT hybrid), Model, Dataset, Training Loop
"""
import json, os

def _split(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": _split(source),
            "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════════════
# SECTION 4: VideoMAE Encoder (Partial FT — Hybrid)
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 4: VideoMAE Encoder — Partial Fine-Tuning (Hybrid)

**Strategy (ADR-001 — Revised):**
- Blocks 0-9: **FROZEN** — intermediate features pre-computed once & cached (768-dim)
- Blocks 10-11: **TRAINABLE** — domain adapter, trained during each fold
- Projection: 768 → 256 (trainable)
- Hybrid approach: blocks 0-9 output cached (~28 MB), blocks 10-11 run live

> **Why hybrid?** Pre-computing frozen block outputs avoids redundant forward passes
> through 10 transformer layers every epoch. Only the 2 trainable blocks + projection
> + temporal model are computed per batch. ~10× faster than online inference.

**Architecture:**
```
[One-time pre-computation]
Raw clips → VideoMAE blocks 0-9 (frozen) → mean pool → cache (768-dim per clip)

[Per-epoch training]
Cached 768-dim → blocks 10-11 (trainable) → layernorm → projection → 256-dim
→ concat(visual_256, pose_16) → Temporal Transformer → BCE
```
"""))

cells.append(code("""# ============================================================
# SECTION 4: VideoMAE Encoder — Partial Fine-Tuning (Hybrid)
# ============================================================

class VideoMAEFrozenEncoder(nn.Module):
    \"\"\"
    Uses full VideoMAE model with output_hidden_states to extract
    intermediate features after block (split_at - 1).
    All parameters frozen; run once and cached.
    \"\"\"

    def __init__(self, videomae_model, split_at: int = 10):
        super().__init__()
        self.model = videomae_model
        self.split_at = split_at
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            pixel_values: (B, T, C, H, W)
        Returns:
            intermediate_features: (B, 768) — mean-pooled after block (split_at-1)
        \"\"\"
        outputs = self.model(pixel_values, output_hidden_states=True)
        # hidden_states tuple: (embedding_output, block0_out, ..., block11_out)
        # Index split_at → output after block (split_at - 1)
        intermediate = outputs.hidden_states[self.split_at]  # (B, seq_len, 768)
        return intermediate.mean(dim=1)  # (B, 768)


class VideoMAEDomainAdapter(nn.Module):
    \"\"\"
    Domain adaptation FFN replacing VideoMAE blocks 10-11.

    With mean-pooled input (seq_length=1), self-attention degenerates
    to identity. Two residual FFN blocks replicate the blocks' capacity
    (768 → 3072 → 768 each), followed by projection (768 → 256).
    \"\"\"

    def __init__(self, input_dim: int = 768, hidden_dim: int = 3072,
                 projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # Block 1 FFN equivalent (residual)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

        # Block 2 FFN equivalent (residual)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

        # Projection: 768 → projection_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        n_ffn = sum(p.numel() for p in self.ffn1.parameters()) + \
                sum(p.numel() for p in self.ffn2.parameters())
        n_proj = sum(p.numel() for p in self.projection.parameters())
        print(f"  Domain adapter FFN: {n_ffn:,} params")
        print(f"  Projection: {n_proj:,} params")
        print(f"  Total trainable adapter: {n_ffn + n_proj:,} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x: (B, 768) — mean-pooled intermediate features from frozen blocks
        Returns:
            adapted: (B, projection_dim) — domain-adapted visual embeddings
        \"\"\"
        x = x + self.ffn1(x)    # Residual block 1
        x = x + self.ffn2(x)    # Residual block 2
        return self.projection(x)  # (B, projection_dim)


# ── Load VideoMAE & create components ──
print("📦 Loading VideoMAE for Partial Fine-Tuning (hybrid)...")
_full_videomae = VideoMAEModel.from_pretrained(CFG["VIDEOMAE_MODEL"])

split_at = min(CFG["TRAINABLE_BLOCKS"])

# Domain adapter — standalone FFN (no VideoMAE block dependency)
domain_adapter = VideoMAEDomainAdapter(
    input_dim=CFG["VIDEOMAE_DIM"],
    projection_dim=CFG["PROJECTION_DIM"],
).to(DEVICE)

# Frozen encoder wraps full model (uses official forward + output_hidden_states)
frozen_encoder = VideoMAEFrozenEncoder(_full_videomae, split_at=split_at).to(DEVICE)

# Verify freeze status
n_frozen = sum(p.numel() for p in frozen_encoder.parameters() if not p.requires_grad)
n_trainable_enc = sum(p.numel() for p in frozen_encoder.parameters() if p.requires_grad)
n_trainable_adapt = sum(p.numel() for p in domain_adapter.parameters() if p.requires_grad)
print(f"\\n✅ VideoMAE hybrid setup complete:")
print(f"   Frozen encoder: {n_frozen:,} params (ALL FROZEN)")
print(f"   Domain adapter: {n_trainable_adapt:,} trainable params")
assert n_trainable_enc == 0, "❌ Frozen encoder has trainable params!"
print("🗑️ Full model will be freed after pre-computation")
"""))

cells.append(code("""# ============================================================
# Clip Extraction & Encoding (standalone functions)
# ============================================================

def extract_clips_from_video(video_path: str, cfg: dict) -> Optional[List[np.ndarray]]:
    \"\"\"Extract clips from a single video. Each clip: (CLIP_LENGTH, H, W, 3).\"\"\"
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (cfg["IMG_SIZE"], cfg["IMG_SIZE"]))
            frames.append(frame)
        cap.release()

        if len(frames) < cfg["CLIP_LENGTH"]:
            return None

        clips = []
        stride = max(1, (len(frames) - cfg["CLIP_LENGTH"]) // cfg["NUM_CLIPS"])
        for start in range(0, len(frames) - cfg["CLIP_LENGTH"] + 1, stride):
            clip = np.array(frames[start:start + cfg["CLIP_LENGTH"]])
            clips.append(clip)
            if len(clips) >= cfg["NUM_CLIPS"] * 2:
                break

        return clips if clips else None
    except Exception:
        return None


@torch.no_grad()
def encode_clips_intermediate(encoder: VideoMAEFrozenEncoder, clips: List[np.ndarray],
                               device: str) -> np.ndarray:
    \"\"\"Encode clips with frozen blocks 0-9. Returns (N, 768) numpy.\"\"\"
    encoder.eval()
    embeddings = []

    for clip in clips:
        # Normalize with ImageNet stats
        clip_f = clip.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        clip_f = (clip_f - mean) / std

        # (T, H, W, C) → (1, T, C, H, W) — HuggingFace VideoMAE format
        tensor = torch.from_numpy(clip_f).permute(0, 3, 1, 2).unsqueeze(0).float()
        tensor = tensor.to(device)

        emb = encoder(tensor)  # (1, 768)
        embeddings.append(emb.cpu().numpy().squeeze())

    return np.array(embeddings)


print("✅ Clip extraction & encoding functions defined")
"""))

cells.append(code("""# ============================================================
# Pre-compute intermediate features (blocks 0-9, cached)
# ============================================================

def precompute_intermediate_features(video_paths, encoder, cfg, device,
                                      cache_path=None):
    \"\"\"
    Pre-compute frozen blocks 0-9 output for all videos.
    Returns list of (n_clips_i, 768) numpy arrays.
    \"\"\"
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"✅ Loaded cached intermediate features: {len(data['features'])} videos")
        return data['features']

    print(f"🔄 Pre-computing intermediate features (blocks 0-{min(cfg['TRAINABLE_BLOCKS'])-1}) "
          f"for {len(video_paths)} videos...")
    all_features = []
    n_failed = 0

    for i, vpath in enumerate(video_paths):
        clips = extract_clips_from_video(vpath, cfg)
        if clips is not None and len(clips) > 0:
            feats = encode_clips_intermediate(encoder, clips, device)
            all_features.append(feats)
        else:
            all_features.append(np.zeros((0, cfg["VIDEOMAE_DIM"]), dtype=np.float32))
            n_failed += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(video_paths)} videos...")

    if cache_path:
        np.save(cache_path, {'features': all_features}, allow_pickle=True)
        print(f"💾 Cached intermediate features")

    n_valid = len(video_paths) - n_failed
    print(f"✅ Intermediate features: {n_valid}/{len(video_paths)} videos encoded")
    return all_features


vis_cache = os.path.join(CFG["RESULTS_DIR"], "intermediate_features_cache.npy")
all_intermediate_features = precompute_intermediate_features(
    data_df["video_path"].values, frozen_encoder, CFG, DEVICE,
    cache_path=vis_cache
)

# Free frozen encoder (and the full model it wraps) — no longer needed
del frozen_encoder
torch.cuda.empty_cache()
print("🗑️ Frozen encoder + full VideoMAE model released from GPU memory")
print(f"📊 Domain adapter remains on {DEVICE} for training")
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 5-6: Temporal Transformer + Classification Head
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Sections 5-6: Temporal Transformer + Classification Head

**Pipeline per video (training):**
```
cached intermediate (768-dim per clip)
→ Domain Adapter (blocks 10-11, trainable) → projection → 256-dim
→ concat(visual_256, pose_16) → 272-dim
→ Linear projection → 256-dim
→ Positional Encoding
→ TransformerEncoder (4 layers, 8 heads, causal mask)
→ Mean pooling → 256-dim
→ FC → Sigmoid → p(lame) ∈ [0, 1]
```

> **Two-LR training:** Domain adapter at `LR_VIDEOMAE` (1e-5),
> temporal model at `LR_HEAD` (1e-4).
"""))

cells.append(code("""# ============================================================
# SECTIONS 5-6: Temporal Transformer + Classification Head
# ============================================================

class CowLamenessModelV32(nn.Module):
    \"\"\"
    Complete lameness detection model with domain adaptation.

    Input: sequence of pre-computed intermediate clip embeddings (768-dim) + pose
    Output: binary probability p(lame)

    Architecture:
        1. Domain adapter: blocks 10-11 (768 → 768) → projection (768 → 256)
        2. Concat with pose: (256 + 16) = 272
        3. Input projection: 272 → hidden_dim
        4. Positional encoding (sinusoidal)
        5. TransformerEncoder with causal mask (4 layers, 8 heads)
        6. Temporal mean pooling
        7. Classification head → sigmoid
    \"\"\"

    def __init__(self, adapter: VideoMAEDomainAdapter, pose_dim: int,
                 hidden_dim: int, num_heads: int, num_layers: int,
                 dropout: float, max_clips: int = 32):
        super().__init__()

        self.adapter = adapter
        visual_dim = adapter.projection[1].out_features  # nn.Linear(768, projection_dim)
        self.input_dim = visual_dim + pose_dim
        self.hidden_dim = hidden_dim

        # Input projection: (visual + pose) → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sinusoidal positional encoding
        self.register_buffer('pos_encoding',
                             self._create_pos_encoding(hidden_dim, max_clips))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _create_pos_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        \"\"\"Upper triangular causal mask (True = masked).\"\"\"
        return torch.triu(torch.ones(seq_len, seq_len, device=device),
                          diagonal=1).bool()

    def forward(self, clip_intermediate: torch.Tensor, clip_pose: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                use_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"
        Args:
            clip_intermediate: (B, N, 768) — cached blocks 0-9 output
            clip_pose:         (B, N, pose_dim) — DLC pose features
            padding_mask:      (B, N) — True where padded
            use_causal:        whether to apply causal attention mask

        Returns:
            logits: (B, 1)
            attention_weights: (B, N) — temporal importance proxy
        \"\"\"
        B, N, D = clip_intermediate.shape

        # Domain adapter: blocks 10-11 + projection per clip
        flat = clip_intermediate.reshape(B * N, D)  # (B*N, 768)
        clip_visual = self.adapter(flat)  # (B*N, projection_dim)
        clip_visual = clip_visual.reshape(B, N, -1)  # (B, N, 256)

        # Concatenate visual + pose
        x = torch.cat([clip_visual, clip_pose], dim=-1)  # (B, N, 272)

        # Project to hidden dim
        x = self.input_proj(x)  # (B, N, hidden)

        # Add positional encoding
        x = x + self.pos_encoding[:, :N, :]

        # Causal mask
        causal_mask = self._get_causal_mask(N, x.device) if use_causal else None

        # Transformer
        x = self.transformer(x, mask=causal_mask,
                            src_key_padding_mask=padding_mask)  # (B, N, hidden)

        # Temporal pooling (mean over non-padded positions)
        if padding_mask is not None:
            valid_mask = ~padding_mask  # (B, N)
            x_masked = x * valid_mask.unsqueeze(-1).float()
            pooled = x_masked.sum(dim=1) / valid_mask.sum(
                dim=1, keepdim=True).float().clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Attention weights proxy for interpretability
        with torch.no_grad():
            attn_weights = torch.norm(x, dim=-1)  # (B, N)
            if padding_mask is not None:
                attn_weights = attn_weights.masked_fill(padding_mask, 0.0)
            attn_weights = F.softmax(attn_weights, dim=-1)

        logits = self.classifier(pooled)  # (B, 1)
        return logits, attn_weights


# ── Initialize model ──
model = CowLamenessModelV32(
    adapter=domain_adapter,
    pose_dim=CFG["POSE_FEAT_DIM"],
    hidden_dim=CFG["HIDDEN_DIM"],
    num_heads=CFG["NUM_HEADS"],
    num_layers=CFG["NUM_LAYERS"],
    dropout=CFG["DROPOUT"],
    max_clips=CFG["NUM_CLIPS"] * 2,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
adapter_params = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
temporal_params = trainable_params - adapter_params
print(f"\\n✅ CowLamenessModelV32 (Partial FT)")
print(f"   Total: {total_params:,} | Trainable: {trainable_params:,}")
print(f"   Adapter (blocks 10-11 + proj): {adapter_params:,} @ LR={CFG['LR_VIDEOMAE']}")
print(f"   Temporal + classifier: {temporal_params:,} @ LR={CFG['LR_HEAD']}")

# Quick shape test
with torch.no_grad():
    dummy_int = torch.randn(2, 8, CFG["VIDEOMAE_DIM"]).to(DEVICE)
    dummy_pose = torch.randn(2, 8, CFG["POSE_FEAT_DIM"]).to(DEVICE)
    out, attn = model(dummy_int, dummy_pose)
    assert out.shape == (2, 1), f"Expected (2,1) got {out.shape}"
    assert attn.shape == (2, 8), f"Expected (2,8) got {attn.shape}"
    print(f"✅ Shape test passed: output={out.shape}, attention={attn.shape}")

# Free shape-test model (fold models are created fresh per fold)
del model
torch.cuda.empty_cache()
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 7: Dataset & DataLoader
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 7: Dataset & DataLoader

Each sample uses **hybrid** features:
- `clip_intermediate ∈ ℝ^(N×768)` — cached blocks 0-9 output (frozen)
- `pose_feat ∈ ℝ^16` — DLC pose features (replicated per clip)
- During training, blocks 10-11 (trainable) adapt the 768-dim features
"""))

cells.append(code("""# ============================================================
# SECTION 7: Dataset & DataLoader (intermediate features)
# ============================================================

class CowLamenessDatasetV32(Dataset):
    \"\"\"
    Dataset using pre-computed intermediate features (blocks 0-9 output).

    Blocks 10-11 (domain adapter) run online during training.
    \"\"\"

    def __init__(self, intermediate_features_list: list, pose_features: np.ndarray,
                 labels: np.ndarray, cfg: dict):
        self.intermediate_features = intermediate_features_list  # list of (n_clips_i, 768)
        self.pose_features = pose_features  # (N_samples, 16)
        self.labels = labels
        self.cfg = cfg

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        vis = self.intermediate_features[idx]   # (n_clips, 768) or (0, 768)
        pose = np.nan_to_num(self.pose_features[idx], nan=0.0)  # NaN → 0 for model input
        label = self.labels[idx]

        n_clips = len(vis)
        target_n = self.cfg["NUM_CLIPS"]
        mask = np.zeros(target_n, dtype=bool)

        # Replicate pose features for each clip
        if n_clips > 0:
            pose_rep = np.tile(pose, (n_clips, 1))
        else:
            vis = np.zeros((0, self.cfg["VIDEOMAE_DIM"]), dtype=np.float32)
            pose_rep = np.zeros((0, self.cfg["POSE_FEAT_DIM"]), dtype=np.float32)

        # Pad/truncate to target_n
        if n_clips >= target_n:
            indices = np.linspace(0, n_clips - 1, target_n, dtype=int)
            vis = vis[indices]
            pose_rep = pose_rep[indices]
        elif n_clips > 0:
            pad_v = np.zeros((target_n - n_clips, self.cfg["VIDEOMAE_DIM"]),
                             dtype=np.float32)
            pad_p = np.zeros((target_n - n_clips, self.cfg["POSE_FEAT_DIM"]),
                             dtype=np.float32)
            vis = np.concatenate([vis, pad_v], axis=0)
            pose_rep = np.concatenate([pose_rep, pad_p], axis=0)
            mask[n_clips:] = True
        else:
            vis = np.zeros((target_n, self.cfg["VIDEOMAE_DIM"]), dtype=np.float32)
            pose_rep = np.zeros((target_n, self.cfg["POSE_FEAT_DIM"]),
                                dtype=np.float32)
            mask[:] = True

        return vis, pose_rep, label, mask


def collate_fn(batch):
    \"\"\"Custom collate: stack arrays and convert to tensors.\"\"\"
    visuals, poses, labels, masks = zip(*batch)
    return (
        torch.tensor(np.array(visuals), dtype=torch.float32),
        torch.tensor(np.array(poses), dtype=torch.float32),
        torch.tensor(np.array(labels), dtype=torch.long),
        torch.tensor(np.array(masks), dtype=torch.bool),
    )

print("✅ Dataset & DataLoader classes defined")
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 8: Training Loop
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 8: Training Loop

- **5-fold subject-level cross-validation** (StratifiedGroupKFold)
- **BCE loss** with class weights for imbalance
- **AdamW** with **2 LR groups**: domain adapter (1e-5), temporal model (1e-4)
- **ReduceLROnPlateau** scheduler
- **Early stopping** on validation loss (patience=7)
- **Gradient clipping** (max_norm=1.0)

> Blocks 10-11 of VideoMAE are fine-tuned at lower LR for domain adaptation.
> Temporal Transformer + classifier are trained at standard LR.
"""))

cells.append(code("""# ============================================================
# SECTION 8: Training Functions
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, cfg):
    \"\"\"Train for one epoch. Returns mean loss.\"\"\"
    model.train()
    total_loss = 0
    n_batches = 0

    for visuals, poses, labels, masks in dataloader:
        visuals = visuals.to(device)
        poses = poses.to(device)
        labels = labels.float().to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits, _ = model(visuals, poses, padding_mask=masks)
        loss = criterion(logits.squeeze(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_causal=True):
    \"\"\"Evaluate model. Returns metrics dict, probs, labels, attention.\"\"\"
    model.eval()

    total_loss = 0
    all_probs, all_labels, all_attns = [], [], []
    n_batches = 0

    for visuals, poses, labels, masks in dataloader:
        visuals = visuals.to(device)
        poses = poses.to(device)
        labels_f = labels.float().to(device)
        masks = masks.to(device)

        logits, attn = model(visuals, poses, padding_mask=masks, use_causal=use_causal)
        loss = criterion(logits.squeeze(-1), labels_f)

        probs = torch.sigmoid(logits.squeeze(-1))
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_attns.extend(attn.cpu().numpy())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(labels_arr, preds_arr),
        "precision": precision_score(labels_arr, preds_arr, zero_division=0),
        "recall": recall_score(labels_arr, preds_arr, zero_division=0),
        "f1": f1_score(labels_arr, preds_arr, zero_division=0),
        "auc": roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.0,
    }

    return metrics, probs_arr, labels_arr, np.array(all_attns)

print("✅ Training functions defined")
"""))

cells.append(code("""# ============================================================
# SECTION 8: 5-Fold Cross-Validation Training
# ============================================================

def run_cross_validation(data_df, all_intermediate_features, pose_features, cfg, device):
    \"\"\"Run 5-fold subject-level stratified CV with Partial FT.\"\"\"
    video_paths = data_df["video_path"].values
    data_labels = data_df["label"].values
    animal_ids = data_df["animal_id"].values

    # Class weights for BCE
    n_healthy = (data_labels == 0).sum()
    n_lame = (data_labels == 1).sum()
    pos_weight = torch.tensor([n_healthy / n_lame]).to(device)
    print(f"📊 Class balance — Healthy: {n_healthy}, Lame: {n_lame}, "
          f"pos_weight: {pos_weight.item():.3f}")

    # CV splitter
    cv = StratifiedGroupKFold(n_splits=cfg["CV_FOLDS"], shuffle=True,
                               random_state=cfg["SEED"])

    fold_results = []
    all_fold_probs = []
    all_fold_labels = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(
            cv.split(video_paths, data_labels, animal_ids)):
        print(f"\\n{'='*60}")
        print(f"FOLD {fold+1}/{cfg['CV_FOLDS']}")
        print(f"{'='*60}")

        # Verify no animal leakage
        train_animals = set(animal_ids[train_idx])
        val_animals = set(animal_ids[val_idx])
        assert len(train_animals & val_animals) == 0, "❌ Animal leakage detected!"
        print(f"✅ No leakage | Train: {len(train_idx)} ({len(train_animals)} animals) | "
              f"Val: {len(val_idx)} ({len(val_animals)} animals)")

        # Create datasets with intermediate features
        train_vis = [all_intermediate_features[i] for i in train_idx]
        val_vis = [all_intermediate_features[i] for i in val_idx]

        train_ds = CowLamenessDatasetV32(
            train_vis, pose_features[train_idx], data_labels[train_idx], cfg
        )
        val_ds = CowLamenessDatasetV32(
            val_vis, pose_features[val_idx], data_labels[val_idx], cfg
        )

        train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                                   shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"],
                                 shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Fresh model for each fold (domain adapter re-initialized from pretrained)
        # Note: domain_adapter is shared reference — we need fresh copies per fold
        # Fresh adapter per fold via deepcopy
        import copy
        fold_adapter = copy.deepcopy(domain_adapter)

        fold_model = CowLamenessModelV32(
            adapter=fold_adapter,
            pose_dim=cfg["POSE_FEAT_DIM"],
            hidden_dim=cfg["HIDDEN_DIM"],
            num_heads=cfg["NUM_HEADS"],
            num_layers=cfg["NUM_LAYERS"],
            dropout=cfg["DROPOUT"],
        ).to(device)

        # Optimizer — 2 LR groups
        adapter_params = list(fold_model.adapter.parameters())
        temporal_params = [p for n, p in fold_model.named_parameters()
                          if not n.startswith("adapter.")]

        optimizer = torch.optim.AdamW([
            {"params": adapter_params, "lr": cfg["LR_VIDEOMAE"]},
            {"params": temporal_params, "lr": cfg["LR_HEAD"]},
        ], weight_decay=cfg["WEIGHT_DECAY"])

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": [],
                   "val_f1": [], "val_auc": []}

        for epoch in range(cfg["EPOCHS"]):
            train_loss = train_one_epoch(fold_model, train_loader,
                                         optimizer, criterion, device, cfg)
            val_metrics, val_probs, val_labels, _ = evaluate(
                fold_model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_auc"].append(val_metrics["auc"])

            scheduler.step(val_metrics["loss"])

            # Early stopping
            if val_metrics["loss"] < best_val_loss - 0.001:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_state = {
                    "model": {k: v.cpu().clone()
                              for k, v in fold_model.state_dict().items()},
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or patience_counter == 0:
                print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_metrics['loss']:.4f} | "
                      f"Acc: {val_metrics['accuracy']:.3f} | "
                      f"F1: {val_metrics['f1']:.3f} | "
                      f"AUC: {val_metrics['auc']:.3f}"
                      + (" ★" if patience_counter == 0 else ""))

            if patience_counter >= cfg["PATIENCE"]:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

        # Load best model and final evaluation
        fold_model.load_state_dict(best_state["model"])
        final_metrics, final_probs, final_labels, final_attns = evaluate(
            fold_model, val_loader, criterion, device)

        fold_results.append({
            "fold": fold + 1,
            "best_epoch": best_state["epoch"] + 1,
            "history": history,
            "fold_probs": final_probs,
            "fold_labels": final_labels,
            "fold_attns": final_attns,
            **final_metrics,
        })
        all_fold_probs.extend(final_probs)
        all_fold_labels.extend(final_labels)
        best_models.append(best_state)

        print(f"\\n📊 Fold {fold+1} Best (epoch {best_state['epoch']+1}):")
        print(f"   Acc: {final_metrics['accuracy']:.4f} | "
              f"F1: {final_metrics['f1']:.4f} | AUC: {final_metrics['auc']:.4f}")

        # Free GPU memory after each fold
        del fold_model, fold_adapter
        torch.cuda.empty_cache()

    return fold_results, np.array(all_fold_probs), np.array(all_fold_labels), best_models

# ═══════════════ RUN TRAINING ═══════════════
print("\\n🚀 Starting 5-Fold Cross-Validation Training (Partial FT)...")
fold_results, all_probs, all_labels, best_models = run_cross_validation(
    data_df, all_intermediate_features, pose_features, CFG, DEVICE
)
print("\\n✅ Training complete!")
"""))

# Save Part 2
notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": cells
}
out = r"c:\Users\HP\Desktop\Clone Repos\CowLameness\Colab_Notebook\_v32_part2.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Part 2 saved: {out} ({len(cells)} cells)")
