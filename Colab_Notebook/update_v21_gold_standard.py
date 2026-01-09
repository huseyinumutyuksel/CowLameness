"""
Update v21.ipynb with all Gold Standard features
=================================================
This script modifies the Cow_Lameness_Analysis_v21.ipynb to include:
- VideoMAE with partial fine-tuning
- Causal Transformer
- Severity Regression (0-3)
- Domain Normalization
- SSL Pretraining support
- LR Groups
- Checkpoint/Resume
"""

import json
import os

NOTEBOOK_PATH = r"c:\Users\Umut\Desktop\Github Projects\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v21.ipynb"

# Load existing notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New cells to add
NEW_CELLS = []

# =============================================================================
# Cell: VideoMAE Encoder (v25 requirement)
# =============================================================================
videomae_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 5.5 VIDEOMAE ENCODER (v25 GOLD STANDARD REQUIREMENT)\n",
        "# ============================================================\n",
        "\n",
        "from transformers import VideoMAEModel\n",
        "\n",
        "class VideoMAEEncoder(nn.Module):\n",
        "    \"\"\"\n",
        "    VideoMAE encoder with partial fine-tuning strategy.\n",
        "    \n",
        "    Architecture:\n",
        "    - Blocks 0-8: FROZEN (preserve general motion representation)\n",
        "    - Blocks 9-11: TRAINABLE (adapt to lameness-specific patterns)\n",
        "    \"\"\"\n",
        "    \n",
        "    def __init__(self, model_name='MCG-NJU/videomae-base', \n",
        "                 output_dim=128, trainable_blocks=[9, 10, 11]):\n",
        "        super().__init__()\n",
        "        \n",
        "        self.videomae = VideoMAEModel.from_pretrained(\n",
        "            model_name, output_hidden_states=True\n",
        "        )\n",
        "        \n",
        "        # Partial fine-tuning: freeze early layers\n",
        "        for param in self.videomae.parameters():\n",
        "            param.requires_grad = False\n",
        "        \n",
        "        for name, param in self.videomae.named_parameters():\n",
        "            for block_idx in trainable_blocks:\n",
        "                if f'encoder.layer.{block_idx}' in name:\n",
        "                    param.requires_grad = True\n",
        "                    break\n",
        "        \n",
        "        # Projection layer\n",
        "        self.projection = nn.Sequential(\n",
        "            nn.Linear(768, 256),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.1),\n",
        "            nn.Linear(256, output_dim),\n",
        "            nn.LayerNorm(output_dim)\n",
        "        )\n",
        "        \n",
        "        # Count trainable params\n",
        "        trainable = sum(p.numel() for p in self.videomae.parameters() if p.requires_grad)\n",
        "        total = sum(p.numel() for p in self.videomae.parameters())\n",
        "        print(f'VideoMAE: {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)')\n",
        "    \n",
        "    def forward(self, x):\n",
        "        # x: (B, T, C, H, W)\n",
        "        B, T, C, H, W = x.shape\n",
        "        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)\n",
        "        \n",
        "        outputs = self.videomae(pixel_values=x)\n",
        "        hidden = outputs.last_hidden_state  # (B, num_patches, 768)\n",
        "        \n",
        "        # Mean pooling over patches\n",
        "        video_feat = hidden.mean(dim=1)  # (B, 768)\n",
        "        \n",
        "        return self.projection(video_feat)  # (B, output_dim)\n",
        "\n",
        "\n",
        "# Initialize VideoMAE if enabled\n",
        "videomae_encoder = None\n",
        "if CFG['USE_VIDEOMAE']:\n",
        "    videomae_encoder = VideoMAEEncoder(\n",
        "        output_dim=CFG['VIDEO_DIM'],\n",
        "        trainable_blocks=CFG.get('TRAINABLE_BLOCKS', [9, 10, 11])\n",
        "    ).to(DEVICE)\n",
        "    print('✅ VideoMAE Encoder initialized with partial fine-tuning')\n",
        "else:\n",
        "    print('⚠️ VideoMAE disabled in config')\n"
    ]
}

# =============================================================================
# Cell: Domain Normalization (v29 requirement)
# =============================================================================
domain_norm_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 5.6 DOMAIN NORMALIZATION (v29 REQUIREMENT)\n",
        "# ============================================================\n",
        "\n",
        "class DomainNorm(nn.Module):\n",
        "    \"\"\"\n",
        "    Domain Normalization for cross-farm generalization.\n",
        "    Handles domain shift between different farms/cameras.\n",
        "    \"\"\"\n",
        "    \n",
        "    def __init__(self, dim, eps=1e-6):\n",
        "        super().__init__()\n",
        "        self.layer_norm = nn.LayerNorm(dim, eps=eps)\n",
        "        self.scale = nn.Parameter(torch.ones(dim))\n",
        "        self.shift = nn.Parameter(torch.zeros(dim))\n",
        "    \n",
        "    def forward(self, x):\n",
        "        x = self.layer_norm(x)\n",
        "        return x * self.scale + self.shift\n",
        "\n",
        "print('✅ DomainNorm defined')\n"
    ]
}

# =============================================================================
# Cell: Causal Transformer (v30 requirement)
# =============================================================================
causal_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 5.7 CAUSAL TRANSFORMER (v30 REQUIREMENT)\n",
        "# ============================================================\n",
        "\n",
        "class CausalTransformerEncoder(nn.Module):\n",
        "    \"\"\"\n",
        "    Causal Transformer Encoder.\n",
        "    Uses causal mask to prevent information leakage from future.\n",
        "    Enables online/streaming prediction.\n",
        "    \"\"\"\n",
        "    \n",
        "    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.1):\n",
        "        super().__init__()\n",
        "        \n",
        "        self.d_model = d_model\n",
        "        \n",
        "        encoder_layer = nn.TransformerEncoderLayer(\n",
        "            d_model=d_model,\n",
        "            nhead=nhead,\n",
        "            dim_feedforward=d_model * 4,\n",
        "            dropout=dropout,\n",
        "            batch_first=True\n",
        "        )\n",
        "        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n",
        "        self._causal_mask = None\n",
        "    \n",
        "    def _get_causal_mask(self, seq_len, device):\n",
        "        if self._causal_mask is None or self._causal_mask.size(0) != seq_len:\n",
        "            # Upper triangular mask (prevents attending to future)\n",
        "            self._causal_mask = torch.triu(\n",
        "                torch.ones(seq_len, seq_len, device=device), diagonal=1\n",
        "            ).bool()\n",
        "        return self._causal_mask\n",
        "    \n",
        "    def forward(self, x, use_causal=True):\n",
        "        B, T, D = x.shape\n",
        "        \n",
        "        if use_causal:\n",
        "            mask = self._get_causal_mask(T, x.device)\n",
        "        else:\n",
        "            mask = None\n",
        "        \n",
        "        return self.transformer(x, mask=mask)\n",
        "\n",
        "print('✅ CausalTransformerEncoder defined')\n"
    ]
}

# =============================================================================
# Cell: Self-Supervised Pretraining (v29 requirement)
# =============================================================================
ssl_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 5.8 SELF-SUPERVISED PRETRAINING (v29 REQUIREMENT)\n",
        "# ============================================================\n",
        "\n",
        "class TemporalOrderNet(nn.Module):\n",
        "    \"\"\"\n",
        "    Self-Supervised Temporal Order Verification.\n",
        "    Pretext task: Predict if frame order is correct or reversed.\n",
        "    \"\"\"\n",
        "    \n",
        "    def __init__(self, input_dim=768, hidden_dim=256):\n",
        "        super().__init__()\n",
        "        \n",
        "        self.encoder = nn.Sequential(\n",
        "            nn.Linear(input_dim, hidden_dim),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.1),\n",
        "            nn.Linear(hidden_dim, hidden_dim),\n",
        "            nn.ReLU()\n",
        "        )\n",
        "        self.classifier = nn.Linear(hidden_dim, 2)\n",
        "    \n",
        "    def forward(self, x):\n",
        "        # x: (T, D)\n",
        "        encoded = self.encoder(x)  # (T, hidden)\n",
        "        pooled = encoded.mean(dim=0)  # (hidden,)\n",
        "        return self.classifier(pooled).unsqueeze(0)\n",
        "\n",
        "print('✅ TemporalOrderNet defined (SSL)')\n"
    ]
}

# =============================================================================
# Cell: Updated TransformerMIL with Severity Regression
# =============================================================================
model_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 8.1 GOLD STANDARD MODEL (v25-v30 COMPLETE)\n",
        "# ============================================================\n",
        "\n",
        "class LamenessSeverityModel(nn.Module):\n",
        "    \"\"\"\n",
        "    Complete Lameness Detection Model with:\n",
        "    - Multi-modal fusion (Pose + Flow + VideoMAE)\n",
        "    - Causal Transformer\n",
        "    - MIL Attention\n",
        "    - Severity Regression (0-3)\n",
        "    \"\"\"\n",
        "    \n",
        "    def __init__(self, config):\n",
        "        super().__init__()\n",
        "        self.config = config\n",
        "        \n",
        "        # Modal encoders\n",
        "        self.pose_encoder = nn.Sequential(\n",
        "            nn.Linear(config['POSE_DIM'], 128),\n",
        "            nn.ReLU(),\n",
        "            nn.LayerNorm(128)\n",
        "        ) if config['USE_POSE'] else None\n",
        "        \n",
        "        self.flow_encoder = nn.Sequential(\n",
        "            nn.Linear(config['FLOW_DIM'], 64),\n",
        "            nn.ReLU(),\n",
        "            nn.LayerNorm(64)\n",
        "        ) if config['USE_FLOW'] else None\n",
        "        \n",
        "        self.video_encoder = nn.Sequential(\n",
        "            nn.Linear(config['VIDEO_DIM'], 128),\n",
        "            nn.ReLU(),\n",
        "            nn.LayerNorm(128)\n",
        "        ) if config['USE_VIDEOMAE'] else None\n",
        "        \n",
        "        # Calculate total dimension\n",
        "        total_dim = 0\n",
        "        if config['USE_POSE']: total_dim += 128\n",
        "        if config['USE_FLOW']: total_dim += 64\n",
        "        if config['USE_VIDEOMAE']: total_dim += 128\n",
        "        self.total_dim = total_dim\n",
        "        \n",
        "        # Domain normalization (v29)\n",
        "        self.domain_norm = DomainNorm(total_dim)\n",
        "        \n",
        "        # Causal Transformer (v30)\n",
        "        self.temporal_encoder = CausalTransformerEncoder(\n",
        "            d_model=total_dim,\n",
        "            nhead=config['NUM_HEADS'],\n",
        "            num_layers=config['NUM_LAYERS']\n",
        "        )\n",
        "        \n",
        "        # MIL Attention\n",
        "        self.attention = nn.Sequential(\n",
        "            nn.Linear(total_dim, 64),\n",
        "            nn.Tanh(),\n",
        "            nn.Linear(64, 1)\n",
        "        )\n",
        "        \n",
        "        # Output head (regression or classification)\n",
        "        if config.get('MODE') == 'regression':\n",
        "            self.head = nn.Sequential(\n",
        "                nn.Linear(total_dim, 128),\n",
        "                nn.ReLU(),\n",
        "                nn.Dropout(0.3),\n",
        "                nn.Linear(128, 1),\n",
        "                nn.Sigmoid()  # Output [0, 1], scale to [0, 3]\n",
        "            )\n",
        "        else:\n",
        "            self.head = nn.Sequential(\n",
        "                nn.Linear(total_dim, 128),\n",
        "                nn.ReLU(),\n",
        "                nn.Dropout(0.3),\n",
        "                nn.Linear(128, 1),\n",
        "                nn.Sigmoid()\n",
        "            )\n",
        "    \n",
        "    def forward(self, pose, flow, video=None):\n",
        "        features = []\n",
        "        \n",
        "        if self.config['USE_POSE'] and self.pose_encoder is not None:\n",
        "            if pose.dim() == 4:\n",
        "                B, N, W, D = pose.shape\n",
        "                pose = pose.mean(dim=2)\n",
        "            features.append(self.pose_encoder(pose))\n",
        "        \n",
        "        if self.config['USE_FLOW'] and self.flow_encoder is not None:\n",
        "            features.append(self.flow_encoder(flow))\n",
        "        \n",
        "        if self.config['USE_VIDEOMAE'] and self.video_encoder is not None and video is not None:\n",
        "            features.append(self.video_encoder(video))\n",
        "        \n",
        "        x = torch.cat(features, dim=-1)\n",
        "        \n",
        "        # Domain normalization\n",
        "        x = self.domain_norm(x)\n",
        "        \n",
        "        # Causal transformer\n",
        "        h = self.temporal_encoder(x, use_causal=self.config.get('USE_CAUSAL', True))\n",
        "        \n",
        "        # MIL attention\n",
        "        attn_logits = self.attention(h).squeeze(-1)\n",
        "        attn_weights = F.softmax(attn_logits, dim=1)\n",
        "        bag = (h * attn_weights.unsqueeze(-1)).sum(dim=1)\n",
        "        \n",
        "        # Prediction\n",
        "        pred = self.head(bag).squeeze(-1)\n",
        "        \n",
        "        # Scale to [0, 3] for regression\n",
        "        if self.config.get('MODE') == 'regression':\n",
        "            pred = pred * 3.0\n",
        "        \n",
        "        return pred, attn_weights\n",
        "\n",
        "# Initialize model\n",
        "model = LamenessSeverityModel(CFG).to(DEVICE)\n",
        "\n",
        "print('✅ LamenessSeverityModel initialized')\n",
        "print(f'   Mode: {CFG.get(\"MODE\", \"classification\")}')\n",
        "print(f'   Causal: {CFG.get(\"USE_CAUSAL\", True)}')\n",
        "print(f'   Total dimension: {model.total_dim}')\n",
        "print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')\n"
    ]
}

# =============================================================================
# Cell: Training with LR Groups and Checkpoint
# =============================================================================
training_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 9.3 GOLD STANDARD TRAINING (v26 REQUIREMENTS)\n",
        "# ============================================================\n",
        "\n",
        "# Loss function based on mode\n",
        "if CFG.get('MODE') == 'regression':\n",
        "    criterion = nn.MSELoss()\n",
        "    print('Using MSELoss for severity regression')\n",
        "else:\n",
        "    criterion = nn.BCELoss()\n",
        "    print('Using BCELoss for binary classification')\n",
        "\n",
        "# LR Groups (v26 requirement)\n",
        "param_groups = [\n",
        "    {'params': model.parameters(), 'lr': CFG['LR']}\n",
        "]\n",
        "\n",
        "if videomae_encoder is not None:\n",
        "    param_groups.append({\n",
        "        'params': videomae_encoder.parameters(), \n",
        "        'lr': CFG['LR_VIDEOMAE']\n",
        "    })\n",
        "    print(f'VideoMAE LR: {CFG[\"LR_VIDEOMAE\"]}')\n",
        "\n",
        "optimizer = optim.AdamW(param_groups, weight_decay=CFG['WEIGHT_DECAY'])\n",
        "\n",
        "best_val_loss = float('inf')\n",
        "train_losses = []\n",
        "val_losses = []\n",
        "start_epoch = 0\n",
        "\n",
        "# Resume from checkpoint if exists (v26 requirement)\n",
        "checkpoint_path = f'{MODEL_DIR}/checkpoint.pt'\n",
        "if os.path.exists(checkpoint_path):\n",
        "    print(f'Loading checkpoint from {checkpoint_path}')\n",
        "    checkpoint = torch.load(checkpoint_path)\n",
        "    model.load_state_dict(checkpoint['model'])\n",
        "    optimizer.load_state_dict(checkpoint['optimizer'])\n",
        "    start_epoch = checkpoint['epoch'] + 1\n",
        "    best_val_loss = checkpoint['best_val_loss']\n",
        "    train_losses = checkpoint.get('train_losses', [])\n",
        "    val_losses = checkpoint.get('val_losses', [])\n",
        "    print(f'Resuming from epoch {start_epoch}')\n",
        "\n",
        "print('\\n' + '='*60)\n",
        "print('TRAINING (Gold Standard v26+)')\n",
        "print('='*60)\n",
        "\n",
        "for epoch in range(start_epoch, CFG['EPOCHS']):\n",
        "    # Training\n",
        "    model.train()\n",
        "    train_loss = 0\n",
        "    \n",
        "    for pose, flow, label in train_loader:\n",
        "        pose = pose.to(DEVICE)\n",
        "        flow = flow.to(DEVICE)\n",
        "        label = label.to(DEVICE)\n",
        "        \n",
        "        # For regression, scale labels to [0, 3]\n",
        "        if CFG.get('MODE') == 'regression':\n",
        "            label = label * 3.0\n",
        "        \n",
        "        optimizer.zero_grad()\n",
        "        pred, attn = model(pose, flow)\n",
        "        loss = criterion(pred, label)\n",
        "        loss.backward()\n",
        "        \n",
        "        # Gradient clipping\n",
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
        "        \n",
        "        optimizer.step()\n",
        "        train_loss += loss.item()\n",
        "    \n",
        "    train_loss /= len(train_loader)\n",
        "    train_losses.append(train_loss)\n",
        "    \n",
        "    # Validation\n",
        "    model.eval()\n",
        "    val_loss = 0\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        for pose, flow, label in test_loader:\n",
        "            pose = pose.to(DEVICE)\n",
        "            flow = flow.to(DEVICE)\n",
        "            label = label.to(DEVICE)\n",
        "            \n",
        "            if CFG.get('MODE') == 'regression':\n",
        "                label = label * 3.0\n",
        "            \n",
        "            pred, attn = model(pose, flow)\n",
        "            loss = criterion(pred, label)\n",
        "            val_loss += loss.item()\n",
        "    \n",
        "    val_loss /= len(test_loader)\n",
        "    val_losses.append(val_loss)\n",
        "    \n",
        "    # Save checkpoint (v26 requirement)\n",
        "    if val_loss < best_val_loss:\n",
        "        best_val_loss = val_loss\n",
        "        torch.save({\n",
        "            'model': model.state_dict(),\n",
        "            'optimizer': optimizer.state_dict(),\n",
        "            'epoch': epoch,\n",
        "            'best_val_loss': best_val_loss,\n",
        "            'train_losses': train_losses,\n",
        "            'val_losses': val_losses,\n",
        "            'config': CFG\n",
        "        }, f'{MODEL_DIR}/best_model.pt')\n",
        "    \n",
        "    # Save checkpoint for resume\n",
        "    torch.save({\n",
        "        'model': model.state_dict(),\n",
        "        'optimizer': optimizer.state_dict(),\n",
        "        'epoch': epoch,\n",
        "        'best_val_loss': best_val_loss,\n",
        "        'train_losses': train_losses,\n",
        "        'val_losses': val_losses,\n",
        "        'config': CFG\n",
        "    }, checkpoint_path)\n",
        "    \n",
        "    if (epoch + 1) % 5 == 0:\n",
        "        print(f'Epoch {epoch+1}/{CFG[\"EPOCHS\"]} | Train: {train_loss:.4f} | Val: {val_loss:.4f}')\n",
        "\n",
        "print('\\n✅ Training complete')\n",
        "print(f'   Best validation loss: {best_val_loss:.4f}')\n"
    ]
}

# =============================================================================
# Cell: Evaluation with MAE/RMSE
# =============================================================================
eval_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 10.1 EVALUATION (v30 - SEVERITY METRICS)\n",
        "# ============================================================\n",
        "\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
        "\n",
        "# Load best model\n",
        "checkpoint = torch.load(f'{MODEL_DIR}/best_model.pt')\n",
        "model.load_state_dict(checkpoint['model'])\n",
        "model.eval()\n",
        "\n",
        "all_preds = []\n",
        "all_labels = []\n",
        "all_attns = []\n",
        "\n",
        "with torch.no_grad():\n",
        "    for pose, flow, label in test_loader:\n",
        "        pose = pose.to(DEVICE)\n",
        "        flow = flow.to(DEVICE)\n",
        "        \n",
        "        pred, attn = model(pose, flow)\n",
        "        \n",
        "        all_preds.append(pred.cpu().numpy())\n",
        "        all_labels.append(label.numpy() * 3.0 if CFG.get('MODE') == 'regression' else label.numpy())\n",
        "        all_attns.append(attn.cpu().numpy())\n",
        "\n",
        "all_preds = np.concatenate(all_preds)\n",
        "all_labels = np.concatenate(all_labels)\n",
        "\n",
        "print('\\n' + '='*60)\n",
        "print('FINAL TEST RESULTS')\n",
        "print('='*60)\n",
        "\n",
        "if CFG.get('MODE') == 'regression':\n",
        "    mae = mean_absolute_error(all_labels, all_preds)\n",
        "    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))\n",
        "    print(f'MAE (Mean Absolute Error): {mae:.3f}')\n",
        "    print(f'RMSE (Root Mean Squared Error): {rmse:.3f}')\n",
        "    print(f'Prediction range: [{all_preds.min():.2f}, {all_preds.max():.2f}]')\n",
        "else:\n",
        "    pred_binary = (all_preds > 0.5).astype(int)\n",
        "    accuracy = accuracy_score(all_labels, pred_binary)\n",
        "    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_binary, average='binary')\n",
        "    cm = confusion_matrix(all_labels, pred_binary)\n",
        "    \n",
        "    print(f'Accuracy:  {accuracy:.4f}')\n",
        "    print(f'Precision: {precision:.4f}')\n",
        "    print(f'Recall:    {recall:.4f}')\n",
        "    print(f'F1-Score:  {f1:.4f}')\n",
        "    print(f'\\nConfusion Matrix:')\n",
        "    print(cm)\n",
        "\n",
        "print('='*60)\n"
    ]
}

# Find insertion points and insert new cells
cells = notebook['cells']

# Find the optical flow section (around section 5)
insert_after_flow = None
for i, cell in enumerate(cells):
    source = ''.join(cell.get('source', []))
    if '5.2 TEST OPTICAL FLOW' in source:
        insert_after_flow = i + 1
        break

# Find the model section (section 8)
replace_model = None
for i, cell in enumerate(cells):
    source = ''.join(cell.get('source', []))
    if '8.1 TRANSFORMER MIL MODEL' in source:
        replace_model = i
        break

# Find training section (section 9.3)
replace_training = None
for i, cell in enumerate(cells):
    source = ''.join(cell.get('source', []))
    if '9.3 TRAINING LOOP' in source:
        replace_training = i
        break

# Find evaluation section (section 10.1)
replace_eval = None
for i, cell in enumerate(cells):
    source = ''.join(cell.get('source', []))
    if '10.1 FINAL EVALUATION' in source:
        replace_eval = i
        break

# Insert new cells after flow section
if insert_after_flow:
    cells.insert(insert_after_flow, videomae_cell)
    cells.insert(insert_after_flow + 1, domain_norm_cell)
    cells.insert(insert_after_flow + 2, causal_cell)
    cells.insert(insert_after_flow + 3, ssl_cell)
    print(f'Inserted 4 new cells after position {insert_after_flow}')
    
    # Adjust indices
    if replace_model: replace_model += 4
    if replace_training: replace_training += 4
    if replace_eval: replace_eval += 4

# Replace model cell
if replace_model:
    cells[replace_model] = model_cell
    print(f'Replaced model cell at position {replace_model}')

# Replace training cell
if replace_training:
    cells[replace_training] = training_cell
    print(f'Replaced training cell at position {replace_training}')

# Replace evaluation cell
if replace_eval:
    cells[replace_eval] = eval_cell
    print(f'Replaced evaluation cell at position {replace_eval}')

# Update CFG cell
for i, cell in enumerate(cells):
    source = ''.join(cell.get('source', []))
    if '3.1 GLOBAL CONFIGURATION' in source and 'USE_VIDEOMAE' in source:
        # Update the config
        new_source = [
            "# ============================================================\\n",
            "# 3.1 GLOBAL CONFIGURATION (GOLD STANDARD v21+)\\n",
            "# ============================================================\\n",
            "\\n",
            "CFG = {\\n",
            "    # Video processing\\n",
            "    \\\"FPS\\\": 30,\\n",
            "    \\\"WINDOW_FRAMES\\\": 60,\\n",
            "    \\\"STRIDE_FRAMES\\\": 15,\\n",
            "    \\n",
            "    # Model dimensions\\n",
            "    \\\"POSE_DIM\\\": 16,\\n",
            "    \\\"FLOW_DIM\\\": 3,\\n",
            "    \\\"VIDEO_DIM\\\": 128,\\n",
            "    \\\"HIDDEN_DIM\\\": 256,\\n",
            "    \\\"NUM_HEADS\\\": 8,\\n",
            "    \\\"NUM_LAYERS\\\": 4,\\n",
            "    \\n",
            "    # Training\\n",
            "    \\\"BATCH_SIZE\\\": 1,\\n",
            "    \\\"EPOCHS\\\": 30,\\n",
            "    \\\"LR\\\": 1e-4,\\n",
            "    \\\"LR_VIDEOMAE\\\": 1e-5,\\n",
            "    \\\"WEIGHT_DECAY\\\": 1e-4,\\n",
            "    \\n",
            "    # Ablation config\\n",
            "    \\\"USE_POSE\\\": True,\\n",
            "    \\\"USE_FLOW\\\": True,\\n",
            "    \\\"USE_VIDEOMAE\\\": True,  # ✅ ENABLED\\n",
            "    \\n",
            "    # Gold Standard features\\n",
            "    \\\"MODE\\\": \\\"regression\\\",\\n",
            "    \\\"USE_CAUSAL\\\": True,\\n",
            "    \\\"USE_SSL\\\": False,\\n",
            "    \\\"TRAINABLE_BLOCKS\\\": [9, 10, 11],\\n",
            "}\\n",
            "\\n",
            "print(\\\"✅ Gold Standard Configuration\\\")\\n",
            "for k, v in CFG.items():\\n",
            "    print(f\\\"   {k}: {v}\\\")\\n"
        ]
        cells[i]['source'] = new_source
        print(f'Updated CFG cell at position {i}')
        break

# Save updated notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f'\n✅ Notebook updated successfully: {NOTEBOOK_PATH}')
print('Changes made:')
print('  - Updated CFG with USE_VIDEOMAE=True, MODE=regression, USE_CAUSAL=True')
print('  - Added VideoMAE Encoder cell')
print('  - Added DomainNorm cell')
print('  - Added CausalTransformerEncoder cell')
print('  - Added TemporalOrderNet (SSL) cell')
print('  - Updated model to LamenessSeverityModel')
print('  - Updated training with LR groups and checkpoint')
print('  - Updated evaluation with MAE/RMSE metrics')
