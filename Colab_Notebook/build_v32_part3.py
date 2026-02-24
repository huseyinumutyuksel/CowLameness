"""
Build Cow_Lameness_Analysis_v32.ipynb — Part 3 (Sections 9-12)
Evaluation, Ablation, Explainability, Results
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
# SECTION 9: Evaluation
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 9: Comprehensive Evaluation (Q1 Journal Standard)

- Confusion matrix (counts + normalized)
- ROC curve (**per-fold** + mean)
- Precision-Recall curve (**per-fold** + mean)
- Per-fold metrics table
- Learning curves
- Statistical significance test + 95% CI
"""))

cells.append(code("""# ============================================================
# SECTION 9: Confusion Matrix
# ============================================================

def plot_confusion_matrix(true_labels, pred_probs, save_path=None):
    \"\"\"Normalized confusion matrix heatmap.\"\"\"
    preds = (pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [(axes[0], cm, "Confusion Matrix (Counts)", "d"),
                                  (axes[1], cm_norm, "Confusion Matrix (Normalized)", ".2%")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                   xticklabels=["Healthy", "Lame"], yticklabels=["Healthy", "Lame"],
                   cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(all_labels, all_probs,
    save_path=os.path.join(CFG["RESULTS_DIR"], "confusion_matrix.png"))
"""))

cells.append(code("""# ============================================================
# ROC & PR Curves (per-fold + mean)
# ============================================================

def plot_roc_and_pr_curves(fold_results, agg_labels, agg_probs, save_path=None):
    \"\"\"ROC and Precision-Recall curves with per-fold detail.\"\"\"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- ROC ---
    ax = axes[0]
    for r in fold_results:
        if "fold_probs" in r and "fold_labels" in r:
            fl = r["fold_labels"]
            fp = r["fold_probs"]
            if len(np.unique(fl)) > 1:
                fpr_f, tpr_f, _ = roc_curve(fl, fp)
                auc_f = roc_auc_score(fl, fp)
                ax.plot(fpr_f, tpr_f, '--', alpha=0.35, linewidth=1,
                       label=f'Fold {r["fold"]} ({auc_f:.3f})')

    fpr, tpr, _ = roc_curve(agg_labels, agg_probs)
    auc_val = roc_auc_score(agg_labels, agg_probs)
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Mean (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.fill_between(fpr, tpr, alpha=0.08, color='blue')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- PR (per-fold + mean) ---
    ax = axes[1]
    for r in fold_results:
        if "fold_probs" in r and "fold_labels" in r:
            fl = r["fold_labels"]
            fp = r["fold_probs"]
            if len(np.unique(fl)) > 1:
                prec_f, rec_f, _ = precision_recall_curve(fl, fp)
                ap_f = average_precision_score(fl, fp)
                ax.plot(rec_f, prec_f, '--', alpha=0.35, linewidth=1,
                       label=f'Fold {r["fold"]} (AP={ap_f:.3f})')

    prec, rec, _ = precision_recall_curve(agg_labels, agg_probs)
    ap = average_precision_score(agg_labels, agg_probs)
    ax.plot(rec, prec, 'r-', linewidth=2.5, label=f'Mean (AP = {ap:.3f})')
    ax.fill_between(rec, prec, alpha=0.1, color='red')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

plot_roc_and_pr_curves(fold_results, all_labels, all_probs,
    save_path=os.path.join(CFG["RESULTS_DIR"], "roc_pr_curves.png"))
"""))

cells.append(code("""# ============================================================
# Per-fold Results Table
# ============================================================

def print_fold_results_table(fold_results):
    \"\"\"Print and display per-fold metrics as a table.\"\"\"
    rows = []
    for r in fold_results:
        rows.append({
            "Fold": r["fold"],
            "Best Epoch": r["best_epoch"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1": f"{r['f1']:.4f}",
            "AUC": f"{r['auc']:.4f}",
        })

    df = pd.DataFrame(rows)

    # Compute mean ± std
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    means = {col: np.mean([r[col] for r in fold_results]) for col in metric_cols}
    stds = {col: np.std([r[col] for r in fold_results]) for col in metric_cols}

    summary_row = {
        "Fold": "Mean±Std",
        "Best Epoch": "-",
    }
    for col in metric_cols:
        key = col.capitalize() if col != "auc" else "AUC"
        summary_row[key] = f"{means[col]:.4f}±{stds[col]:.4f}"

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    print("\\n" + "="*80)
    print("📊 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    save_path = os.path.join(CFG["RESULTS_DIR"], "fold_results.csv")
    df.to_csv(save_path, index=False)
    print(f"💾 Saved to {save_path}")

    return df, means, stds

results_df, means, stds = print_fold_results_table(fold_results)
"""))

cells.append(code("""# ============================================================
# Learning Curves (all folds)
# ============================================================

def plot_learning_curves(fold_results, save_path=None):
    \"\"\"Training/validation loss and metrics over epochs for each fold.\"\"\"
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds, 3, figsize=(18, 4 * n_folds))
    if n_folds == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(fold_results):
        h = r["history"]
        epochs = range(1, len(h["train_loss"]) + 1)

        # Loss
        axes[i, 0].plot(epochs, h["train_loss"], 'b-', label='Train')
        axes[i, 0].plot(epochs, h["val_loss"], 'r-', label='Val')
        axes[i, 0].set_title(f'Fold {r["fold"]} — Loss')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[i, 1].plot(epochs, h["val_acc"], 'g-', label='Val Acc')
        axes[i, 1].set_title(f'Fold {r["fold"]} — Accuracy')
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        # AUC
        axes[i, 2].plot(epochs, h["val_auc"], 'm-', label='Val AUC')
        axes[i, 2].set_title(f'Fold {r["fold"]} — AUC')
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

    plt.suptitle("Learning Curves (5-Fold CV)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_learning_curves(fold_results,
    save_path=os.path.join(CFG["RESULTS_DIR"], "learning_curves.png"))
"""))

cells.append(code("""# ============================================================
# Statistical Significance
# ============================================================

print("\\n📊 Statistical Significance Analysis")
print("="*50)

# Classification report
preds = (all_probs >= 0.5).astype(int)
print("\\nClassification Report:")
print(classification_report(all_labels, preds, target_names=["Healthy", "Lame"]))

# Paired t-test across folds (accuracy vs chance=0.5)
fold_accs = [r["accuracy"] for r in fold_results]
fold_aucs = [r["auc"] for r in fold_results]

t_acc, p_acc = stats.ttest_1samp(fold_accs, 0.5)
t_auc, p_auc = stats.ttest_1samp(fold_aucs, 0.5)
print(f"Accuracy vs chance (0.5): t={t_acc:.3f}, p={p_acc:.6f} "
      f"{'***' if p_acc<0.001 else '**' if p_acc<0.01 else '*' if p_acc<0.05 else 'ns'}")
print(f"AUC vs chance (0.5):      t={t_auc:.3f}, p={p_auc:.6f} "
      f"{'***' if p_auc<0.001 else '**' if p_auc<0.01 else '*' if p_auc<0.05 else 'ns'}")

# 95% CI
ci_acc = stats.t.interval(0.95, len(fold_accs)-1, loc=np.mean(fold_accs), scale=stats.sem(fold_accs))
ci_auc = stats.t.interval(0.95, len(fold_aucs)-1, loc=np.mean(fold_aucs), scale=stats.sem(fold_aucs))
print(f"\\n95% CI Accuracy: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
print(f"95% CI AUC:      [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 10: Ablation Study
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 10: Ablation Study

Compare 4 configurations using **pre-computed intermediate features**:

| Config | Visual (768→256) | Pose (16D) | Adapter FT | Description |
|--------|:---:|:---:|:---:|-------------|
| A | ✅ | ✅ | ✅ | Full model (Partial FT baseline) |
| B | ✅ | zeroed | ✅ | VideoMAE only |
| C | zeroed | ✅ | ✅ | Pose only |
| D | ✅ (frozen proj) | ✅ | ❌ | Frozen VideoMAE (no domain adapter) |

All configs use the same LR scheduler for fair comparison.
"""))

cells.append(code("""# ============================================================
# SECTION 10: Ablation Study
# ============================================================

def run_ablation_config(config_name, all_int_feats, pose_feats,
                        data_df, cfg, device,
                        zero_visual=False, zero_pose=False, use_causal=True,
                        freeze_adapter=False):
    \"\"\"
    Run a single ablation configuration using pre-computed intermediate features.
    Features are zeroed out to test component contribution — model
    architecture stays identical (same input dim) for fair comparison.
    \"\"\"
    print(f"\\n{'='*60}")
    print(f"ABLATION: {config_name}")
    print(f"  Visual:  {'ZEROED' if zero_visual else 'ON'}")
    print(f"  Pose:    {'ZEROED' if zero_pose else 'ON'}")
    print(f"  Adapter: {'FROZEN' if freeze_adapter else 'TRAINABLE'}")
    print(f"  Causal:  {'ON' if use_causal else 'OFF'}")
    print(f"{'='*60}")

    import copy
    data_labels = data_df["label"].values
    animal_ids = data_df["animal_id"].values
    video_paths = data_df["video_path"].values

    n_healthy = (data_labels == 0).sum()
    n_lame = (data_labels == 1).sum()
    pos_weight = torch.tensor([n_healthy / n_lame]).to(device)

    cv = StratifiedGroupKFold(n_splits=cfg["CV_FOLDS"], shuffle=True,
                               random_state=cfg["SEED"])

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(
            cv.split(video_paths, data_labels, animal_ids)):

        # Apply feature zeroing
        train_vis = [np.zeros_like(all_int_feats[i]) if zero_visual
                     else all_int_feats[i] for i in train_idx]
        val_vis = [np.zeros_like(all_int_feats[i]) if zero_visual
                   else all_int_feats[i] for i in val_idx]

        train_pose = (np.zeros_like(pose_feats[train_idx]) if zero_pose
                      else pose_feats[train_idx])
        val_pose = (np.zeros_like(pose_feats[val_idx]) if zero_pose
                    else pose_feats[val_idx])

        train_ds = CowLamenessDatasetV32(
            train_vis, train_pose, data_labels[train_idx], cfg)
        val_ds = CowLamenessDatasetV32(
            val_vis, val_pose, data_labels[val_idx], cfg)

        train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                                   shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"],
                                 shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Fresh adapter + model per fold
        # Fresh adapter per fold via deepcopy
        abl_adapter = copy.deepcopy(domain_adapter)

        if freeze_adapter:
            for p in abl_adapter.parameters():
                p.requires_grad = False

        abl_model = CowLamenessModelV32(
            adapter=abl_adapter,
            pose_dim=cfg["POSE_FEAT_DIM"],
            hidden_dim=cfg["HIDDEN_DIM"],
            num_heads=cfg["NUM_HEADS"],
            num_layers=cfg["NUM_LAYERS"],
            dropout=cfg["DROPOUT"],
        ).to(device)

        # Optimizer with scheduler
        trainable_params = [p for p in abl_model.parameters() if p.requires_grad]
        if not freeze_adapter:
            adapter_p = [p for p in abl_model.adapter.parameters() if p.requires_grad]
            temporal_p = [p for n, p in abl_model.named_parameters()
                         if not n.startswith("adapter.") and p.requires_grad]
            optimizer = torch.optim.AdamW([
                {"params": adapter_p, "lr": cfg["LR_VIDEOMAE"]},
                {"params": temporal_p, "lr": cfg["LR_HEAD"]},
            ], weight_decay=cfg["WEIGHT_DECAY"])
        else:
            optimizer = torch.optim.AdamW(
                trainable_params, lr=cfg["LR_HEAD"],
                weight_decay=cfg["WEIGHT_DECAY"])

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_metrics = None

        for epoch in range(cfg["EPOCHS"]):
            abl_model.train()
            for vis_b, pose_b, lbl_b, mask_b in train_loader:
                vis_b = vis_b.to(device)
                pose_b = pose_b.to(device)
                lbl_b = lbl_b.float().to(device)
                mask_b = mask_b.to(device)

                optimizer.zero_grad()
                logits, _ = abl_model(vis_b, pose_b, padding_mask=mask_b,
                                      use_causal=use_causal)
                loss = criterion(logits.squeeze(-1), lbl_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(abl_model.parameters(),
                                               cfg["GRAD_CLIP"])
                optimizer.step()

            val_m, _, _, _ = evaluate(abl_model, val_loader, criterion, device,
                                      use_causal=use_causal)

            scheduler.step(val_m["loss"])

            if val_m["loss"] < best_val_loss - 0.001:
                best_val_loss = val_m["loss"]
                patience_counter = 0
                best_metrics = val_m.copy()
            else:
                patience_counter += 1

            if patience_counter >= cfg["PATIENCE"]:
                break

        fold_metrics.append(best_metrics)
        print(f"  Fold {fold+1}: Acc={best_metrics['accuracy']:.3f} "
              f"F1={best_metrics['f1']:.3f} AUC={best_metrics['auc']:.3f}")

        # Free GPU memory after each ablation fold
        del abl_model, abl_adapter
        torch.cuda.empty_cache()

    # Average
    result = {"config": config_name}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [m[key] for m in fold_metrics]
        result[key] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

    return result


# Run ablation study
print("\\n🔬 Starting Ablation Study (4 configs × 5 folds)")
print("Using pre-computed intermediate features + domain adapter.\\n")

ablation_results = []

# Config A: Full model (use results from main training)
ablation_results.append({
    "config": "A: Full (Partial FT + Pose)",
    "accuracy": f"{means['accuracy']:.4f}±{stds['accuracy']:.4f}",
    "precision": f"{means['precision']:.4f}±{stds['precision']:.4f}",
    "recall": f"{means['recall']:.4f}±{stds['recall']:.4f}",
    "f1": f"{means['f1']:.4f}±{stds['f1']:.4f}",
    "auc": f"{means['auc']:.4f}±{stds['auc']:.4f}",
})

# Config B: VideoMAE only (zero pose)
result_b = run_ablation_config(
    "B: VideoMAE Only", all_intermediate_features, pose_features,
    data_df, CFG, DEVICE, zero_pose=True)
ablation_results.append(result_b)

# Config C: Pose only (zero visual)
result_c = run_ablation_config(
    "C: Pose Only", all_intermediate_features, pose_features,
    data_df, CFG, DEVICE, zero_visual=True)
ablation_results.append(result_c)

# Config D: Frozen VideoMAE (no domain adapter training)
result_d = run_ablation_config(
    "D: Frozen VideoMAE", all_intermediate_features, pose_features,
    data_df, CFG, DEVICE, freeze_adapter=True)
ablation_results.append(result_d)
"""))

cells.append(code("""# ============================================================
# Ablation Results Table
# ============================================================

abl_df = pd.DataFrame(ablation_results)
print("\\n" + "="*80)
print("📊 ABLATION STUDY RESULTS")
print("="*80)
print(abl_df.to_string(index=False))
print("="*80)

abl_df.to_csv(os.path.join(CFG["RESULTS_DIR"], "ablation_results.csv"), index=False)
print(f"💾 Saved ablation results")
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 11: Explainability
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 11: Explainability & Feature Analysis

- **Temporal attention bar chart** — which clips are most important (3 correct + 3 incorrect)
- **Pose feature importance** via label correlation
- **Feature statistical comparison** with Welch's t-test and Cohen's d
"""))

cells.append(code("""# ============================================================
# SECTION 11: Temporal Attention Visualization
# ============================================================

def plot_temporal_attention(fold_results, save_path=None):
    \"\"\"
    Temporal attention bar charts for 3 correct + 3 incorrect predictions.
    Shows which clips the model focuses on for each video.
    \"\"\"
    # Use last fold results (most complete)
    last_fold = fold_results[-1]
    if "fold_attns" not in last_fold:
        print("⚠️ No attention data available for visualization")
        return

    attns = last_fold["fold_attns"]   # (N_val, num_clips)
    probs = last_fold["fold_probs"]
    labels = last_fold["fold_labels"]

    preds = (probs >= 0.5).astype(int)
    correct_mask = preds == labels
    incorrect_mask = ~correct_mask

    # Select up to 3 correct and 3 incorrect
    correct_idx = np.where(correct_mask)[0][:3]
    incorrect_idx = np.where(incorrect_mask)[0][:3]

    n_correct = len(correct_idx)
    n_incorrect = len(incorrect_idx)
    n_total = n_correct + n_incorrect

    if n_total == 0:
        print("⚠️ No predictions to visualize")
        return

    fig, axes = plt.subplots(2, max(n_correct, n_incorrect, 1),
                              figsize=(6 * max(n_correct, n_incorrect, 1), 8))
    if max(n_correct, n_incorrect, 1) == 1:
        axes = axes.reshape(2, 1)

    # Correct predictions
    for i in range(max(n_correct, n_incorrect, 1)):
        # Top row: correct
        ax = axes[0, i]
        if i < n_correct:
            idx = correct_idx[i]
            attn_vals = attns[idx]
            true_label = "Lame" if labels[idx] == 1 else "Healthy"
            prob_val = probs[idx]
            colors = ['#2ecc71'] * len(attn_vals)
            ax.bar(range(len(attn_vals)), attn_vals, color=colors, alpha=0.8)
            ax.set_title(f"✅ Correct: {true_label}\\np={prob_val:.3f}",
                        fontsize=10, fontweight='bold')
            ax.set_xlabel("Clip Index", fontsize=9)
            ax.set_ylabel("Attention Weight", fontsize=9)
            ax.set_ylim(0, max(attn_vals) * 1.2)
        else:
            ax.set_visible(False)

        # Bottom row: incorrect
        ax = axes[1, i]
        if i < n_incorrect:
            idx = incorrect_idx[i]
            attn_vals = attns[idx]
            true_label = "Lame" if labels[idx] == 1 else "Healthy"
            pred_label = "Lame" if preds[idx] == 1 else "Healthy"
            prob_val = probs[idx]
            colors = ['#e74c3c'] * len(attn_vals)
            ax.bar(range(len(attn_vals)), attn_vals, color=colors, alpha=0.8)
            ax.set_title(f"❌ Wrong: True={true_label}, Pred={pred_label}\\n"
                        f"p={prob_val:.3f}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Clip Index", fontsize=9)
            ax.set_ylabel("Attention Weight", fontsize=9)
            ax.set_ylim(0, max(attn_vals) * 1.2)
        else:
            ax.set_visible(False)

    plt.suptitle("Temporal Attention Weights (Which Clips Matter?)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_temporal_attention(fold_results,
    save_path=os.path.join(CFG["RESULTS_DIR"], "temporal_attention.png"))
"""))

cells.append(code("""# ============================================================
# Pose Feature Statistical Analysis
# ============================================================

def pose_feature_ttest_table(features, data_labels, feature_names, save_path=None):
    \"\"\"Welch's t-test for each pose feature: healthy vs lame.\"\"\"
    rows = []
    for i, name in enumerate(feature_names):
        h = features[data_labels == 0, i]
        l = features[data_labels == 1, i]
        h = h[~np.isnan(h)]  # Exclude NaN (missing data)
        l = l[~np.isnan(l)]

        if len(h) > 5 and len(l) > 5:
            t_stat, p_val = stats.ttest_ind(h, l, equal_var=False)
            pooled_std = np.sqrt((np.var(h) + np.var(l)) / 2 + 1e-8)
            effect_size = (np.mean(l) - np.mean(h)) / pooled_std
            rows.append({
                "Feature": name,
                "Healthy (mean±std)": f"{np.mean(h):.4f}±{np.std(h):.4f}",
                "Lame (mean±std)": f"{np.mean(l):.4f}±{np.std(l):.4f}",
                "t-stat": f"{t_stat:.3f}",
                "p-value": f"{p_val:.6f}",
                "Cohen's d": f"{effect_size:.3f}",
                "Sig.": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            })

    df = pd.DataFrame(rows)
    print("\\n📊 Pose Feature Statistical Comparison (Welch's t-test)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    if save_path:
        df.to_csv(save_path, index=False)
    return df

ttest_df = pose_feature_ttest_table(
    pose_features, data_df["label"].values, pose_extractor.FEATURE_NAMES,
    save_path=os.path.join(CFG["RESULTS_DIR"], "pose_feature_ttest.csv")
)
"""))

cells.append(code("""# ============================================================
# Feature Importance (correlation with label)
# ============================================================

def plot_feature_importance(features, data_labels, feature_names, save_path=None):
    \"\"\"
    Feature importance: absolute correlation between each pose feature
    and the binary label.
    \"\"\"
    fig, ax = plt.subplots(figsize=(12, 6))

    importance = []
    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        valid_mask = ~np.isnan(feat_vals)
        if valid_mask.sum() > 5 and np.std(feat_vals[valid_mask]) > 0:
            corr = abs(np.corrcoef(feat_vals[valid_mask], data_labels[valid_mask])[0, 1])
        else:
            corr = 0.0
        importance.append(corr)

    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = [importance[i] for i in sorted_idx]

    colors = ['#e74c3c' if v > 0.1 else '#3498db' if v > 0.05 else '#95a5a6'
              for v in sorted_vals]
    ax.barh(range(len(sorted_names)), sorted_vals, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('|Correlation with Label|', fontsize=12)
    ax.set_title('Pose Feature Importance (Correlation with Lameness)',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_feature_importance(
    pose_features, data_df["label"].values, pose_extractor.FEATURE_NAMES,
    save_path=os.path.join(CFG["RESULTS_DIR"], "feature_importance.png")
)
"""))

# ═══════════════════════════════════════════════════════════════
# SECTION 12: Results Summary
# ═══════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 12: Results Summary & Model Export
"""))

cells.append(code("""# ============================================================
# SECTION 12: Final Results Summary
# ============================================================

print("\\n" + "="*80)
print("🏆 FINAL RESULTS SUMMARY — Cow Lameness Analysis v32")
print("="*80)

print(f"\\n📐 Architecture:")
print(f"   VideoMAE (blocks 0-9 frozen, 10-11 trainable) → {CFG['PROJECTION_DIM']}D")
print(f"   + DLC Pose ({CFG['POSE_FEAT_DIM']}D)")
print(f"   → Temporal Transformer ({CFG['NUM_LAYERS']}L, {CFG['NUM_HEADS']}H)")
print(f"   → Binary Classification (Healthy/Lame)")

print(f"\\n📊 Dataset:")
print(f"   Total videos: {len(data_df)}")
print(f"   Healthy: {(data_df['label']==0).sum()} | Lame: {(data_df['label']==1).sum()}")
print(f"   Unique animals: {data_df['animal_id'].nunique()}")
print(f"   Validation: {CFG['CV_FOLDS']}-fold subject-level CV")

print(f"\\n🎯 Performance (Mean ± Std across {CFG['CV_FOLDS']} folds):")
print(f"   Accuracy:  {means['accuracy']:.4f} ± {stds['accuracy']:.4f}")
print(f"   Precision: {means['precision']:.4f} ± {stds['precision']:.4f}")
print(f"   Recall:    {means['recall']:.4f} ± {stds['recall']:.4f}")
print(f"   F1 Score:  {means['f1']:.4f} ± {stds['f1']:.4f}")
print(f"   AUC-ROC:   {means['auc']:.4f} ± {stds['auc']:.4f}")

target_met = means['accuracy'] >= 0.80
print(f"\\n{'✅' if target_met else '❌'} Target accuracy ≥ 80%: {'MET' if target_met else 'NOT MET'}")

print(f"\\n📁 All results saved to: {CFG['RESULTS_DIR']}")
print("="*80)
"""))

cells.append(code("""# ============================================================
# Save Best Model
# ============================================================

# Select best fold by F1
best_fold_idx = np.argmax([r["f1"] for r in fold_results])
best_model_state = best_models[best_fold_idx]

save_dict = {
    "cfg": CFG,
    "model_state": best_model_state["model"],
    "fold_results": [{k: v for k, v in r.items()
                      if k not in ("history", "fold_probs", "fold_labels", "fold_attns")}
                     for r in fold_results],
    "means": means,
    "stds": stds,
    "pose_feature_names": pose_extractor.FEATURE_NAMES,
}

model_path = os.path.join(CFG["RESULTS_DIR"], "best_model_v32.pth")
torch.save(save_dict, model_path)
print(f"💾 Best model saved: {model_path}")
print(f"   Best fold: {best_fold_idx + 1} (F1={fold_results[best_fold_idx]['f1']:.4f})")

# Save all results as JSON
import json as json_lib
results_summary = {
    "version": "v32",
    "architecture": "PartialFT_VideoMAE(blocks10-11) + DLC_Pose + TemporalTransformer",
    "classification": "binary",
    "dataset_size": len(data_df),
    "cv_folds": CFG["CV_FOLDS"],
    "means": {k: float(v) for k, v in means.items()},
    "stds": {k: float(v) for k, v in stds.items()},
}
with open(os.path.join(CFG["RESULTS_DIR"], "results_summary.json"), "w") as f:
    json_lib.dump(results_summary, f, indent=2)

print("\\n✅ All artifacts saved. Notebook execution complete.")
"""))

# Save Part 3
notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": cells
}
out = r"c:\Users\HP\Desktop\Clone Repos\CowLameness\Colab_Notebook\_v32_part3.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Part 3 saved: {out} ({len(cells)} cells)")
