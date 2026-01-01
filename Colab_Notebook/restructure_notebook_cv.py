"""
Restructure Colab Notebook for Academic-Standard 5-Fold CV
"""
import json
from pathlib import Path

# Load notebook
notebook_path = Path(r"c:\Users\HP\Desktop\CowLameness\Colab_Notebook\Cow_Lameness_Analysis_v19.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("Loading notebook v19...")
print(f"Total cells: {len(notebook['cells'])}")

# Find and update cells
cells = notebook['cells']

# ============================================================================
# CHANGE 1: Update Data Preparation Cell (Cell with "Prepare Data for Training")
# ============================================================================
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '## 6. Prepare Data for Training' in source:
            print(f"\n✅ Found 'Prepare Data' section at cell {i}")
            
            # Update the next code cell
            if i+1 < len(cells) and cells[i+1].get('cell_type') == 'code':
                cells[i+1]['source'] = [
                    "# Academic Standard: Train/Test split only\n",
                    "# Test set is held out for final evaluation (NEVER used in training/tuning)\n",
                    "X = np.array([d['features'] for d in dataset])\n",
                    "y = np.array([d['label'] for d in dataset])\n",
                    "\n",
                    "# Split into Train (85%) and Test (15%)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y, \n",
                    "    test_size=0.15,  # 15% for final test evaluation\n",
                    "    stratify=y, \n",
                    "    random_state=42\n",
                    ")\n",
                    "\n",
                    "# Standardize features\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "print(f\"📊 Data Split (Academic Standard):\")\n",
                    "print(f\"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)\")\n",
                    "print(f\"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)\")\n",
                    "print(f\"\\n   ✅ Test set isolated for final evaluation\")\n",
                    "print(f\"   ✅ 5-Fold CV will be performed on Train set for hyperparameter tuning\")"
                ]
                print(f"   ✅ Updated data preparation cell {i+1}")
            break

# ============================================================================
# CHANGE 2: Update 5-Fold CV Cell (Cell with "Training with 5-Fold CV")
# ============================================================================
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '## 8. Training with 5-Fold CV' in source:
            print(f"\n✅ Found '5-Fold CV' section at cell {i}")
            
            # Update markdown title
            cells[i]['source'] = ["## 8. Hyperparameter Tuning with 5-Fold CV"]
            
            # Update the next code cell
            if i+1 < len(cells) and cells[i+1].get('cell_type') == 'code':
                cells[i+1]['source'] = [
                    "# Hyperparameter configurations to test\n",
                    "configs_to_test = [\n",
                    "    {'hidden_dim': 256, 'num_heads': 8, 'num_layers': 4, 'dropout': 0.3},  # Default\n",
                    "    {'hidden_dim': 512, 'num_heads': 8, 'num_layers': 6, 'dropout': 0.2},  # Larger\n",
                    "    {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.4},  # Smaller\n",
                    "]\n",
                    "\n",
                    "print(\"=\"*60)\n",
                    "print(\"HYPERPARAMETER TUNING WITH 5-FOLD CROSS-VALIDATION\")\n",
                    "print(\"=\"*60)\n",
                    "print(f\"Testing {len(configs_to_test)} configurations\")\n",
                    "print(f\"Each configuration evaluated with 5-Fold CV\")\n",
                    "print(\"=\"*60)\n",
                    "\n",
                    "# Track best configuration\n",
                    "best_cv_score = 0\n",
                    "best_params = None\n",
                    "all_results = []\n",
                    "\n",
                    "for config_idx, config in enumerate(configs_to_test):\n",
                    "    print(f\"\\n{'='*60}\")\n",
                    "    print(f\"Configuration {config_idx+1}/{len(configs_to_test)}\")\n",
                    "    print(f\"  hidden_dim={config['hidden_dim']}, num_heads={config['num_heads']}\")\n",
                    "    print(f\"  num_layers={config['num_layers']}, dropout={config['dropout']}\")\n",
                    "    print(f\"{'='*60}\")\n",
                    "    \n",
                    "    # 5-Fold CV for this configuration\n",
                    "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                    "    fold_results = []\n",
                    "    \n",
                    "    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):\n",
                    "        print(f\"  Fold {fold+1}/5...\", end=' ')\n",
                    "        \n",
                    "        # Create model with current config\n",
                    "        model = TemporalTransformer(\n",
                    "            input_dim=X_train_scaled.shape[1],\n",
                    "            hidden_dim=config['hidden_dim'],\n",
                    "            num_heads=config['num_heads'],\n",
                    "            num_layers=config['num_layers']\n",
                    "        ).to(device)\n",
                    "        \n",
                    "        # Train on this fold\n",
                    "        best_model, val_acc = train_model(\n",
                    "            model, \n",
                    "            X_train_scaled[train_idx], y_train[train_idx],\n",
                    "            X_train_scaled[val_idx], y_train[val_idx],\n",
                    "            epochs=20\n",
                    "        )\n",
                    "        \n",
                    "        fold_results.append(val_acc)\n",
                    "        print(f\"Accuracy: {val_acc:.4f}\")\n",
                    "    \n",
                    "    # Calculate mean CV score\n",
                    "    mean_cv_score = np.mean(fold_results)\n",
                    "    std_cv_score = np.std(fold_results)\n",
                    "    \n",
                    "    print(f\"\\n  📊 Configuration {config_idx+1} Results:\")\n",
                    "    print(f\"     Mean CV Accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}\")\n",
                    "    print(f\"     Fold Accuracies: {[f'{acc:.4f}' for acc in fold_results]}\")\n",
                    "    \n",
                    "    all_results.append({\n",
                    "        'config': config,\n",
                    "        'mean_cv_score': mean_cv_score,\n",
                    "        'std_cv_score': std_cv_score,\n",
                    "        'fold_results': fold_results\n",
                    "    })\n",
                    "    \n",
                    "    # Track best configuration\n",
                    "    if mean_cv_score > best_cv_score:\n",
                    "        best_cv_score = mean_cv_score\n",
                    "        best_params = config\n",
                    "        print(f\"     ✅ New best configuration!\")\n",
                    "\n",
                    "print(f\"\\n{'='*60}\")\n",
                    "print(\"HYPERPARAMETER TUNING COMPLETE\")\n",
                    "print(f\"{'='*60}\")\n",
                    "print(f\"Best Configuration:\")\n",
                    "print(f\"  {best_params}\")\n",
                    "print(f\"  CV Accuracy: {best_cv_score:.4f}\")\n",
                    "print(f\"{'='*60}\")"
                ]
                print(f"   ✅ Updated 5-Fold CV cell {i+1}")
            break

# ============================================================================
# CHANGE 3: Update Final Test Evaluation Cell
# ============================================================================
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '## 9. Final Test Evaluation' in source:
            print(f"\n✅ Found 'Final Test Evaluation' section at cell {i}")
            
            # Update markdown title
            cells[i]['source'] = ["## 9. Final Model Training & Test Evaluation"]
            
            # Update the next code cell
            if i+1 < len(cells) and cells[i+1].get('cell_type') == 'code':
                cells[i+1]['source'] = [
                    "print(\"\\n\" + \"=\"*60)\n",
                    "print(\"FINAL MODEL TRAINING\")\n",
                    "print(\"=\"*60)\n",
                    "print(f\"Training with best hyperparameters on FULL training set\")\n",
                    "print(f\"  Parameters: {best_params}\")\n",
                    "print(f\"  Training samples: {len(X_train_scaled)}\")\n",
                    "print(\"=\"*60 + \"\\n\")\n",
                    "\n",
                    "# Create final model with best hyperparameters\n",
                    "final_model = TemporalTransformer(\n",
                    "    input_dim=X_train_scaled.shape[1],\n",
                    "    hidden_dim=best_params['hidden_dim'],\n",
                    "    num_heads=best_params['num_heads'],\n",
                    "    num_layers=best_params['num_layers']\n",
                    ").to(device)\n",
                    "\n",
                    "# Train on FULL training set (no validation split)\n",
                    "criterion = nn.CrossEntropyLoss()\n",
                    "optimizer = optim.Adam(final_model.parameters(), lr=0.001)\n",
                    "\n",
                    "X_train_t = torch.FloatTensor(X_train_scaled).to(device)\n",
                    "y_train_t = torch.LongTensor(y_train).to(device)\n",
                    "\n",
                    "for epoch in range(30):\n",
                    "    final_model.train()\n",
                    "    optimizer.zero_grad()\n",
                    "    outputs = final_model(X_train_t)\n",
                    "    loss = criterion(outputs, y_train_t)\n",
                    "    loss.backward()\n",
                    "    optimizer.step()\n",
                    "    \n",
                    "    if (epoch + 1) % 5 == 0:\n",
                    "        print(f\"  Epoch {epoch+1}/30 - Loss: {loss.item():.4f}\")\n",
                    "\n",
                    "print(\"\\n✅ Final model training complete\")\n",
                    "\n",
                    "# Evaluate on held-out test set\n",
                    "print(\"\\n\" + \"=\"*60)\n",
                    "print(\"EVALUATING ON HELD-OUT TEST SET\")\n",
                    "print(\"=\"*60)\n",
                    "\n",
                    "final_model.eval()\n",
                    "with torch.no_grad():\n",
                    "    X_test_t = torch.FloatTensor(X_test_scaled).to(device)\n",
                    "    test_outputs = final_model(X_test_t)\n",
                    "    test_preds = test_outputs.argmax(dim=1).cpu().numpy()\n",
                    "    test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()[:, 1]\n",
                    "\n",
                    "# Calculate metrics\n",
                    "accuracy = accuracy_score(y_test, test_preds)\n",
                    "precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='binary')\n",
                    "cm = confusion_matrix(y_test, test_preds)\n",
                    "auc = roc_auc_score(y_test, test_probs)\n",
                    "\n",
                    "print(f\"\\n{'='*60}\")\n",
                    "print(\"FINAL TEST RESULTS\")\n",
                    "print(f\"{'='*60}\")\n",
                    "print(f\"Test Accuracy:  {accuracy:.4f}\")\n",
                    "print(f\"Precision:      {precision:.4f}\")\n",
                    "print(f\"Recall:         {recall:.4f}\")\n",
                    "print(f\"F1-Score:       {f1:.4f}\")\n",
                    "print(f\"ROC-AUC:        {auc:.4f}\")\n",
                    "print(f\"\\nConfusion Matrix:\")\n",
                    "print(cm)\n",
                    "print(f\"{'='*60}\")\n",
                    "\n",
                    "# Academic reporting\n",
                    "print(f\"\\n{'='*60}\")\n",
                    "print(\"ACADEMIC SUMMARY\")\n",
                    "print(f\"{'='*60}\")\n",
                    "print(f\"Cross-Validation (Training Set):\")\n",
                    "print(f\"  Mean Accuracy: {best_cv_score:.4f}\")\n",
                    "print(f\"\\nFinal Test Set Performance:\")\n",
                    "print(f\"  Accuracy: {accuracy:.4f}\")\n",
                    "print(f\"  F1-Score: {f1:.4f}\")\n",
                    "print(f\"  ROC-AUC:  {auc:.4f}\")\n",
                    "print(f\"{'='*60}\")\n",
                    "\n",
                    "# Save comprehensive results\n",
                    "metrics = {\n",
                    "    'methodology': '5-Fold Cross-Validation for Hyperparameter Tuning',\n",
                    "    'data_split': {\n",
                    "        'train_size': len(X_train),\n",
                    "        'test_size': len(X_test),\n",
                    "        'train_percentage': 85,\n",
                    "        'test_percentage': 15\n",
                    "    },\n",
                    "    'best_hyperparameters': best_params,\n",
                    "    'cv_results': {\n",
                    "        'mean_accuracy': float(best_cv_score),\n",
                    "        'all_configurations': all_results\n",
                    "    },\n",
                    "    'test_results': {\n",
                    "        'accuracy': float(accuracy),\n",
                    "        'precision': float(precision),\n",
                    "        'recall': float(recall),\n",
                    "        'f1_score': float(f1),\n",
                    "        'roc_auc': float(auc),\n",
                    "        'confusion_matrix': cm.tolist()\n",
                    "    },\n",
                    "    'features_used': list(config['features'].keys()),\n",
                    "    'timestamp': datetime.now().isoformat()\n",
                    "}\n",
                    "\n",
                    "with open(f\"{OUTPUT_DIR}/metrics.json\", 'w') as f:\n",
                    "    json.dump(metrics, f, indent=2)\n",
                    "\n",
                    "torch.save({\n",
                    "    'model_state_dict': final_model.state_dict(),\n",
                    "    'scaler': scaler,\n",
                    "    'config': config,\n",
                    "    'best_params': best_params,\n",
                    "    'cv_score': best_cv_score,\n",
                    "    'test_metrics': metrics['test_results']\n",
                    "}, f\"{OUTPUT_DIR}/models/best_model_multimodal.pth\")\n",
                    "\n",
                    "print(\"\\n✅ Results saved to:\")\n",
                    "print(f\"   - {OUTPUT_DIR}/metrics.json\")\n",
                    "print(f\"   - {OUTPUT_DIR}/models/best_model_multimodal.pth\")"
                ]
                print(f"   ✅ Updated final evaluation cell {i+1}")
            break

# Save updated notebook as v20
output_path = notebook_path.parent / "Cow_Lameness_Analysis_v20.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

print(f"\n{'='*60}")
print("NOTEBOOK RESTRUCTURING COMPLETE")
print(f"{'='*60}")
print(f"✅ Created: {output_path.name}")
print(f"\nKey Changes:")
print(f"  1. Data split: Train/Test (85/15) - Validation set removed")
print(f"  2. 5-Fold CV: Used for hyperparameter tuning")
print(f"  3. Final model: Trained on full training set with best params")
print(f"  4. Test evaluation: Comprehensive academic reporting")
print(f"\nAcademic Standards:")
print(f"  ✅ Test set isolated (never used in training/tuning)")
print(f"  ✅ CV used for hyperparameter selection")
print(f"  ✅ Final model uses full training data")
print(f"  ✅ Both CV and test metrics reported")
print(f"{'='*60}")
