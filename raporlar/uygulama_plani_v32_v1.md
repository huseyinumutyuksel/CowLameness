# Cow Lameness Analysis v32 — Uygulama Planı

## Amaç

Sıfırdan, temiz ve sabitlenmiş bir Google Colab notebook (`Cow_Lameness_Analysis_v32.ipynb`) oluşturmak. Q1 dergi hedefli, binary sınıflandırma (Sağlıklı/Topal), DLC pose feature'ları + VideoMAE (partial FT) + Temporal Transformer mimarisi.

---

## Sabitlenmiş Mimari Kararlar (ADR)

| Karar | Seçim | Gerekçe |
|-------|-------|---------|
| **Sınıflandırma** | Binary (0/1) | Veri seti sadece klasör seviyesinde etiketli |
| **Pose framework** | DeepLabCut (SuperAnimal) | 1167 çıktı hazır; MMPose gelecekte opsiyonel |
| **VideoMAE stratejisi** | Partial FT (son 2 blok) | Domain adaptasyonu; ADR ile sabitleniyor |
| **Temporal model** | Temporal Transformer | Clip dizisinden video-level karar |
| **MIL Attention** | ❌ Yok | Temporal Transformer ile redundant |
| **Loss** | BCE + class weight | Binary; sınıf dengesizliği (642:525) |
| **Validation** | 5-fold subject-level CV | Q1 dergi standardı |

---

## Notebook Bölümleri ve Proposed Changes

### [NEW] [Cow_Lameness_Analysis_v32.ipynb](file:///c:/Users/HP/Desktop/Clone%20Repos/CowLameness/Colab_Notebook/Cow_Lameness_Analysis_v32.ipynb)

Notebook aşağıdaki hücrelerden oluşacaktır:

---

### Bölüm 1 — Environment & Config

- GPU kontrol, seed ayarları (reproducibility)
- Tüm hiperparametreler tek `CFG` dictionary'de
- Google Drive mount

```python
CFG = {
    "SEED": 42,
    "NUM_CLIPS": 8,           # Video başına clip sayısı
    "CLIP_LENGTH": 16,        # Her clip'te frame sayısı
    "VIDEOMAE_DIM": 768,      # VideoMAE hidden dim
    "POSE_FEAT_DIM": 16,      # DLC'den çıkarılan feature sayısı
    "HIDDEN_DIM": 256,        # Temporal Transformer hidden dim
    "NUM_HEADS": 8,
    "NUM_LAYERS": 4,
    "DROPOUT": 0.3,
    "TRAINABLE_BLOCKS": [10, 11],  # VideoMAE son 2 blok
    "BATCH_SIZE": 8,
    "EPOCHS": 40,
    "LR_VIDEOMAE": 1e-5,
    "LR_HEAD": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "PATIENCE": 7,
    "CV_FOLDS": 5,
    "POSE_FRAMEWORK": "deeplabcut",  # Gelecekte "mmpose" da seçilebilir
}
```

---

### Bölüm 2 — Data Discovery & Subject ID Extraction

- Video klasörlerinden yol listesi ve label (0=Sağlıklı, 1=Topal)
- Video adından `animal_id` çıkarma (subject-level split için)
- DLC CSV dosyalarıyla eşleştirme
- Veri seti istatistikleri tablosu

---

### Bölüm 3 — Pose Feature Extraction (DLC)

Mevcut [gait_features.py](file:///c:/Users/HP/Desktop/Clone%20Repos/CowLameness/Colab_Notebook/gait_features.py) modülü sıfırdan yeniden yazılacak — doğru SuperAnimal keypoint isimlerini kullanarak.

**Çıkarılacak 16 özellik (clip-level):**

| # | Feature | Kaynak Keypoint | Klinik Anlam |
|---|---------|----------------|-------------|
| 1-2 | `head_vertical_displacement_mean/std` | Head | Head bob — topallık göstergesi |
| 3-4 | `spine_angle_mean/std` | Withers + Spine + Tail | Sırt eğriliği varyansı |
| 5-6 | `step_duration_left_mean/std` | Left front/hind hoof | Sol adım zamanlaması |
| 7-8 | `step_duration_right_mean/std` | Right front/hind hoof | Sağ adım zamanlaması |
| 9 | `temporal_asymmetry_ratio` | Hooves | Sol-sağ asimetri |
| 10-11 | `hip_sway_amplitude/range` | Left/Right hip | Kalça sallanması |
| 12-13 | `stride_length_left/right_cv` | Hooves | Adım uzunluğu tutarlılığı |
| 14 | `knee_angle_asymmetry` | Hip-Knee-Hoof | Eklem açısı asimetrisi |
| 15 | `step_frequency` | Hooves | Adım sıklığı |
| 16 | `back_curvature_variance` | Withers-Spine-Tail | Kamburluk değişimi |

Çıkarım sonucu: her clip için `pose_feat ∈ ℝ¹⁶`

> [!IMPORTANT]
> SuperAnimal-Quadruped modelinin gerçek keypoint isimleri CSV header'dan dinamik olarak okunacak. Hardcoded index yok.

---

### Bölüm 4 — VideoMAE Encoder (Partial Fine-Tuning)

```
VideoMAE (MCG-NJU/videomae-base)
├── Patch embedding + Blocks 0-9: FROZEN
├── Blocks 10-11: TRAINABLE (requires_grad=True)
└── Projection: 768 → 256
```

- `forward()`: Video clip → mean pooling over spatial tokens → `clip_visual ∈ ℝ²⁵⁶`
- Runtime assertion: frozen blokların gradient almadığını doğrula

---

### Bölüm 5 — Clip Embedding Oluşturma

Her clip için:

```python
clip_embedding = concat(clip_visual, pose_feat)  # ℝ^(256+16) = ℝ^272
```

Her video `NUM_CLIPS` adet clip → `clip_sequence ∈ ℝ^(8 × 272)`

---

### Bölüm 6 — Temporal Transformer + Classification Head

```
clip_sequence (B, 8, 272)
  → Linear projection → (B, 8, 256)
  → Positional Encoding
  → TransformerEncoder (4 layers, 8 heads, causal mask)
  → Mean pooling over time → (B, 256)
  → FC → Dropout → FC → Sigmoid → prediction ∈ [0, 1]
```

**Causal mask:** `torch.triu` ile gelecek clip'lere bakmayı engelle (online inference desteği).

---

### Bölüm 7 — Dataset & DataLoader

- `CowLamenessDatasetV32`: video path → clip'lere böl → VideoMAE embed + Pose feature çıkar
- Collate function: padding + attention mask
- Subject-level split fonksiyonu (`animal_id` bazlı)

---

### Bölüm 8 — Training Loop

- 5-fold subject-level cross-validation
- BCE loss + class weight (642:525 dengesizliği)
- AdamW, iki LR grubu: VideoMAE blokları (1e-5), geri kalan (1e-4)
- Early stopping: validation loss (patience=7)
- Best model checkpoint (en düşük val_loss)
- Gradient clipping (max_norm=1.0)
- Epoch bazlı: train_loss, val_loss, val_accuracy, val_f1 loglama

---

### Bölüm 9 — Evaluation (Q1 Dergi Seviyesi)

Her fold ve genel ortalama için:

| Çıktı | Format |
|-------|--------|
| **Confusion Matrix** | Heatmap (normalize edilmiş) |
| **ROC Curve** | 5 fold + mean AUC |
| **Precision-Recall Curve** | 5 fold |
| **Metrics Table** | Accuracy, Precision, Recall, F1, AUC ± std |
| **Per-fold Results** | Tablo |
| **Statistical Test** | Paired t-test (fold sonuçları) |
| **Classification Report** | sklearn format |

---

### Bölüm 10 — Ablation Study

| Konfigürasyon | Açıklama |
|---------------|----------|
| **VideoMAE only** | Pose feature'lar çıkarılır |
| **Pose only** | VideoMAE çıkarılır, sadece gait feature → Transformer |
| **Combined (full)** | Hem VideoMAE hem pose |
| **Frozen VideoMAE** | Partial FT yerine tamamen frozen |

Her ablation: aynı 5-fold CV, aynı metrikler.

---

### Bölüm 11 — Explainability & Visualizations

- **Temporal Attention:** Hangi clip'ler yüksek dikkat aldı? (bar chart)
- **Pose Feature Importance:** Her feature'ın sınıflandırmaya katkısı
- **Örnek Videolar:** 3 doğru + 3 yanlış tahmin, attention overlay ile
- **Feature Distribution:** Sağlıklı vs Topal için boxplot (her pose feature)
- **t-test tablosu:** Her pose feature için p-value

---

### Bölüm 12 — Results Summary & Academic Tables

- Final performans tablosu (mean ± std across folds)
- Ablation karşılaştırma tablosu
- Comparison with literature (if applicable)
- Best model kaydetme (Drive'a)

---

## Kapsam Dışı (Bu Versiyonda YAPILMAYACAK)

| Yapılmayacak | Neden |
|-------------|-------|
| MMPose entegrasyonu | Kod desteği olacak ama aktif kullanılmayacak |
| CORAL ordinal regression | Binary sınıflandırma kararı alındı |
| MIL Attention | Temporal Transformer ile redundant |
| Multi-modal fusion (optical flow, SAM, YOLO) | Gereksiz karmaşıklık |
| Rapor oluşturma kodu | Ayrı adım |

---

## Dosya Yapısı

```
Colab_Notebook/
├── Cow_Lameness_Analysis_v32.ipynb  ← YENİ (tek dosya, self-contained)
└── (eski dosyalar dokunulmayacak)
```

> [!NOTE]
> v32 notebook **self-contained** olacak — dışarıdan import etmeyecek.
> Tüm class ve fonksiyonlar notebook içinde tanımlanacak.
> Bu, Colab'da tek tıkla çalıştırılabilirlik sağlar ve reproducibility garanti eder.

---

## Verification Plan

### Otomatik Doğrulama (Notebook İçi Assertion'lar)

Notebook içinde aşağıdaki assertion hücreleri olacaktır:

1. **Determinism test:** Aynı seed → aynı model ağırlıkları
2. **Subject split verification:** `assert len(set(train_ids) & set(test_ids)) == 0`
3. **VideoMAE freeze check:** Block 0–9'daki tüm parametrelerin `requires_grad=False`
4. **Pose feature shape:** Her clip için `pose_feat.shape == (16,)`
5. **Clip embedding shape:** `clip_embedding.shape == (B, NUM_CLIPS, 272)`
6. **Mask validation:** Padding mask'in attention'a doğru uygulandığını doğrula
7. **Label balance:** Train/val split'teki sınıf dağılımını raporla

### Manuel Doğrulama (Kullanıcı Tarafından)

1. Notebook'u Google Colab'da açın (T4 GPU veya daha iyisi)
2. Runtime → Run all ile tüm hücreleri sırayla çalıştırın
3. Bölüm 2 çıktısını doğrulayın: 642 sağlıklı + 525 topal video listelenmeli
4. Bölüm 3 çıktısını doğrulayın: Pose feature dağılım boxplot'ları anlamlı görünmeli
5. Bölüm 8 çıktısını izleyin: Train/val loss eğrileri → overfitting kontrolü
6. Bölüm 9 tablolarını inceleyin: Accuracy >80% hedefi karşılanıyor mu?
7. Bölüm 10 ablation tablosunu doğrulayın: Combined > single modality olmalı
8. Tüm görsellerin (confusion matrix, ROC, boxplot) akademik kalitede olduğunu onaylayın

> [!CAUTION]
> Notebook Colab ortamında çalışacak şekilde tasarlanmıştır. Lokal çalıştırma desteklenmez.
> Videolar ve DLC çıktıları Google Drive'da olmalıdır.
