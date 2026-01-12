v26 Notebook Creation - inceleme6.md Hatalarının Düzeltilmesi
Amaç
Cow_Lameness_Analysis_v25.ipynb
'deki doğru kısımları koruyarak ve 
inceleme6.md
'de belirtilen tüm kritik hataları düzelterek Cow_Lameness_Analysis_v26.ipynb oluşturmak.

Temel Değişiklikler (inceleme6.md'den)
1️⃣ VideoMAE Mimarisi (KRİTİK FIX)
v25'teki Hata: VideoMAE çıktısını "frame embedding'ler" gibi kullanıyor, ancak VideoMAE patch token üretiyor (mekansal + zamansal karışık).

v26 Çözümü:

VideoMAE tamamen frozen (partial fine-tuning yok)
VideoMAE sadece feature extractor olarak kullanılacak
Çıkış: temporal + spatial pooling → tek clip embedding
- VideoMAE → temporal token pooling → MIL
+ Video → sabit uzunluklu clip'ler → her clip → frozen VideoMAE → clip embedding → temporal transformer
2️⃣ Temporal Model Konumu
v25'teki Hata: Temporal attention patch-level representation'lar üzerinde çalışıyor.

v26 Çözümü:

Clip-level architecture:
Video → N adet sabit uzunluklu clip (her biri 16 frame)
Her clip → frozen VideoMAE → tek clip embedding (768-dim)
Clip embedding dizisi → causal temporal transformer
Mask her forward'ta zorunlu
3️⃣ Ordinal Regression (CORAL)
v25'teki: CORAL var ama doğru kullanılmamış olabilir.

v26: CORAL düzgün implemente edilecek:

Output: K-1 sigmoid (3 sigmoid for 4 classes)
Loss: BCE
Prediction: sum(sigmoid(logits) > 0.5)
4️⃣ Subject-Level Split
v25'teki: Var ama doğrulanmalı.

v26:

animal_id bazlı split (aynı inek asla hem train hem test'te olmayacak)
Explicit verification assertion
5️⃣ Fusion Çıkarılacak (KRİTİK)
v25'teki Hata: RGB + Pose + Flow fusion var ama:

Güçlü gerekçe yok
Ablation yok
Makaleyi zayıflatıyor
v26 Çözümü:

Sadece RGB + Temporal Modeling
Pose/Flow tamamen çıkarılacak
Daha sade, katkısı net bir çalışma
Proposed Changes
[Colab Notebook]
[NEW] 
Cow_Lameness_Analysis_v26.ipynb
Yeni notebook şu bölümleri içerecek:

Environment & Imports - v25'ten korunacak (deterministic seeding)
Paths - Sadece VIDEO_DIR gerekli (POSE_DIR kaldırılacak)
Config - Fusion parametreleri kaldırılacak, VideoMAE fully frozen
Temporal Sorting - v25'ten korunacak
VideoMAE Feature Extractor (YENİ)
Fully frozen VideoMAE
Video → clip'ler → clip embedding'ler
Clip-Level Temporal Transformer (YENİ)
Clip embedding'ler üzerinde çalışacak
Mandatory mask her forward'ta
CORAL Ordinal Loss - v25'ten iyileştirilecek
Model v26 (YENİ)
Fusion yok
Clip-level architecture
Subject-Level Split - v25'ten korunacak + doğrulama
Training Loop - Basitleştirilmiş (fusion yok)
Evaluation & Clinical Report - v25'ten korunacak
Verification Plan
Automated Tests
Bu bir Colab notebook olduğu için otomatik test yok. Ancak notebook içinde assertion'lar olacak:

Determinism test: Aynı seed ile aynı sonuçlar
Subject split verification: assert len(train_cows & test_cows) == 0
VideoMAE frozen verification: Tüm VideoMAE parametrelerinin requires_grad=False olduğunu doğrula
Mask discipline: Her forward'ta mask kullanıldığını doğrula
Manual Verification
Notebook'u Colab'da açın
Hücreleri sırasıyla çalıştırın
Her ✅ çıktısının doğru olduğunu doğrulayın
Final verification hücresinin tüm fix'leri onayladığını kontrol edin
v25'ten Korunacak Özellikler
✅ Deterministic seeding
✅ Temporal sorting (sorted_frames)
✅ Subject-level split algoritması
✅ CORAL loss temel yapısı
✅ Clinical explainability (lameness signs, visualize)
✅ Collate function (padding + mask)
✅ Evaluation metrics (MAE, F1, confusion matrix)
v25'ten Kaldırılacak/Değiştirilecek
❌ Fusion (AblationFusion class) → Kaldırılacak
❌ Partial fine-tuning → VideoMAE fully frozen
❌ 3-group optimizer → 2-group (frozen/head)
❌ Pose/Flow processing → Kaldırılacak
❌ VideoMAETemporalEncoder → ClipLevelVideoMAE olarak yeniden tasarlanacak