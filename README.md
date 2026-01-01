# 🐄 Cow Lameness Detection Project / İneklerde Topallık Tespiti Projesi

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DeepLabCut](https://img.shields.io/badge/DeepLabCut-2.3.9-green)](http://www.mackenziemathislab.org/deeplabcut)
[![MMPose](https://img.shields.io/badge/MMPose-1.3.1-orange)](https://github.com/open-mmlab/mmpose)

**Production-ready, academically rigorous system for automated lameness detection in dairy cattle using pose estimation and deep learning.**

**İnek sütünde topallık tespiti için pose tahmini ve derin öğrenme kullanan, üretime hazır, akademik olarak titiz sistem.**

---

## 📋 Table of Contents / İçindekiler

**English:**
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [Results](#results)
- [Citation](#citation)

**Türkçe:**
- [Proje Genel Bakış](#proje-genel-bakış-tr)
- [Mimari](#mimari-tr)
- [Veri Seti](#veri-seti-tr)
- [Kurulum](#kurulum-tr)
- [Kullanım İş Akışı](#kullanım-iş-akışı-tr)
- [Sonuçlar](#sonuçlar-tr)

---

## 🎯 Project Overview

This project implements a **dual-framework pose estimation** approach to detect lameness in dairy cattle through gait analysis. The system processes individual cow videos to identify biomechanical abnormalities associated with lameness.

### Key Features

✅ **Dual Pose Estimation**: DeepLabCut SuperAnimal-Quadruped + MMPose for robustness  
✅ **Hybrid Architecture**: Local processing + Cloud analysis (Google Colab)  
✅ **Academic Rigor**: 5-fold cross-validation, statistical tests (t-tests), explainable AI  
✅ **Production Ready**: Automated batch processing, resume capability, comprehensive logging  
✅ **Zero Manual Labeling**: Uses only folder-based classification (Healthy/Lame)  

### Methodology

1. **Local Pose Estimation** (DeepLabCut & MMPose)
2. **Drive Synchronization** (CSV/H5 outputs only)
3. **Cloud Training** (Google Colab - single gold-standard notebook)
4. **Local Report Generation** (Academic paper with metrics & visualizations)

---

## 🏗️ Architecture

```
LOCAL (Windows):
├── DeepLabCut/
│   ├── .venv/                    # Isolated Python environment
│   ├── requirements.txt          # Pinned dependencies
│   ├── setup_environment.ps1     # Setup script (UV package manager)
│   ├── process_videos.py         # Batch processing with test mode
│   ├── outputs/                  # Pose CSV/H5 files
│   └── README.md
├── MMPose/
│   ├── .venv/
│   ├── requirements.txt
│   ├── setup_environment.ps1
│   ├── process_videos.py
│   ├── outputs/
│   └── README.md
├── sync/
│   └── sync_to_drive.py          # Upload outputs to Google Drive
├── report_generation/
│   ├── generate_report.py        # Academic report generator
│   ├── report_template.md
│   └── requirements.txt
└── Colab_Notebook/
    └── Cow_Lameness_Analysis_v18.ipynb  # Training & analysis (run in Colab)

GOOGLE DRIVE:
└── Inek Topallik Tespiti Parcalanmis Inek Videolari/
    ├── cow_single_videos/
    │   ├── Saglikli/  (642 videos)
    │   └── Topal/     (525 videos)
    └── outputs/
        ├── deeplabcut/            # Uploaded CSVs from local
        ├── mmpose/                # Uploaded CSVs from local
        └── colab_results/         # Training outputs from Colab
```

---

## 📊 Dataset

- **Total Videos**: 1167
  - **Healthy (Sağlıklı)**: 642 videos
  - **Lame (Topal)**: 525 videos
- **Labeling**: Folder-based only (no manual keypoint annotation)
- **Video Location**: Google Drive (`cow_single_videos/{Saglikli,Topal}/`)
- **Local Video Path**: `C:\Users\HP\Desktop\Yeni klasör\CowLameness_v15\cow_single_videos\`

### Data Split (in Colab notebook)

- **Training**: 70% (~817 videos)
- **Validation**: 15% (~175 videos)
- **Test**: 15% (~175 videos)

---

## 🚀 Installation

### Prerequisites

- **OS**: Windows 10/11
- **Python**: 3.8 - 3.10 (3.9 recommended)
- **GPU**: NVIDIA GPU with CUDA 11.x (highly recommended for DeepLabCut/MMPose)
- **Package Manager**: [UV](https://astral.sh/uv) (auto-installed by setup scripts)
- **Google Account**: For Google Drive and Colab access

### Step 1: Clone Repository

```powershell
git clone <repository-url>
cd CowLameness
```

### Step 2: Setup DeepLabCut Environment

```powershell
cd DeepLabCut
.\setup_environment.ps1
```

**This will**:
- Install UV package manager (if not present)
- Create `.venv` virtual environment
- Install DeepLabCut 2.3.9 with NumPy 1.23.5 (compatibility)
- Verify installation

### Step 3: Setup MMPose Environment

```powershell
cd ..\MMPose
.\setup_environment.ps1
```

---

## 📖 Usage Workflow

### Phase 1: DeepLabCut Pose Estimation (LOCAL)

**Test Mode** (required before batch processing):

```powershell
cd DeepLabCut
.\.venv\Scripts\Activate.ps1
python process_videos.py --test
```

**Expected Output**: `outputs/cow_0001_DLC_SuperAnimal.csv`

**Batch Mode** (after test approval):

```powershell
python process_videos.py --batch
```

⏱️ **Estimated Time**: 35-60 hours for 1167 videos  
💾 **Output**: 1167 CSV files in `DeepLabCut/outputs/`

### Phase 2: MMPose Pose Estimation (LOCAL)

Same workflow as DeepLabCut:

```powershell
cd ..\MMPose
.\.venv\Scripts\Activate.ps1
python process_videos.py --test    # Test first
python process_videos.py --batch   # Then batch
```

⏱️ **Estimated Time**: 20-40 hours  
💾 **Output**: 1167 CSV files in `MMPose/outputs/`

### Phase 3: Sync to Google Drive

```powershell
cd ..\sync
python sync_to_drive.py
```

**Uploads**:
- `DeepLabCut/outputs/*.csv` → Drive `/outputs/deeplabcut/`
- `MMPose/outputs/*.csv` → Drive `/outputs/mmpose/`

⚠️ **Note**: Only CSV/H5 files are uploaded (NOT videos - they're already in Drive)

### Phase 4: Training & Analysis (GOOGLE COLAB)

1. Open `Colab_Notebook/Cow_Lameness_Analysis_v18.ipynb` in Google Colab
2. Run all cells sequentially
3. Notebook will:
   - Load pose CSVs from Drive
   - Extract 169 biomechanical features
   - Perform train/val/test split (70/15/15)
   - Run statistical analysis (t-tests)
   - Train Transformer model with 5-fold CV
   - Evaluate on held-out test set
   - Generate explainable AI visualizations
   - Save all outputs to `Drive/outputs/colab_results/`

⏱️ **Estimated Time**: 3-5 hours

### Phase 5: Generate Academic Report (LOCAL)

```powershell
cd report_generation
python generate_report.py
```

**Output**: `outputs/academic_report.md` (and optionally `academic_report.pdf`)

---

## 📈 Results

*(To be updated after implementation)*

### Expected Performance

- **Baseline Target**: >70% accuracy
- **Goal**: >80% accuracy with statistical significance (p < 0.05)

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Per-class performance analysis

---

## 📚 Citation

```bibtex
@misc{cow_lameness_detection_2026,
  title={Automated Lameness Detection in Dairy Cattle using Dual-Framework Pose Estimation},
  author={[Your Name]},
  year={2026},
  note={Deep Learning-based Gait Analysis System}
}
```

### References

1. Mathis, A., et al. (2018). "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning." *Nature Neuroscience*, 21(9), 1281-1289.
2. Ye, S., et al. (2024). "SuperAnimal pretrained pose estimation models for behavioral analysis." *Nature Communications*.
3. Contributors, M. (2020). "OpenMMLab Pose Estimation Toolbox and Benchmark." https://github.com/open-mmlab/mmpose

---

## 🤝 Contributing

This is an academic research project. For questions or collaboration inquiries, please open an issue.

---

## 📄 License

*(To be determined)*

---

# 🇹🇷 Türkçe Dokümantasyon

## Proje Genel Bakış (TR)

Bu proje, süt ineklerinde topallığı yürüyüş analizi yoluyla tespit etmek için **çift çerçeveli pose tahmini** yaklaşımı uygular. Sistem, bireysel inek videolarını işleyerek topallıkla ilişkili biyomekanik anormallikleri tespit eder.

### Temel Özellikler

✅ **Çift Pose Tahmini**: Sağlamlık için DeepLabCut SuperAnimal-Quadruped + MMPose  
✅ **Hibrit Mimari**: Yerel işleme + Bulut analizi (Google Colab)  
✅ **Akademik Titizlik**: 5-kat çapraz doğrulama, istatistiksel testler (t-testleri), açıklanabilir AI  
✅ **Üretime Hazır**: Otomatik toplu işleme, devam edebilme yeteneği, kapsamlı loglama  
✅ **Sıfır Manuel Etiketleme**: Yalnızca klasör tabanlı sınıflandırma kullanır (Sağlıklı/Topal)

---

## Mimari (TR)

Sistem 3 katmanlı yapıda çalışır:

1. **Yerel İşleme (Windows)**
   - DeepLabCut ve MMPose ile pose tahmini
   - Çıktılar (CSV/H5) yerel olarak kaydedilir

2. **Drive Senkronizasyonu**
   - Sadece CSV/H5 dosyaları Drive'a yüklenir
   - Videolar zaten Drive'da (tekrar yüklenmez)

3. **Colab Analizi**
   - Eğitim ve test Google Colab'da
   - Tek altın standart notebook

4. **Yerel Rapor Oluşturma**
   - Akademik rapor yerel bilgisayarda üretilir
   - Colab'dan indirilen çıktıları kullanır

---

## Veri Seti (TR)

- **Toplam Video**: 1167 adet
  - **Sağlıklı**: 642 video
  - **Topal**: 525 video
- **Etiketleme**: Sadece klasör bazlı (manuel keypoint anotasyonu yok)
- **Bölünme**: %70 eğitim, %15 doğrulama, %15 test

---

## Kurulum (TR)

### Gereksinimler

- **İşletim Sistemi**: Windows 10/11
- **Python**: 3.8 - 3.10 (3.9 önerilir)
- **GPU**: NVIDIA GPU (CUDA 11.x) - DeepLabCut/MMPose için şiddetle önerilir
- **Google Hesabı**: Drive ve Colab erişimi için

### Adım 1: Repository'yi Klonlayın

```powershell
git clone <repository-url>
cd CowLameness
```

### Adım 2: DeepLabCut Ortamını Kurun

```powershell
cd DeepLabCut
.\setup_environment.ps1
```

### Adım 3: MMPose Ortamını Kurun

```powershell
cd ..\MMPose
.\setup_environment.ps1
```

---

## Kullanım İş Akışı (TR)

### Faz 1: DeepLabCut Pose Tahmini (YEREL)

**Test Modu** (toplu işlemeden önce zorunlu):

```powershell
cd DeepLabCut
.\.venv\Scripts\Activate.ps1
python process_videos.py --test
```

**Toplu İşleme Modu** (test onayından sonra):

```powershell
python process_videos.py --batch
```

⏱️ **Tahmini Süre**: 1167 video için 35-60 saat

### Faz 2: MMPose Pose Tahmini (YEREL)

DeepLabCut ile aynı iş akışı.

### Faz 3: Google Drive'a Senkronizasyon

```powershell
cd ..\sync
python sync_to_drive.py
```

### Faz 4: Eğitim ve Analiz (GOOGLE COLAB)

`Colab_Notebook/Cow_Lameness_Analysis_v18.ipynb` dosyasını Google Colab'da açın ve tüm hücreleri çalıştırın.

### Faz 5: Akademik Rapor Oluşturma (YEREL)

```powershell
cd report_generation
python generate_report.py
```

---

## Sonuçlar (TR)

*(Uygulama sonrası güncellenecek)*

### Beklenen Performans

- **Hedef**: >%70 doğruluk
- **İdeal**: >%80 doğruluk (istatistiksel anlamlılık ile)

---

## İletişim

Sorularınız için issue açabilirsiniz.

---

**Last Updated**: 2026-01-01  
**Version**: 1.0
