#!/usr/bin/env python3
"""Add inceleme9.md enhancements to v29 notebook."""

import json
from pathlib import Path

# Load v29
notebook_path = Path("Cow_Lameness_Analysis_v29.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Enhancement 1: Update header with stronger clinical focus
header_cell = nb['cells'][0]
header_cell['source'] = [
    "# 🐄 V29 - CLINICAL LAMENESS ESTIMATION (Gold Standard)\n\n"
    "## Ana Problem Tanımı\n\n"
    "**Hedef:** İneklerde topallık (lameness) video kayıtlarından **animal-level ordinal tahmin** yapmak.\n\n"
    "**Kritik Ayrımlar:**\n"
    "- ❌ Frame-level prediction DEĞİL\n"
    "- ❌ Clip-level prediction DEĞİL\n"
    "- ✅ Animal-level prediction (klinik olarak anlamlı)\n\n"
    "---\n\n"
    "## Klinik Zaman Penceresi\n\n"
    "**Gereksinim:** Her örnek **en az 2 yürüyüş döngüsü** (~6-10 saniye) içermeli.\n\n"
    "**Gerekçe:**\n"
    "- Tek yürüyüş döngüsü = ~2-3 saniye (sığırlarda)\n"
    "- Asimetri ve periyodik bozukluk tespiti için minimum 2 döngü gerekli\n"
    "- Video ≠ yürüyüş epizodu (duruş ve dönüş anları hariç tutulur)\n\n"
    "---\n\n"
    "## Uygulama Disiplini Garantileri\n\n"
    "| Nokta | Garanti Mekanizması |\n"
    "|-------|---------------------|\n"
    "| VideoMAE CLS | `extract_cls_embedding()` izole fonksiyon + assertion |\n"
    "| Temporal Mask | Custom `StrictMaskedTransformer` with `-inf` guarantee |\n"
    "| Clip Ordering | `assert_temporal_order()` per batch |\n"
    "| CORAL | `coral_encode_strict()` - raw label ASLA loss'a girmez |\n"
    "| Subject Split | Yapısal sıralama: **Cell 4** split → **Cell 11** clip |\n\n"
    "---\n\n"
    "## Akademik Gerekçeler\n\n"
    "**Q: Why is VideoMAE frozen?**\n"
    "> \"VideoMAE is frozen to preserve pretrained spatiotemporal representations, "
    "while disease-specific temporal dynamics are modeled explicitly at the clip level. "
    "Fine-tuning on small medical datasets risks overfitting and losing generalization.\"\n\n"
    "**Q: Why external temporal modeling?**\n"
    "> \"VideoMAE operates on fixed 16-frame clips (~0.5s). Gait assessment requires "
    "observing patterns across multiple clips (6-10 seconds). The Temporal Transformer "
    "captures long-range dynamics beyond VideoMAE's temporal scope.\"\n\n"
    "**Q: What is the unit of prediction?**\n"
    "> \"The proposed model predicts the **ordinal lameness score at the animal level** "
    "based on a short walking video segment. This aligns with veterinary practice, where "
    "lameness is assessed for the animal as a whole, not individual frames.\"\n\n"
    "---\n\n"
    "## Klinik Skor Mapping (CORAL → Clinic)\n\n"
    "| CORAL | Class | Türkçe | Clinical Finding | Action |\n"
    "|-------|-------|--------|------------------|--------|\n"
    "| 0 | Healthy | Sağlıklı | Normal gait | Routine |\n"
    "| 1 | Mild | Hafif | Head bob, shortened stride | Monitor |\n"
    "| 2 | Moderate | Orta | Asymmetric gait, weight shifting | Vet required |\n"
    "| 3 | Severe | Şiddetli | Arched back, reluctance to walk | URGENT Vet |"
]

# Enhancement 2: Add animal-level aggregation cell after evaluation (before final cell)
animal_aggregation_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Animal-Level Aggregation Strategy\n\n"
        "**Problem:** Aynı hayvana ait birden fazla video olabilir.\n\n"
        "**Çözüm:** Animal-level aggregation ile nihai tahmin yapılır.\n\n"
        "### Inference Stratejisi\n\n"
        "```python\n"
        "def animal_level_inference(model, animal_videos, aggregation='mean'):\n"
        '    """\n'
        "    Animal-level prediction by aggregating multiple video predictions.\n"
        "    \n"
        "    Args:\n"
        "        animal_videos: List of videos for the same animal\n"
        "        aggregation: 'mean', 'median', or 'worst_case'\n"
        "    \n"
        "    Returns:\n"
        "        animal_level_score: Single ordinal score for the animal\n"
        '    """\n'
        "    video_scores = []\n"
        "    \n"
        "    for video in animal_videos:\n"
        "        clips, mask = extract_clips(video)\n"
        "        logits, _ = model(clips, mask)\n"
        "        score = coral_loss.predict(logits)\n"
        "        video_scores.append(score)\n"
        "    \n"
        "    if aggregation == 'mean':\n"
        "        animal_score = round(np.mean(video_scores))\n"
        "    elif aggregation == 'median':\n"
        "        animal_score = int(np.median(video_scores))\n"
        "    elif aggregation == 'worst_case':\n"
        "        animal_score = max(video_scores)  # Highest lameness\n"
        "    \n"
        "    return animal_score\n"
        "```\n\n"
        "### Training Stratejisi\n\n"
        "**Kural:** Her epoch'ta aynı `animal_id`'den tek örnek kullanılır.\n\n"
        "**Gerekçe:**\n"
        "- Aynı hayvanın çoklu videoları model'i bias edebilir\n"
        "- Animal-level generalization garantisi için gerekli\n\n"
        "**Not:** Mevcut implementasyonda her video ayrı örnek olarak kullanılıyor. "
        "Production sistemde animal-level sampling uygulanmalıdır."
    ]
}

# Insert before the last cell (verification cell)
nb['cells'].insert(-1, animal_aggregation_cell)

# Save enhanced v29
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✅ v29 enhanced with inceleme9.md improvements:")
print("   - Updated header with clinical focus")
print("   - Added clinical time window definition") 
print("   - Added animal-level aggregation strategy")
print("   - Strengthened academic justifications")
print("   - Clarified prediction unit")
print(f"\n📄 Saved to: {notebook_path}")
