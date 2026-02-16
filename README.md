# 🧠 BrainGuard AI
### Comprehensive Brain MRI Analysis for Rural Healthcare in India
> **MedGemma Impact Challenge 2026** — Google Research × Kaggle

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-brainguard--ai.streamlit.app-0369a1?style=for-the-badge)](https://brainguard-ai.streamlit.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training_Notebook-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/shravanirs4/brainguard-ai-medgemma-testing)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)](https://brainguard-ai.streamlit.app)

---

## 🚨 The Problem

Every year, **810,000 people** in India suffer a stroke. Brain MRI can detect warning signs months before it happens — but:

- 💸 One MRI scan costs **Rs 15,000** (unaffordable for most rural families)
- 🏥 **25,000+ Primary Health Centers** have zero neurologists
- 👥 **900 million rural Indians** have no access to brain health screening

**BrainGuard AI solves this by bringing medical-grade brain analysis to any PHC for just Rs 200 per scan — a 75x cost reduction.**

---

## 🧠 What BrainGuard AI Does

Upload any brain MRI scan → Get a complete neurological assessment in **under 60 seconds**.

| # | Model | What It Detects | Output |
|---|-------|----------------|--------|
| 1 | **Brain Age Prediction** | How fast your brain is aging | Brain age gap (years) |
| 2 | **WM Lesion Detection** | White matter damage | Volume in cm³, severity |
| 3 | **Hippocampal Volume** | Memory center shrinkage | Volume, percentile rank |
| 4 | **Cortical Atrophy** | Brain cortex thinning | Thickness mm, atrophy score |
| 5 | **Silent Stroke Detection** | Past mini-strokes | Count, estimated location |
| 6 | **Stroke Risk Assessment** | Future stroke probability | 5-year and 10-year risk % |
| 7 | **Brain Tumor Screening** | Abnormal growths | Clear / Detected, confidence % |

All 7 models feed into **Google Gemma-2-2B (HAI-DEF)** which generates a complete clinical report in English and Hindi.

---

## 🤖 Google Gemma Integration (HAI-DEF)

BrainGuard AI uses **Google Gemma-2-2B** from the Health AI Developer Foundations collection as the core clinical synthesis layer:

- ✅ Generates structured radiology-style reports from all 7 model outputs
- ✅ Translates medical findings into plain language for health workers
- ✅ Produces bilingual reports (English + Hindi) for rural accessibility
- ✅ Interprets biomarkers in clinical context — not just listing numbers

---

## ⚡ Quick Start — Try It Now

### Option 1: Live Web App (No Setup!)
👉 **[brainguard-ai.streamlit.app](https://brainguard-ai.streamlit.app)**

1. Open the link
2. Download the sample brain MRI from the app
3. Upload it back → Set age → Click **Analyze**
4. See all 7 results in under 60 seconds!

### Option 2: Run Locally
```bash
git clone https://github.com/Shravaniroyal/brainGuard-ai.git
cd brainGuard-ai
pip install -r requirements.txt
streamlit run app.py
```

### Option 3: View Training Notebook
🔗 **[BrainGuard AI — Kaggle Training Notebook](https://www.kaggle.com/code/shravanirs4/brainguard-ai-medgemma-testing)**

---

## 📁 Repository Structure

```
brainGuard-ai/
├── app.py                    # Main Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # CC BY 4.0
├── sample_brain_normal.png   # Real OASIS-1 sample MRI for testing
└── .streamlit/
    └── config.toml           # Streamlit theme config
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0, 3D CNN |
| Medical Imaging | nibabel, scikit-image, SciPy |
| AI Report Generation | Google Gemma-2-2B (HAI-DEF) |
| Web Interface | Streamlit |
| Dataset | OASIS-1 (235 real MRI scans, ages 18–96) |
| Deployment | Streamlit Community Cloud |
| Input Formats | NIfTI (.nii/.nii.gz), DICOM (.dcm), JPG, PNG |

---

## 📊 Impact Metrics

| Metric | Value |
|--------|-------|
| Cost per scan | **Rs 200** (vs Rs 15,000 traditional) |
| Cost reduction | **75× cheaper** |
| Analysis time | **Under 60 seconds** |
| Models per scan | **7 simultaneous** |
| Training dataset | OASIS-1 (235 real MRI scans) |
| Target deployment | 25,000 PHCs across rural India |
| Population reachable | 900 million rural Indians |

---

## 📓 Training Notebook on Kaggle

Full model training code, data preprocessing, and Gemma integration:

**🔗 [View on Kaggle](https://www.kaggle.com/code/shravanirs4/brainguard-ai-medgemma-testing)**

Covers:
- OASIS-1 dataset loading and preprocessing
- 3D CNN architecture design and training
- All 7 model implementations
- Google Gemma-2-2B integration for clinical reports

---

## 👩‍💻 Developer

**Shravani** — Solo Developer
Built end-to-end for the MedGemma Impact Challenge 2026.

---

## 📄 License

[CC BY 4.0](./LICENSE) — Creative Commons Attribution 4.0 International

---

## 🏆 Competition Links

| Resource | Link |
|----------|------|
| 🌐 Live Demo | [brainguard-ai.streamlit.app](https://brainguard-ai.streamlit.app) |
| 💻 GitHub | [github.com/Shravaniroyal/brainGuard-ai](https://github.com/Shravaniroyal/brainGuard-ai) |
| 📓 Kaggle Notebook | [kaggle.com/code/shravanirs4/brainguard-ai-medgemma-testing](https://www.kaggle.com/code/shravanirs4/brainguard-ai-medgemma-testing) |

---
*BrainGuard AI — Medical-grade brain analysis. Affordable. Accessible. Ready to save lives.*