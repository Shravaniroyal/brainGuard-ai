# 🧠 NeuroScan AI
### Comprehensive Brain MRI Analysis with 7 AI Models

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://neuroscan-ai.streamlit.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-MedGemma_Challenge-blue)](https://kaggle.com/competitions/med-gemma-impact-challenge)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)

---

## 🎯 The Problem
- **810,000** strokes per year in rural India
- **₹15,000** average MRI cost — unaffordable for most
- **25,000** Primary Health Centers with zero neurologists
- **900 million** rural Indians without access to brain health screening

## ✅ The Solution
NeuroScan AI analyzes a brain MRI scan with **7 specialized AI models** in under 60 seconds — for just **₹200** (75x cheaper).

---

## 🤖 7 AI Models

| # | Model | Technology | Output |
|---|-------|-----------|--------|
| 1 | Brain Age Prediction | 3D CNN (PyTorch) | Predicted age, brain age gap |
| 2 | WM Lesion Detection | Image processing | Volume (cm³), severity |
| 3 | Hippocampal Volume | Morphological analysis | Volume, percentile |
| 4 | Cortical Atrophy | Feature extraction | Thickness, atrophy score |
| 5 | Silent Stroke Detection | Rule-based classifier | Count, locations |
| 6 | Stroke Risk Prediction | XGBoost + Framingham | 5-yr & 10-yr risk % |
| 7 | Brain Tumor Screening | Connected components | Detected/Clear, volume |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/neuroscan-ai.git
cd neuroscan-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open: **http://localhost:8501**

---

## 📁 Supported File Formats
- ✅ NIfTI (.nii, .nii.gz) — 3D brain volume
- ✅ JPG / PNG / BMP — 2D brain slice
- ✅ DICOM (.dcm) — medical scanner format

---

## 💡 Tech Stack
- **Frontend**: Streamlit
- **Deep Learning**: PyTorch (3D CNN)
- **Medical AI**: Google Gemma-2-2B
- **Medical Imaging**: nibabel, scikit-image
- **Dataset**: OASIS-1 (235 subjects, real MRI scans)

---

## 📊 Performance
- Brain Age MAE: ~21 years (trained on 235 subjects)
- Successfully tested on OASIS-1 and OASIS Alzheimer's datasets
- All 7 models working in production

---

## 🌍 Impact
| Metric | Value |
|--------|-------|
| Cost per scan | ₹200 (vs ₹15,000) |
| Cost reduction | 75x cheaper |
| Target deployment | 25,000 PHCs |
| Population reached | 900M rural Indians |

---

## 🏆 Competition
Submitted to **MedGemma Impact Challenge 2026** by Google Research on Kaggle.

**Developer:** Shravani  
**Track:** Main Track + Agentic Workflow Prize

---

## 📄 License
CC BY 4.0 — Free to use with attribution.