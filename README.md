# 🛡️ DEEPFAKE SENTINEL (v3.1)
### Dual-Domain Neural XAI, Biometric Iris HUD & Multimodal Gemini Forensic Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface)](https://huggingface.co/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.5_Flash_Vision-8E75B2.svg?logo=google)](https://ai.google.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black.svg?logo=three.js)](https://threejs.org/)
[![Chart.js](https://img.shields.io/badge/Chart.js-v4.4-FF6384.svg?logo=chartdotjs)](https://www.chartjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🔬 Executive Overview
**Deepfake Sentinel** is a state-of-the-art synthetic media detection and digital provenance verification system engineered to counter modern generative deepfakes (GANs, Diffusion models, Face Swaps). 

Unlike conventional single-backbone black-box classifiers that struggle on cross-generator generalization, Deepfake Sentinel employs a **multi-modal fusion architecture** integrating **Vision Transformers (ViT-B/16)**, **EfficientNet-B3 (399k)**, **2D Fast Fourier Transform (FFT) optics**, **biometric iris/pupil corneal specular physics**, **C2PA & SynthID provenance verification**, and **Google Gemini Multimodal Vision AI reasoning**.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  │   Input Image / Video / Live Webcam   │
                                  +---------------------------------------+
                                                      │
                                            [MTCNN Face Alignment]
                                                      │
       ┌───────────────────────┬──────────────────────┴───────────────────────┬───────────────────────┐
       ▼                       ▼                                              ▼                       ▼
┌───────────────┐     ┌──────────────────┐                           ┌─────────────────┐     ┌─────────────────┐
│ Spatial Model │     │ Frequency Optics │                           │  Biometric HUD  │     │ Provenance Core │
│ ViT-B/16 +    │     │ 2D Log-FFT Power │                           │ Iris Symmetry & │     │ C2PA Manifest & │
│ EfficientNet  │     │ Azimuthal Decay  │                           │ Corneal Glints  │     │ SynthID Parsing │
└───────┬───────┘     └────────┬─────────┘                           └────────┬────────┘     └────────┬────────┘
        │                      │                                              │                       │
        └──────────────────────┼──────────────────────┬───────────────────────┴───────────────────────┘
                               │                      │
                               ▼                      ▼
                    ┌─────────────────────┐┌─────────────────────┐
                    │ Multi-Modal Bayesian││  Active Learning    │
                    │   Fusion Engine     ││  Feedback Memory    │
                    └──────────┬──────────┘└──────────┬──────────┘
                               │                      │
                               ▼                      ▼
                    ┌────────────────────────────────────────────┐
                    │       Google Gemini Vision Reasoner        │
                    │ (1-Line Description + Visual XAI Verdict)  │
                    └─────────────────────┬──────────────────────┘
                                          │
                                          ▼
                    ┌────────────────────────────────────────────┐
                    │ Dark Obsidian Luxe 3D WebGL Dashboard HUD  │
                    └────────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

1. **Dual-Domain Neural Classification**:
   - **Spatial Domain**: Vision Transformer (`ViT-B/16`) with Test-Time Augmentation (TTA) + `EfficientNet-B3` (399k weights).
   - **Frequency Domain**: 2D Fast Fourier Transform (FFT) log-magnitude power spectrum highlighting GAN deconvolution checkerboard patterns and high-frequency energy decay anomalies.
2. **Biometric Iris & Corneal Specularity HUD**:
   - Extracts bilateral iris RoIs and computes pupil ellipticity circularity.
   - Measures corneal specular reflection asymmetry across bilateral light source vectors to catch impossible generative lighting.
3. **Multimodal Google Gemini Forensic Vision**:
   - Directly feeds the visual face specimen to Gemini Vision to provide a concise 1-line scene/subject description and authoritative Real vs. Fake judgment with physical reasoning.
4. **Active Human-in-the-Loop Feedback Memory**:
   - 64-bit DCT perceptual hash (`pHash`) memory engine allowing instant verdict override and continuous model memorization.
5. **Interactive Model Visualizations (Chart.js)**:
   - Live modal popup rendering Multi-Modal Feature Weights, 2D FFT Radial Decay Curves, Biometric Iris Symmetry Radar, and ROC-AUC Confusion Matrices.
6. **Provenance & Cryptographic Metadata**:
   - Scans for Coalition for Content Provenance and Authenticity (C2PA) manifests, Google SynthID watermarks, and EXIF camera sensor stamps.
7. **Adversarial Robustness Evaluation**:
   - Fast Gradient Sign Method (FGSM) real-time perturbation engine with interactive $\epsilon$ slider.
8. **Dark Obsidian Luxe 3D Interface**:
   - Three.js WebGL particle field reacting to scan verdicts (Idle $\to$ Cyan Scanning $\to$ Emerald Authentic $\to$ Crimson Synthetic).
   - 1-Click exportable JSON Forensic Audit Certificate.
   - Built-in Curated Benchmark Specimen Gallery.

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/HarshvShah/Deepfake-Sentinel.git
cd Deepfake-Sentinel
```

### 3. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your Google AI Studio Gemini API Key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Launch the Application
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
```
Open your browser and navigate to **http://127.0.0.1:8080**.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Dual-domain forensic scan on image (supports FGSM perturbation & Gemini key) |
| `POST` | `/predict-video` | Multi-frame temporal consistency & jitter scan on video uploads |
| `POST` | `/predict-frame` | Low-latency single-frame inference for live webcam stream |
| `POST` | `/feedback` | Submit human ground-truth feedback to active perceptual memory |
| `GET` | `/feedback-stats` | Get active feedback memory count and recorded history |
| `POST` | `/test-gemini-key` | Test Google Gemini API Key validity with live diagnostics |
| `GET` | `/model-metrics` | Telemetry for Chart.js model visualization graphs |
| `GET` | `/samples` | Curated benchmark specimen gallery list |
| `GET` | `/health` | Server status and active compute backend (CPU/CUDA) |

---

## 🛡️ Security & Privacy
- **Credential Protection**: `.env` and `kaggle.json` are excluded from version control via `.gitignore`.
- **Memory Safety**: In-memory buffer processing without unencrypted disk caching of user uploads.
- **Client-Side Storage**: Gemini API keys can be provided dynamically in the browser and stored locally in browser `localStorage`.

---

## 📄 License
Distributed under the [MIT License](LICENSE).
