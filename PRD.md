# Product Requirements Document (PRD)
## Deepfake Sentinel: Trustworthy & Explainable AI Forensic Platform

---

### Document Information
- **Project Name**: Deepfake Sentinel (v2.0)
- **Document Version**: 2.0.0
- **Status**: Approved / In-Development
- **Target Audience**: Core AI Engineers, Research Reviewers, Academic Evaluators, Forensic Analysts

---

## 1. Executive Summary & Problem Context

### 1.1 The "99.7% False Positive" Domain Gap Pathology
A common failure mode in deepfake detection models is catastrophic overfitting to dataset-specific artifacts (compression profiles, synthetic sensor noise, specific color grading, and fixed camera resolutions). When deployed to live uncompressed webcam feeds or real-world uncurated imagery, baseline classifiers undergo extreme **domain shift**—erroneously classifying clean, real human faces as **99.7% Fake**.

### 1.2 Product Vision
**Deepfake Sentinel** is an open-architecture, academic-grade, and enterprise-ready deepfake verification framework. The system eliminates domain shift by pairing **landmark-guided facial extraction (MTCNN)**, **aggressive artifact-simulation data augmentation**, **dual-domain classification (Spatial Vision Transformer/EfficientNet + 2D FFT Spectral Optics)**, and **Explainable AI (Grad-CAM)** to visually justify every forensic verdict.

```
                                  [ Input Media ]
                             (Image / Live Webcam / MP4)
                                         │
                                         ▼
                            [ MTCNN Facial Landmarking ]
                          (Margin 40px, Scale Alignment)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ Spatial Domain Pipeline ]               [ Spectral Domain Pipeline ]
      ┌─────────────────────────────┐           ┌─────────────────────────────┐
      │ • ViT-B/16 / EfficientNet   │           │ • 2D Fast Fourier Transform │
      │ • Test-Time Augmentation    │           │ • Log-Magnitude Spectrum    │
      │ • Grad-CAM Token Heatmap    │           │ • High-Freq Anomaly Ratio   │
      └──────────────┬──────────────┘           └──────────────┬──────────────┘
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          ▼
                             [ Forensic Fusion Engine ]
                                          │
                                          ▼
                       [ Glass-Box Diagnostic Output ]
               (Prediction + Confidence + Heatmap + FFT + Audit)
```

---

## 2. Codebase & Directory Structure Analysis

A comprehensive audit of the workspace reveals a clean, decoupled full-stack architecture:

```
X:\AIMINI
├── .gitignore                    # Credential and artifact isolation
├── kaggle.json                   # Kaggle API credentials (secured via gitignore)
├── README.md                     # Architecture overview & forensic spec
├── requirements.txt              # Core PyTorch, Torchvision, FastAPI, ViT, Grad-CAM dependencies
├── dataset/
│   └── Dataset/
│       ├── fake/                 # Curated manipulated facial frames (PNG format)
│       └── real/                 # Curated authentic facial frames (PNG format)
├── backend/
│   ├── main.py                   # FastAPI application, REST endpoints, CORS & static file mount
│   ├── model.py                  # Vision Transformer (ViT-B/16) / Spatial Classifier with TTA
│   ├── preprocessor.py           # MTCNN landmark-aware tight face extraction
│   ├── xai.py                    # PyTorch Grad-CAM interpretability wrapper & reshape transforms
│   ├── frequency.py              # 2D Fast Fourier Transform (FFT) & spectral energy anomaly calculator
│   └── robustness.py             # Fast Gradient Sign Method (FGSM) adversarial stress testing
└── frontend/
    └── static/
        └── index.html            # Dark Obsidian Luxe WebGL 3D dashboard (Three.js + Tailwind + Lucide)
```

### Module Responsibilities & Audit Findings:
1. **`backend/preprocessor.py`**:
   - Uses `facenet_pytorch.MTCNN` with a 40px margin and 256x256 target resolution.
   - Solves facial cutoff and misalignment issues inherent to traditional OpenCV Haar cascades.
2. **`backend/model.py`**:
   - Employs a pre-trained Vision Transformer (`dima806/deepfake_vs_real_image_detection`) evaluated with Test-Time Augmentation (TTA) via horizontal flips.
   - Can also support modified `EfficientNet-B0` / `Xception` backbones with custom BCEWithLogits heads.
3. **`backend/xai.py`**:
   - Implements PyTorch Grad-CAM hooks over the final layer norm / convolutional feature maps.
   - Custom `reshape_transform` converts 1D ViT patch tokens back into 2D spatial feature representations $(B, C, H, W)$.
4. **`backend/frequency.py`**:
   - Computes 2D Fast Fourier Transform (FFT) shifted log-magnitude power spectra.
   - Calculates high-frequency energy ratio to spot GAN deconvolution grids and diffusion upsampling checkerboard patterns.
5. **`backend/robustness.py`**:
   - Real-time FGSM adversarial perturbation engine to verify model robustness under adversarial evasion noise ($\epsilon \in [0.01, 0.15]$).
6. **`backend/main.py`**:
   - High-throughput asynchronous FastAPI server exposing `/predict`, `/health`, and `/samples` with in-memory Base64 image serialization.
7. **`frontend/static/index.html`**:
   - Cybernetic 3D particle background (Three.js), live webcam streaming, upload modal, FGSM attack slider, and one-click JSON Forensic Audit Certificate exporter.

---

## 3. Product Objectives & Target Metrics

| Metric Category | Target Requirement | Benchmark Success Metric |
| :--- | :--- | :--- |
| **Face Extraction** | Landmark-aligned facial cropping across varied yaw, pitch, and lighting conditions. | **> 95%** extraction success rate without jawline truncation. |
| **Webcam Generalization** | Elimination of domain shift false positives on raw, uncompressed camera streams. | **< 15%** False Positive Rate (FPR) on live webcam streams (targeting $>85\%$ Real confidence). |
| **Model Accuracy** | Cross-dataset discrimination on public benchmarks (FaceForensics++, 140k Real/Fake). | **> 95%** Detection AUC-ROC and F1-Score. |
| **Explainability (XAI)** | Grad-CAM activation maps must localize facial boundary blending, eye artifacts, or mouth warps. | Saliency maps focused $>80\%$ within facial perimeter rather than background noise. |
| **Inference Latency** | Full pipeline processing (MTCNN + ViT/EfficientNet + FFT + GradCAM). | **< 250 ms** on CUDA GPU / **< 800 ms** on modern 8-core CPU. |
| **Adversarial Resilience** | Model retention of true classification under bounded perturbation ($\epsilon = 0.03$). | **> 80%** retention of true positive rate under FGSM attacks. |

---

## 4. Functional Requirements (FR)

### FR-1: Robust Facial Extraction & Landmarking
- **FR-1.1**: The system must accept arbitrary resolution images (`.jpg`, `.png`, `.webp`) and video frames (`.mp4`).
- **FR-1.2**: Must use Multi-Task Cascaded Convolutional Networks (MTCNN) to detect 5-point facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner).
- **FR-1.3**: Automatic fall-back to center-weighted aspect-ratio cropping if no face is detected with threshold $>0.90$.

### FR-2: Domain-Shift Invariant Augmentation Engine (Training Pipeline)
To eliminate the 99.7% webcam false positive bug, training data pipelines must enforce aggressive domain-generalization augmentations:
- **FR-2.1 (Gaussian Blur & Downsampling)**: Random blur kernel ($k \in [3, 7]$) to neutralize sensor sharpness memorization.
- **FR-2.2 (JPEG Compression Simulation)**: Dynamic compression artifacts (quality factor $Q \in [40, 95]$).
- **FR-2.3 (Photometric Jitter)**: Random brightness ($\pm 0.3$), contrast ($\pm 0.3$), saturation ($\pm 0.3$), and hue ($\pm 0.1$).
- **FR-2.4 (Random Erasing / Cutout)**: Random rectangular occlusion ($p=0.2$, area ratio $[0.02, 0.2]$) to enforce global morphological reasoning over localized pixel memorization.

### FR-3: Dual-Domain Classification Engine
- **FR-3.1 (Spatial Domain)**: Pass normalized $224 \times 224$ (or $256 \times 256$) facial tensor into a deep neural backbone (ViT-B/16 or EfficientNet-B0 with BCEWithLogitsLoss).
- **FR-3.2 (Test-Time Augmentation)**: Average spatial predictions across canonical and horizontally flipped orientations to reduce variance.
- **FR-3.3 (Spectral Domain)**: Execute 2D FFT on grayscale facial matrix, compute shifted log-magnitude power spectrum:
  $$\mathcal{M}(u, v) = 20 \log_{10} (|\mathcal{F}\{\mathcal{I}\}(u, v)| + \epsilon)$$
- **FR-3.4 (Spectral Anomaly Scoring)**: Quantify high-frequency power distribution outside center radius $R = \min(H, W)/4$. High-frequency anomalies indicate GAN upsampling checkerboards or diffusion autoencoder residuals.

### FR-4: Explainable AI (XAI) & Grad-CAM Integration
- **FR-4.1**: Compute gradients of target class logit $y^c$ with respect to feature activation maps $A^k$:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
- **FR-4.2**: Generate localized coarse localization map:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **FR-4.3**: Normalize, apply `COLORMAP_JET` / `COLORMAP_VIRIDIS`, and blend with the original facial crop at $0.5$ alpha transparency.

### FR-5: Adversarial Evasion Lab (FGSM)
- **FR-5.1**: Provide on-the-fly Fast Gradient Sign Method generation:
  $$\tilde{x} = x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(\theta, x, y)\right)$$
- **FR-5.2**: Enable interactive UI perturbation control with slider $\epsilon \in [0.01, 0.15]$ to stress-test classifier stability.

### FR-6: Multi-Specimen UI & Diagnostic Visualization
- **FR-6.1**: Composite side-by-side diagnostic visualization rendering:
  1. *Aligned Facial Crop*
  2. *Grad-CAM Explainability Overlay*
  3. *2D FFT Spectral Log-Magnitude Heatmap*
- **FR-6.2**: Provide three execution modes: **Image Upload**, **Live Browser Webcam Stream**, and **Curated Test Specimens**.
- **FR-6.3**: 1-Click exportable **JSON Forensic Audit Certificate** containing timestamp, confidence scores, spectral ratio, attack status, and Base64 diagnostic hashes.

---

## 5. Non-Functional Requirements (NFR)

### NFR-1: Reliability & Robustness
- **NFR-1.1**: The backend must gracefully catch and handle non-face inputs, corrupted image headers, or invalid multipart payloads without crashing (HTTP 400/500 with descriptive JSON error payload).
- **NFR-1.2**: Thread-safe model inference supporting asynchronous concurrent prediction requests.

### NFR-2: Memory & Data Privacy
- **NFR-2.1**: **Zero Disk-Footprint**: All uploaded images and webcam frames must be processed in-memory using `io.BytesIO` buffers. No persistent raw biometric data saved to disk.
- **NFR-2.2**: Strict gitignore policy ensuring dataset archives, training checkpoints, and API credentials (`kaggle.json`) are excluded from version control.

### NFR-3: Portability & Cross-Platform Compatibility
- **NFR-3.1**: Platform-agnostic device selection (automatic detection of `cuda`, `mps`, or `cpu`).
- **NFR-3.2**: Zero frontend build tooling required: standard Tailwind CDN, Vanilla ES6 JavaScript, Lucide icons, and Three.js running natively in modern web browsers.

---

## 6. Implementation & Roadmap Phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PROJECT PHASES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Ingestion & Augmentation Pipeline                              │
│   • Set up Kaggle 140k dataset dataloaders                              │
│   • Implement MTCNN crop & landmark alignment                           │
│   • Apply aggressive blur, JPEG, and color jitter augmentations        │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 2: Backbone Optimization & Domain Calibration                     │
│   • Benchmark ViT-B/16 vs. EfficientNet-B0                              │
│   • Train with CosineAnnealingLR & AdamW                                │
│   • Optimize on validation loss & cross-dataset validation              │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 3: Dual-Domain Forensic Fusion & XAI Engine                       │
│   • Grad-CAM backward hook registration                                 │
│   • 2D FFT spectral high-frequency ratio calculator                     │
│   • FGSM adversarial robustness perturbation engine                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Phase 4: Full-Stack Integration & Webcam Validation                     │
│   • FastAPI endpoints (/predict, /health, /samples)                     │
│   • Obsidian Luxe 3D WebGL Dashboard                                    │
│   • Live webcam verification: verify real faces output < 15% Fake      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Sign-off & Revision History

| Version | Date | Author / Agent | Changes / Notes |
| :--- | :--- | :--- | :--- |
| `1.0.0` | Initial Sprint | AI Lead | Baseline CNN classifier with Haar Cascades (suffered 99.7% webcam false positive). |
| `2.0.0` | Current Release | Deepfake Sentinel Team / Anti Gravity | Dual-Domain architecture, MTCNN alignment, ViT/EfficientNet with TTA, 2D FFT, Grad-CAM XAI, FGSM Lab, and 3D WebGL Dashboard. |
