# System Architecture Specification
# Deepfake Sentinel: Dual-Domain Neural XAI & Multimodal Forensic Platform

## 1. Architectural Philosophy
Deepfake Sentinel addresses the core failure mode of modern synthetic media detectors: **overfitting to specific generator artifacts**. Generative models (StyleGAN, Diffusion, Face Swaps) continuously evolve, causing single-backbone deep neural networks trained strictly on spatial pixels to fail when presented with novel architectures or basic post-processing compressions.

Deepfake Sentinel implements an orthogonal, multi-domain defense-in-depth architecture:
1. **Spatial Representation**: Captures semantic coherence and micro-texture blending seams using Vision Transformers (`ViT-B/16`) and convolutional feature pyramids (`EfficientNet-B3`).
2. **Frequency Domain Optics**: Employs 2D Fast Fourier Transform (`2D FFT`) log-magnitude power spectra to detect high-frequency energy decay anomalies and transposed convolution checkerboard artifacts.
3. **Physical & Biometric Invariants**: Evaluates natural biological constraints (bilateral corneal specular reflection symmetry and pupil circularity) that generative models fail to render realistically.
4. **Multimodal LLM Semantic Reasoner**: Leverages Google Gemini Vision as an expert forensic judge to pinpoint anomalous regions (hands, ears, hair, background perspective) and generate human-readable explanations.
5. **Continuous Active Learning Memory**: Provides a zero-latency feedback calibration engine keyed by 64-bit discrete cosine transform perceptual hashes (`pHash`) and SHA-256 signatures.

---

## 2. High-Level System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CYBERNETIC WEBGL HUD PRESENTATION                     │
│               (HTML5 + Vanilla ES6 + Tailwind CSS + Three.js)               │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Single Image Scan   │  Temporal Video Scan  │  Live Biometric Webcam     │
│  • Specimen Benchmarks │  FGSM Adversarial HUD │  Model Visualizer Modal    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP REST / Multipart-FormData
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FASTAPI BACKEND SERVICE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Pydantic v2 Payload Validation & CORS Middleware                         │
│  • Multi-Threading Daemon Engine (CPU / CUDA Auto-Targeting)                │
│  • Dynamic Google Gemini API Key Injection & Session Passthrough           │
└──────────┬───────────────────────────┬───────────────────────────┬──────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ 1. SPATIAL DOMAIN   │     │ 2. FREQUENCY DOMAIN │     │ 3. BIOMETRIC DOMAIN │
├─────────────────────┤     ├─────────────────────┤     ├─────────────────────┤
│ • MTCNN 5-Pt Align  │     │ • Grayscale 2D FFT  │     │ • Bilateral Iris RoI│
│ • ViT-B/16 Global   │     │ • Log Power Spectra │     │ • Corneal Glints    │
│ • EfficientNet-B3   │     │ • Azimuthal Profile │     │ • Pupil Circularity │
│ • Dual-Crop TTA     │     │ • High-Freq Slope   │     │ • Symmetry Distance │
└──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: MULTI-MODAL DECISION FUSION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Bayesian Weighting: 55% Gemini + 20% ViT + 15% EffNet + 5% FFT + 5% Iris │
│ • Decision Boundary Threshold: 0.45 Fake Probability                        │
│ • Epistemic Confidence Interval & Uncertainty Estimation                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 STAGE 5: EXPLAINABILITY & PROVENANCE AUDIT                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Grad-CAM Attention Heatmap Projection on ViT Self-Attention Tokens        │
│ • Primary Anomaly Region Localization (Pinpoint Suspect Box & Label)        │
│ • C2PA Cryptographic Signature & Google SynthID Watermark Parsing           │
│ • 64-bit DCT pHash Ground-Truth Active Learning Memory Cache               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Breakdown

### 3.1 Spatial Feature Extraction Subsystem (`backend/model.py`)
- **MTCNN Landmark Alignment**: Detects 5 canonical facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner). Rotates and scales facial regions to an affine-aligned $256 \times 256$ crop.
- **Vision Transformer (`google/vit-base-patch16-224`)**: Operates on global uncropped context to inspect lighting vectors, hair boundaries, background warps, and clothing seams. Processes $16 \times 16$ pixel non-overlapping patches with self-attention.
- **EfficientNet-B3 Backbone**: Processes normalized aligned facial crops. Employs inverted residual blocks with squeeze-and-excitation attention to detect micro-texture blending boundaries.
- **Test-Time Augmentation (TTA)**: Evaluates canonical orientation plus horizontal mirror flips, computing invariant feature embeddings.

### 3.2 Frequency Domain Analysis Subsystem (`backend/model.py`)
- **2D Fast Fourier Transform**: Converts spatial intensity matrices $I(x,y)$ into complex spatial frequency spectra $F(u,v)$.
- **Log-Magnitude Normalization**: Computes $\log(1 + |F(u,v)|)$ with center DC frequency shift.
- **Azimuthal Radial Averaging**: Integrates energy across concentric radial frequencies $r = \sqrt{u^2 + v^2}$. Real camera sensors exhibit natural $1/f^\alpha$ power decay; GAN and diffusion upsampling filters exhibit anomalous high-frequency energy spikes.

### 3.3 Biometric Optics & Corneal Specularity Subsystem (`backend/model.py`)
- **Siamese Iris Isolation**: Isolates ocular bounding boxes from MTCNN landmarks.
- **Pupil Ellipticity Ratio**: Computes major vs. minor axis ratio of pupil contours. Real pupils retain circular geometry under non-pathological conditions; generative algorithms warp pupil perimeters.
- **Corneal Specular Glint Vector**: Measures catchlight ray angles relative to the iris center. Real portraits feature symmetrical corneal highlights from shared scene illumination; deepfakes show divergent catchlights.

### 3.4 Multimodal Gemini Vision Reasoner (`backend/explainability.py`)
- Transmits full-resolution specimens to Google Gemini Vision models (`gemini-3.1-flash-lite`, `gemini-2.5-flash`, `gemini-3.5-flash`).
- Prompts extract structured forensic breakdowns:
  - 1-line plain-language scene description.
  - Authoritative classification (`REAL` vs. `FAKE`).
  - Pinpointed primary anomaly region (e.g., *Ear-Hair Interface*, *Corneal Glint Divergence*, *Skin Micro-Texture Smoothing*).
  - Normalized bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$.

### 3.5 Active Learning & Provenance Subsystem (`backend/main.py`)
- **DCT Perceptual Hash (`pHash`)**: Computes 64-bit frequency hash of resized 32x32 luminance arrays. Hashes differing by Hamming distance $\le 4$ match under compression, resizing, and color jitter.
- **SHA-256 Byte Verification**: Exact byte hash ensures tamper-evident specimen identity.
- **Cryptographic Provenance**: Scans byte manifests for C2PA JUMBF chunks and Google SynthID watermark markers.

---

## 4. Dataflow Lifecycles

### 4.1 Image Forensic Analysis Flow
1. **Intake**: Client submits multipart image with optional client-side Gemini API key and FGSM toggle.
2. **Hash Lookup**: Server queries `backend/data/feedback_memory.json` by SHA-256 and pHash. If previously calibrated, overrides inference with ground truth.
3. **Face Alignment**: MTCNN checks for facial landmarks; falls back to center crop if non-facial.
4. **Spatial Scoring**: Forward pass through ViT-B/16 and EfficientNet-B3.
5. **Spectral Scoring**: Computes 2D FFT, radial decay curve, and frequency fake metric.
6. **Biometric Scoring**: Computes corneal glint asymmetry and pupil circularity.
7. **Gemini Consultation**: If API key is present, requests visual forensic judgment.
8. **Bayesian Decision Fusion**: Blends scores according to calibrated weights.
9. **Grad-CAM Extraction**: Computes gradient backprop on ViT feature tokens.
10. **JSON Delivery**: Returns comprehensive verdict, confidence, explainability heatmaps, and audit certificate.

### 4.2 Temporal Video Forensic Flow
1. **Stream Sampling**: Samples video stream at 5 fps across up to 30 frames.
2. **Per-Frame Alignment**: Tracks facial bounding boxes across frames.
3. **Temporal Inconsistency Tracking**: Computes frame-to-frame feature variance and eye-blink interval cadence.
4. **Aggregation**: Aggregates anomalous frame ratio and highlights timestamp intervals with detected deepfake artifacts.
