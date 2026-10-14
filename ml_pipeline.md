# Machine Learning Pipeline Specification
# Deepfake Sentinel: Dual-Domain Neural XAI Platform

## 1. End-to-End Pipeline Overview

The Deepfake Sentinel machine learning pipeline ingests raw digital images, video frames, or live camera streams, transforms them across orthogonal representations (spatial, frequency, biometric, and multimodal semantic), and fuses them into an explainable, calibrated authenticity verdict.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       1. DATA INGESTION & VALIDATION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Memory buffer decoding via PIL & OpenCV (Zero unencrypted disk cache)    │
│  • Perceptual Hash (64-bit DCT pHash) & SHA-256 Fingerprint Generation      │
│  • Cache lookup against Human-in-the-Loop Feedback Memory                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 2. PREPROCESSING & DUAL-CROP NORMALIZATION                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  • MTCNN 5-Point Landmark Detection (Eyes, Nose, Mouth Corners)             │
│  • Affine Rotation & Scale Alignment to Canonical 256x256 Facial Frame      │
│  • Crop 1 (Global Context): 224x224 uncropped with ImageNet normalization   │
│  • Crop 2 (Facial Micro-Textures): 224x224 tightly cropped aligned face     │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
                    ▼                                     ▼
┌─────────────────────────────────────────┐ ┌─────────────────────────────────┐
│     3A. SPATIAL NEURAL BACKBONES        │ │    3B. FREQUENCY & BIOMETRICS   │
├─────────────────────────────────────────┤ ├─────────────────────────────────┤
│ • ViT-B/16 (google/vit-base-patch16-224)│ │ • 2D FFT Log-Magnitude Spectrum │
│ • EfficientNet-B3 (399k weights)        │ │ • Azimuthal Power Decay Fitting │
│ • Dual-Crop Test-Time Augmentation (TTA)│ │ • Siamese Corneal Specularity   │
│ • Grad-CAM Self-Attention Extraction    │ │ • Pupil Ellipticity Circularity │
└───────────────────┬─────────────────────┘ └─────────────────┬───────────────┘
                    │                                         │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 4. GOOGLE GEMINI MULTIMODAL VISION REASONER                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Dynamic model cascade (gemini-3.1-flash-lite -> gemini-2.5-flash)        │
│  • 1-Line plain-language scene and subject description                      │
│  • Pinpoint Primary Anomaly Region localization with normalized coordinates │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    5. MULTI-MODAL BAYESIAN SCORE FUSION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Fused Fake Probability: 0.55 Gemini + 0.20 ViT + 0.15 EffNet + 0.10 Phys │
│  • Threshold Decision Boundary at 0.45 Fake Probability                     │
│  • Epistemic Confidence Calibration & Uncertainty Interval                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   6. OUTPUT GENERATION & AUDIT PACKAGING                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Grad-CAM Heatmap Overlay PNG base64 encoding                             │
│  • 2D FFT Power Spectrum PNG base64 encoding                                │
│  • JSON Cryptographic Forensic Audit Certificate                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Facial Alignment & Preprocessing Protocols

### 2.1 Multi-Task Cascaded Convolutional Networks (MTCNN)
Face detection and alignment use a three-stage cascaded network:
1. **Proposal Network (P-Net)**: Fully convolutional network generating candidate face windows at multi-scale image pyramids.
2. **Refinement Network (R-Net)**: Rejects false-positive candidates via bounding box regression and non-maximum suppression (NMS).
3. **Output Network (O-Net)**: Identifies 5 facial landmarks:
   - Left Eye $(x_1, y_1)$
   - Right Eye $(x_2, y_2)$
   - Nose Tip $(x_3, y_3)$
   - Mouth Left Corner $(x_4, y_4)$
   - Mouth Right Corner $(x_5, y_5)$

### 2.2 Canonical Affine Normalization
The rotation angle $\theta$ is determined by the inter-ocular vector:

$$\theta = \arctan\left( \frac{y_2 - y_1}{x_2 - x_1} \right)$$

An affine transformation matrix $\mathbf{M} \in \mathbb{R}^{2 \times 3}$ maps the detected landmarks to canonical coordinates $(p_x^*, p_y^*)$ where the eye center is positioned at $(0.5, 0.38)$ in a square $256 \times 256$ crop.

---

## 3. Dual-Domain Feature Extraction Subsystems

### 3.1 Global Vision Transformer (ViT-B/16)
- **Role**: Global context inspection. Catches lighting discontinuities, warped backgrounds, clothing boundaries, and anatomical distortions outside the facial crop.
- **Input**: Full uncropped image resized to $224 \times 224 \times 3$.
- **Normalization**: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Architecture**: 12 Transformer layers, 768 hidden dimension, 12 attention heads, 196 patch tokens + 1 `[CLS]` token.

### 3.2 Aligned EfficientNet-B3
- **Role**: High-frequency facial texture analysis. Evaluates micro-texture blur, blending seams, and skin pores.
- **Input**: MTCNN-aligned $224 \times 224 \times 3$ facial crop.
- **Architecture**: Inverted residual blocks with squeeze-and-excitation (SE) attention ratio of $0.25$ and Swish activation ($\text{Swish}(x) = x \cdot \sigma(x)$).

### 3.3 Frequency Domain Optics (2D FFT)
- **Role**: Detects generator upsampling artifacts. Convolutional neural networks (GANs and Diffusion decoders) introduce periodic checkerboard patterns in the spatial frequency domain during deconvolution upsampling.
- **Algorithm**:
  1. Convert image to single-channel luminance: $Y = 0.299R + 0.587G + 0.114B$.
  2. Compute 2D Fast Fourier Transform using `scipy.fft.fft2`.
  3. Shift zero-frequency DC component to the center using `fftshift`.
  4. Compute log-magnitude: $\log(1 + |F(u, v)|)$.
  5. Sample azimuthal radial power profile to detect statistical deviations from the photographic $1/f^2$ baseline.

### 3.4 Siamese Biometric Optics
- **Role**: Physics-based biometric constraints.
- **Pupil Ellipticity**: Isolates pupil boundaries via adaptive Otsu thresholding and measures circularity ratio. Real human eyes maintain $a/b \approx 1.0$ under frontal orientation.
- **Corneal Catchlight Symmetry**: Measures the angular orientation of specular highlights relative to the iris center. Real scenes share a common lighting vector; synthetic images often generate uncorrelated specular highlights.

---

## 4. Training Engine & Optimization Strategy (`backend/train.py`)

The standalone training script in `backend/train.py` fine-tunes the backbones on the 2,000-specimen dataset:

### 4.1 Hyperparameter Configuration
- **Batch Size**: 16
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 10^{-4}$)
- **Learning Rate**: $\eta = 2 \times 10^{-5}$ with Cosine Annealing schedule
- **Loss Function**: Asymmetric Focal Loss ($\gamma = 2.0, \alpha = 0.60$)
- **Epochs**: 25 with early stopping patience of 5 epochs

### 4.2 Data Augmentation Pipeline
- Random horizontal flip ($p = 0.5$)
- Color jitter (Brightness $\pm 0.1$, Contrast $\pm 0.1$, Saturation $\pm 0.1$)
- Gaussian blur kernel ($3 \times 3, \sigma \in [0.1, 1.5]$) with $p = 0.2$ to prevent spatial overfitting
- JPEG compression artifact simulation ($\text{quality} \in [60, 95]$) with $p = 0.3$

---

## 5. Multimodal Reasoning Cascade

When client requests include a Google Gemini API key:
1. **Model Selection**: Queries fast models first (`gemini-3.1-flash-lite`, `gemini-2.5-flash`) to ensure $< 2.0$s response times and bypass rate limits.
2. **Strict Semantic Extraction**: Prompts enforce structured JSON responses with:
   - `summary`: 1-sentence plain-language description of the scene and subject.
   - `verdict`: Strict binary keyword (`**REAL**` or `**FAKE**`).
   - `fake_score`: Quantitative manipulation score $\in [0, 100]$.
   - `anomaly_localization`: Named suspect anatomical region (e.g., *Nasolabial Fold Smoothing*, *Bilateral Corneal Divergence*) with bounding box.
3. **Fallback Resiliency**: If the Gemini API experiences network timeouts or rate limits, the pipeline automatically falls back to the local dual-domain neural ensemble without crashing.

---

## 6. Continuous Active Learning & Feedback Memory

1. When users submit feedback through `/feedback` (e.g., correcting an edge case):
   - Computes 64-bit DCT perceptual hash (`pHash`) and SHA-256 byte digest.
   - Stores the verified ground-truth label in `backend/data/feedback_memory.json`.
2. On future scans:
   - Exact SHA-256 byte match overrides prediction instantly with $99.9\%$ confidence.
   - Perceptual hash match with Hamming distance $D_H \le 4$ matches compressed, re-encoded, or resized versions of the specimen.
