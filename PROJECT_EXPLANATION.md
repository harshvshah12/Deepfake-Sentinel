# Deepfake Sentinel: Comprehensive Project Explanation & Architecture Walkthrough

This document provides an end-to-end guide to **Deepfake Sentinel**, explaining the core problem, features, machine learning models, mathematical formulations, explainability tools, and production viability. It is structured for academic reviewers, technical recruiters, and forensic evaluators.

---

## 1. Executive Summary: The Deepfake Detection Challenge

Generative deepfakes created by Generative Adversarial Networks (StyleGAN2/3), Latent Diffusion Models (Stable Diffusion XL, Midjourney v6, FLUX.1), and face-swapping pipelines (DeepFaceLab, SimSwap) have advanced beyond the detection capabilities of simple convolutional neural networks.

### The Core Failure Mode of Prior Art:
1. **Single-Backbone Overfitting**: Classifiers trained strictly on spatial RGB pixels learn high-frequency generator-specific artifacts. When tested on a new generator or after mild JPEG compression, their accuracy drops by $15\% - 30\%$.
2. **Black-Box Opacity**: Traditional models output a raw number (e.g., `0.87 Fake`) with zero explanation of *where* the manipulation occurred or *why* the model believes it is synthetic.
3. **Fragility to Post-Processing**: Common social media operations (resizing, re-encoding, minor color grading) strip superficial artifacts, causing false negatives.

### How Deepfake Sentinel Solves This:
Deepfake Sentinel uses an **orthogonal multi-domain defense architecture** that simultaneously analyzes:
- **Spatial Global Semantics** (`ViT-B/16`)
- **Facial Micro-Textures** (`EfficientNet-B3`)
- **Physical Frequency Optics** (`2D Fast Fourier Transform`)
- **Biological Invariants** (Bilateral corneal specular reflections & pupil circularity)
- **Multimodal Visual Reasoning** (`Google Gemini Vision`)
- **Continuous Memory** (64-bit DCT perceptual hashing active learning)

---

## 2. What Each Feature Is & How It Works

### 2.1 Single Image Forensic Scanner
- **What it is**: The core diagnostic view where users can upload any digital photograph or portrait to receive an instantaneous, multi-layered forensic inspection report.
- **How it works**: The backend decodes the image in memory, detects and aligns faces using MTCNN, passes the specimen through spatial backbones (ViT and EfficientNet), computes the 2D FFT power spectrum, calculates corneal glint symmetry, and optionally queries Google Gemini Vision. The results are fused into a calibrated authenticity verdict with an epistemic confidence rating ($50\% - 99.9\%$).

### 2.2 Pinpoint Primary Anomaly Region Localization
- **What it is**: An amber glowing badge and bounding box overlay that highlights the exact anatomical area causing the model to flag the image as synthetic (e.g., *Ear-Hair Interface*, *Corneal Glint Divergence*, *Skin Micro-Texture Smoothing*).
- **How it works**: Google Gemini Vision parses the high-resolution image and returns normalized coordinates $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ paired with a plain-language physical explanation of the broken optical rule.

### 2.3 Tri-Panel Explainable AI (XAI) Visualizer
- **What it is**: Three synchronized visualizers that provide visual proof of the diagnosis:
  1. **Grad-CAM Attention Heatmap**: Projects self-attention token weights directly over the face, highlighting areas where the neural network focused.
  2. **2D Log-FFT Frequency Spectrum**: Displays spatial frequency power distribution, revealing synthetic grid peaks or abnormal high-frequency energy decay.
  3. **Biometric Iris HUD**: Displays zoomed bilateral iris crops with detected pupil contours and corneal specular highlight vectors.

### 2.4 Multi-Frame Temporal Video Deepfake Inspector
- **What it is**: A dedicated video analysis tab that evaluates MP4/AVI videos for temporal flickering and biometric drift across time.
- **How it works**: Samples video frames at uniform intervals, tracks face landmarks across consecutive frames, computes inter-frame feature variance and eye-blink cadence anomalies, and pinpoints exact timestamp intervals containing manipulated frames.

### 2.5 Cybernetic Real-Time Webcam Stream
- **What it is**: A live biometric scanner that streams webcam frames directly into the lightweight inference engine at low latency.
- **How it works**: Captures video frames via HTML5 Canvas, submits them to `/predict-frame`, tracks corneal glint stability, and reflects real-time authenticity status on the cybernetic 3D HUD.

### 2.6 FGSM Adversarial Robustness Sandbox
- **What it is**: An interactive security module allowing users to test the model's resilience against intentional adversarial attacks designed to fool AI detectors.
- **How it works**: Uses the Fast Gradient Sign Method (FGSM) to inject imperceptible mathematically computed noise vectors ($\epsilon = 0.01 - 0.15$) and displays the resulting perturbation map alongside the updated model confidence.

### 2.7 Curated Benchmark Specimen Gallery
- **What it is**: A pre-loaded gallery of real and synthetic images across GANs, Diffusion, and real camera captures.
- **How it works**: Allows 1-click loading of benchmark specimens from `dataset/` directly into the diagnostic pipeline to test cross-generator generalization in real time.

### 2.8 Active Learning Feedback Memory Loop
- **What it is**: A human-in-the-loop continuous calibration engine that allows forensic analysts to correct model predictions.
- **How it works**: When a correction is submitted, the system computes the image's SHA-256 byte digest and 64-bit DCT perceptual hash (`pHash`), caching the ground truth in `backend/data/feedback_memory.json`. Any future scan of that image (even if resized or compressed) matches via Hamming distance $\le 4$ and immediately outputs the verified ground truth with $99.9\%$ confidence.

---

## 3. Why Each Machine Learning Model Was Chosen

| Component | Architecture | Why Chosen |
| :--- | :--- | :--- |
| **MTCNN** | 3-stage Cascaded CNN | Fast, deterministic 5-point facial landmark alignment that eliminates head-tilt and scale variations across wild images. |
| **ViT-B/16** | Vision Transformer (12 layers, 196 patches) | Unlike CNNs that only look at local receptive fields, Transformers use self-attention across the whole uncropped image to catch global lighting inconsistencies and warped backgrounds. |
| **EfficientNet-B3** | Inverted Residual CNN (399k weights) | Squeeze-and-excitation attention blocks excel at identifying micro-texture boundaries, blending seams, and skin pore smoothing on the tight facial crop. |
| **2D Fast Fourier Transform** | Mathematical SciPy transform | Completely independent of learned weights; captures physical camera sensor physics ($1/f^\alpha$ power decay) and detects transposed convolution checkerboard grids. |
| **Siamese Iris Biometrics** | Corneal Ray-Tracing Physics | Tests optical symmetry of light reflections. Since generative models generate eyes independently, catchlights frequently point in divergent directions. |
| **Google Gemini Vision** | Multimodal LLM Cascade | Acts as an expert forensic judge, combining deep visual reasoning with natural language explanations and localized suspect bounding boxes. |

---

## 4. How the Multi-Modal Ensemble Reached 98.7% Accuracy

The model was evaluated against 2,000 verified forensic specimens (documented in [`dataset.md`](dataset.md)):

1. **Baseline CNNs (ResNet-50 / MesoNet)**: Achieved $71.4\% - 82.6\%$ accuracy, struggling significantly on modern diffusion models (SDXL/Midjourney) due to high-frequency convolution kernel confusion.
2. **Vision Transformer (ViT-B/16)**: Lifted spatial accuracy to $91.2\%$ by capturing global patch relationships.
3. **Dual-Domain Neural Architecture**: Combining spatial ViT + EfficientNet with 2D FFT optics achieved $94.6\%$ accuracy and reduced JPEG compression degradation to less than $-4\%$.
4. **Bayesian Multimodal Fusion (+ Gemini Vision)**: Combining the local neural backbones with Google Gemini Vision achieved **$98.7\%$ overall accuracy**, **$0.998$ ROC-AUC**, and **$99.0\%$ recall**.

---

## 5. Security, Provenance & Privacy Architecture
- **In-Memory Zero Disk Cache**: User uploads are processed entirely in volatile memory buffers and never written to permanent disk without encryption.
- **C2PA & SynthID Provenance**: Inspects digital image manifests for Coalition for Content Provenance and Authenticity (C2PA) cryptographic signatures and Google SynthID watermarks.
- **Client-Side Key Isolation**: Gemini API keys are never hardcoded on the server; clients can inject their own keys stored exclusively in browser `localStorage`.
