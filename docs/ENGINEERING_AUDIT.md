# Engineering Audit — Deepfake Sentinel
**Document Version:** 1.0.0  
**Audit Date:** 2026-10-15  
**Baseline Git Commit:** `31adfce` (`docs(research): add publication-grade IEEE research paper and update documentation index`)  
**Working Tree Status:** Clean on `main`  
**Host Environment:** Windows (x86_64), Python 3.10, PyTorch CPU execution mode, Port 8080 active.  

---

## 1. Executive Summary

Deepfake Sentinel is an applied machine learning and digital forensics platform designed to evaluate synthetic media authenticity across images and video. The current repository contains functional multi-model inference pipelines (EfficientNet-B3 + HuggingFace ViT-B/16), frequency-domain analysis (2D FFT power spectrum), facial landmark extraction (MTCNN), perceptual hashing (pHash/dHash/aHash/SHA-256), and an interactive single-page frontend.

However, an exhaustive line-by-line engineering audit reveals critical architectural vulnerabilities, mathematical disparities, heuristic shortcuts masquerading as production models, and hardcoded decision overrides. These issues undermine technical credibility and scientific rigor.

This audit establishes the baseline truth of the codebase across seven mandatory categories:
1. **Verified Existing Functionality**
2. **Incorrect or Unreliable Functionality**
3. **Heuristic or Experimental Features**
4. **Documented but Unimplemented Features**
5. **Real-World Execution Failure Points**
6. **External Credentials, Services, and Hardware Dependencies**
7. **Regression Risks and Breaking Changes**

---

## 2. Exhaustive Audit by Functional Category

### Category 1: Verified Existing Functionality (Working as Intended)
- **Face Detection and Landmark Tracking (`backend/preprocessor.py`)**:
  - `FaceExtractor` utilizes `facenet_pytorch.MTCNN` with multi-scale pyramid detection.
  - Correctly extracts 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth).
  - Gracefully falls back to centered square crops when no human face is detected.
- **Dual-Backbone Spatial Inference (`backend/model.py`)**:
  - `EfficientNet-B3` loaded from `backend/weights/deepfakefusion/best_efficientnet_b3_deepfake_detector_399k.pth` (43.3 MB state dictionary). Correctly modifies classifier head to single logit output.
  - `ViT-B/16` loaded via Hugging Face `dima806/deepfake_vs_real_image_detection`.
  - Test-Time Augmentation (TTA) via horizontal flipping works across both backbones.
- **2D Fourier FFT Magnitude Spectrum (`backend/frequency.py`)**:
  - 2D Fast Fourier Transform (`np.fft.fft2` + `np.fft.fftshift`) computes log-magnitude power spectrum.
  - High-frequency power ratio and 8-bin azimuthal radial profile are dynamically calculated.
- **Grad-CAM Visual Attention (`backend/xai.py`)**:
  - `pytorch_grad_cam.GradCAM` targets the final convolutional layer of EfficientNet (`model.features[-1]`).
  - Successfully overlays thermal activation maps onto the extracted face crop.
- **Continuous Feedback Memory (`backend/feedback_memory.py`)**:
  - Stores user ground-truth corrections in `backend/data/feedback_memory.json`.
  - Performs dual lookup: exact SHA-256 and perceptual pHash matching with Hamming distance threshold ($\le 10$).
- **API Server & Routing (`backend/main.py`)**:
  - FastAPI serves `/health`, `/samples`, `/sample-image`, `/model-metrics`, `/predict`, `/predict-frame`, and `/feedback`.
  - Static files mounted correctly at `/static` and root `/` returns `frontend/static/index.html`.

---

### Category 2: Functionality That Exists But Is Incorrect or Unreliable

#### 2.1 Arbitrary Watermark Byte-Matching Override
- **Location:** `backend/main.py` lines 345–351; `backend/provenance.py` lines 216–254.
- **Flaw:** `detect_visual_watermarks_and_logos()` searches raw byte strings for keywords such as `b"gemini"`, `b"adobe"`, `b"synthid"`, or `b"deepmind"`.
- **Impact:** If any matching byte sequence is found, `backend/main.py` completely bypasses the machine learning models and hardcodes:
  ```python
  fused_fake_prob = 0.999
  fused_real_prob = 0.001
  prediction = "Fake"
  confidence = 0.999
  ```
- **Why this is wrong:** An authentic photo exported with Adobe Photoshop, or containing ordinary metadata mentioning "adobe", is instantly branded as 99.9% Fake with 99.9% confidence. This is unscientific and causes catastrophic false positives.

#### 2.2 False C2PA Cryptographic Signature Assertion
- **Location:** `backend/provenance.py` lines 103–129.
- **Flaw:** Searches for 4-byte markers (`b"c2pa"`, `b"jumb"`, `b"c2bi"`). If found, it outputs:
  ```python
  "validation_status": "VALID_SIGNATURE"
  "signature_issuer": "C2PA Compliant Hardware Authority"
  "c2pa_score": 0.98
  ```
- **Impact:** It performs zero cryptographic certificate validation, public key verification, or SHA-256 hash tree validation. Furthermore, `backend/main.py` line 369 uses this unverified match to force `fused_fake_raw = min(fused_fake_raw, 0.06)`, falsely classifying images as 94% real.

#### 2.3 Divergent Decision Fusion Logic
- **Disparity:**
  - `backend/model.py:predict_single`: $0.60 \times \text{EffNet} + 0.40 \times \text{ViT}$.
  - `backend/main.py:predict`: $0.40 \times \text{ViT} + 0.30 \times \text{EffNet} + 0.15 \times \text{Spectral} + 0.10 \times \text{Iris} + 0.05 \times \text{Color}$.
  - `test_bayesian_fusion.py`: Adds heuristic deltas ($\pm 0.20$, $\pm 0.15$) onto raw fake score.
- **Impact:** Different endpoints and test scripts give conflicting predictions for the exact same image. There is no unified mathematical contract.

#### 2.4 Destructive Model Training Loop
- **Location:** `backend/train.py` lines 140–145.
- **Flaw:** Calling `train.py` overwrites `best_efficientnet_b3_deepfake_detector_399k.pth` directly after only 2 epochs without preserving backups or validating against a frozen benchmark holdout.

---

### Category 3: Features That Are Only Heuristic or Experimental

#### 3.1 Untrained Siamese Iris Neural Network
- **Location:** `backend/iris.py` lines 11–47, 57–60.
- **Finding:** `SiameseIrisNetwork` is defined as a 3-layer CNN. In `IrisBiometricEngine.__init__`, it is instantiated with `SiameseIrisNetwork().to(self.device)` without loading any checkpoint weights.
- **Reality:** The Siamese iris embeddings are computed using completely random weights. While the pupil circularity and corneal specular offset are heuristic geometric computer vision routines, the "Siamese deep feature embedding" is unvalidated. Yet, `/model-metrics` advertises this as a "96.4% Accuracy Biometric Neural Network".

#### 3.2 2D FFT Checkerboard Heuristic
- **Location:** `backend/frequency.py` lines 56, 82.
- **Finding:** High frequency ratio is simply multiplied by an arbitrary factor of $2.8$ (`raw_high_freq_ratio * 2.8`). High-frequency noise from JPEG compression or physical camera grain is routinely misflagged as a "GAN upsampling artifact".

#### 3.3 Color Space Discontinuity
- **Location:** `backend/color_forensics.py` lines 52–57.
- **Finding:** Linearly combines channel Pearson correlation, YCbCr standard deviation differences, and Laplacian variance without empirical calibration against a labeled dataset.

---

### Category 4: Features Documented But Not Implemented
- **Full Video Photoplethysmography (rPPG)**: Documented in research architecture notes as reading blood flow pulse waves from facial skin; current `backend/video.py` only runs frame-by-frame static ViT/FFT and averages probabilities.
- **Server-Side Multi-Page PDF Forensic Report**: Mentioned in product requirements; no PDF generator exists in the repository.
- **True C2PA Manifest Verification**: Documented as compliant with C2PA 1.3 standards; currently only byte pattern matching without public key verification.
- **Visual Logo Localizer**: Function is named `detect_visual_watermarks_and_logos`, but does not perform template matching, OCR, or bounding-box localization on image pixels.

---

### Category 5: Real-World Execution Failure Points
1. **Large Video Memory Exhaustion**: Uploading a 4K 60FPS video loads frames into RAM without streaming or downsampling, risking process crashes.
2. **Missing Bounding Boxes on Synthetic Graphics**: Passing non-face graphics (e.g. landscapes or synthetic documents) to facial routines causes empty landmark arrays, which default to hardcoded dummy constants (0.85, 10, 0.1).
3. **No Inconclusive Decision State**: The decision threshold is binary (`Fake if prob > 0.45/0.48 else Real`). If models disagree violently (e.g. ViT = 0.95, EffNet = 0.05), the system produces an overconfident output instead of flagging `INCONCLUSIVE` or requesting human review.

---

### Category 6: External Dependencies and Hardware Constraints
- **Gemini Multimodal API**: Requires a valid `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Without it, graceful fallback must generate deterministic, evidence-grounded reports.
- **GPU Acceleration**: Defaults to `cpu` when CUDA is unavailable. Inference latency on CPU ranges between 350ms–900ms per image; TTA doubles this.
- **Hugging Face Hub**: Requires network access on first run to download `dima806/deepfake_vs_real_image_detection` (cached locally in `~/.cache/huggingface`).

---

### Category 7: Changes That Could Break Existing Behaviour
- Modifying `/predict` output JSON keys without updating `frontend/static/index.html` will break client-side rendering.
- Removing `b"gemini"` or `b"adobe"` overrides will change existing synthetic test scores for files containing those strings (this change is required for scientific correctness, but test expectations must be updated).
- Unifying `fused_fake_prob` will alter the final probability displayed on the UI.

---

## 3. Immediate Action Items
1. Unify all inference and fusion pipelines into a single canonical source of truth with calibrated confidence and an explicit `INCONCLUSIVE` state.
2. Isolate the untrained `SiameseIrisNetwork` from decision weighting and accurately label it as an experimental prototype.
3. Remove hardcoded byte-matching overrides from the Bayesian fusion pipeline.
4. Build the sequential 14-stage Investigation Workspace on the frontend.
5. Implement server-side multi-page PDF forensic report generation via ReportLab.
