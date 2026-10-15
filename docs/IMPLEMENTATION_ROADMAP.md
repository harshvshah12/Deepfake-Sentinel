# Implementation Roadmap — Deepfake Sentinel
**Product Name:** Deepfake Sentinel  
**Target Milestone:** Production-Grade Trustworthy Forensic Intelligence Platform  
**Architecture Contract:** Unified Inference Engine, Traceable Evidence Pipeline, Calibrated Fusion, Multi-Page PDF Reporting  

---

## Roadmap Overview & Execution Matrix

| Goal | Milestone | Status | Key Deliverables |
|---|---|---|---|
| **Goal 0** | Baseline Audit & Reproduction | **IN PROGRESS** | `docs/ENGINEERING_AUDIT.md`, `docs/IMPLEMENTATION_ROADMAP.md`, `docs/BASELINE_TEST_RESULTS.md` |
| **Goal 1** | Model Integrity & Fusion Correctness | **PLANNED** | Unified `ForensicDecisionEngine`, removal of byte overrides, calibration curves, `INCONCLUSIVE` boundary |
| **Goal 2** | Animated Investigation Workspace | **PLANNED** | Sequential 14-Stage HUD in frontend, real progress dispatch, interactive evidence inspection tabs |
| **Goal 3** | Physiological & Temporal rPPG | **PLANNED** | Green/CHROM blood volume pulse extraction, FFT power spectrum of cardiac rhythm, SNR filtering |
| **Goal 4** | Watermark & Provenance Reality | **PLANNED** | Transparent C2PA manifest parser, SynthID spectral adapter, visual logo template matching |
| **Goal 5** | Dataset Integrity & Training Pipeline | **PLANNED** | Strict manifest registry, split verification (zero data leakage), non-destructive checkpointing |
| **Goal 6** | Multi-Page Forensic PDF Report | **PLANNED** | ReportLab PDF engine (3–5 pages: Summary, Preprocessing, Dual Models, Physics, Provenance) |
| **Goal 7** | Video Temporal Consistency | **PLANNED** | Inter-frame optical flow variance, blink cadence detector, landmark stability metric |
| **Goal 8** | Security, Privacy & Reliability | **PLANNED** | Strict file limits (50MB), MIME sniff validation, path sanitization, temp file cleanup |
| **Goal 9** | Playwright E2E Verification | **PLANNED** | Visual regressions, user upload workflows, error state tests, PDF download test |
| **Goal 10** | Performance & Deployment | **PLANNED** | Latency profiling, asset compression, Dockerfile hardening, Vercel/Render alignment |
| **Goal 11** | Documentation & Model Cards | **PLANNED** | Model Cards, Dataset Cards, Ethical Disclosure, IEEE synchronization |

---

## Phased Execution Strategy

### Phase 1: Baseline Establishment (Goal 0)
1. Complete system audit of all components, dependencies, and model weights.
2. Execute baseline evaluations on 50 authentic and synthetic samples; record metrics in `docs/BASELINE_TEST_RESULTS.md`.
3. Create `docs/IMPLEMENTATION_LOG.md` to track each subsequent goal.

### Phase 2: Scientific Core & Decision Engine (Goal 1, Goal 4)
1. Build `backend/fusion.py` defining `ForensicDecisionEngine`:
   - Calibrated sigmoid scaling for spatial neural outputs (EfficientNet-B3 + ViT-B/16).
   - Dynamic weight allocation based on input validity (e.g. face detected vs landscape).
   - Anomaly score normalization for physics signals (2D FFT, pupil circularity, corneal reflections).
   - Explicit 3-tier verdict: `AUTHENTIC` ($P(\text{Fake}) < 0.40$), `SYNTHETIC` ($P(\text{Fake}) > 0.60$), and `INCONCLUSIVE` ($0.40 \le P \le 0.60$ or severe model discordance).
2. Refactor `backend/provenance.py` and remove destructive hardcoded overrides:
   - Separate metadata assertions from pixel-level models.
   - Accurately describe C2PA status (`NO_MANIFEST`, `UNVERIFIED_MANIFEST_PRESENT`, `VALID_CRYPTOGRAPHIC_SIGNATURE`).
   - Clearly delineate heuristic Fourier watermarking from official Google SynthID API capabilities.

### Phase 3: Investigation Workspace UI (Goal 2)
1. Restructure `frontend/static/index.html` to present the **14-Stage Forensic Investigation Pipeline**:
   - Stage 1: Asset Intake & Cryptographic Hashing (SHA-256 / pHash)
   - Stage 2: EXIF & Sensor Provenance Inspection
   - Stage 3: C2PA & Content Authenticity Manifest Audit
   - Stage 4: High-Frequency Digital Watermark & SynthID Scan
   - Stage 5: Multi-Scale MTCNN Facial Landmark Extraction
   - Stage 6: Bilateral Iris & Corneal Specular Alignment
   - Stage 7: Pupil Ellipticity & Boundary Regularity
   - Stage 8: 2D Fourier Power Spectrum & Azimuthal Profiling
   - Stage 9: Color Co-occurrence & Chrominance Covariance
   - Stage 10: EfficientNet-B3 Deep Spatial Convolution
   - Stage 11: Vision Transformer (ViT-B/16) Patch Attention
   - Stage 12: Grad-CAM Interpretability & Artifact Heatmap
   - Stage 13: Multi-Modal Bayesian Decision Fusion
   - Stage 14: Forensic Verdict Formulation & Report Synthesis
2. Implement true sequential visualization: stages activate as backend telemetry streams, eliminating fake timer animations.

### Phase 4: Physiological Video Forensics (Goal 3, Goal 7)
1. Extend `backend/video.py` with Chrominance-based rPPG (`CHROM` algorithm) to extract blood volume pulse (BVP) from cheek regions across video frames.
2. Calculate heart rate power spectrum and signal-to-noise ratio (SNR); synthetic deepfakes lack coherent cardiac periodicity.
3. Compute inter-frame face alignment jitter and eye blink frequency.

### Phase 5: Server-Side PDF Forensic Report (Goal 6)
1. Implement `backend/report_generator.py` using `reportlab`:
   - Page 1: Official Header, Executive Verdict, Confidence Index, Cryptographic Hashes, Thumbnail with Bounding Box.
   - Page 2: Preprocessing Evidence, Facial Landmarks, Bilateral Iris RoIs, Corneal Glint Vectors.
   - Page 3: Model Inferences (EfficientNet vs ViT), Grad-CAM Heatmap, 2D FFT Frequency Magnitude Chart.
   - Page 4: Provenance Metadata, EXIF breakdown, C2PA Manifest analysis, Limitations & Forensic Disclaimers.
2. Expose `GET /report/{analysis_id}` and `POST /generate-report` with automated download in frontend.

### Phase 6: System Hardening, Playwright E2E & Documentation (Goal 5, 8, 9, 10, 11)
1. Add upload sanitization, payload size limits (50MB), MIME sniff detection, and temp file cleanup.
2. Build end-to-end Playwright tests verifying upload, investigation sequence, inspection tabs, and PDF export.
3. Update `README.md`, sync `C:\projects\orchestra-brain`, and autonomously push to GitHub.
