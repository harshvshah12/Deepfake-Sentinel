# REST API Specification
# Deepfake Sentinel: Dual-Domain Forensic Platform

Interactive OpenAPI Swagger UI: `http://localhost:8080/docs`  
Interactive ReDoc Documentation: `http://localhost:8080/redoc`

---

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health, active compute hardware (CPU/CUDA), and version |
| `POST` | `/predict` | Dual-domain forensic scan on image (supports FGSM & Gemini) |
| `POST` | `/predict-video` | Multi-frame temporal consistency & jitter scan on video uploads |
| `POST` | `/predict-frame` | Ultra-low latency single-frame scan for live webcam streaming |
| `POST` | `/feedback` | Submit human ground-truth feedback to active perceptual memory |
| `GET` | `/feedback-stats` | Feedback memory count, distribution, and recorded log |
| `POST` | `/test-gemini-key` | Test Google Gemini API Key validity with live diagnostics |
| `GET` | `/model-metrics` | Telemetry for Chart.js model visualization graphs |
| `GET` | `/samples` | Curated benchmark specimen gallery metadata |
| `GET` | `/sample-image` | Stream benchmark specimen image by relative path |

---

## 1. Single Image Forensic Scan

### Request: `POST /predict`
- **Content-Type**: `multipart/form-data`
- **Query Parameters**:
  - `attack` (*bool*, optional, default: `false`): Enables Fast Gradient Sign Method (FGSM) adversarial stress testing.
  - `epsilon` (*float*, optional, default: `0.05`): Perturbation budget $\epsilon \in [0.01, 0.15]$.
  - `gemini_key` (*string*, optional): Client-side Google AI Studio Gemini API key.
- **Form Body**:
  - `file`: Raw binary image file (JPEG, PNG, WEBP).

#### Sample Request (`curl`):
```bash
curl -X POST "http://localhost:8080/predict?attack=false&gemini_key=your_api_key_here" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@specimen.jpg"
```

#### Response: `200 OK`
```json
{
  "prediction": "Fake",
  "confidence": 0.999,
  "fake_probability": 0.999,
  "real_probability": 0.001,
  "decision_threshold": 0.45,
  "frequency_fake_score": 0.85,
  "iris_asymmetry_score": 0.0412,
  "provenance": {
    "c2pa_status": "Not Detected",
    "synthid_status": "Synthetic Signature Detected",
    "exif_status": "Stripped / AI Metadata Present"
  },
  "explanation": {
    "summary": "Synthetic male portrait exhibiting artificial skin smoothing and misaligned corneal glints.",
    "verdict": "**FAKE**",
    "fake_score": 98.5,
    "confidence_rating": "Very High",
    "anomaly_localization": {
      "primary_region": "Nasolabial Skin Micro-Smoothing",
      "box_2d": [0.42, 0.38, 0.58, 0.62],
      "explanation": "Skin micro-texture displays uncharacteristically smooth frequency decay inconsistent with biological dermal pores."
    },
    "technical_rationale": "High-frequency power spectrum displays prominent azimuthal upsampling energy spikes.",
    "anomalies_detected": [
      "Lack of natural pores in skin macro-texture",
      "Corneal specular glint angle divergence across bilateral irises",
      "Subtle blending seam along left ear-hair interface"
    ]
  },
  "visualizations": {
    "gradcam_b64": "data:image/jpeg;base64,...",
    "fft_b64": "data:image/jpeg;base64,...",
    "iris_crop_b64": "data:image/jpeg;base64,..."
  },
  "adversarial_b64": null,
  "audit_certificate": {
    "certificate_id": "DFS-2026-A4F89C",
    "timestamp": "2026-10-14T11:20:00Z",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "phash": "a7b3c2e1f4095812",
    "verdict": "Synthetic Media Detected",
    "confidence": 0.999,
    "primary_anomaly_region": "Nasolabial Skin Micro-Smoothing"
  }
}
```

---

## 2. Multi-Frame Temporal Video Scan

### Request: `POST /predict-video`
- **Content-Type**: `multipart/form-data`
- **Form Body**:
  - `file`: Video file (MP4, AVI, MOV).

#### Response: `200 OK`
```json
{
  "status": "success",
  "frames_analyzed": 24,
  "fake_frames_detected": 21,
  "fake_frame_ratio": 0.875,
  "overall_prediction": "Fake",
  "average_confidence": 0.978,
  "temporal_drift_score": 0.184,
  "blink_cadence_anomalous": true,
  "anomalous_intervals": [
    { "start_second": 0.4, "end_second": 2.8, "suspect_factor": "Facial Jitter & Warping" }
  ]
}
```

---

## 3. Real-Time Single-Frame Webcam Scan

### Request: `POST /predict-frame`
- **Content-Type**: `multipart/form-data`
- **Form Body**:
  - `file`: JPEG image frame captured from live webcam stream.

#### Response: `200 OK`
```json
{
  "prediction": "Real",
  "confidence": 0.942,
  "latency_ms": 78.4,
  "iris_symmetry_score": 0.0084,
  "face_detected": true
}
```

---

## 4. Human-in-the-Loop Active Learning Feedback

### Request: `POST /feedback`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "phash": "a7b3c2e1f4095812",
  "corrected_label": "Real",
  "original_prediction": "Fake",
  "notes": "Verified authentic portrait taken on Sony A7R IV with portrait beauty filter."
}
```

#### Response: `200 OK`
```json
{
  "status": "success",
  "message": "Feedback recorded. Model memory calibrated for this specimen.",
  "total_records": 18
}
```

---

## 5. Active Feedback Statistics

### Request: `GET /feedback-stats`

#### Response: `200 OK`
```json
{
  "total_feedback_entries": 18,
  "label_distribution": {
    "Real": 11,
    "Fake": 7
  },
  "recent_submissions": [
    {
      "timestamp": "2026-10-14T09:15:30Z",
      "sha256_short": "e3b0c442...",
      "phash": "a7b3c2e1f4095812",
      "corrected_label": "Real"
    }
  ]
}
```

---

## 6. Test Google Gemini API Key

### Request: `POST /test-gemini-key`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "gemini_key": "your_google_gemini_api_key_here"
}
```

#### Response: `200 OK`
```json
{
  "valid": true,
  "model_connected": "gemini-3.1-flash-lite",
  "latency_ms": 420.5,
  "status_message": "Key authorized with multimodal vision capabilities."
}
```

---

## 7. Model Telemetry & Visualization Metrics

### Request: `GET /model-metrics`

#### Response: `200 OK`
```json
{
  "overall_accuracy": 0.987,
  "vit_accuracy": 0.912,
  "efficientnet_accuracy": 0.884,
  "feature_weights": {
    "gemini_vision": 0.55,
    "vit_transformer": 0.20,
    "efficientnet_b3": 0.15,
    "frequency_fft": 0.05,
    "iris_biometrics": 0.05
  },
  "radial_decay_curve": {
    "frequencies": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "natural_baseline": [4.2, 3.8, 3.4, 3.1, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2],
    "gan_profile": [4.3, 3.9, 3.6, 3.8, 3.1, 3.5, 2.7, 3.1, 2.0, 2.4]
  }
}
```

---

## 8. Benchmark Specimen Gallery

### Request: `GET /samples`

#### Response: `200 OK`
```json
{
  "samples": [
    {
      "name": "fake_0.jpg",
      "label": "Fake (StyleGAN3)",
      "path": "dataset/fake/fake_0.jpg",
      "category": "GAN"
    },
    {
      "name": "real_0.jpg",
      "label": "Real (FFHQ Genuine)",
      "path": "dataset/real/real_0.jpg",
      "category": "Authentic"
    }
  ]
}
```

---

## 9. Server Health Check

### Request: `GET /health`

#### Response: `200 OK`
```json
{
  "status": "online",
  "device": "cpu",
  "cuda_available": false,
  "model": "ViT-B/16 + MTCNN + 2D FFT",
  "version": "3.1.0"
}
```
