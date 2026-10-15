# Baseline Test Results — Deepfake Sentinel
**Execution Date:** 2026-10-15  
**Evaluation Script:** `test_bayesian_fusion.py`  
**Test Specimen Sample:** 50 Random Specimens (25 Authentic, 25 Synthetic) from `dataset/real/` and `dataset/fake/`  
**Inference Hardware:** CPU execution (Intel / AMD x86_64)  
**TTA Enabled:** True (horizontal reflection averaging)  

---

## 1. Summary Metrics

| Metric | Measured Baseline | Previously Documented Claim | Delta / Discrepancy |
|---|---|---|---|
| **Authentic (Real) Accuracy** | **72.0%** (18/25) | 99.6% | **-27.6% (7 False Positives)** |
| **Synthetic (Fake) Accuracy** | **96.0%** (24/25) | 99.8% | **-3.8% (1 False Negative)** |
| **Overall System Accuracy** | **84.0%** (42/50) | 99.8% | **-15.8% Overclaim** |
| **Mean Inference Latency (CPU)** | **420ms / specimen** | 87ms | Measured on CPU without CUDA |

---

## 2. Per-Specimen Baseline Execution Trace

### 2.1 Authentic (Real) Cohort (25 Specimens)
| Specimen ID | Ground Truth | Prediction | Real Prob | Fake Prob | Verdict Status |
|---|---|---|---|---|---|
| `real_0.jpg` | Real | Real | 0.811 | 0.189 | PASS |
| `real_1.jpg` | Real | Real | 0.961 | 0.039 | PASS |
| `real_10.jpg` | Real | Real | 0.786 | 0.214 | PASS |
| `real_100.jpg` | Real | **Fake** | 0.064 | 0.936 | **FAIL (False Positive)** |
| `real_1000.jpg` | Real | Real | 0.557 | 0.443 | PASS (Borderline) |
| `real_1001.jpg` | Real | **Fake** | 0.143 | 0.857 | **FAIL (False Positive)** |
| `real_1002.jpg` | Real | **Fake** | 0.171 | 0.829 | **FAIL (False Positive)** |
| `real_1003.jpg` | Real | **Fake** | 0.156 | 0.844 | **FAIL (False Positive)** |
| `real_1004.jpg` | Real | **Fake** | 0.005 | 0.995 | **FAIL (False Positive)** |
| `real_1005.jpg` | Real | **Fake** | 0.003 | 0.997 | **FAIL (False Positive)** |
| `real_1006.jpg` | Real | Real | 0.617 | 0.383 | PASS |
| `real_1007.jpg` | Real | Real | 0.998 | 0.002 | PASS |
| `real_1008.jpg` | Real | Real | 0.544 | 0.456 | PASS (Borderline) |
| `real_1009.jpg` | Real | Real | 0.962 | 0.038 | PASS |
| `real_101.jpg` | Real | **Fake** | 0.426 | 0.574 | **FAIL (False Positive)** |
| `real_1010.jpg` | Real | Real | 0.604 | 0.396 | PASS |
| `real_1011.jpg` | Real | Real | 0.626 | 0.374 | PASS |
| `real_1012.jpg` | Real | Real | 0.898 | 0.101 | PASS |
| `real_1013.jpg` | Real | Real | 0.938 | 0.062 | PASS |
| `real_1014.jpg` | Real | Real | 0.937 | 0.063 | PASS |
| `real_1015.jpg` | Real | Real | 0.850 | 0.150 | PASS |
| `real_1016.jpg` | Real | Real | 0.918 | 0.082 | PASS |
| `real_1017.jpg` | Real | Real | 0.950 | 0.050 | PASS |
| `real_1018.jpg` | Real | Real | 0.996 | 0.004 | PASS |
| `real_1019.jpg` | Real | Real | 0.963 | 0.037 | PASS |

**Cohort Real Accuracy: 18 / 25 (72.0%)**

---

### 2.2 Synthetic (Fake) Cohort (25 Specimens)
| Specimen ID | Ground Truth | Prediction | Real Prob | Fake Prob | Synthetic Iris Flag | Verdict Status |
|---|---|---|---|---|---|---|
| `fake_0.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1.jpg` | Fake | Fake | 0.026 | 0.974 | False | PASS |
| `fake_10.jpg` | Fake | Fake | 0.007 | 0.993 | False | PASS |
| `fake_100.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1000.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1001.jpg` | Fake | Fake | 0.072 | 0.928 | False | PASS |
| `fake_1002.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1003.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1004.jpg` | Fake | Fake | 0.199 | 0.801 | False | PASS |
| `fake_1005.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1006.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1007.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1008.jpg` | Fake | Fake | 0.022 | 0.978 | False | PASS |
| `fake_1009.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_101.jpg` | Fake | **Real** | 0.639 | 0.361 | False | **FAIL (False Negative)** |
| `fake_1010.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1011.jpg` | Fake | Fake | 0.081 | 0.919 | False | PASS |
| `fake_1012.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1013.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1014.jpg` | Fake | Fake | 0.020 | 0.980 | False | PASS |
| `fake_1015.jpg` | Fake | Fake | 0.290 | 0.710 | False | PASS |
| `fake_1016.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1017.jpg` | Fake | Fake | 0.001 | 0.999 | True | PASS |
| `fake_1018.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |
| `fake_1019.jpg` | Fake | Fake | 0.001 | 0.999 | False | PASS |

**Cohort Fake Accuracy: 24 / 25 (96.0%)**

---

## 3. Root Cause Analysis of False Positives

1. **Over-Sensitivity to JPEG Compression & Grain:**
   - Samples `real_1001.jpg` through `real_1005.jpg` are genuine human faces, but they feature compression artifacts and uneven lighting.
   - The 2D FFT spectral analyzer in `backend/frequency.py` treats high-frequency noise from standard lossy compression as a GAN upsampling checkerboard artifact (`anomaly_score > 0.50`), adding a $+0.15$ delta to the fake probability.
2. **Untrained Iris Embedding Perturbation:**
   - In `backend/iris.py`, the random weights in `SiameseIrisNetwork` along with extreme pupil thresholding occasionally yield high corneal glint asymmetry on natural outdoor photos where sun reflections or shadows appear in one eye.
3. **Rigid Binary Threshold:**
   - With a hard boundary at $0.48$, specimens with ambiguous signals are classified as `Fake` instead of `INCONCLUSIVE` or requesting secondary human review.
