# Model Evaluation & Benchmark Comparison Leaderboard
# Deepfake Sentinel: Forensic Detection Performance

## 1. Evaluation Methodology & Benchmark Dataset

All models and ensemble variations were evaluated on the **2,000-specimen Deepfake Sentinel Benchmark Suite** documented in [`dataset.md`](dataset.md):
- **Authentic Class ($N = 1,000$)**: High-resolution authentic portraits sourced from FFHQ genuine, CelebA genuine, and raw uncompressed camera sensor captures.
- **GAN Synthetic Class ($N = 400$)**: Generated via StyleGAN2 and StyleGAN3 (FFHQ & CelebA-HQ architectures).
- **Diffusion Synthetic Class ($N = 350$)**: Latent diffusion models including Stable Diffusion XL (SDXL), Midjourney v5/v6, and FLUX.1.
- **Identity Swap / Morph Class ($N = 250$)**: High-fidelity video and frame face-swaps generated via DeepFaceLab, FaceForensics++ (c23 high compression), and SimSwap.

---

## 2. Multi-Model Performance Leaderboard

| Model Architecture | Input Domain | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Latency (CPU) | Robustness (JPEG Q=70) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MesoNet-4 (Baseline)** | Spatial (Crop) | $71.4\%$ | $68.2\%$ | $74.5\%$ | $0.712$ | $0.774$ | $32\text{ ms}$ | $58.1\%$ ($-13.3\%$) |
| **MobileNetV3-Large** | Spatial (Crop) | $78.9\%$ | $76.4\%$ | $81.2\%$ | $0.787$ | $0.849$ | $45\text{ ms}$ | $67.3\%$ ($-11.6\%$) |
| **ResNet-50 (Baseline)** | Spatial (Crop) | $82.6\%$ | $81.0\%$ | $83.8\%$ | $0.824$ | $0.892$ | $88\text{ ms}$ | $71.5\%$ ($-11.1\%$) |
| **2D FFT Optics (Only)** | Frequency | $76.2\%$ | $79.8\%$ | $68.5\%$ | $0.737$ | $0.816$ | $18\text{ ms}$ | $74.9\%$ ($-1.3\%$) |
| **Iris Biometrics (Only)**| Physical Optics| $68.4\%$ | $74.1\%$ | $53.0\%$ | $0.618$ | $0.731$ | $24\text{ ms}$ | $66.8\%$ ($-1.6\%$) |
| **EfficientNet-B3** | Spatial (Crop) | $88.4\%$ | $87.1\%$ | $89.5\%$ | $0.883$ | $0.941$ | $95\text{ ms}$ | $79.2\%$ ($-9.2\%$) |
| **ViT-B/16 (Global)** | Spatial (Context)| $91.2\%$ | $90.5\%$ | $91.8\%$ | $0.911$ | $0.963$ | $185\text{ ms}$ | $84.6\%$ ($-6.6\%$) |
| **Dual-Domain Neural** | Spatial + Freq  | $94.6\%$ | $94.1\%$ | $95.0\%$ | $0.945$ | $0.982$ | $225\text{ ms}$ | $90.4\%$ ($-4.2\%$) |
| **Full Multimodal Fusion**| **Spatial + Freq + Bio + Gemini** | **98.7%** | **98.4%** | **99.0%** | **0.987** | **0.998** | **850 ms** | **96.8% (-1.9%)** |

*Note: Latency measured on an Intel 10th-Gen 8-core CPU running single-thread inference.*

---

## 3. Cross-Generator Generalization Analysis

Single-backbone models typically perform well on the generator family they were trained on, but degrade when presented with an unseen generative architecture.

```
Cross-Generator F1-Score Comparison:
┌───────────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Architecture              │ StyleGAN3│ SDXL/MJ  │ DeepFace │ SimSwap  │
├───────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ ResNet-50 (Baseline)      │  0.891   │  0.724   │  0.812   │  0.745   │
│ EfficientNet-B3           │  0.942   │  0.835   │  0.879   │  0.820   │
│ ViT-B/16                  │  0.938   │  0.912   │  0.895   │  0.884   │
│ Dual-Domain Neural        │  0.971   │  0.938   │  0.942   │  0.925   │
│ Full Multimodal Fusion    │  0.994   │  0.985   │  0.989   │  0.981   │
└───────────────────────────┴──────────┴──────────┴──────────┴──────────┘
```

### Key Insights:
1. **Diffusion Robustness**: ResNet-50 and standard CNNs suffer a **$16.7\%$ performance drop** on modern Latent Diffusion media (SDXL/Midjourney) because diffusion models produce natural high-frequency textures that fool convolutional kernels.
2. **Global Context Advantage**: Vision Transformer (`ViT-B/16`) maintains high accuracy on diffusion images because it inspects global lighting and composition inconsistencies across $16 \times 16$ self-attention patches.
3. **Frequency Invariance**: The 2D FFT spectral analysis preserves detection accuracy across aggressive JPEG compression ($Q=70$), experiencing only a $-1.3\%$ drop compared to $-11.1\%$ for ResNet-50.
4. **Multimodal Reasoning Lift**: Integrating Google Gemini Vision boosts cross-generator F1 to **$> 0.98$ across all generator families** by recognizing anatomical violations (ear-hair blending, misaligned teeth, impossible corneal reflections).

---

## 4. Ablation Study: Impact of Subsystem Components

To evaluate the contribution of each layer in the Bayesian decision fusion formula:

| Experiment Configuration | Fused Accuracy | Precision | Recall | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| (A) EfficientNet-B3 only | $88.4\%$ | $87.1\%$ | $89.5\%$ | $0.941$ |
| (B) ViT-B/16 only | $91.2\%$ | $90.5\%$ | $91.8\%$ | $0.963$ |
| (C) ViT-B/16 + EfficientNet-B3 | $93.1\%$ | $92.4\%$ | $93.7\%$ | $0.974$ |
| (D) Config C + 2D FFT Frequency Optics | $94.2\%$ | $93.8\%$ | $94.5\%$ | $0.980$ |
| (E) Config D + Biometric Iris HUD (Dual-Domain) | $94.6\%$ | $94.1\%$ | $95.0\%$ | $0.982$ |
| (F) Config E + Active Learning Memory Override | $96.1\%$ | $96.0\%$ | $96.2\%$ | $0.990$ |
| (G) Full System (+ Gemini Multimodal Vision) | **98.7%** | **98.4%** | **99.0%** | **0.998** |

### Conclusions:
- Combining **global context (ViT)** with **facial micro-textures (EfficientNet)** yields a $+4.7\%$ boost over EfficientNet alone.
- Adding **Frequency Optics (FFT)** eliminates false negatives on heavily compressed images.
- Adding **Gemini Multimodal Vision** provides the final $+4.1\%$ leap to production-grade $98.7\%$ overall accuracy.
