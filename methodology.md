# Scientific Forensic Methodology & Research Protocols
# Deepfake Sentinel: Dual-Domain Neural Platform

## 1. Research Hypothesis & Problem Statement

### 1.1 Hypothesis Formulation
- **Primary Hypothesis ($H_1$)**: Deepfake generators (GANs, Latent Diffusion, Autoencoders) optimize predominantly for perceptual realism in the spatial RGB domain, leaving measurable mathematical artifacts in the frequency domain (checkerboard deconvolution energy spikes) and physical biometric domain (corneal specular reflection asymmetry).
- **Secondary Hypothesis ($H_2$)**: A multi-modal Bayesian decision fusion combining global self-attention (ViT), local convolutional micro-textures (EfficientNet), physical optics (2D FFT), biometric invariants (iris corneal catchlights), and multimodal LLM visual reasoning (Gemini Vision) significantly outperforms any individual single-domain detector on out-of-distribution cross-generator generalization.

---

## 2. Experimental Design & Partitioning

To ensure zero data leakage and rigorous scientific evaluation:
1. **Total Specimens**: 2,000 verified images categorized into Authentic ($N = 1,000$), GAN Synthetics ($N = 400$), Diffusion Synthetics ($N = 350$), and Identity Morphs ($N = 250$).
2. **Partitioning**:
   - **Training Set ($70\%$, $N = 1,400$)**: Used for backbone fine-tuning with Asymmetric Focal Loss ($\gamma = 2.0$).
   - **Validation Set ($15\%$, $N = 300$)**: Used for hyperparameter tuning and early stopping.
   - **Holdout Test Set ($15\%$, $N = 300$)**: Completely unseen test partition strictly reserved for final benchmark metrics.
3. **Identity Separation**: All CelebA and FFHQ specimens were partitioned by identity group, ensuring that no individual's face in the training set appears in the validation or test sets.

---

## 3. Preprocessing & Normalization Protocols

1. **Facial Alignment**: MTCNN detects 5 facial landmarks. An affine transformation normalizes inter-pupillary distance to 70 pixels and aligns the eye axis horizontally.
2. **Dual-Crop Formulation**:
   - **Crop A (Contextual)**: Scaled full image preserving background, neck, ears, and clothing for global Vision Transformer attention.
   - **Crop B (Aligned RoI)**: Canonical $256 \times 256$ square facial crop for convolutional texture analysis.
3. **Numerical Standardization**: Tensors normalized via ImageNet channel moments:
   - $\mu = [0.485, 0.456, 0.406]$
   - $\sigma = [0.229, 0.224, 0.225]$

---

## 4. Evaluation Metrics & Statistical Significance

Detectors are evaluated on:
1. **Accuracy**: Overall fraction of correct classifications at threshold $\tau = 0.45$.
2. **Precision & Recall**: Precision evaluates false alarm cost; Recall measures synthetic catch rate.
3. **Area Under the ROC Curve (ROC-AUC)**: Threshold-independent evaluation of ranking ability across varying operational false positive rates.
4. **Brier Score**: Evaluates probability calibration accuracy $\text{BS} = \frac{1}{N} \sum (P_i - y_i)^2$.
5. **Compression Robustness**: Relative F1-score retention when images undergo lossy JPEG compression with quality factor $Q = 70$.
