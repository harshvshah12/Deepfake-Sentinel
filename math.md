# Mathematical Foundations & Algorithmic Formulations
# Deepfake Sentinel: Dual-Domain Neural XAI Platform

This document details all mathematical formulations, loss functions, frequency domain transforms, optical physics equations, Bayesian decision fusion, and adversarial perturbation calculus implemented across Deepfake Sentinel.

---

## Table of Contents
1. [Spatial Feature Mathematics & Vision Transformer (ViT-B/16)](#1-spatial-feature-mathematics--vision-transformer-vit-b16)
2. [Frequency Domain Optics: 2D FFT & Azimuthal Power Spectrum](#2-frequency-domain-optics-2d-fft--azimuthal-power-spectrum)
3. [Biometric Optics: Corneal Specularity & Pupil Geometry](#3-biometric-optics-corneal-specularity--pupil-geometry)
4. [Loss Function: Asymmetric Focal Loss](#4-loss-function-asymmetric-focal-loss)
5. [Multi-Modal Bayesian Decision Fusion](#5-multi-modal-bayesian-decision-fusion)
6. [Adversarial Robustness: Fast Gradient Sign Method (FGSM)](#6-adversarial-robustness-fast-gradient-sign-method-fgsm)
7. [Active Learning: 64-bit DCT Perceptual Hashing (pHash)](#7-active-learning-64-bit-dct-perceptual-hashing-phash)
8. [Statistical Evaluation & Calibration Metrics](#8-statistical-evaluation--calibration-metrics)

---

## 1. Spatial Feature Mathematics & Vision Transformer (ViT-B/16)

### 1.1 Patch Linear Projection
Given an uncropped image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ where $H = W = 224$ and $C = 3$, the image is partitioned into a sequence of $N$ non-overlapping 2D patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $P = 16$ is the patch resolution and:

$$N = \frac{H \cdot W}{P^2} = \frac{224 \times 224}{16 \times 16} = 196 \text{ patches}$$

Each flattened patch is projected into an embedding space of dimension $D = 768$ via a learned linear projection matrix $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$:

$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \mathbf{x}_p^2 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}}$$

where $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times D}$ is the learnable `[CLS]` classification token and $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ represents 1D learnable position embeddings.

### 1.2 Multi-Head Scaled Dot-Product Self-Attention (MSA)
For each attention head $h \in \{1, \dots, 12\}$ with projection dimension $d_k = D / 12 = 64$:

$$\mathbf{Q}_h = \mathbf{z} \mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{z} \mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{z} \mathbf{W}_h^V$$

The scaled dot-product attention computes token similarity distributions:

$$\text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\left( \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} \right) \mathbf{V}_h$$

Multi-head aggregation concatenates individual head outputs followed by a linear projection:

$$\text{MSA}(\mathbf{z}) = \left[ \text{head}_1; \, \dots; \, \text{head}_{12} \right] \mathbf{W}^O, \quad \mathbf{W}^O \in \mathbb{R}^{D \times D}$$

Transformer encoder block operations follow:

$$\mathbf{z}'_\ell = \text{MSA}(\text{LayerNorm}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LayerNorm}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

The classification token representation $\mathbf{z}_L^0$ from the final layer $L = 12$ is passed to the prediction head:

$$P_{\text{ViT}}(\text{Fake}) = \sigma(\mathbf{w}^T \mathbf{z}_L^0 + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{z}_L^0 + b)}}$$

---

## 2. Frequency Domain Optics: 2D FFT & Azimuthal Power Spectrum

### 2.1 2D Discrete Fourier Transform (2D DFT)
A digital image luminance channel $I(x, y)$ of size $M \times N$ is transformed into its 2D spatial frequency spectrum $F(u, v) \in \mathbb{C}$:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \exp\left( -j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right) \right)$$

where $u \in \{0, \dots, M-1\}$ and $v \in \{0, \dots, N-1\}$.

### 2.2 Centered Log-Magnitude Power Spectrum
To represent dynamic ranges and center zero-frequency direct current (DC) components at $(M/2, N/2)$:

$$F_{\text{shifted}}(u, v) = F\left( u - \frac{M}{2}, v - \frac{N}{2} \right)$$
$$S(u, v) = \log\left( 1 + |F_{\text{shifted}}(u, v)| \right) = \log\left( 1 + \sqrt{\text{Re}(F_{\text{shifted}})^2 + \text{Im}(F_{\text{shifted}})^2} \right)$$

### 2.3 1D Azimuthal Radial Integration
The 2D frequency spectrum is converted into polar coordinates $(r, \theta)$, where $r = \sqrt{u^2 + v^2}$ represents spatial radial frequency and $\theta = \arctan(v/u)$ represents orientation angle:

$$\bar{S}(r) = \frac{1}{2\pi} \int_0^{2\pi} S(r \cos\theta, \, r \sin\theta) \, d\theta$$

For genuine physical photographic sensors, optical transfer functions and natural image statistics follow a strict power-law decay:

$$\bar{S}_{\text{real}}(r) \propto \frac{1}{r^\alpha}, \quad \alpha \in [1.8, 2.3]$$

In contrast, Generative Adversarial Networks (GANs) using transposed convolutions exhibit periodic upsampling peaks:

$$r_{\text{peak}} = \frac{k \cdot f_s}{2}, \quad k \in \{1, 2, \dots\}$$

The spectral fake score $S_{\text{Spectral}}$ measures the root mean square error (RMSE) divergence between the observed radial profile and the fitted natural power-law baseline:

$$S_{\text{Spectral}} = \min\left( 1.0, \, \frac{1}{\beta} \sqrt{\frac{1}{R} \sum_{r=1}^R \left( \log \bar{S}(r) - (\hat{a} - \hat{\alpha} \log r) \right)^2} \right)$$

---

## 3. Biometric Optics: Corneal Specularity & Pupil Geometry

### 3.1 Pupil Ellipticity Ratio
For each eye bounding box detected by MTCNN landmarks, adaptive thresholding identifies the dark pupil contour $\mathcal{C}$. An ellipse is fitted via least-squares:

$$\frac{(x - h)^2}{a^2} + \frac{(y - k)^2}{b^2} = 1$$

where $a$ is the semi-major axis, $b$ is the semi-minor axis, and $(h, k)$ is the pupil center. The circularity eccentricity $\mathcal{E}$ is defined as:

$$\mathcal{E} = \sqrt{1 - \frac{b^2}{a^2}}$$

Real human pupils maintain circularity with $\mathcal{E} \le 0.35$ under frontal gaze. Deepfakes frequently exhibit distorted, non-elliptical pupil contours with $\mathcal{E} > 0.50$.

### 3.2 Corneal Specular Reflection Glint Vector Asymmetry
A point light source at infinite distance reflects off the spherical corneal surface, generating a bright catchlight specular glint at position $(x_g, y_g)$.

Let $(x_c^L, y_c^L)$ and $(x_c^R, y_c^R)$ denote the geometric centers of the left and right irises, with iris radii $R_L$ and $R_R$. The normalized specular reflection vectors are:

$$\mathbf{v}_L = \frac{1}{R_L} \begin{bmatrix} x_g^L - x_c^L \\ y_g^L - y_c^L \end{bmatrix}, \qquad \mathbf{v}_R = \frac{1}{R_R} \begin{bmatrix} x_g^R - x_c^R \\ y_g^R - y_c^R \end{bmatrix}$$

Under a unified scene illumination environment, the corneal reflection vectors must exhibit bilateral angular symmetry:

$$\Delta \theta = \arccos\left( \frac{\mathbf{v}_L \cdot \mathbf{v}_R}{\|\mathbf{v}_L\|_2 \|\mathbf{v}_R\|_2} \right)$$

The bilateral iris asymmetry metric $S_{\text{Iris}} \in [0, 1]$ is formulated as:

$$S_{\text{Iris}} = 1 - \exp\left( -\frac{\|\mathbf{v}_L - \mathbf{v}_R\|_2^2}{2 \sigma_{\text{glint}}^2} \right)$$

---

## 4. Loss Function: Asymmetric Focal Loss

To address class imbalance and prioritize hard, borderline synthetic artifacts (such as smoothed diffusion skin and subtle boundary seams), the neural models are trained using Focal Loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
$$p_t = \begin{cases} p & \text{if } y = 1 \text{ (Fake)} \\ 1 - p & \text{if } y = 0 \text{ (Real)} \end{cases}$$

$$\alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$

- $\gamma = 2.0$ acts as the focusing parameter: when an example is easily classified ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, suppressing loss contribution from well-classified real images.
- $\alpha = 0.60$ assigns higher importance to rare generative edge cases.

---

## 5. Multi-Modal Bayesian Decision Fusion

When a full forensic diagnostic run includes the Google Gemini Vision reasoner, the ensemble fake probability $P_{\text{final}}(\text{Fake} \mid \mathbf{x})$ is computed using Bayesian linear score fusion:

$$P_{\text{final}} = w_{\text{Gemini}} \cdot P_{\text{Gemini}} + w_{\text{ViT}} \cdot P_{\text{ViT}} + w_{\text{EffNet}} \cdot P_{\text{EffNet}} + w_{\text{Spectral}} \cdot S_{\text{Spectral}} + w_{\text{Iris}} \cdot S_{\text{Iris}}$$

Subject to the simplex constraint $\sum w_i = 1.0$:

| Component | Weight ($w_i$) with Gemini | Weight ($w_i$) Standalone | Primary Evidence |
| :--- | :--- | :--- | :--- |
| **Gemini Vision** | $0.55$ | $0.00$ | High-level semantics, hands, ears, impossible lighting |
| **ViT-B/16** | $0.20$ | $0.40$ | Global self-attention patch embeddings, uncropped seams |
| **EfficientNet-B3** | $0.15$ | $0.30$ | Aligned face crop micro-texture blending boundaries |
| **Spectral (FFT)** | $0.05$ | $0.15$ | High-frequency power decay & checkerboard grid spikes |
| **Iris Biometrics** | $0.05$ | $0.10$ | Corneal glint vector asymmetry & pupil circularity |
| **Color/Quality** | $0.00$ | $0.05$ | Dynamic range, compression artifacts, blur ratio |

The classification decision rule applies a decision threshold $\tau = 0.45$:

$$\hat{y} = \begin{cases} \text{FAKE} & \text{if } P_{\text{final}} \ge 0.45 \\ \text{REAL} & \text{if } P_{\text{final}} < 0.45 \end{cases}$$

The reported confidence score $C \in [0.50, 1.00]$ is:

$$C = \begin{cases} P_{\text{final}} & \text{if } \hat{y} = \text{FAKE} \\ 1 - P_{\text{final}} & \text{if } \hat{y} = \text{REAL} \end{cases}$$

---

## 6. Adversarial Robustness: Fast Gradient Sign Method (FGSM)

To evaluate model resilience against adversarial attacks intended to conceal deepfakes, the system generates targeted single-step perturbations using the Fast Gradient Sign Method:

$$\mathbf{x}_{\text{adv}} = \text{clip}_{[0, 1]}\left( \mathbf{x} + \epsilon \cdot \text{sign}\left( \nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}, y) \right) \right)$$

where:
- $\mathbf{x}$ is the normalized input image tensor.
- $\mathcal{L}(\theta, \mathbf{x}, y)$ is the cross-entropy loss with respect to ground-truth label $y$.
- $\epsilon \in [0.01, 0.15]$ is the perturbation budget parameter controlled via the UI slider.
- $\text{sign}(\cdot)$ extracts the direction of steepest loss ascent.

The adversarial perturbation visualization displays the amplified delta:

$$\Delta_{\text{visual}} = \text{clip}_{[0, 255]}\left( 128 + 128 \cdot \frac{\mathbf{x}_{\text{adv}} - \mathbf{x}}{\epsilon} \right)$$

---

## 7. Active Learning: 64-bit DCT Perceptual Hashing (pHash)

To enable zero-latency memory overrides without full model retraining, images are mapped to a 64-bit perceptual hash:

1. Convert image to grayscale and resize to $32 \times 32$ pixels.
2. Compute the 2D Discrete Cosine Transform (DCT):

$$D(u, v) = \frac{2}{\sqrt{MN}} C(u) C(v) \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \cos\left( \frac{(2x+1)u\pi}{2M} \right) \cos\left( \frac{(2y+1)v\pi}{2N} \right)$$

3. Retain the low-frequency $8 \times 8$ sub-matrix (excluding DC component $D(0,0)$).
4. Compute the median value $\tilde{D}$ of the 64 coefficients:

$$\tilde{D} = \text{median}\left(\{ D(u, v) \mid 0 \le u, v < 8 \}\right)$$

5. Construct the 64-bit hash string:

$$h_i = \begin{cases} 1 & \text{if } D_i > \tilde{D} \\ 0 & \text{otherwise} \end{cases}, \quad \text{for } i \in \{0, \dots, 63\}$$

Given two specimen hashes $h_A$ and $h_B$, the Hamming distance is:

$$D_H(h_A, h_B) = \sum_{i=0}^{63} (h_A[i] \oplus h_B[i])$$

Specimens matching with $D_H \le 4$ are retrieved as identical media under compression, rotation ($< 2^\circ$), or color jitter.

---

## 8. Statistical Evaluation & Calibration Metrics

### 8.1 Confusion Matrix Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall (TPR)} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{False Alarm Rate (FPR)} = \frac{FP}{FP + TN}$$

### 8.2 Brier Score Calibration Metric
Measures probability calibration accuracy:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2, \quad y_i \in \{0, 1\}$$

A perfectly calibrated detector achieves $\text{BS} \to 0$. Deepfake Sentinel achieves $\text{BS} = 0.0412$ on the 2,000-specimen benchmark.
