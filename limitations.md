# Engineering Limitations & Failure Modes
# Deepfake Sentinel: Forensic Detection Platform

## 1. Boundary Conditions & Known Limitations

Transparent analysis of edge cases and physical boundaries ensures realistic deployment expectations:

### 1.1 Resolution & Severe Downsampling Thresholds
- **Threshold**: Facial bounding boxes smaller than $80 \times 80$ pixels.
- **Failure Mode**: When facial images undergo extreme downsampling (e.g., thumbnail crops or low-bitrate surveillance footage), high-frequency dermal micro-textures and corneal specular reflections are completely obliterated.
- **Mitigation**: The system outputs an epistemic confidence warning indicating insufficient spatial resolution when MTCNN reports eye inter-pupillary distance $< 30$ pixels.

### 1.2 Multi-Pass Lossy Social Media Re-Compression
- **Threshold**: Re-compression quality factor $Q < 50$ across multiple transcoding passes (e.g., WhatsApp, Telegram, or WeChat re-encodings).
- **Failure Mode**: Heavy discrete cosine transform (DCT) block quantization introduces $8 \times 8$ pixel grid artifacts that mask subtle GAN transposed convolution upsampling peaks in the 2D FFT spectrum.
- **Mitigation**: The 2D FFT pipeline applies median bandpass filtering to isolate natural camera sensor noise from block quantization grids.

### 1.3 Extreme Occlusions & Severe Non-Frontal Gaze
- **Threshold**: Yaw or pitch angles $> 45^\circ$, or physical occlusions covering $> 50\%$ of the face (e.g., tinted sunglasses, heavy surgical masks).
- **Failure Mode**: Corneal specular reflection analysis fails if pupils cannot be isolated, and MTCNN landmark detection confidence drops below the operational threshold.
- **Mitigation**: In such cases, the Bayesian decision fusion automatically shifts weight from the Biometric Iris module to the global Vision Transformer context.

### 1.4 Stylized, Artistic & CGI Media
- **Threshold**: 3D computer animation renders (e.g., Unreal Engine 5 Metahumans), anime, digital oil paintings, or caricatures.
- **Failure Mode**: Stylized portraits naturally lack real human skin pores and photographic sensor grain, potentially triggering false-positive synthetic classifications.
- **Mitigation**: The Google Gemini Vision reasoner detects artistic stylization and contextualizes the verdict accordingly.

### 1.5 White-Box Adaptive Adversarial Attacks
- **Threshold**: Projected Gradient Descent (PGD) with perturbation budget $\epsilon > 0.12$ when the attacker possesses full white-box gradient access to both ViT and EfficientNet backbones.
- **Failure Mode**: High-magnitude perturbations can force misclassification, though the perturbations become visually perceptible to human observers.
