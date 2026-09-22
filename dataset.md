# Deepfake Sentinel — Comprehensive Forensic Dataset Registry

*Last Updated: 2026-08-22 | Total Verified Specimens: 2,000*

## Dataset Distribution Overview

| Category | Class | Specimen Count | Primary Generative Architecture / Source | Preprocessing Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Authentic Portraits** | Real | 1,000 | FFHQ, CelebA Genuine, Unsplash Sensor Captures | Full Portrait + MTCNN Aligned Crop |
| **GAN Synthetics** | Fake | 400 | StyleGAN2 / StyleGAN3 (FFHQ & CelebA-HQ) | 256x256 MTCNN Aligned Face Crop |
| **Diffusion Synthetics** | Fake | 350 | Stable Diffusion XL, Midjourney v5/v6, FLUX.1 | Full Uncropped + Aligned Facial RoI |
| **FaceSwap / Morphed** | Fake | 250 | DeepFaceLab, FaceForensics++, SimSwap | Boundary Mask + Margin Crop |
| **TOTAL** | — | **2,000** | **Multi-Modal Forensic Distribution** | **Dual-Domain (Spatial + Context)** |

---

## Detailed Provenance & Source Registry

### 1. Authentic Human Baseline (dataset/real/)
- **Source Repositories**:
  - [NVIDIA Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
  - [CelebFaces Attributes (CelebA)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - High-resolution uncompressed optical camera captures (Canon EOS, Sony Alpha, iPhone RAW).
- **Physical Characteristics**: Authentic Bayer color filter array (CFA) demosaicing, natural 1/f Fourier power spectrum decay, bilateral corneal specular catchlight symmetry, natural skin porosity and subsurface scattering.
- **Preprocessing**: 
  - Standardized to 3-channel RGB.
  - Paired MTCNN 5-point landmark facial crops ( \times 256$) and uncropped full-scene context ( \times 1024$).

### 2. Generative Adversarial Networks (dataset/fake/ — Sub-Distribution A)
- **Source Repositories**:
  - [NVIDIA StyleGAN3 (Alias-Free GAN)](https://github.com/NVlabs/stylegan3)
  - [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
- **Artifact Signatures**: 
  - Periodic checkerboard frequency peaks in 2D FFT magnitude spectrum caused by transposed convolution upsampling.
  - Corneal reflection asymmetry (different light source geometries reflected in left vs. right eye).
  - Background morphing and ear/earring geometric blending anomalies.

### 3. Latent Diffusion Models (dataset/fake/ — Sub-Distribution B)
- **Source Models**:
  - Stable Diffusion XL (SDXL 1.0)
  - Midjourney v5.2 / v6.0
  - FLUX.1 [schnell / dev]
  - Google Imagen 3 / Gemini
- **Artifact Signatures**: 
  - Subtle spatial residual noise distributions in YCrCb chrominance channels.
  - Micro-smoothing of epidermal textures (airbrushed plastic sheen).
  - High-frequency phase distribution shifts detectable via Daubechies DWT sub-band energy ratios.

### 4. Deepfake Face-Swap & Re-Faced Morphs (dataset/fake/ — Sub-Distribution C)
- **Source Frameworks**:
  - [FaceForensics++ (Deepfakes, FaceSwap, Face2Face, NeuralTextures)](https://github.com/ondyari/FaceForensics)
  - [DeepFaceLab (DFL)](https://github.com/iperov/DeepFaceLab)
  - [SimSwap & Roop ReActor](https://github.com/neuralchen/SimSwap)
- **Artifact Signatures**:
  - Boundary gradient discontinuities along the blending mask seam (jawline, forehead border).
  - Color space mismatch between swapped face RoI and original background neck/ears.
  - Blur and resolution mismatch across boundary interfaces.

---

## Preprocessing & Normalization Protocol

1. **Facial Alignment**: Multi-Task Cascaded Convolutional Networks (MTCNN) detects facial bounding box and 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner).
2. **Affine Transformation**: Images are rotationally normalized based on inter-pupillary line horizontal alignment.
3. **Dual-Domain Resizing**:
   - **Backbone 1 (EfficientNet-B3)**: Tight aligned facial crop resized to  \times 256$ with ImageNet normalization $.
   - **Backbone 2 (ViT-B/16)**: Full contextual portrait resized to  \times 224$ patches ( \times 16$).
4. **Augmentation**: Test-Time Augmentation (TTA) with bilateral horizontal flipping, subtle color jittering, and Gaussian noise simulation.

---

## Quality Assurance & Validation Standards

- **Corrupted Image Elimination**: Checked with PIL header validation + OpenCV decoding.
- **Aspect Ratio Integrity**: Bounding boxes padded by 25% margin to preserve chin and hairline context without aspect distortion.
- **Zero Label Leakage**: Hash-based deduplication (SHA-256 and pHash) guarantees zero specimen overlap between train and test evaluation splits.
