# Model inventory

Recorded from the checkpoints and configs on this machine on 16 Oct 2026. Training-set metrics below are the numbers stored inside the EfficientNet file. They were not remeasured on `dataset/` until `docs/local_evaluation.json` says otherwise.

## EfficientNet-B3

- File: `backend/weights/deepfakefusion/best_efficientnet_b3_deepfake_detector_399k.pth` (43,372,031 bytes)
- Architecture: torchvision `efficientnet_b3`, classifier replaced with one linear unit
- Weights load: `state_dict` key, 574 tensors, `eval()` after load
- Input: 256×256, ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Label map in the checkpoint: `real=0`, `fake=1`
- Output: one logit. The scan uses `sigmoid(logit)` as a fake-class score. It is not temperature-scaled on this machine.
- Checkpoint operating threshold: `0.43`
- Checkpoint version string: `deepfakefusion_v2_sixsource`
- Checkpoint `final_test_metrics` (author's training job, not this folder): accuracy 0.9375, precision 0.9313, recall 0.9446, F1 0.9379, ROC-AUC 0.9877
- Failure: if the file is missing or the state dict does not load, the model status is `available: false` and no score is invented

## ViT-B/16

- Checkpoint: Hugging Face `dima806/deepfake_vs_real_image_detection`
- Local config id2label: `0 = Real`, `1 = Fake`
- Input: the checkpoint processor, image size 224
- Output: softmax probability of the Fake index. The index is read from `id2label`, not hard-coded.
- Task: deepfake-versus-real image classification. A high score is not evidence that every diffusion model was in the training set.
- This deployment has no separate measured accuracy for this checkpoint on `dataset/`.
- Failure: if the weights cannot be loaded, status is `available: false`

## MTCNN

- Package: `facenet_pytorch.MTCNN`
- Role: face box and five landmarks. If detection misses, the scan center-crops and says the face was not detected.

## Not neural detectors

| Module | What it does | In the score? |
| --- | --- | --- |
| 2D FFT | High-frequency energy and an 8-bin radial profile | No |
| Color statistics | Channel correlation and YCbCr means inside the face box | No |
| Iris geometry | Pupil and specular measurements when landmarks exist | No |
| Siamese iris network | Random weights. Labeled `UNTRAINED_RANDOM_WEIGHTS` | No |
| Visible corner text | Connected components that look like a text line. OCR is unavailable | No |
| SynthID | `NOT_CHECKED`. No official verifier is installed | No |
| C2PA | Metadata-region marker only. Never reported as a valid signature | No |
| Gemini text | Optional commentary when a key is present. Not a probability | No |
| User feedback | Stored and shown. Does not override the score | No |

## Fusion

Unweighted mean of the neural fake-scores that actually ran. Inconclusive when the two scores differ by 0.45 or more, or when the mean is between 0.40 and 0.60. No face keeps a strong agreement, and a weak or disagreeing result stays inconclusive.
