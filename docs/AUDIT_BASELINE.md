# Baseline audit

Date: 16 Oct 2026. Tree: `X:\AImini` (`harshvshah12/Deepfake-Sentinel`). The empty `X:\deep` folder is not the application.

## What was already there

- FastAPI app in `backend/main.py`, one-page UI in `frontend/static/index.html`
- Endpoints used by that page: `/health`, `/samples`, `/sample-image`, `/predict`, `/predict-frame`, `/feedback`, `/feedback-stats`, `/model-metrics`, `/test-gemini-key`
- Detectors: EfficientNet-B3 checkpoint, ViT-B/16 from Hugging Face, MTCNN, FFT, Grad-CAM, EXIF, a byte search that was called C2PA and SynthID, corner edges called a watermark, untrained iris network
- Dataset folders: `dataset/real` (1000 JPEG files) and `dataset/fake` (1000 JPEG files). No identity manifest.
- Tests at the repo root: `test_bayesian_fusion.py`, `test_ai_vs_real.py`, `test_candidates.py`. They are scripts, not a pytest suite.
- An earlier local script run, recorded in `docs/BASELINE_TEST_RESULTS.md` on 15 Oct 2026, scored 18/25 real and 24/25 fake on the first sorted JPEGs. Those figures belong to the old fusion, not to a published benchmark.

## Failures found before this change

1. `/model-metrics` returned accuracy, AUC, a confusion matrix, and an ROC curve that were not measured here.
2. The image page, if `/predict` failed, built a result in the browser: random SHA-256, a C2PA issuer, and a SynthID flag taken from the filename.
3. JSON export minted another random hash.
4. The chart modal did the same if `/model-metrics` failed, including an iris radar with fixed numbers.
5. Grad-CAM failure painted a red circle and returned it as the heatmap.
6. Missed eye landmarks were filled with circularity 0.85 and specular area 10.
7. SynthID was inferred from high-frequency energy and from byte strings anywhere in the file, including compressed JPEG data.
8. A C2PA byte hit could be described as a valid signature. The UI said "cryptographically signed".
9. User feedback replaced the model score with 0.98 or 0.999.
10. `/sample-image` trusted the query path.
11. EfficientNet and ViT were averaged with FFT, iris, and color using weights that were not fit on a held-out split in this repo.
12. The PDF generator existed and was not wired to a download route.
13. `backend/train.py` saved over the production checkpoint path.

## What this pass changed

The scan record, the API, the PDF, and the page now share one decision: the mean of the neural scores that loaded. Auxiliary measurements stay in the record with `validated_contribution: false`. Model load failure stays `Unavailable`.

## Still blocked

- Official SynthID verifier: not installed, no credential
- C2PA signature check: `c2pa` package is not installed
- OCR: `pytesseract` is not installed
- FaceForensics++, DFDC, and Celeb-DF: not downloaded. Access is gated. See `dataset/metadata/EXTERNAL_DATASETS.md`
- No identity-level split exists for the 2,000 local JPEGs, so a score on that folder is not a leakage-safe benchmark
