# Final verification

Date: 16 Oct 2026. Repository: `X:\AImini`. No commit was created in this session.

## What a scan does now

An image upload is checked by file bytes, decoded, hashed, and run through face detection, optional eye-line rotation, EfficientNet-B3, and ViT-B/16. The decision is the unweighted mean of the neural fake-scores that loaded. If the models disagree by 0.45 or more, or the mean is between 0.40 and 0.60, the result is inconclusive. If neither model loads, the result is `Unavailable` and no score is filled in.

FFT, color, iris geometry, visible text marks, C2PA markers, SynthID, Gemini text, and past user labels are stored beside the decision. Their weight on the score is 0.

## Tests that ran

| Command | Result |
| --- | --- |
| `python -m pytest tests/test_forensic_core.py` | 14 passed after the watermark threshold and PDF text fixes |
| `python -m pytest tests/test_api_scan.py` | 5 passed in 24.2s. Health, bad file, path escape, real JPEG scan, PDF download, metrics without an invented ROC |
| `python scripts/evaluate_local.py --limit 8` | 16 folder-labeled images. Precision 0.727, recall 1.0, F1 0.842, ROC-AUC 0.984. Not a held-out benchmark. See `docs/MODEL_EVALUATION_REPORT.md` |
| `npx playwright test` | 3 passed in 27.6s. Upload of `dataset/real/real_0.jpg`, timeline contains `sha256` and `final_decision`, mobile width 390, reduced motion |

## Measured model behavior

- EfficientNet checkpoint loads. Label map `real=0`, `fake=1`. Sigmoid of one logit. Author threshold 0.43 is reported, not used as a silent override.
- ViT id2label on disk: `0=Real`, `1=Fake`. The fake index is read from that map.
- Smoke sample on 16 local JPEGs is in `docs/local_evaluation.json`. Three folder-real images were called fake.

## Performance on this CPU

The API suite, including model load and one 96×96 JPEG, finished in 24.2s. The 16-image smoke evaluation finished in about 40s including model load. Playwright's full server start plus one real JPEG finished in 27.6s. Per-image timings are on each scan record under `timings`. No GPU was used.

## Known limits

- SynthID is `NOT_CHECKED`. The official verifier is not installed.
- C2PA finds a metadata marker or reports `NO_MANIFEST`. It does not check a signature. The `c2pa` package is not installed.
- OCR is unavailable (`pytesseract` is not installed). Visible marks are text-like corner components.
- Color features were not ablated on a held-out set, so they do not affect the score.
- The 2,000 local JPEGs have no identity or source-video ids. External datasets were not downloaded.
- Video uses the mean of per-frame neural scores. There is no temporal network. Frame-to-frame variation is not treated as proof.
- Scans live in process memory only (about 24 records) and are not written to disk.
- The old root `index.html` was copied from `frontend/static/index.html` so the static copy no longer invents a hash when the API is down.

## Files touched

Backend: `decision.py`, `fusion.py`, `pipeline.py`, `model.py`, `main.py`, `security.py`, `scan_store.py`, `pdf_report.py`, `provenance.py`, `preprocessor.py`, `xai.py`, `iris.py`, `color_forensics.py`, `frequency.py`, `explainability.py`, `video.py`, `train.py`.

UI: `frontend/static/index.html` and `index.html`.

Tests and scripts: `tests/test_forensic_core.py`, `tests/test_api_scan.py`, `tests/e2e/scan.spec.js`, `playwright.config.js`, `scripts/evaluate_local.py`, `scripts/dataset_integrity.py`, `scripts/acquire_external_datasets.py`.

Docs: `AUDIT_BASELINE.md`, `MODEL_INVENTORY.md`, `MODEL_EVALUATION_REPORT.md`, `IMPLEMENTATION_ROADMAP.md`, this file.
