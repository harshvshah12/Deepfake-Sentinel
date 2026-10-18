# Model evaluation

## What was measured here

Command: `python scripts/evaluate_local.py --limit 8`

Date: 16 Oct 2026. Device: CPU. Test-time flip was off. Decision: unweighted mean of EfficientNet-B3 and ViT-B/16 fake-scores. FFT, color, iris, provenance, and watermarks did not enter the score.

Files: the first 8 sorted JPEGs in `dataset/real` and the first 8 in `dataset/fake`. The label is the folder name. These files have no identity, source-video, or generator manifest, so this is not a leakage-safe split and not a cross-dataset test.

| Measure | Value |
| --- | --- |
| Images scored | 16 |
| Precision on the fake class | 0.727 |
| Recall on the fake class | 1.000 |
| F1 on the fake class | 0.842 |
| ROC-AUC | 0.984 |
| Confusion matrix (rows true real, true fake; columns predicted real, predicted fake) | `[[5, 3], [0, 8]]` |

Inconclusive predictions were counted as not-fake for that matrix. Three of eight folder-real images were called fake. All eight folder-fake images were called fake.

Do not quote these sixteen images as the system's accuracy.

## What was not measured

- Precision-recall AUC
- Calibration (no reliability diagram, no temperature fit)
- Inference latency distribution beyond the per-scan `timings` field
- Cross-dataset generalization
- An ablation of color features. They stay at weight 0 until a held-out comparison exists
- FaceForensics++, DFDC, Celeb-DF, SynthForensics, DeepfakeBench. Those archives were not downloaded

## Checkpoint numbers that are not this test

The EfficientNet file contains its author's `final_test_metrics`: accuracy 0.9375, F1 0.9379, ROC-AUC 0.9877. That is the training job's test split, not `dataset/` in this repository. The UI must not present it as a live score. `/model-metrics` labels it `checkpoint_embedded_training_run` and leaves `roc_curve` null.

## Earlier local script

`docs/BASELINE_TEST_RESULTS.md` records a 15 Oct 2026 run of the previous fusion on 50 files (72% real, 96% fake, 84% overall). That fusion mixed auxiliary scores into the decision. It is historical, not the current engine.
