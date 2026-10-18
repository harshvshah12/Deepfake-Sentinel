# Implementation roadmap

| Milestone | State | Evidence |
| --- | --- | --- |
| 0 Baseline audit | Done | `docs/AUDIT_BASELINE.md`, `docs/MODEL_INVENTORY.md` |
| 1 Detection engine | Done | `backend/decision.py`, `backend/pipeline.py`, `tests/test_forensic_core.py` |
| 2 Dataset and validation | Partial | Local inventory of 1000/1000 folder-labeled JPEGs. External sets not downloaded. Smoke numbers in `docs/MODEL_EVALUATION_REPORT.md` |
| 3 Forensic timeline | Done for images | `POST /scan` SSE events with scan id, stage id, status, timestamp. Playwright covers one upload |
| 4 Color analysis | Measured, not used | `backend/color_forensics.py`. No ablation showed a benefit, so the weight is 0 |
| 5 Watermarks and provenance | Done within the tools on this machine | Visible text-line detector, SynthID `NOT_CHECKED`, C2PA marker vs missing marker. Tests in `tests/test_forensic_core.py` |
| 6 Explainability | Done | Grad-CAM or `UNAVAILABLE`, FFT image, evidence ledger on the scan |
| 7 PDF report | Done | `GET /report/{scan_id}` from the stored scan. Tests build real, fake, inconclusive, and failed reports |
| 8 Video | Partial | Per-frame neural mean, video metadata, variance reported and not treated as proof. No temporal network |
| 9 Frontend | Done on the existing page | Timeline, inconclusive state, PDF and JSON from the scan, drag-and-drop, reduced motion |
| 10 API and privacy | Done for the listed checks | Content sniffing, size limits, dataset path confinement, 30 requests/minute, in-memory retention |
| 11 Playwright | See the verification report | `tests/e2e/scan.spec.js` |
| 12 Release audit | See `docs/FINAL_VERIFICATION_REPORT.md` | |

## Not claimed

- A calibrated probability
- A universal diffusion detector
- Blood flow or health from a still
- A legal finding
