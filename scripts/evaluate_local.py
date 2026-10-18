"""Smoke evaluation on folder-labeled local images.

Labels come from dataset/real and dataset/fake, not from how an image looks.
These files do not carry source-identity metadata, so this is not a
leakage-safe benchmark and must not be shown as a leaderboard score.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from backend.model import EnsembleModel
from backend.preprocessor import FaceExtractor
from backend.decision import ForensicDecisionEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out", default="docs/local_evaluation.json")
    args = parser.parse_args()
    real = sorted(Path("dataset/real").glob("*.jpg"))[: args.limit]
    fake = sorted(Path("dataset/fake").glob("*.jpg"))[: args.limit]
    if not real or not fake:
        payload = {"status": "NOT_MEASURED", "reason": "dataset/real or dataset/fake has no JPEG files."}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(payload)
        return

    ensemble = EnsembleModel(device="cpu")
    extractor = FaceExtractor(device="cpu")
    y_true = []
    y_score = []
    y_pred = []
    for label, paths in ((0, real), (1, fake)):
        for path in paths:
            image = Image.open(path).convert("RGB")
            meta = extractor.extract_face_and_landmarks(image)
            eff, vit, _detail = ensemble.predict_backbones(meta["face_img"], use_tta=False)
            decision = ForensicDecisionEngine.fuse_evidence(
                eff_fake_prob=eff,
                vit_fake_prob=vit,
                face_detected=bool(meta.get("detected")),
                eff_available=eff is not None,
                vit_available=vit is not None,
            )
            score = decision.get("fake_probability")
            if score is None:
                continue
            y_true.append(label)
            y_score.append(score)
            y_pred.append(1 if decision["prediction"] == "Fake" else 0)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    payload = {
        "status": "SMOKE_SAMPLE",
        "limitation": (
            "Folder names were used as labels. The local files have no identity or source-video manifest, "
            "so this is not a held-out, leakage-safe, or cross-dataset benchmark. "
            "Inconclusive predictions are counted as not-fake for this confusion matrix only."
        ),
        "images_per_class_requested": args.limit,
        "images_scored": len(y_true),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": matrix,
        "decision": "unweighted mean of neural fake-scores; auxiliary features excluded",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
