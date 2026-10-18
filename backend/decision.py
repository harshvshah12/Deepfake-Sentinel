"""Neural-only decision engine.

Auxiliary measurements (FFT, color, iris, provenance, watermarks, user labels,
and language-model comments) are reported beside the decision. They do not
change the probability unless a held-out calibration file explicitly enables
them. No such file is shipped with this repository.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class ForensicDecisionEngine:
    CONFIDENT_REAL_THRESHOLD = 0.40
    CONFIDENT_FAKE_THRESHOLD = 0.60
    HIGH_DISCORDANCE_DELTA = 0.45
    METHOD = (
        "Unweighted mean of available neural fake-class scores. "
        "A result is inconclusive when the models disagree by 0.45 or more, "
        "or when the mean falls between 0.40 and 0.60. "
        "FFT, color, iris geometry, provenance, watermarks, user feedback, "
        "and Gemini text do not change this score."
    )

    @classmethod
    def fuse_evidence(
        cls,
        eff_fake_prob: Optional[float] = None,
        vit_fake_prob: Optional[float] = None,
        spectral_score: float = 0.0,
        iris_data: Optional[Dict[str, Any]] = None,
        color_data: Optional[Dict[str, Any]] = None,
        face_detected: bool = True,
        provenance_data: Optional[Dict[str, Any]] = None,
        c2pa_data: Optional[Dict[str, Any]] = None,
        synthid_data: Optional[Dict[str, Any]] = None,
        gemini_fake_score: Optional[float] = None,
        feedback_override: Optional[Dict[str, Any]] = None,
        eff_available: bool = True,
        vit_available: bool = True,
    ) -> Dict[str, Any]:
        del spectral_score, iris_data, color_data, provenance_data, c2pa_data, synthid_data, gemini_fake_score

        warnings = []
        if feedback_override:
            warnings.append(
                "A previous user label exists for a similar file. "
                "That label is not verified ground truth and does not change the model score."
            )

        available = []
        if eff_available and eff_fake_prob is not None:
            available.append(("efficientnet_b3", float(np.clip(eff_fake_prob, 0.0, 1.0))))
        elif not eff_available:
            warnings.append("EfficientNet-B3 weights were not loaded. Its score was not invented.")
        if vit_available and vit_fake_prob is not None:
            available.append(("vit_b16", float(np.clip(vit_fake_prob, 0.0, 1.0))))
        elif not vit_available:
            warnings.append("ViT-B/16 weights were not loaded. Its score was not invented.")

        weights = {name: 0.0 for name, _ in (("efficientnet_b3", 0), ("vit_b16", 0))}
        if not available:
            return cls._unavailable(warnings)

        for name, _prob in available:
            weights[name] = round(1.0 / len(available), 4)

        scores = [prob for _name, prob in available]
        fused_fake = float(np.mean(scores))
        fused_real = float(1.0 - fused_fake)
        discordance = abs(available[0][1] - available[1][1]) if len(available) == 2 else None

        if discordance is None:
            discordance_level = "UNKNOWN"
            warnings.append("Only one neural model produced a score. Uncertainty is higher.")
        elif discordance >= cls.HIGH_DISCORDANCE_DELTA:
            discordance_level = "HIGH"
        elif discordance >= 0.25:
            discordance_level = "MODERATE"
        else:
            discordance_level = "LOW"

        if not face_detected:
            warnings.append(
                "No face was detected. These checkpoints were trained on facial images, "
                "so a non-face photograph is outside the documented training task."
            )

        disagree = discordance is not None and discordance >= cls.HIGH_DISCORDANCE_DELTA
        if disagree:
            prediction = "Inconclusive"
            badge = "INCONCLUSIVE — MODELS DISAGREE"
            inconclusive = True
            confidence = float(max(0.5, 1.0 - discordance))
            notes = (
                f"EfficientNet fake-score {available[0][1]:.3f} and ViT fake-score {available[1][1]:.3f} "
                f"differ by {discordance:.3f}. The mean is reported, and the verdict is inconclusive."
            )
        elif fused_fake >= cls.CONFIDENT_FAKE_THRESHOLD and (face_detected or fused_fake >= 0.85):
            prediction = "Fake"
            badge = "SYNTHETIC (NEURAL SCORE)"
            inconclusive = False
            confidence = fused_fake
            notes = f"Mean neural fake-score is {fused_fake:.3f}, at or above {cls.CONFIDENT_FAKE_THRESHOLD:.2f}."
        elif fused_fake <= cls.CONFIDENT_REAL_THRESHOLD and (face_detected or fused_fake <= 0.15):
            prediction = "Real"
            badge = "LOWER NEURAL FAKE-SCORE"
            inconclusive = False
            confidence = fused_real
            notes = f"Mean neural fake-score is {fused_fake:.3f}, at or below {cls.CONFIDENT_REAL_THRESHOLD:.2f}."
        else:
            prediction = "Inconclusive"
            badge = "INCONCLUSIVE"
            inconclusive = True
            confidence = float(max(fused_fake, fused_real))
            if not face_detected:
                notes = (
                    f"Mean neural fake-score is {fused_fake:.3f}, and no face was detected. "
                    "The result stays inconclusive."
                )
            else:
                notes = (
                    f"Mean neural fake-score is {fused_fake:.3f}, inside the open interval "
                    f"({cls.CONFIDENT_REAL_THRESHOLD:.2f}, {cls.CONFIDENT_FAKE_THRESHOLD:.2f})."
                )

        if len(available) == 1 and prediction != "Inconclusive":
            inconclusive = False

        uncertainty_reasons = list(warnings)
        if inconclusive:
            uncertainty_reasons.append(notes)
        level = "high" if inconclusive or discordance_level in {"HIGH", "UNKNOWN"} else (
            "moderate" if discordance_level == "MODERATE" else "low"
        )

        return {
            "prediction": prediction,
            "verdict_category": (
                "Synthetic" if prediction == "Fake" else "Authentic" if prediction == "Real" else "Inconclusive"
            ),
            "verdict_badge": badge,
            "fake_probability": round(fused_fake, 4),
            "real_probability": round(fused_real, 4),
            "confidence": round(confidence, 4),
            "authenticity_index": round(fused_real * 100, 1),
            "discordance_level": discordance_level,
            "discordance_value": None if discordance is None else round(discordance, 4),
            "is_inconclusive": inconclusive or prediction == "Inconclusive",
            "feedback_applied": False,
            "decision_notes": notes,
            "method": cls.METHOD,
            "score_semantics": "uncalibrated_neural_mean",
            "weights_used": weights,
            "raw_signals": {
                "eff_fake_prob": None if eff_fake_prob is None else round(float(eff_fake_prob), 4),
                "vit_fake_prob": None if vit_fake_prob is None else round(float(vit_fake_prob), 4),
            },
            "uncertainty": {"level": level, "reasons": uncertainty_reasons},
            "warnings": warnings,
            "auxiliary_weights": {
                "spectral": 0.0,
                "iris": 0.0,
                "color": 0.0,
                "provenance": 0.0,
                "watermark": 0.0,
                "gemini": 0.0,
                "user_feedback": 0.0,
            },
        }

    @classmethod
    def _unavailable(cls, warnings: list) -> Dict[str, Any]:
        notes = "No neural model produced a score. This is not a prediction."
        return {
            "prediction": "Unavailable",
            "verdict_category": "Unavailable",
            "verdict_badge": "MODEL UNAVAILABLE",
            "fake_probability": None,
            "real_probability": None,
            "confidence": None,
            "authenticity_index": None,
            "discordance_level": "UNKNOWN",
            "discordance_value": None,
            "is_inconclusive": True,
            "feedback_applied": False,
            "decision_notes": notes,
            "method": cls.METHOD,
            "score_semantics": "unavailable",
            "weights_used": {},
            "raw_signals": {"eff_fake_prob": None, "vit_fake_prob": None},
            "uncertainty": {"level": "high", "reasons": warnings + [notes]},
            "warnings": warnings,
            "auxiliary_weights": {},
        }
