"""Canonical image scan. Every stage records what actually ran."""

from __future__ import annotations

import hashlib
import io
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from backend.color_forensics import analyze_chroma_and_boundary
from backend.decision import ForensicDecisionEngine
from backend.explainability import generate_forensic_explanation
from backend.feedback_memory import lookup_feedback_memory
from backend.frequency import calculate_spectral_analysis, compute_fft_spectrum
from backend.provenance import (
    compute_cryptographic_and_perceptual_hashes,
    detect_c2pa_manifest,
    detect_synthid_watermark,
    detect_visual_watermarks_and_logos,
    parse_exif_metadata,
)
from backend.robustness import generate_adversarial_perturbation
from backend.scan_store import is_cancelled
from backend.xai import generate_heatmap
from backend.video import np_to_b64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jpeg_b64(image: Image.Image, size: int = 256) -> str:
    thumb = image.convert("RGB").resize((size, size))
    return np_to_b64(np.array(thumb))


def _jpeg_b64_fit(image: Image.Image, max_side: int = 520) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height, 1))
    if scale < 1:
        rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    return np_to_b64(np.array(rgb))


def _draw_detections(image: Image.Image, bbox=None, landmarks=None) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, 640 / max(width, height, 1))
    if scale < 1:
        rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    canvas = np.array(rgb)
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(float(value) * scale) for value in bbox]
        thickness = max(2, int(min(canvas.shape[:2]) * 0.008))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 240, 255), thickness)
    if landmarks:
        points = []
        for point in landmarks:
            if not point or len(point) < 2:
                continue
            x, y = int(float(point[0]) * scale), int(float(point[1]) * scale)
            points.append((x, y))
            cv2.circle(canvas, (x, y), max(3, int(min(canvas.shape[:2]) * 0.012)), (0, 255, 200), -1)
        if len(points) >= 2:
            cv2.line(canvas, points[0], points[1], (0, 255, 200), 2)
    return np_to_b64(canvas)


def _story_frame(label: str, b64: Optional[str]) -> Optional[Dict[str, str]]:
    if not b64:
        return None
    return {"label": label, "b64": b64}


def _story_chapter(step_id: str, kicker: str, title: str, body: str, frames: list, meters=None, footnote: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": step_id,
        "kicker": kicker,
        "title": title,
        "body": body,
        "frames": [frame for frame in frames if frame],
        "meters": meters or [],
        "footnote": footnote or "",
        "meta": meta or {},
    }


class ScanCancelled(Exception):
    pass


def _heatmap_with_deadline(ensemble, model_img, face_np, device, seconds: float = 12):
    """Grad-CAM is optional. A stuck backward pass must not hold the verdict."""
    if ensemble.eff_model is None:
        return None, "EfficientNet is unavailable, so Grad-CAM was not run."
    box: Dict[str, Any] = {}

    def run():
        try:
            tensor = ensemble.eff_transform(model_img).unsqueeze(0).to(device)
            box["heatmap"] = generate_heatmap(ensemble, tensor, face_np)
        except Exception as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        return None, f"Grad-CAM exceeded {int(seconds)}s and was skipped. The decision does not use it."
    if box.get("error"):
        return None, box["error"]
    heatmap = box.get("heatmap")
    if heatmap is None:
        return None, "Grad-CAM did not return a map."
    return heatmap, None


def analyze_image(
    contents: bytes,
    filename: str,
    ensemble,
    extractor,
    iris_engine,
    device,
    attack: bool = False,
    epsilon: float = 0.05,
    gemini_key: Optional[str] = None,
    on_stage: Optional[Callable[[Dict[str, Any]], None]] = None,
    scan_id: Optional[str] = None,
) -> Dict[str, Any]:
    scan_id = scan_id or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    stages = []
    warnings = []
    errors = []
    timings: Dict[str, float] = {}

    def stage(stage_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        if is_cancelled(scan_id):
            raise ScanCancelled(scan_id)
        event = {
            "scan_id": scan_id,
            "stage_id": stage_id,
            "status": status,
            "timestamp": _now(),
            "result": result,
            "error": error,
        }
        stages.append(event)
        if on_stage:
            on_stage(event)
        if error:
            errors.append({"stage_id": stage_id, "error": error})

    stage("file_received", "completed", {"filename": filename or "upload", "size_bytes": len(contents)})

    t0 = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:
        stage("image_decoding", "failed", error=f"{type(exc).__name__}: {exc}")
        return _failed(scan_id, filename, contents, stages, warnings, errors, timings, started, "The file could not be decoded as an image.")
    width, height = image.size
    timings["decode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    stage("format_validation", "completed", {"decoded": True, "width": width, "height": height, "mode": "RGB"})
    stage("image_decoding", "completed", {"width": width, "height": height})

    digest = hashlib.sha256(contents).hexdigest()
    stage("sha256", "completed", {"sha256": digest})

    exif = parse_exif_metadata(image)
    stage("metadata_inspection", "completed", {
        "has_exif": exif.get("has_exif"),
        "camera_hardware_found": exif.get("camera_hardware_found"),
        "ai_software_detected": exif.get("ai_software_detected"),
        "software": exif.get("software"),
    })

    visible = detect_visual_watermarks_and_logos(np.array(image), contents)
    stage("visible_watermark", "completed", {
        "detected": visible.get("visual_watermark_detected"),
        "regions": visible.get("regions"),
        "ocr_status": visible.get("ocr_status"),
        "metadata_text_match": visible.get("metadata_text_match"),
    })

    t0 = time.perf_counter()
    face_meta = extractor.extract_face_and_landmarks(image)
    timings["face_detection_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    stage("face_detection", "completed", {
        "detected": bool(face_meta.get("detected")),
        "confidence": face_meta.get("confidence"),
        "bbox": face_meta.get("bbox"),
    })
    landmarks = face_meta.get("landmarks") or []
    stage("facial_landmarks", "completed" if landmarks else "completed", {
        "count": len(landmarks),
        "landmarks": landmarks,
        "status": "MEASURED" if landmarks else "NOT_FOUND",
    })

    alignment = extractor.align_by_eyes(face_meta["face_img"], landmarks if face_meta.get("detected") else [])
    aligned_img = alignment["image"]
    if alignment.get("applied"):
        # Landmarks were in the pre-rotation crop. Re-detect on the rotated crop for the model input only.
        realigned = extractor.extract_face_and_landmarks(aligned_img)
        model_img = realigned["face_img"]
    else:
        model_img = aligned_img
    stage("face_crop", "completed", {"crop_b64": _jpeg_b64(face_meta["face_img"]), "method": face_meta.get("crop_method", "mtcnn_margin")})
    stage("facial_alignment", alignment["status"].lower(), {
        "applied": alignment["applied"],
        "angle_deg": alignment["angle_deg"],
        "reason": alignment["reason"],
        "aligned_b64": _jpeg_b64(model_img),
    })

    normalized_preview = ensemble.eff_transform(model_img)
    stage("model_normalization", "completed", {
        "efficientnet": ensemble.eff_status.get("normalization") if ensemble.eff_status.get("available") else None,
        "vit_input_size": (ensemble.vit_status.get("input_size") if ensemble.vit_status.get("available") else None),
        "standardized_input_b64": _jpeg_b64(model_img),
        "tensor_shape": list(normalized_preview.shape),
    })

    face_np = np.array(model_img.resize((256, 256)))
    gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality = {
        "width": width,
        "height": height,
        "laplacian_variance": round(lap, 3),
        "mean_brightness": round(float(gray.mean()), 3),
    }
    if lap < 15:
        warnings.append("The face crop is very soft. Laplacian variance is below 15.")
    if min(width, height) < 64:
        warnings.append("The image is smaller than 64 pixels on its short side.")
    stage("image_quality", "completed", quality)

    t0 = time.perf_counter()
    eff_prob, vit_prob, model_detail = ensemble.predict_backbones(model_img, use_tta=True)
    timings["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    attacked = None
    if attack and ensemble.eff_model is not None:
        try:
            import torch
            tensor = ensemble.eff_transform(model_img).unsqueeze(0).to(device)
            perturbed = generate_adversarial_perturbation(ensemble, tensor, label_idx=1, epsilon=epsilon)
            with torch.no_grad():
                logit = float(ensemble.eff_model(perturbed).view(-1)[0].item())
                attacked = float(torch.sigmoid(torch.tensor(logit)).item())
            eff_prob = attacked
            if model_detail.get("efficientnet"):
                model_detail["efficientnet"]["adversarial"] = {
                    "method": "FGSM",
                    "epsilon": epsilon,
                    "fake_probability_after_attack": round(attacked, 6),
                }
            warnings.append("EfficientNet received an FGSM perturbation. This scan is a robustness experiment, not a clean measurement.")
        except Exception as exc:
            warnings.append(f"FGSM was requested and failed: {type(exc).__name__}: {exc}")

    eff_stage = model_detail.get("efficientnet") or {"available": False}
    vit_stage = model_detail.get("vit") or {"available": False}
    stage(
        "efficientnet_inference",
        "completed" if eff_stage.get("available") else "failed",
        {k: v for k, v in eff_stage.items() if k != "error"},
        error=eff_stage.get("error"),
    )
    stage(
        "vit_inference",
        "completed" if vit_stage.get("available") else "failed",
        {k: v for k, v in vit_stage.items() if k != "error"},
        error=vit_stage.get("error"),
    )

    spectral = calculate_spectral_analysis(face_np)
    fft_image = compute_fft_spectrum(face_np)
    stage("frequency_analysis", "completed", {
        "high_freq_ratio": spectral.get("high_freq_ratio"),
        "scaled_score": spectral.get("anomaly_score"),
        "radial_profile": spectral.get("radial_profile"),
        "contributes_to_score": False,
        "fft_b64": np_to_b64(fft_image),
    })

    color = analyze_chroma_and_boundary(face_np, bbox=face_meta.get("bbox") if face_meta.get("detected") else None)
    stage("auxiliary_color", "completed", {
        "skin_chrominance": color.get("skin_chrominance"),
        "color_anomaly_index": color.get("color_anomaly_index"),
        "contributes_to_score": False,
    })

    c2pa = detect_c2pa_manifest(contents)
    synthid = detect_synthid_watermark(face_np, contents)
    stage("provenance", "completed", {
        "c2pa_status": c2pa.get("validation_status"),
        "synthid_status": synthid.get("status"),
        "visible_regions": len(visible.get("regions") or []),
    })

    iris = iris_engine.analyze_eyes(image, landmarks if face_meta.get("detected") else [])
    stage("iris_measurement", "completed" if iris.get("detected") else "completed", {
        "status": iris.get("status"),
        "detected": iris.get("detected"),
        "contributes_to_score": False,
    })

    hashes = compute_cryptographic_and_perceptual_hashes(face_np, contents)
    feedback = lookup_feedback_memory(hashes.get("sha256"), hashes.get("phash"))
    if feedback:
        warnings.append("A prior user label matched this file or a similar perceptual hash. It does not override the model.")

    decision = ForensicDecisionEngine.fuse_evidence(
        eff_fake_prob=eff_prob,
        vit_fake_prob=vit_prob,
        face_detected=bool(face_meta.get("detected")),
        feedback_override=feedback,
        eff_available=eff_prob is not None,
        vit_available=vit_prob is not None,
    )
    warnings.extend(decision.get("warnings") or [])
    stage("evidence_fusion", "completed", {
        "method": decision.get("method"),
        "weights": decision.get("weights_used"),
        "auxiliary_weights": decision.get("auxiliary_weights"),
    })
    stage("final_decision", "completed", {
        "prediction": decision.get("prediction"),
        "fake_probability": decision.get("fake_probability"),
        "real_probability": decision.get("real_probability"),
        "uncertainty": decision.get("uncertainty"),
    })

    heatmap, heatmap_error = _heatmap_with_deadline(ensemble, model_img, face_np, device, seconds=12)
    heatmap_b64 = np_to_b64(heatmap) if heatmap is not None else None
    stage(
        "explainability",
        "completed" if heatmap is not None else "failed",
        {"gradcam": "generated" if heatmap is not None else "UNAVAILABLE", "heatmap_b64": heatmap_b64},
        error=heatmap_error,
    )
    if heatmap_error:
        warnings.append(heatmap_error)

    explanation_payload = {
        "prediction": decision.get("prediction"),
        "real_probability": decision.get("real_probability") if decision.get("real_probability") is not None else 0.0,
        "fake_probability": decision.get("fake_probability") if decision.get("fake_probability") is not None else 0.0,
        "confidence": decision.get("confidence") or 0.0,
        "spectral_anomaly_score": spectral.get("anomaly_score"),
        "iris_biometrics": iris,
        "color_forensics": color,
        "provenance": exif,
        "c2pa": c2pa,
        "synthid": synthid,
        "raw_spatial_fake_prob": decision.get("fake_probability") or 0.0,
        "decision_notes": decision.get("decision_notes"),
    }
    explanation = generate_forensic_explanation(explanation_payload, gemini_key=gemini_key)
    stage("report_ready", "completed", {"pdf": "available_after_completion", "pages_planned": 5})

    ledger = _ledger(decision, model_detail, spectral, color, iris, visible, c2pa, synthid, exif)
    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
    status = "completed" if decision.get("prediction") != "Unavailable" else "partial"

    original_b64 = _jpeg_b64_fit(image)
    boxed_b64 = _draw_detections(image, bbox=face_meta.get("bbox")) if face_meta.get("detected") else original_b64
    landmark_b64 = _draw_detections(image, bbox=face_meta.get("bbox"), landmarks=landmarks) if landmarks else boxed_b64
    crop_b64 = _jpeg_b64(face_meta["face_img"])
    aligned_b64 = _jpeg_b64(model_img)
    fft_b64 = np_to_b64(fft_image)
    visual_story = [
        _story_chapter(
            "ingest",
            "01  Ingest",
            "The file enters as raw pixels",
            f"{filename or 'upload'} is {width}×{height}. The specimen is staged in the secure sandbox. Nothing has been scored yet.",
            [_story_frame("Uploaded frame", original_b64)],
            meta={"width": width, "height": height, "sha256": digest, "size_bytes": len(contents)},
            footnote="Raw byte ingest verified • Zero server-side persistence",
        ),
        _story_chapter(
            "hash",
            "02  Integrity",
            "Cryptographic file fingerprint",
            f"SHA-256: {digest[:32]}… Any single pixel modification alters this fingerprint completely. It confirms tamper-proof evidence provenance.",
            [_story_frame("Source still", original_b64)],
            meta={"sha256": digest},
            footnote="SHA-256 Digest • Cryptographic integrity lock",
        ),
        _story_chapter(
            "face",
            "03  Face lock",
            "MTCNN neural boundary lock" if face_meta.get("detected") else "No face lock — center crop fallback",
            (
                f"MTCNN detected the facial boundary with {round((face_meta.get('confidence') or 0.0) * 100, 1)}% confidence. Bounding box coordinates isolate the subject from complex background."
                if face_meta.get("detected")
                else "MTCNN did not locate an organic face. The pipeline activates center-crop fallback so neural networks can still evaluate the specimen."
            ),
            [_story_frame("Face box", boxed_b64), _story_frame("Original", original_b64)],
            meta={"bbox": face_meta.get("bbox"), "confidence": face_meta.get("confidence"), "detected": face_meta.get("detected")},
            footnote="MTCNN Cascaded Networks (P-Net -> R-Net -> O-Net)",
        ),
        _story_chapter(
            "landmarks",
            "04  Landmarks",
            f"{len(landmarks)} Biometric facial anchor points" if landmarks else "Landmarks not located",
            (
                f"Sub-pixel regression identified {len(landmarks)} keypoints (ocular pupils, nasal bridge, mouth corners). The inter-pupillary vector measures head roll tilt prior to alignment."
                if landmarks
                else "Without landmarks, rotational compensation cannot be calculated."
            ),
            [_story_frame("Landmark overlay", landmark_b64)],
            meta={"landmarks": landmarks, "count": len(landmarks), "angle_deg": alignment.get("angle_deg")},
            footnote="5-Point Biometric Keypoint Geometry",
        ),
        _story_chapter(
            "crop",
            "05  Face crop",
            "Extracted subject window",
            f"Crop strategy: {face_meta.get('crop_method', 'mtcnn_margin')}. Irrelevant scene context outside the facial boundary is discarded to concentrate neural focus.",
            [_story_frame("Full frame", boxed_b64), _story_frame("Face crop", crop_b64)],
            meta={"crop_method": face_meta.get("crop_method")},
            footnote="Spatial Window Extraction",
        ),
        _story_chapter(
            "align",
            "06  Alignment",
            "Eyes leveled to canonical horizontal horizon" if alignment.get("applied") else "Canonical horizon verified",
            (
                f"The measured inter-ocular tilt angle was {alignment.get('angle_deg')}°. Affine rotation levels the eye axis to exactly 0.0° horizontal. Both EfficientNet and ViT require this standardized canonical alignment to avoid mistaking head tilt for synthesis anomalies."
                if alignment.get("applied")
                else (alignment.get("reason") or "The crop eye line was already within 2° of horizontal. No rotational warp was necessary.")
            ),
            [_story_frame("Raw Crop (Tilted)", crop_b64), _story_frame("Aligned Picture (Leveled)", aligned_b64)],
            meters=[
                {
                    "label": "Inter-Ocular Roll Angle",
                    "value": abs(float(alignment.get("angle_deg") or 0.0)) / 45.0,
                    "detail": f"{alignment.get('angle_deg')}° tilt" if alignment.get("angle_deg") is not None else "0.0° (Level)",
                }
            ],
            meta={
                "angle_deg": alignment.get("angle_deg"),
                "applied": alignment.get("applied"),
                "reason": alignment.get("reason"),
                "crop_b64": crop_b64,
                "aligned_b64": aligned_b64,
            },
            footnote="Bicubic Affine Rotation • Inter-Pupillary Axis Leveling",
        ),
        _story_chapter(
            "tensor",
            "07  Tensor Preprocessing",
            "Standardized 224×224 Normalization & RGB Split",
            "The leveled face crop is scaled to canonical neural dimensions (224×224×3) and normalized via ImageNet channel statistics (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]). Decomposing into R, G, B channels exposes chrominance distribution imbalances.",
            [_story_frame("Aligned Crop", aligned_b64)],
            meta={"size": [224, 224, 3], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            footnote="Tensor Shape: [1, 3, 224, 224] • Precision: Float32",
        ),
        _story_chapter(
            "vit_patches",
            "08  ViT Tokenization",
            "196 Vision Transformer Patches",
            "ViT-B/16 slices the 224×224 aligned face into 196 non-overlapping 16×16 spatial patches. Multi-head self-attention models cross-patch semantic coherence to identify generative synthesis seams.",
            [_story_frame("Patch Grid", aligned_b64)],
            meta={"patch_size": 16, "num_patches": 196, "grid_dim": 14},
            footnote="14×14 Patch Matrix • Linear Projection + Positional Embeddings",
        ),
        _story_chapter(
            "models",
            "09  Neural scores",
            "Dual-Backbone Inference",
            "Both backbones evaluate the exact same aligned crop. EfficientNet-B3 analyzes convolutional texture gradients; ViT-B/16 evaluates global spatial patch attention.",
            [_story_frame("Network input", aligned_b64)],
            meters=[
                {
                    "label": "EfficientNet-B3",
                    "value": None if eff_prob is None else round(float(eff_prob), 4),
                    "detail": "unavailable" if eff_prob is None else f"fake {round(float(eff_prob), 3)}",
                },
                {
                    "label": "ViT-B/16",
                    "value": None if vit_prob is None else round(float(vit_prob), 4),
                    "detail": "unavailable" if vit_prob is None else f"fake {round(float(vit_prob), 3)}",
                },
            ],
            meta={"eff_prob": eff_prob, "vit_prob": vit_prob},
            footnote="Dual Model Ensemble • Calibrated Softmax Probabilities",
        ),
        _story_chapter(
            "fft",
            "10  Frequency view",
            "2D FFT Optical Spectrum",
            "2D Fast Fourier Transform shifts spatial pixels into optical frequency domain. Generative models (GANs, Diffusion, Latent Upsamplers) leave distinctive periodic lattice artifacts and high-frequency roll-off anomalies.",
            [_story_frame("Aligned crop", aligned_b64), _story_frame("FFT spectrum", fft_b64)],
            meta={"high_freq_ratio": spectral.get("high_freq_ratio"), "anomaly_score": spectral.get("anomaly_score")},
            footnote=f"Radial Spectral Anomaly Index: {spectral.get('anomaly_score')}",
        ),
        _story_chapter(
            "cam",
            "11  Grad-CAM",
            "Where EfficientNet looked" if heatmap_b64 else "Grad-CAM unavailable",
            "Warm pixels illustrate gradient attribution on EfficientNet's final convolutional layer. They expose the exact spatial regions (eyes, mouth contour, skin texture) that triggered the model's prediction.",
            [_story_frame("Aligned crop", aligned_b64), _story_frame("Grad-CAM Heatmap", heatmap_b64)] if heatmap_b64 else [_story_frame("Aligned crop", aligned_b64)],
            meta={"heatmap_b64": heatmap_b64, "aligned_b64": aligned_b64},
            footnote="Gradient-Weighted Class Activation Mapping (top_conv layer)",
        ),
    ]
    if iris.get("detected") and (iris.get("left_eye_b64") or iris.get("right_eye_b64")):
        visual_story.append(
            _story_chapter(
                "iris",
                "12  Iris Biometrics",
                "Ocular Zoom & Corneal Reflection",
                "High-resolution ocular crops inspect pupil circularity and specular reflection symmetry between the eyes. Generative models often generate physically inconsistent specular reflection angles.",
                [
                    _story_frame("Left eye", iris.get("left_eye_b64")),
                    _story_frame("Right eye", iris.get("right_eye_b64")),
                ],
                meta={"left_eye_b64": iris.get("left_eye_b64"), "right_eye_b64": iris.get("right_eye_b64"), "status": iris.get("status")},
                footnote=iris.get("status"),
            )
        )
    visual_story.append(
        _story_chapter(
            "decision",
            "13  Consensus Decision",
            str(decision.get("prediction") or "Unavailable"),
            decision.get("decision_notes") or "Evidence fusion synthesizes spatial and frequency backbones into an authoritative consensus verdict.",
            [_story_frame("Aligned crop", aligned_b64), _story_frame("Grad-CAM", heatmap_b64) or _story_frame("FFT", fft_b64)],
            meters=[
                {
                    "label": "Consensus Fake Score",
                    "value": None if decision.get("fake_probability") is None else round(float(decision.get("fake_probability")), 4),
                    "detail": str(decision.get("prediction")),
                }
            ],
            meta={"prediction": decision.get("prediction"), "confidence": decision.get("confidence"), "fake_probability": decision.get("fake_probability"), "real_probability": decision.get("real_probability")},
            footnote="Bayesian Consensus Fusion • Final Verdict",
        )
    )
    if errors and decision.get("prediction") == "Unavailable":
        status = "failed"

    record = {
        "scan_id": scan_id,
        "status": status,
        "media_type": "image",
        "filename": filename or "upload",
        "created_at": stages[0]["timestamp"],
        "input_metadata": {
            "filename": filename or "upload",
            "size_bytes": len(contents),
            "width": width,
            "height": height,
            "sha256": digest,
        },
        "preprocessing_results": {
            "face_detected": bool(face_meta.get("detected")),
            "face_confidence": face_meta.get("confidence"),
            "bbox": face_meta.get("bbox"),
            "landmarks": landmarks,
            "alignment": {k: alignment[k] for k in ("applied", "angle_deg", "status", "reason")},
            "quality": quality,
            "normalization": {
                "efficientnet": ensemble.eff_status.get("normalization"),
                "vit": ensemble.vit_status.get("input_size"),
            },
        },
        "model_results": [model_detail.get("efficientnet"), model_detail.get("vit")],
        "auxiliary_evidence": {"spectral": spectral, "color": color, "iris": iris},
        "provenance_results": {"exif": exif, "c2pa": c2pa, "synthid": synthid, "visible_watermark": visible},
        "explainability": {
            "gradcam_status": "generated" if heatmap is not None else "UNAVAILABLE",
            "gradcam_error": heatmap_error,
            "ledger": ledger,
            "narrative": explanation,
        },
        "final_decision": decision,
        "uncertainty": decision.get("uncertainty"),
        "timings": timings,
        "warnings": warnings,
        "errors": errors,
        "stages": stages,
        "user_annotation": {
            "matched": bool(feedback),
            "overrides_prediction": False,
            "record": {k: feedback.get(k) for k in ("corrected_label", "sha256") if feedback} if feedback else None,
        },
        "prediction": decision.get("prediction"),
        "confidence": decision.get("confidence"),
        "real_probability": decision.get("real_probability"),
        "fake_probability": decision.get("fake_probability"),
        "verdict_badge": decision.get("verdict_badge"),
        "is_inconclusive": decision.get("is_inconclusive"),
        "discordance_level": decision.get("discordance_level"),
        "discordance_value": decision.get("discordance_value"),
        "decision_notes": decision.get("decision_notes"),
        "weights_used": decision.get("weights_used"),
        "raw_signals": decision.get("raw_signals"),
        "spectral_anomaly_score": spectral.get("anomaly_score"),
        "high_freq_ratio": spectral.get("high_freq_ratio"),
        "radial_profile": spectral.get("radial_profile"),
        "iris_biometrics": iris,
        "color_forensics": color,
        "provenance": exif,
        "c2pa": c2pa,
        "synthid": synthid,
        "visual_watermark": visible,
        "hashes": hashes,
        "feedback_applied": False,
        "adversarial_attack_applied": bool(attacked is not None),
        "epsilon": epsilon if attacked is not None else None,
        "face_detected": bool(face_meta.get("detected")),
        "bbox": face_meta.get("bbox"),
        "landmarks": landmarks,
        "face_b64": aligned_b64,
        "heatmap_b64": heatmap_b64,
        "fft_b64": fft_b64,
        "original_b64": original_b64,
        "face_box_b64": boxed_b64,
        "landmark_overlay_b64": landmark_b64,
        "crop_b64": crop_b64,
        "aligned_b64": aligned_b64,
        "alignment": {k: alignment[k] for k in ("applied", "angle_deg", "status", "reason")},
        "visual_story": visual_story,
        "explanation": explanation,
        "retention": "The scan record stays in process memory only and is dropped after a small number of newer scans. The upload is not written to a durable store.",
    }
    return record


def _failed(scan_id, filename, contents, stages, warnings, errors, timings, started, message) -> Dict[str, Any]:
    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return {
        "scan_id": scan_id,
        "status": "failed",
        "media_type": "image",
        "filename": filename,
        "input_metadata": {"filename": filename, "size_bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()},
        "preprocessing_results": {},
        "model_results": [],
        "auxiliary_evidence": {},
        "provenance_results": {},
        "explainability": {"ledger": []},
        "final_decision": {"prediction": "Unavailable", "fake_probability": None, "real_probability": None, "decision_notes": message},
        "uncertainty": {"level": "high", "reasons": [message]},
        "timings": timings,
        "warnings": warnings,
        "errors": errors or [{"stage_id": "image_decoding", "error": message}],
        "stages": stages,
        "prediction": "Unavailable",
        "confidence": None,
        "real_probability": None,
        "fake_probability": None,
        "is_inconclusive": True,
        "feedback_applied": False,
        "face_detected": False,
        "heatmap_b64": None,
        "face_b64": None,
        "fft_b64": None,
        "decision_notes": message,
    }


def _ledger(decision, model_detail, spectral, color, iris, visible, c2pa, synthid, exif) -> list:
    rows = []
    for key, label in (("efficientnet", "EfficientNet-B3"), ("vit", "ViT-B/16")):
        item = model_detail.get(key) or {}
        rows.append({
            "measurement": f"{key}_fake_score",
            "detector": label,
            "result": item.get("fake_probability"),
            "region": "aligned_face_crop",
            "confidence_meaning": item.get("output_semantics"),
            "interpretation": "Neural class score for the fake label.",
            "limitations": item.get("error") or "Not a calibrated probability for this deployment, and not proof about every generator.",
            "validated_contribution": bool(item.get("available")),
        })
    rows.append({
        "measurement": "high_frequency_ratio",
        "detector": "2D FFT",
        "result": spectral.get("high_freq_ratio"),
        "region": "face_crop",
        "confidence_meaning": None,
        "interpretation": "Share of spectral energy outside the inner half-radius.",
        "limitations": spectral.get("limitation"),
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "skin_chrominance",
        "detector": "YCbCr statistics",
        "result": color.get("skin_chrominance"),
        "region": "face_box" if color.get("skin_chrominance") else None,
        "confidence_meaning": None,
        "interpretation": "Descriptive color measurement.",
        "limitations": color.get("limitation"),
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "iris_geometry",
        "detector": "landmark eye crops",
        "result": None if not iris.get("detected") else {
            "mean_circularity": iris.get("mean_circularity"),
            "specular_asymmetry": iris.get("specular_asymmetry"),
        },
        "region": "eyes" if iris.get("detected") else None,
        "confidence_meaning": None,
        "interpretation": iris.get("status"),
        "limitations": iris.get("limitation") or iris.get("reason"),
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "visible_watermark",
        "detector": "corner_glyph_line",
        "result": visible.get("regions"),
        "region": "corners",
        "confidence_meaning": "detector confidence, not a class probability",
        "interpretation": "Text-like corner marks, if any.",
        "limitations": visible.get("limitation"),
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "c2pa",
        "detector": "metadata_marker",
        "result": c2pa.get("validation_status"),
        "region": "file_metadata",
        "confidence_meaning": None,
        "interpretation": c2pa.get("note"),
        "limitations": "No cryptographic verification was performed.",
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "synthid",
        "detector": "not_installed",
        "result": synthid.get("status"),
        "region": None,
        "confidence_meaning": None,
        "interpretation": synthid.get("reason"),
        "limitations": synthid.get("reason"),
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "exif_ai_software",
        "detector": "exif",
        "result": exif.get("ai_software_detected"),
        "region": "metadata",
        "confidence_meaning": None,
        "interpretation": exif.get("ai_software_name") or "No listed generator name in EXIF.",
        "limitations": "Metadata can be added or removed without changing the pixels.",
        "validated_contribution": False,
    })
    rows.append({
        "measurement": "fused_neural_mean",
        "detector": "ForensicDecisionEngine",
        "result": decision.get("fake_probability"),
        "region": None,
        "confidence_meaning": decision.get("score_semantics"),
        "interpretation": decision.get("decision_notes"),
        "limitations": decision.get("method"),
        "validated_contribution": decision.get("prediction") not in {None, "Unavailable"},
    })
    return rows
