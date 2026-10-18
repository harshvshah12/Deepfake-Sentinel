import asyncio
import io
import os
import json
import time
import base64
import glob
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Query, Body, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.preprocessor import FaceExtractor
from backend.model import EnsembleModel
from backend.xai import generate_heatmap
from backend.frequency import compute_fft_spectrum, calculate_spectral_analysis
from backend.robustness import generate_adversarial_perturbation
from backend.video import process_video_file, np_to_b64
from backend.iris import IrisBiometricEngine
from backend.color_forensics import analyze_chroma_and_boundary
from backend.provenance import (
    parse_exif_metadata,
    detect_c2pa_manifest,
    detect_synthid_watermark,
    detect_visual_watermarks_and_logos,
    compute_cryptographic_and_perceptual_hashes
)
from backend.explainability import generate_forensic_explanation, call_gemini_forensic_reasoner
from backend.feedback_memory import save_user_feedback, lookup_feedback_memory, get_feedback_stats
from backend.fusion import ForensicDecisionEngine
from backend.pipeline import analyze_image, ScanCancelled
from backend.scan_store import save_scan, get_scan, cancel_scan
from backend.security import UploadRejected, classify_upload, resolve_dataset_file, SlidingRateLimiter
from backend.pdf_report import generate_scan_pdf
from backend.pdf_html_generator import generate_report_html

# Auto-load .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().replace('"', '').replace("'", ""))
    except Exception:
        pass

app = FastAPI(
    title="Deepfake Sentinel — Trustworthy XAI Forensic Engine",
    description="Next-Generation AI Image Detection & Provenance Verification Platform with C2PA, SynthID, 2D Fourier Spectral Optics & Grad-CAM Interpretability.",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hardware & Model Services
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = FaceExtractor(device=device)
ensemble = EnsembleModel(device=device)
iris_engine = IrisBiometricEngine(device=device)
rate_limiter = SlidingRateLimiter(limit=30, window_seconds=60.0)
DATASET_ROOT = Path("dataset")
TEST_ROOT = Path("test_images")

class FramePayload(BaseModel):
    image: str
    attack: bool = False
    epsilon: float = 0.05

class FeedbackPayload(BaseModel):
    sha256: str
    phash: str
    corrected_label: str
    original_prediction: Optional[str] = "Unknown"
    metadata: Optional[Dict[str, Any]] = None

class GeminiTestPayload(BaseModel):
    api_key: str

@app.get("/")
def read_root():
    return FileResponse("frontend/static/index.html")

@app.get("/health")
def health():
    return {
        "status": "online",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model": "EfficientNet-B3 + ViT-B/16",
        "efficientnet_loaded": bool(ensemble.eff_status.get("available")),
        "vit_loaded": bool(ensemble.vit_status.get("available")),
        "version": "3.3.0"
    }

@app.get("/samples")
def get_samples():
    real_tests = sorted(glob.glob("test_images/real/*.*"))[:6]
    fake_tests = sorted(glob.glob("test_images/fake/*.*"))[:6]
    real_train = sorted(glob.glob("dataset/real/*.jpg") + glob.glob("dataset/real/*.png"))[:4]
    fake_train = sorted(glob.glob("dataset/fake/*.jpg") + glob.glob("dataset/fake/*.png"))[:4]
    
    real_images = real_tests + real_train
    fake_images = fake_tests + fake_train
    
    return {
        "real": [{"id": f"real_{i}", "name": Path(p).name, "url": f"/sample-image?path={p}"} for i, p in enumerate(real_images)],
        "fake": [{"id": f"fake_{i}", "name": Path(p).name, "url": f"/sample-image?path={p}"} for i, p in enumerate(fake_images)]
    }

@app.get("/sample-image")
def get_sample_image(path: str = Query(...)):
    try:
        sample = resolve_dataset_file(path, DATASET_ROOT, extra_roots=[TEST_ROOT])
    except UploadRejected as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    return FileResponse(sample)

@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    """
    Learns and persists user corrections into continuous feedback memory.
    """
    res = save_user_feedback(
        sha256=payload.sha256,
        phash=payload.phash,
        corrected_label=payload.corrected_label,
        original_prediction=payload.original_prediction,
        metadata=payload.metadata
    )
    return res

@app.get("/feedback-stats")
def feedback_stats():
    return get_feedback_stats()

@app.post("/test-gemini-key")
def test_gemini_key(payload: GeminiTestPayload):
    """
    Tests if a Google Gemini API Key is valid by calling gemini-1.5-flash with a probe payload.
    """
    mock_payload = {
        "prediction": "Real",
        "real_probability": 0.99,
        "fake_probability": 0.01,
        "spectral_anomaly_score": 0.12,
        "iris_biometrics": {"specular_asymmetry": 0.02, "mean_circularity": 0.88},
        "provenance": {"camera_hardware_found": True},
        "c2pa": {"validation_status": "VALID_SIGNATURE"},
        "synthid": {"synthid_detected": False}
    }
    try:
        verdict = call_gemini_forensic_reasoner(mock_payload, payload.api_key)
        return {
            "success": True,
            "message": "Google Gemini API Key is active & operational!",
            "sample_verdict": verdict
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/model-metrics")
def get_model_metrics():
    """
    Returns comprehensive data visualization metrics and verified performance benchmarks for all models.
    """
    local_path = Path("docs/local_evaluation.json")
    local_eval = None
    if local_path.exists():
        try:
            local_eval = json.loads(local_path.read_text(encoding="utf-8"))
        except Exception:
            local_eval = {"status": "UNREADABLE"}
    checkpoint_test = (ensemble.eff_checkpoint_meta or {}).get("final_test_metrics")
    return {
        "disclaimer": (
            "EfficientNet figures below were stored inside the training checkpoint. "
            "They are not a new measurement of this repository's dataset folder. "
            "No ROC curve is invented when a local evaluation file is absent. "
            "FFT, color, and iris measurements are not accuracy claims."
        ),
        "models": [
            {
                "name": "EfficientNet-B3",
                "loaded": bool(ensemble.eff_status.get("available")),
                "checkpoint_test_metrics": checkpoint_test,
                "metrics_source": "checkpoint_embedded_training_run" if checkpoint_test else "NOT_AVAILABLE",
            },
            {
                "name": "ViT-B/16 dima806/deepfake_vs_real_image_detection",
                "loaded": bool(ensemble.vit_status.get("available")),
                "id2label": ensemble.vit_id2label,
                "local_accuracy": None,
                "metrics_source": "NOT_MEASURED_IN_THIS_DEPLOYMENT",
            },
        ],
        "feature_weights": {"EfficientNet-B3": 50, "ViT-B/16": 50} if ensemble.eff_status.get("available") and ensemble.vit_status.get("available") else {},
        "fft_azimuthal_curve": None,
        "roc_curve": None,
        "confusion_matrix": (checkpoint_test or {}).get("confusion_matrix"),
        "confusion_matrix_source": "efficientnet_checkpoint_final_test" if checkpoint_test else None,
        "local_evaluation": local_eval or {"status": "NOT_MEASURED"},
    }

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    attack: bool = Query(False),
    epsilon: float = Query(0.05),
    gemini_key: Optional[str] = Query(None),
):
    allowed, _remaining = rate_limiter.allow(request.client.host if request.client else "unknown", time.time())
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many scans. Wait a minute and try again.")
    filename = file.filename or "upload"
    content_type = file.content_type or ""
    contents = await file.read()
    try:
        kind = classify_upload(filename, content_type, contents)
    except UploadRejected as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    try:
        if kind == "video":
            result = process_video_file(
                video_bytes=contents,
                extractor=extractor,
                ensemble_model=ensemble,
                device=device,
                max_frames=12,
                attack=attack,
                epsilon=epsilon,
            )
            result["filename"] = filename
            result["explanation"] = generate_forensic_explanation(result, gemini_key=gemini_key)
            result["score_semantics"] = "mean_of_per_frame_neural_scores"
            return JSONResponse(_jsonable(result))
        record = analyze_image(
            contents=contents,
            filename=filename,
            ensemble=ensemble,
            extractor=extractor,
            iris_engine=iris_engine,
            device=device,
            attack=attack,
            epsilon=epsilon,
            gemini_key=gemini_key,
        )
        save_scan(record)
        return JSONResponse(_jsonable(record))
    except ScanCancelled:
        raise HTTPException(status_code=409, detail="Scan cancelled.")
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="The scan failed. The server log has the traceback.")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, bytes):
        return None
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _stream_event(event):
    result = dict(event.get("result") or {})
    slim = {key: item for key, item in result.items() if not str(key).endswith("b64")}
    public = dict(event)
    public["result"] = slim
    return {"type": "stage", **_jsonable(public)}


def _stream_scan(contents, filename, attack, epsilon, gemini_key, scan_id):
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def publish(item):
        loop.call_soon_threadsafe(events.put_nowait, item)

    def on_stage(event):
        publish(_stream_event(event))

    def worker():
        try:
            record = analyze_image(
                contents=contents,
                filename=filename,
                ensemble=ensemble,
                extractor=extractor,
                iris_engine=iris_engine,
                device=device,
                attack=attack,
                epsilon=epsilon,
                gemini_key=gemini_key,
                on_stage=on_stage,
                scan_id=scan_id,
            )
            save_scan(record)
            publish({"type": "complete", "scan": _jsonable(record)})
        except ScanCancelled:
            publish({"type": "error", "message": "Scan cancelled."})
        except Exception:
            import traceback
            traceback.print_exc()
            publish({"type": "error", "message": "The scan failed before a decision was produced."})
        finally:
            publish(None)

    threading.Thread(target=worker, daemon=True).start()

    async def generate():
        while True:
            try:
                item = await asyncio.wait_for(events.get(), timeout=2.0)
            except asyncio.TimeoutError:
                yield "data: " + json.dumps({"type": "heartbeat"}) + "\n\n"
                continue
            if item is None:
                break
            yield "data: " + json.dumps(item, default=str) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/scan")
async def scan_media(
    request: Request,
    file: UploadFile = File(...),
    attack: bool = Query(False),
    epsilon: float = Query(0.05),
    gemini_key: Optional[str] = Query(None),
):
    allowed, _remaining = rate_limiter.allow(request.client.host if request.client else "unknown", time.time())
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many scans. Wait a minute and try again.")
    filename = file.filename or "upload"
    contents = await file.read()
    try:
        kind = classify_upload(filename, file.content_type or "", contents)
    except UploadRejected as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    if kind != "image":
        raise HTTPException(status_code=400, detail="The live timeline currently accepts an image. Use /predict for video.")
    import uuid
    scan_id = uuid.uuid4().hex[:16]
    return _stream_scan(contents, filename, attack, epsilon, gemini_key, scan_id)


@app.post("/scan/{scan_id}/cancel")
def cancel(scan_id: str):
    cancel_scan(scan_id)
    return {"scan_id": scan_id, "cancel_requested": True}


@app.get("/scan/{scan_id}")
def read_scan(scan_id: str):
    record = get_scan(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan not found. Records stay in memory only.")
    return record


@app.get("/report/{scan_id}")
def download_report(scan_id: str, inline: bool = False):
    record = get_scan(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan not found. Records stay in memory only.")
    pdf = generate_scan_pdf(record)
    filename = f"Sentinel-Forensic-Audit-{scan_id[:8]}.pdf"
    disposition = "inline" if inline else "attachment"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"{disposition}; filename={filename}"},
    )


@app.get("/report/{scan_id}/html", response_class=HTMLResponse)
def view_report_html(scan_id: str):
    """Renders the live interactive 4-page HTML forensic evidence dossier directly in the browser."""
    record = get_scan(scan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Scan not found. Records stay in memory only.")
    return HTMLResponse(content=generate_report_html(record))


@app.get("/demo-report", response_class=HTMLResponse)
def demo_report_page():
    """Returns the live interactive 4-page HTML forensic report dossier with all 7 graphs."""
    report_file = os.path.join(os.path.dirname(__file__), "..", "test_report.html")
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    sample = {"scan_id": "demo-sample-01", "prediction": "Fake", "confidence": 0.978}
    return HTMLResponse(content=generate_report_html(sample))


@app.get("/demo-report/pdf")
def demo_report_pdf():
    """Returns the 4-page publication-grade PDF dossier with inline browser display."""
    artifact_pdf = r"C:\Users\harsh\.gemini\antigravity\brain\bbcd98a0-900e-4b53-8e4b-456b213399b5\sentinel_sample_forensic_report.pdf"
    if os.path.exists(artifact_pdf):
        with open(artifact_pdf, "rb") as f:
            pdf_bytes = f.read()
    else:
        sample = {"scan_id": "demo-sample-01", "prediction": "Fake", "confidence": 0.978}
        pdf_bytes = generate_scan_pdf(sample)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=Deepfake-Sentinel-4Page-Audit.pdf"},
    )


@app.post("/export-report-pdf")
async def export_report_pdf(payload: Dict[str, Any] = Body(...)):
    """Exports a publication-grade 4-page forensic PDF dossier directly from a scan record."""
    try:
        pdf_bytes = generate_scan_pdf(payload)
        scan_id = payload.get("scan_id", "live")[:10]
        filename = f"Sentinel-Forensic-Audit-{scan_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")


@app.post("/predict-frame")
async def predict_frame(payload: FramePayload = Body(...)):
    """Ultra-fast live webcam analysis endpoint."""
    try:
        header, encoded = payload.image.split(",", 1) if "," in payload.image else ("", payload.image)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        face_meta = extractor.extract_face_and_landmarks(image)
        
        if not face_meta["detected"]:
            return {
                "face_detected": False,
                "prediction": "No Face",
                "real_probability": None,
                "fake_probability": None,
                "confidence": None,
                "authenticity_index": None,
                "spectral_anomaly_score": None,
                "note": "No face was detected, so no score was produced.",
            }
            
        face_img = face_meta["face_img"]
        face_np = face_meta["face_np"]
        
        # Spatial inference
        real_p, fake_p, pred, conf = ensemble.predict_single(face_img, use_tta=False)
        spectral_data = calculate_spectral_analysis(face_np)
        iris_data = iris_engine.analyze_eyes(image, face_meta.get("landmarks", []))
        
        # Check active learning feedback
        hash_data = compute_cryptographic_and_perceptual_hashes(face_np)
        feedback_override = lookup_feedback_memory(hash_data.get("sha256"), hash_data.get("phash"))
        
        return {
            "face_detected": True,
            "prediction": pred,
            "real_probability": real_p,
            "fake_probability": fake_p,
            "confidence": conf,
            "authenticity_index": None if real_p is None else round(real_p * 100, 1),
            "user_annotation": None if not feedback_override else {
                "overrides_prediction": False,
                "corrected_label": feedback_override.get("corrected_label"),
            },
            "spectral_anomaly_score": spectral_data["anomaly_score"],
            "bbox": face_meta["bbox"],
            "iris_biometrics": iris_data,
            "hashes": hash_data
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount static frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
