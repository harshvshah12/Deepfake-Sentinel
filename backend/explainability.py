import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

def generate_forensic_explanation(payload: Dict[str, Any], gemini_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a structured, evidence-backed forensic reasoning report explaining why
    an image/video is classified as Synthetic (AI-Generated) or Authentic (Real).
    Optionally calls Google Gemini API for deep multimodal forensic reasoning if API key is present.
    """
    pred = payload.get("prediction", "Real")
    is_fake = (pred == "Fake")
    real_p = payload.get("real_probability", 0.9)
    fake_p = payload.get("fake_probability", 0.1)
    confidence = payload.get("confidence", max(real_p, fake_p))
    
    iris = payload.get("iris_biometrics", {})
    spec = payload.get("spectral_anomaly_score", 0.1)
    color = payload.get("color_forensics", {})
    prov = payload.get("provenance", {})
    synthid = payload.get("synthid", {})
    c2pa = payload.get("c2pa", {})
    
    findings = []
    
    # --- 1. Provenance, C2PA & EXIF Metadata ---
    c2pa_status = c2pa.get("validation_status", "NO_MANIFEST")
    findings.append(f"C2PA: {c2pa_status}. This is not a signature verification, and it does not change the neural score.")
    if prov.get("ai_software_detected"):
        findings.append(f"EXIF names generator software ({prov.get('ai_software_name')}). Metadata can be edited separately from the pixels.")
    elif prov.get("camera_hardware_found"):
        findings.append(f"EXIF includes camera tags ({prov.get('make')} {prov.get('model')}). Camera tags do not prove the pixels are unedited.")
    else:
        findings.append("EXIF camera tags were not found. Missing metadata does not mean the image is synthetic.")

    if synthid.get("status") == "NOT_CHECKED" or synthid.get("synthid_detected") is None:
        findings.append("SynthID: NOT CHECKED. No official verifier is configured, and frequency energy was not used as a substitute.")
    elif synthid.get("synthid_detected"):
        findings.append("SynthID: a verifier reported a watermark. See the provenance block for the raw status.")

    # --- 3. Biometric Iris & Physical Highlights (Hu et al. ICASSP) ---
    if iris.get("detected") and iris.get("specular_asymmetry") is not None:
        asym = float(iris.get("specular_asymmetry"))
        findings.append(
            f"Eye geometry was measured (specular area asymmetry {asym:.3f}). "
            "This is unvalidated and is not a medical or authenticity conclusion."
        )
    else:
        findings.append("Eye geometry: NOT MEASURED.")

    # --- 4. 2D FFT Frequency Optics (Dzanic et al. NeurIPS) ---
    findings.append(
        f"FFT high-frequency score {float(spec):.4f} is an uncalibrated measurement. "
        "It is not treated as proof of a GAN checkerboard."
    )

    notes = payload.get("decision_notes") or ""
    if pred == "Unavailable":
        summary = "No neural model produced a score. Nothing was substituted."
        findings.append(summary)
    elif pred == "Inconclusive":
        summary = f"Inconclusive. {notes}"
        findings.append(summary)
    elif is_fake:
        summary = f"Neural mean fake-score {float(fake_p):.3f}. {notes}"
        findings.append(summary)
    else:
        summary = f"Neural mean fake-score {float(fake_p):.3f}, labeled Real by the decision thresholds. {notes}"
        findings.append(summary)

    # --- 6. Optional Gemini API Reasoning ---
    active_key = (gemini_key or "").strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    gemini_insight = None
    gemini_verdict = None
    gemini_fake_score = None
    anomaly_localization = {
        "primary_region": "Whole Face Composition",
        "region_box": None,
        "anomaly_details": "Analyzed across spatial neural activations and 2D Fourier spectral frequency distribution."
    }
    
    if active_key:
        try:
            gemini_insight = call_gemini_forensic_reasoner(payload, active_key)
            if gemini_insight:
                # Parse Primary Anomaly Region, Box, and Verdict if returned by Gemini
                for line in gemini_insight.split("\n"):
                    line_s = line.strip()
                    line_l = line_s.lower()
                    
                    if line_l.startswith("forensic verdict:") or "forensic verdict:" in line_l:
                        after_colon = line_s.split(":", 1)[1].strip()
                        verdict_token = after_colon.replace("*", "").replace(".", "").strip().upper()
                        if verdict_token.startswith("REAL") or verdict_token.startswith("AUTHENTIC") or verdict_token.startswith("GENUINE"):
                            gemini_verdict = "Real"
                            gemini_fake_score = None
                        elif verdict_token.startswith("FAKE") or verdict_token.startswith("SYNTHETIC") or verdict_token.startswith("AI") or verdict_token.startswith("MANIPULATED"):
                            gemini_verdict = "Fake"
                            gemini_fake_score = None
                    elif line_l.startswith("primary anomaly region:") or line_l.startswith("anomaly region:"):
                        anomaly_localization["primary_region"] = line_s.split(":", 1)[1].strip()
                    elif line_l.startswith("anomaly explanation:"):
                        anomaly_localization["anomaly_details"] = line_s.split(":", 1)[1].strip()
                    elif line_l.startswith("anomaly details:"):
                        anomaly_localization["anomaly_details"] = line_s.split(":", 1)[1].strip()
                    elif line_l.startswith("region box:"):
                        try:
                            box_str = line_s.split(":", 1)[1].strip().replace("[", "").replace("]", "")
                            coords = [int(float(c.strip())) for c in box_str.split(",") if c.strip()]
                            if len(coords) == 4:
                                anomaly_localization["region_box"] = coords
                        except Exception:
                            pass
        except Exception as e:
            gemini_insight = f"Gemini API Notice: {str(e)}"

    return {
        "verdict_title": "SYNTHETIC AI-GENERATED MEDIA" if is_fake else "AUTHENTIC OPTICAL MEDIA",
        "summary": summary,
        "key_findings": findings,
        "authenticity_confidence_index": round(real_p * 100, 1),
        "ai_generation_index": round(fake_p * 100, 1),
        "gemini_reasoning": gemini_insight,
        "gemini_verdict": gemini_verdict,
        "gemini_fake_score": gemini_fake_score,
        "anomaly_localization": anomaly_localization
    }

def call_gemini_forensic_reasoner(payload: Dict[str, Any], api_key: str) -> Optional[str]:
    """
    Calls Google Gemini Multimodal Vision API to describe the visual specimen in one line,
    provide an expert visual judgment on whether it feels REAL or FAKE, and pinpoint the exact
    suspect region/coordinates responsible for the verdict.
    """
    cleaned_key = api_key.strip().replace('"', '').replace("'", "")
    models = ["gemini-2.5-flash"]
    
    parts = []
    
    # 1. Attach Visual Image Data (Prefer full uncropped image to catch corner logos/watermarks and peripheral artifacts)
    img_b64 = payload.get("full_image_b64") or payload.get("face_b64")
    if img_b64:
        raw_b64 = img_b64.split(",")[-1] if "," in img_b64 else img_b64
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": raw_b64
            }
        })
    
    # 2. Forensic Vision & Telemetry Prompt
    prompt_text = (
        "You are an Elite Forensic Vision Specialist & AI Watermark Detection System.\n"
        "Carefully inspect this full uncropped image for authenticity, manipulation, morphing, or AI generation.\n\n"
        "Provide your forensic analysis in this exact structured format:\n\n"
        "Visual Description: [Describe in one clear line who/what is depicted in the image, their pose, setting, and background]\n\n"
        "Forensic Verdict: [State **REAL** or **FAKE**, followed by 1-2 authoritative sentences explaining your judgment, citing visible logos, skin texture, lighting, and anatomy]\n\n"
        "Primary Anomaly Region: [State the exact anatomical or visual region responsible for the verdict, e.g. 'Left Eye Corneal Catchlight', 'Nasolabial Skin Micro-Smoothing', 'Jawline Blending Seam', 'Ear-Hair Interface', 'Teeth Alignment', 'Background Frequency Distortion', or 'Organic Sensor Texture']\n\n"
        "Region Box: [ymin, xmin, ymax, xmax] normalized as integer percentages 0-100 covering that specific region\n\n"
        "Anomaly Explanation: [Explain specifically what physical or optical rule is broken or verified in that exact region]\n\n"
        f"Automated Model Diagnostic Reference:\n"
        f"- Spatial Ensemble Probability: {payload.get('fake_probability', 0.5)*100:.1f}% Fake\n"
        f"- Spectral Anomaly Score: {payload.get('spectral_anomaly_score')}\n"
    )
    parts.append({"text": prompt_text})
    
    body = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 800
        }
    }
    data_bytes = json.dumps(body).encode("utf-8")
    
    last_error = ""
    for model_name in models:
        target_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={cleaned_key}"
        req = urllib.request.Request(
            target_url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": cleaned_key
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    c_parts = candidates[0]["content"].get("parts", [])
                    if c_parts:
                        return c_parts[0].get("text", "").strip()
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="ignore")
            try:
                err_json = json.loads(err_text)
                last_error = err_json.get("error", {}).get("message", err_text[:120])
            except Exception:
                last_error = f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            last_error = str(e)
                
    return f"Gemini Analysis Unavailable ({last_error})"
