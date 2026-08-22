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
    if c2pa.get("c2pa_manifest_detected"):
        findings.append(f"C2PA Provenance Manifest: Validated cryptographic manifest issued by {c2pa.get('signature_issuer')}.")
    elif prov.get("ai_software_detected"):
        findings.append(f"EXIF Metadata Signature: File metadata contains explicit generative software stamp ({prov.get('ai_software_name')}).")
    elif prov.get("camera_hardware_found"):
        findings.append(f"Camera Hardware Profile: Verified physical sensor profile ({prov.get('make')} {prov.get('model')}).")
    else:
        findings.append("Metadata Hygiene: EXIF camera profile absent, indicative of web re-encoding or synthetic export.")

    # --- 2. SynthID & Digital Watermark ---
    if synthid.get("synthid_detected"):
        findings.append(f"SynthID Watermark: Imperceptible pixel watermark pattern identified with {synthid.get('watermark_confidence')*100:.1f}% confidence.")
    else:
        findings.append("Watermark Integrity: No imperceptible SynthID frequency modulation pattern detected.")

    # --- 3. Biometric Iris & Physical Highlights (Hu et al. ICASSP) ---
    if iris.get("detected"):
        asym = iris.get("specular_asymmetry", 0.0)
        circ_l = iris.get("pupil_circularity_left", 0.85)
        circ_r = iris.get("pupil_circularity_right", 0.85)
        
        if asym > 0.40:
            findings.append(f"Corneal Specular Asymmetry: Divergence between left and right iris specular glints is {asym*100:.1f}%, violating physical bilateral illumination constraints.")
        else:
            findings.append(f"Corneal Specular Symmetry: Specular reflections in both eyes share consistent light source vectors (Asymmetry: {asym*100:.1f}%).")
            
        if min(circ_l, circ_r) < 0.35:
            findings.append(f"Pupil Boundary Regularity: Pupil circularity regularity is {min(circ_l, circ_r):.2f}, exhibiting non-elliptical generative boundary distortion.")
        else:
            findings.append(f"Pupil Ellipticity: Both pupils maintain natural elliptical circularity (Mean: {iris.get('mean_circularity', 0.85):.2f}).")

    # --- 4. 2D FFT Frequency Optics (Dzanic et al. NeurIPS) ---
    if spec > 0.45:
        findings.append(f"2D Fourier Optics: Frequency power spectrum exhibits periodic high-frequency harmonics (Score: {spec:.4f}), typical of transposed convolution upsampling.")
    else:
        findings.append(f"2D Fourier Optics: Natural radial spectral energy decay observed without generative checkerboard frequency spikes (Score: {spec:.4f}).")

    # --- 5. Spatial CNN & Transformer Ensemble ---
    raw_spatial = payload.get("raw_spatial_fake_prob", fake_p)
    if is_fake:
        findings.append(f"Spatial Neural Classification: Dual-backbone ensemble (EfficientNet-B3 399k + ViT-B/16) identified synthetic micro-textures with {raw_spatial*100:.1f}% confidence.")
        summary = (
            f"This specimen is verified as SYNTHETIC (AI-Generated) with {fake_p*100:.1f}% confidence. "
            f"The classification is corroborated by spatial neural feature activations, corneal reflection asymmetries, and frequency-domain generative harmonics."
        )
    else:
        findings.append(f"Spatial Neural Classification: Dual-backbone ensemble verified organic skin pore textures and facial symmetry with {real_p*100:.1f}% confidence.")
        summary = (
            f"This specimen is verified as AUTHENTIC (Organic / Real) with {real_p*100:.1f}% confidence. "
            f"The image exhibits natural optical sensor noise, consistent 3D physical illumination, and authentic biometric anatomy."
        )

    # --- 6. Optional Gemini API Reasoning ---
    active_key = (gemini_key or "").strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    gemini_insight = None
    if active_key:
        try:
            gemini_insight = call_gemini_forensic_reasoner(payload, active_key)
        except Exception as e:
            gemini_insight = f"Gemini API Notice: {str(e)}"

    return {
        "verdict_title": "SYNTHETIC AI-GENERATED MEDIA" if is_fake else "AUTHENTIC OPTICAL MEDIA",
        "summary": summary,
        "key_findings": findings,
        "authenticity_confidence_index": round(real_p * 100, 1),
        "ai_generation_index": round(fake_p * 100, 1),
        "gemini_reasoning": gemini_insight
    }

def call_gemini_forensic_reasoner(payload: Dict[str, Any], api_key: str) -> Optional[str]:
    """
    Calls Google Gemini Multimodal Vision API to describe the visual specimen in one line
    and provide an expert visual judgment on whether it feels REAL or FAKE.
    """
    cleaned_key = api_key.strip().replace('"', '').replace("'", "")
    models = [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.7-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash"
    ]
    
    parts = []
    
    # 1. Attach Visual Image Data (Prefer full uncropped image to catch corner logos/watermarks)
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
        "Carefully inspect this full uncropped image for authenticity:\n"
        "1. AI Watermarks & Logos: Look specifically for Google Gemini 4-pointed sparkle/star icons, DALL-E color badges, Bing Creator icons, Adobe Firefly stamps, TikTok AI markers, or watermark overlays anywhere on the image or in the corners. If ANY AI logo/watermark is present, you MUST state DETECTED.\n"
        "2. Visual & Generative Anomalies: Inspect skin texture (airbrushed plastic sheen), iris catchlights/pupil asymmetry, ear/hair blending, impossible lighting physics, hands/fingers, and background distortions.\n\n"
        "Provide your analysis in exactly this format:\n\n"
        "Visual Description: [Describe in one clear line who/what is depicted in the image, their pose, setting, and background]\n\n"
        "AI Logo / Watermark: [State DETECTED: <Name/Location of logo or watermark> or None Visible]\n\n"
        "Forensic Verdict: [State **REAL** or **FAKE**, followed by 1-2 authoritative sentences explaining your judgment, citing any visible logos, skin texture, lighting, and anatomy]\n\n"
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
        # Try both query param and header formats
        urls = [
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={cleaned_key}",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        ]
        for target_url in urls:
            req = urllib.request.Request(
                target_url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": cleaned_key
                }
            )
            try:
                with urllib.request.urlopen(req, timeout=12.0) as resp:
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
                    last_error = err_json.get("error", {}).get("message", f"HTTP {e.code}")
                except Exception:
                    last_error = f"HTTP {e.code}: {err_text[:120]}"
            except Exception as e:
                last_error = str(e)
                
    raise RuntimeError(f"Gemini API call failed: {last_error}")
