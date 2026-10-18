import io
import hashlib
import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import Dict, Any

def compute_cryptographic_and_perceptual_hashes(image_np: np.ndarray, file_bytes: bytes = None) -> Dict[str, str]:
    """
    Computes cryptographic SHA-256 and 64-bit perceptual hashes (pHash, dHash, aHash)
    for digital asset fingerprinting, reverse lookup, and provenance verification.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Cryptographic SHA-256 Hash
    sha256 = hashlib.sha256(file_bytes if file_bytes else image_np.tobytes()).hexdigest()
    
    # 2. Average Hash (aHash)
    resized_a = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = resized_a.mean()
    ahash_bin = "".join(["1" if p > avg else "0" for p in resized_a.flatten()])
    ahash_hex = f"{int(ahash_bin, 2):016x}"
    
    # 3. Difference Hash (dHash)
    resized_d = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized_d[:, 1:] > resized_d[:, :-1]
    dhash_bin = "".join(["1" if p else "0" for p in diff.flatten()])
    dhash_hex = f"{int(dhash_bin, 2):016x}"
    
    # 4. DCT Perceptual Hash (pHash)
    resized_p = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized_p)
    dct_low = dct[:8, :8]
    median_dct = np.median(dct_low[1:, 1:])
    phash_bin = "".join(["1" if p > median_dct else "0" for p in dct_low.flatten()])
    phash_hex = f"{int(phash_bin, 2):016x}"
    
    return {
        "sha256": sha256,
        "phash": phash_hex,
        "dhash": dhash_hex,
        "ahash": ahash_hex
    }

def parse_exif_metadata(image_pil: Image.Image) -> Dict[str, Any]:
    """
    Extracts and audits EXIF tags, camera hardware profiles, software stamps, and timestamp consistency.
    """
    exif_data = {}
    ai_software_detected = False
    ai_software_name = ""
    camera_hardware_found = False
    
    ai_keywords = [
        "midjourney", "stable diffusion", "dall-e", "comfyui", "automatic1111",
        "novelai", "adobe firefly", "generative fill", "invokeai", "flux",
        "stablediffusion", "civitai", "diffusers", "leonardo.ai"
    ]
    
    try:
        raw_exif = image_pil._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_data[tag_name] = str(value)
                
                # Check for camera hardware tags
                if tag_name in ["Make", "Model", "LensModel", "FocalLength", "ISOSpeedRatings"]:
                    camera_hardware_found = True
                    
                # Check for AI Software signatures
                val_lower = str(value).lower()
                for kw in ai_keywords:
                    if kw in val_lower:
                        ai_software_detected = True
                        ai_software_name = kw.title()
    except Exception:
        pass
        
    # Provenance score based on metadata completeness
    if ai_software_detected:
        provenance_score = 0.05
    elif camera_hardware_found:
        provenance_score = 0.95
    elif len(exif_data) > 0:
        provenance_score = 0.60
    else:
        provenance_score = 0.40

    return {
        "has_exif": len(exif_data) > 0,
        "camera_hardware_found": camera_hardware_found,
        "ai_software_detected": ai_software_detected,
        "ai_software_name": ai_software_name,
        "make": exif_data.get("Make", "Unknown Camera"),
        "model": exif_data.get("Model", "Unknown Model"),
        "software": exif_data.get("Software", "Unknown"),
        "datetime": exif_data.get("DateTimeOriginal", exif_data.get("DateTime", "Unknown")),
        "provenance_score": round(provenance_score, 2),
        "raw_tags": {k: v for k, v in list(exif_data.items())[:12]}
    }

def detect_c2pa_manifest(file_bytes: bytes) -> Dict[str, Any]:
    """
    Look for a C2PA/JUMBF marker in metadata bytes only.

    A marker is not a signature check. This environment does not import a C2PA
    trust-list library, so a present marker stays unverified and a missing
    marker stays "no manifest". Neither result changes the neural score.
    """
    from backend.security import metadata_region

    region = metadata_region(file_bytes or b"")
    markers = [b"c2pa", b"jumb", b"c2bi", b"urn:c2pa"]
    has_c2pa = any(sig in region.lower() if sig != b"jumb" else sig in region for sig in markers)
    # b"c2pa" search is case-sensitive in the raw box; also accept the lowercase spelling in XMP.
    has_c2pa = any(sig in region or sig in region.lower() for sig in (b"c2pa", b"C2PA", b"jumb", b"JUMB"))

    if has_c2pa:
        return {
            "c2pa_manifest_detected": True,
            "claim_generator": "JUMBF or c2pa marker present in the metadata region",
            "signature_issuer": None,
            "validation_status": "MANIFEST_PRESENT_SIGNATURE_NOT_VERIFIED",
            "signature_checked": False,
            "trust_status": "NOT_VERIFIED",
            "content_binding": "NOT_CHECKED",
            "contributes_to_score": False,
            "note": "Byte presence is not a valid signature. The c2pa library is not installed.",
        }
    return {
        "c2pa_manifest_detected": False,
        "claim_generator": None,
        "signature_issuer": None,
        "validation_status": "NO_MANIFEST",
        "signature_checked": False,
        "trust_status": "NO_MANIFEST",
        "content_binding": "NOT_CHECKED",
        "contributes_to_score": False,
        "note": "No C2PA marker was found in the metadata region. Missing metadata does not mean the image is fake.",
    }

def detect_synthid_watermark(image_np: np.ndarray, file_bytes: bytes = b"") -> Dict[str, Any]:
    """Official SynthID verification is not available here. Do not invent one from pixel statistics."""
    del image_np, file_bytes
    return {
        "status": "NOT_CHECKED",
        "synthid_detected": None,
        "watermark_type": None,
        "watermark_confidence": None,
        "detection_method": "not_checked",
        "contributes_to_score": False,
        "reason": (
            "The official Google SynthID verifier is not installed and no SynthID credential is configured. "
            "High-frequency energy is not treated as a SynthID result."
        ),
    }

_METADATA_TAGS = (
    (b"stable diffusion", "Stable Diffusion named in metadata"),
    (b"midjourney", "Midjourney named in metadata"),
    (b"dall-e", "DALL-E named in metadata"),
    (b"adobe firefly", "Adobe Firefly named in metadata"),
    (b"synthid", "SynthID named in metadata text"),
)


def _glyph_line(roi: np.ndarray) -> Dict[str, Any]:
    """A text-like line is several similarly sized components on one baseline."""
    if roi.size < 64:
        return {"hit": False}
    blur = cv2.GaussianBlur(roi, (0, 0), 1.0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(thr)
    glyphs = []
    height, width = roi.shape[:2]
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if 20 <= area <= 2200 and 12 <= bh <= int(height * 0.85) and 3 <= bw <= int(width * 0.5):
            glyphs.append((int(x), int(y), int(bw), int(bh)))
    if len(glyphs) < 4:
        return {"hit": False, "glyph_count": len(glyphs)}
    tops = [g[1] for g in glyphs]
    heights = [g[3] for g in glyphs]
    if max(tops) - min(tops) > height * 0.4:
        return {"hit": False, "glyph_count": len(glyphs)}
    if max(heights) > min(heights) * 3.5:
        return {"hit": False, "glyph_count": len(glyphs)}
    xs = [g[0] for g in glyphs]
    span = max(xs) - min(xs)
    if span < width * 0.25:
        return {"hit": False, "glyph_count": len(glyphs)}
    x0 = min(g[0] for g in glyphs)
    y0 = min(g[1] for g in glyphs)
    x1 = max(g[0] + g[2] for g in glyphs)
    y1 = max(g[1] + g[3] for g in glyphs)
    return {
        "hit": True,
        "glyph_count": len(glyphs),
        "local_bbox": [x0, y0, x1, y1],
        "confidence": round(min(0.9, 0.55 + 0.05 * (len(glyphs) - 4)), 3),
    }


def detect_visual_watermarks_and_logos(image_np: np.ndarray, file_bytes: bytes = b"") -> Dict[str, Any]:
    """
    Visible text-like marks in the corners, plus generator names in metadata only.

    OCR is unavailable without Tesseract. Ordinary corner texture is not enough
    to call a watermark. Compressed pixel bytes are not searched for keywords.
    """
    from backend.security import metadata_region

    regions = []
    visual_detected = False
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
        h, w = gray.shape[:2]
        corner_h = max(16, int(h * 0.16))
        corner_w = max(16, int(w * 0.28))
        boxes = {
            "bottom_right": (h - corner_h, h, w - corner_w, w),
            "bottom_left": (h - corner_h, h, 0, corner_w),
            "top_right": (0, corner_h, w - corner_w, w),
            "top_left": (0, corner_h, 0, corner_w),
        }
        for name, (y1, y2, x1, x2) in boxes.items():
            roi = gray[y1:y2, x1:x2]
            found = _glyph_line(roi)
            if not found.get("hit"):
                continue
            local = found["local_bbox"]
            regions.append({
                "location": name,
                "bbox": [int(x1 + local[0]), int(y1 + local[1]), int(x1 + local[2]), int(y1 + local[3])],
                "glyph_count": found["glyph_count"],
                "confidence": found["confidence"],
                "detector": "corner_glyph_line",
            })
            visual_detected = True
    except Exception as exc:
        regions.append({"error": f"{type(exc).__name__}: {exc}"})

    metadata_match = None
    region = metadata_region(file_bytes or b"")
    lowered = region.lower()
    for sig, name in _METADATA_TAGS:
        if sig in lowered:
            metadata_match = name
            break

    return {
        "watermark_detected": visual_detected,
        "visual_watermark_detected": visual_detected,
        "regions": regions,
        "visual_location": regions[0]["location"] if regions and "location" in regions[0] else None,
        "visual_bbox": regions[0]["bbox"] if regions and "bbox" in regions[0] else None,
        "visual_confidence": regions[0]["confidence"] if regions and "confidence" in regions[0] else 0.0,
        "metadata_text_match": metadata_match,
        "metadata_signature_detected": metadata_match is not None,
        "watermark_type": "visible_text_like_mark" if visual_detected else None,
        "location": regions[0]["location"] if regions and "location" in regions[0] else None,
        "confidence": regions[0]["confidence"] if regions and "confidence" in regions[0] else None,
        "ocr_status": "UNAVAILABLE",
        "ocr_reason": "pytesseract is not installed, so this check does not read the characters.",
        "contributes_to_score": False,
        "detector_status": "RAN",
        "limitation": "A visible mark is not proof the image is synthetic. Absence of a mark is not proof it is authentic.",
    }
