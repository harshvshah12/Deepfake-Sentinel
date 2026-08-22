import os
import json
from typing import Optional, Dict, Any

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "data", "feedback_memory.json")
os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

def hamming_distance(hex1: str, hex2: str) -> int:
    try:
        n1 = int(hex1, 16)
        n2 = int(hex2, 16)
        return bin(n1 ^ n2).count("1")
    except Exception:
        return 999

def save_user_feedback(
    sha256: str,
    phash: str,
    corrected_label: str,
    original_prediction: str,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Saves user feedback and corrections into persistent memory for active online learning.
    Future scans of this image (or duplicates matched by SHA-256 or pHash) will permanently use this verdict.
    """
    memory = {}
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            memory = {}

    entry = {
        "sha256": sha256,
        "phash": phash,
        "corrected_label": corrected_label,
        "original_prediction": original_prediction,
        "confidence": confidence,
        "metadata": metadata or {}
    }
    
    memory[sha256] = entry
    
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
        
    return {
        "status": "learned",
        "message": f"Feedback successfully learned. Specimen will permanently classify as '{corrected_label}'.",
        "entry": entry
    }

def lookup_feedback_memory(sha256: Optional[str], phash: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Checks if a specimen has prior human feedback recorded via exact SHA-256 or perceptual pHash (Hamming <= 10).
    """
    if not os.path.exists(MEMORY_FILE):
        return None
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
            
        # 1. Exact SHA-256 Match
        if sha256 and sha256 in memory:
            return memory[sha256]
            
        # 2. Perceptual pHash Similarity Match (Hamming distance <= 10 for compressed/resized images)
        if phash:
            best_match = None
            min_dist = 999
            for k, v in memory.items():
                stored_phash = v.get("phash")
                if stored_phash:
                    dist = hamming_distance(phash, stored_phash)
                    if dist <= 10 and dist < min_dist:
                        min_dist = dist
                        best_match = v
            if best_match:
                return best_match
    except Exception:
        return None
        
    return None

def get_feedback_stats() -> Dict[str, Any]:
    """Returns feedback memory statistics."""
    if not os.path.exists(MEMORY_FILE):
        return {"total_feedback_entries": 0, "entries": []}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
        return {
            "total_feedback_entries": len(memory),
            "entries": list(memory.values())
        }
    except Exception:
        return {"total_feedback_entries": 0, "entries": []}
