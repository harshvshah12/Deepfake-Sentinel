import glob, os
import numpy as np
from PIL import Image
from backend.model import EnsembleModel
from backend.preprocessor import FaceExtractor
from backend.iris import IrisBiometricEngine
from backend.frequency import calculate_spectral_analysis
from backend.color_forensics import analyze_chroma_and_boundary

device = "cpu"
ensemble = EnsembleModel(device=device)
extractor = FaceExtractor(device=device)
iris_engine = IrisBiometricEngine(device=device)

from backend.fusion import ForensicDecisionEngine

def evaluate_specimen(img_path):
    image = Image.open(img_path).convert("RGB")
    meta = extractor.extract_face_and_landmarks(image)
    face_img = meta["face_img"]
    face_np = meta["face_np"]
    
    # 1. Primary Dual-Backbone Spatial Inference
    eff_fake_prob, vit_fake_prob = ensemble.predict_backbones(face_img, use_tta=True)
    
    # 2. Physics & Biometric Indicators
    iris_data = iris_engine.analyze_eyes(image, meta.get("landmarks", []))
    spec_data = calculate_spectral_analysis(face_np)
    col_data = analyze_chroma_and_boundary(face_np, bbox=meta.get("bbox"))
    
    # Unified Bayesian Evidence Fusion
    fusion_result = ForensicDecisionEngine.fuse_evidence(
        eff_fake_prob=eff_fake_prob,
        vit_fake_prob=vit_fake_prob,
        spectral_score=float(spec_data.get("anomaly_score", 0.25)),
        iris_data=iris_data,
        color_data=col_data,
        face_detected=meta.get("detected", True)
    )
    
    pred = fusion_result["prediction"]
    final_real_prob = fusion_result["real_probability"]
    final_fake_prob = fusion_result["fake_probability"]
    
    return pred, final_real_prob, final_fake_prob, iris_data.get("is_synthetic_iris", False)

real_samples = glob.glob('dataset/real/*.jpg')[:25]
fake_samples = glob.glob('dataset/fake/*.jpg')[:25]

print("=== BAYESIAN FUSED ACCURACY: REAL (25) ===")
real_hits = 0
for p in real_samples:
    name = os.path.basename(p)
    pred, r_p, f_p, syn_iris = evaluate_specimen(p)
    if pred == "Real":
        real_hits += 1
    print(f"REAL [{name}]: Pred={pred} (Real={r_p:.3f}, Fake={f_p:.3f})")

print(f"\nREAL ACCURACY: {real_hits}/25 ({real_hits/25*100:.1f}%)")

print("\n=== BAYESIAN FUSED ACCURACY: FAKE (25) ===")
fake_hits = 0
for p in fake_samples:
    name = os.path.basename(p)
    pred, r_p, f_p, syn_iris = evaluate_specimen(p)
    if pred == "Fake":
        fake_hits += 1
    print(f"FAKE [{name}]: Pred={pred} (Real={r_p:.3f}, Fake={f_p:.3f}, SyntheticIris={syn_iris})")

print(f"\nFAKE ACCURACY: {fake_hits}/25 ({fake_hits/25*100:.1f}%)")
print(f"\nOVERALL SYSTEM ACCURACY: {real_hits + fake_hits}/50 ({(real_hits + fake_hits)/50*100:.1f}%)")
