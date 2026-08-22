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

def evaluate_specimen(img_path):
    image = Image.open(img_path).convert("RGB")
    meta = extractor.extract_face_and_landmarks(image)
    face_img = meta["face_img"]
    face_np = meta["face_np"]
    
    # 1. Primary Dual-Backbone Spatial Inference
    raw_real, raw_fake, _, _ = ensemble.predict_single(face_img, use_tta=True)
    
    # 2. Physics & Biometric Indicators
    iris_data = iris_engine.analyze_eyes(image, meta.get("landmarks", []))
    spec_data = calculate_spectral_analysis(face_np)
    col_data = analyze_chroma_and_boundary(face_np, bbox=meta.get("bbox"))
    
    iris_score = float(iris_data.get("iris_anomaly_score", 0.25))
    spectral_score = float(spec_data.get("anomaly_score", 0.25))
    color_score = float(col_data.get("color_anomaly_index", 0.20))
    
    # Bayesian Evidence Modulation
    # If biometric iris or spectral physics strongly indicates manipulation (>0.50), boost fake score
    # If biometric iris and spectral physics confirm organic coherence (<0.35), boost real score
    biometric_delta = 0.0
    if iris_score > 0.48:
        biometric_delta += 0.20 * (iris_score - 0.48)
    elif iris_score < 0.32:
        biometric_delta -= 0.15 * (0.32 - iris_score)
        
    if spectral_score > 0.50:
        biometric_delta += 0.15 * (spectral_score - 0.50)
    elif spectral_score < 0.30:
        biometric_delta -= 0.10 * (0.30 - spectral_score)
        
    final_fake_prob = float(np.clip(raw_fake + biometric_delta, 0.001, 0.999))
    final_real_prob = float(1.0 - final_fake_prob)
    
    pred = "Fake" if final_fake_prob > 0.48 else "Real"
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
