import io
import os
import base64
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List

from backend.frequency import compute_fft_spectrum, calculate_spectral_analysis
from backend.xai import generate_heatmap
from backend.iris import IrisBiometricEngine
from backend.fusion import ForensicDecisionEngine

iris_engine = IrisBiometricEngine(device='cpu')

def np_to_b64(img_arr: np.ndarray, quality: int = 85) -> str:
    img = Image.fromarray(img_arr.astype('uint8'))
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG', quality=quality)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_video_file(
    video_bytes: bytes,
    extractor,
    ensemble_model,
    device,
    max_frames: int = 12,
    attack: bool = False,
    epsilon: float = 0.05
) -> Dict[str, Any]:
    """
    Performs multi-frame temporal forensic analysis on uploaded video streams.
    Extracts keyframes, executes spatial ViT classification + TTA, 2D FFT spectral optics,
    Grad-CAM explainability, and computes temporal consistency metrics.
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    try:
        sampled_data = extractor.extract_video_frames(tmp_path, max_frames=max_frames)
        
        if not sampled_data:
            raise ValueError("Unable to decode video stream or video contains zero valid frames.")

        frame_results = []
        fake_probs = []
        real_probs = []
        spectral_scores = []
        
        for item in sampled_data:
            face_img = item['face_img']
            face_np = item['face_np']
            frame_idx = item['frame_idx']
            timestamp_sec = item['timestamp_sec']
            
            # 1. Dual-backbone spatial inference
            eff_fake_prob, vit_fake_prob = ensemble_model.predict_backbones(face_img, use_tta=True)
            
            # 2. Spectral 2D Fourier Analysis
            spectral_data = calculate_spectral_analysis(face_np)
            fft_spectrum = compute_fft_spectrum(face_np)
            
            # 3. Iris & Corneal Reflection Analysis
            iris_data = iris_engine.analyze_eyes(face_img, item.get('landmarks', []))
            
            # Frame-level Unified Bayesian Fusion
            frame_fusion = ForensicDecisionEngine.fuse_evidence(
                eff_fake_prob=eff_fake_prob,
                vit_fake_prob=vit_fake_prob,
                spectral_score=spectral_data['anomaly_score'],
                iris_data=iris_data,
                face_detected=item.get('detected', True)
            )
            fused_fake = frame_fusion['fake_probability']
            fused_real = frame_fusion['real_probability']
            frame_pred = frame_fusion['prediction']
            frame_conf = frame_fusion['confidence']
            
            # 4. Explainable AI Heatmap
            input_tensor = ensemble_model.eff_transform(face_img).unsqueeze(0).to(device)
            try:
                heatmap = generate_heatmap(ensemble_model, input_tensor, face_np)
            except Exception:
                heatmap = face_np
            
            face_resized = cv2.resize(face_np, (224, 224))
            
            frame_info = {
                'frame_idx': frame_idx,
                'timestamp_sec': timestamp_sec,
                'prediction': frame_pred,
                'confidence': round(frame_conf, 4),
                'fake_probability': fused_fake,
                'real_probability': fused_real,
                'spectral_anomaly_score': spectral_data['anomaly_score'],
                'iris_anomaly_score': float(iris_data.get('iris_anomaly_score', 0.25)),
                'face_detected': item.get('detected', True),
                'face_b64': np_to_b64(face_resized),
                'heatmap_b64': np_to_b64(heatmap),
                'fft_b64': np_to_b64(fft_spectrum)
            }
            
            frame_results.append(frame_info)
            fake_probs.append(fused_fake)
            real_probs.append(fused_real)
            spectral_scores.append(spectral_data['anomaly_score'])

        # Temporal Aggregation and Variance Metrics
        mean_fake_prob = float(np.mean(fake_probs))
        mean_real_prob = float(np.mean(real_probs))
        max_fake_spike = float(np.max(fake_probs))
        temporal_variance = float(np.var(fake_probs))
        mean_spectral_score = float(np.mean(spectral_scores))

        # Calibrated overall prediction using ForensicDecisionEngine boundaries
        if mean_fake_prob >= ForensicDecisionEngine.CONFIDENT_FAKE_THRESHOLD or (max_fake_spike > 0.85 and temporal_variance > 0.04):
            overall_prediction = 'Fake'
            overall_confidence = mean_fake_prob
            verdict_badge = "SYNTHETIC (AI-GENERATED VIDEO)"
            is_inconclusive = False
        elif mean_fake_prob <= ForensicDecisionEngine.CONFIDENT_REAL_THRESHOLD:
            overall_prediction = 'Real'
            overall_confidence = mean_real_prob
            verdict_badge = "AUTHENTIC (REAL VIDEO)"
            is_inconclusive = False
        else:
            overall_prediction = 'Inconclusive'
            overall_confidence = float(max(mean_real_prob, mean_fake_prob))
            verdict_badge = "BORDERLINE / INCONCLUSIVE VIDEO"
            is_inconclusive = True

        # Select primary representative diagnostic frame
        if overall_prediction == 'Fake':
            primary_idx = int(np.argmax(fake_probs))
        else:
            primary_idx = int(np.argmax(real_probs))
        primary_frame = frame_results[primary_idx]

        return {
            'media_type': 'video',
            'prediction': overall_prediction,
            'confidence': round(overall_confidence, 4),
            'real_probability': round(mean_real_prob, 4),
            'fake_probability': round(mean_fake_prob, 4),
            'spectral_anomaly_score': round(mean_spectral_score, 4),
            'temporal_variance': round(temporal_variance, 4),
            'frames_analyzed': len(frame_results),
            'adversarial_attack_applied': attack,
            'epsilon': epsilon,
            'face_b64': primary_frame['face_b64'],
            'heatmap_b64': primary_frame['heatmap_b64'],
            'fft_b64': primary_frame['fft_b64'],
            'trajectory': frame_results
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
