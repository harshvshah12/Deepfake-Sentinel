import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Tuple, Optional

class SiameseIrisNetwork(nn.Module):
    """
    Lightweight Siamese Neural Network to extract bilateral iris & periocular feature embeddings.
    Evaluates physiological cross-eye coherence and feature distance.
    """
    def __init__(self):
        super(SiameseIrisNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 128x1x1
        )
        self.fc = nn.Linear(128, 64)
        
    def forward_one(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)
        return F.normalize(self.fc(feat), p=2, dim=1)
        
    def forward(self, left_eye_t, right_eye_t):
        feat_l = self.forward_one(left_eye_t)
        feat_r = self.forward_one(right_eye_t)
        # Cosine similarity between left and right eye feature embeddings
        similarity = F.cosine_similarity(feat_l, feat_r, dim=1)
        return similarity

class IrisBiometricEngine:
    """
    Next-Generation Biometric Iris & Corneal Specular Reflection Forensic Engine.
    Implements:
    - 1. Pupil Boundary Ellipticity & Circularity Index
    - 2. Corneal Specular Highlight Bilateral Consistency (ICASSP Hu et al.)
    - 3. Bilateral Sclera Chrominance Variance (L*a*b*)
    - 4. Siamese Iris Deep Feature Coherence
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.siamese_net = SiameseIrisNetwork().to(self.device)
        self.siamese_net.eval()

    def analyze_eyes(self, image_pil: Image.Image, landmarks: list) -> Dict[str, Any]:
        img_np = np.array(image_pil)
        h, w = img_np.shape[:2]

        if not landmarks or len(landmarks) < 2:
            return {
                'detected': False,
                'pupil_circularity_left': 0.85,
                'pupil_circularity_right': 0.85,
                'mean_circularity': 0.85,
                'specular_area_left': 10,
                'specular_area_right': 10,
                'specular_asymmetry': 0.1,
                'sclera_chroma_mismatch': 0.1,
                'siamese_coherence': 0.90,
                'iris_anomaly_score': 0.15,
                'is_synthetic_iris': False,
                'left_eye_b64': '',
                'right_eye_b64': ''
            }

        left_eye_pt = landmarks[0]
        right_eye_pt = landmarks[1]
        
        # Determine adaptive eye RoI size based on inter-ocular distance
        iod = float(np.linalg.norm(np.array(left_eye_pt) - np.array(right_eye_pt)))
        box_size = max(20, int(iod * 0.35))

        def extract_eye_roi(pt):
            cx, cy = int(pt[0]), int(pt[1])
            x1 = max(0, cx - box_size)
            y1 = max(0, cy - box_size)
            x2 = min(w, cx + box_size)
            y2 = min(h, cy + box_size)
            crop = img_np[y1:y2, x1:x2]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                return np.zeros((64, 64, 3), dtype=np.uint8), 0.85, 0, (0, 0), 0.0
                
            crop_64 = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(crop_64, cv2.COLOR_RGB2GRAY)
            
            # --- 1. Pupil Extraction & Circularity ---
            pupil_thresh = np.percentile(gray, 14)
            _, pupil_mask = cv2.threshold(gray, pupil_thresh, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circularity = 0.82
            pupil_center = (32, 32)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0 and area > 10:
                    circularity = float((4 * np.pi * area) / (perimeter ** 2))
                    circularity = np.clip(circularity, 0.0, 1.0)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    pupil_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # --- 2. Corneal Specular Highlights Extraction ---
            spec_thresh = max(195, np.percentile(gray, 96))
            _, spec_mask = cv2.threshold(gray, spec_thresh, 255, cv2.THRESH_BINARY)
            spec_area = int(np.sum(spec_mask > 0))
            
            spec_contours, _ = cv2.findContours(spec_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            spec_offset = (0, 0)
            if spec_contours:
                sc = max(spec_contours, key=cv2.contourArea)
                sM = cv2.moments(sc)
                if sM["m00"] > 0:
                    sc_center = (int(sM["m10"] / sM["m00"]), int(sM["m01"] / sM["m00"]))
                    spec_offset = (sc_center[0] - pupil_center[0], sc_center[1] - pupil_center[1])

            # --- 3. Sclera Chrominance Variance in LAB ---
            lab = cv2.cvtColor(crop_64, cv2.COLOR_RGB2LAB)
            a_mean = float(np.mean(lab[:, :, 1]))
            b_mean = float(np.mean(lab[:, :, 2]))
            chroma_mean = float(np.sqrt(a_mean**2 + b_mean**2))

            # Draw forensic overlay on eye RoI for UI HUD
            annotated_roi = crop_64.copy()
            cv2.circle(annotated_roi, pupil_center, 6, (0, 255, 200), 1)
            if spec_area > 0:
                cv2.circle(annotated_roi, (pupil_center[0] + spec_offset[0], pupil_center[1] + spec_offset[1]), 3, (255, 80, 80), -1)

            return annotated_roi, float(circularity), int(spec_area), spec_offset, chroma_mean

        left_roi, c_l, s_l, off_l, chr_l = extract_eye_roi(left_eye_pt)
        right_roi, c_r, s_r, off_r, chr_r = extract_eye_roi(right_eye_pt)

        # Specular Highlight Asymmetry
        max_spec = max(s_l, s_r, 1)
        spec_area_asymmetry = abs(s_l - s_r) / max_spec

        # Specular Vector / Angle Inconsistency
        vec_l = np.array(off_l, dtype=np.float32)
        vec_r = np.array(off_r, dtype=np.float32)
        norm_l = np.linalg.norm(vec_l)
        norm_r = np.linalg.norm(vec_r)
        if norm_l > 0 and norm_r > 0:
            cos_sim = float(np.dot(vec_l, vec_r) / (norm_l * norm_r))
            spec_vector_inconsistency = max(0.0, 1.0 - (cos_sim + 1.0) / 2.0)
        else:
            spec_vector_inconsistency = 0.15

        # Pupil Circularity Anomaly
        mean_circularity = (c_l + c_r) / 2.0
        pupil_irregularity = float(np.clip(1.0 - (mean_circularity / 0.85), 0.0, 1.0))

        # Sclera Chrominance Discrepancy
        chroma_diff = abs(chr_l - chr_r) / max(chr_l, chr_r, 1.0)

        # Siamese Deep Feature Embedding Coherence
        with torch.no_grad():
            t_l = torch.from_numpy(left_roi).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            t_r = torch.from_numpy(right_roi).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            siamese_sim = float(self.siamese_net(t_l, t_r).item())

        # Synthesize Overall Iris Biometric Anomaly Index
        iris_anomaly = float(np.clip(
            (pupil_irregularity * 0.35) + 
            (spec_area_asymmetry * 0.30) + 
            (spec_vector_inconsistency * 0.20) + 
            (chroma_diff * 0.15),
            0.0, 1.0
        ))

        def roi_to_b64(arr):
            im = Image.fromarray(arr.astype('uint8'))
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=90)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            'detected': True,
            'pupil_circularity_left': round(c_l, 3),
            'pupil_circularity_right': round(c_r, 3),
            'mean_circularity': round(mean_circularity, 3),
            'specular_area_left': s_l,
            'specular_area_right': s_r,
            'specular_asymmetry': round(spec_area_asymmetry, 3),
            'specular_vector_inconsistency': round(spec_vector_inconsistency, 3),
            'sclera_chroma_mismatch': round(float(chroma_diff), 3),
            'siamese_coherence': round(siamese_sim, 3),
            'iris_anomaly_score': round(iris_anomaly, 3),
            'is_synthetic_iris': iris_anomaly > 0.52,
            'left_eye_b64': roi_to_b64(left_roi),
            'right_eye_b64': roi_to_b64(right_roi)
        }
