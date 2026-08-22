import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any

class FaceExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        # Initialize MTCNN with multi-scale support and tight landmark tracking
        self.mtcnn = MTCNN(
            image_size=256,
            margin=40,
            min_face_size=15,
            keep_all=False,
            device=self.device,
            post_process=True,
            thresholds=[0.3, 0.4, 0.4]
        )

    def extract_face(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Extracts aligned face tensor (3, 256, 256) normalized to [-1, 1]."""
        try:
            w, h = image.size
            detect_img = image if min(w, h) >= 256 else image.resize((256, 256), Image.Resampling.BILINEAR)
            face_tensor = self.mtcnn(detect_img)
            return face_tensor
        except Exception:
            return None

    def extract_face_and_landmarks(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extracts face crop, bounding box, 5-point facial landmarks, and confidence score.
        Falls back to center-crop if face detection misses.
        """
        w, h = image.size
        try:
            # If image is small, upscale slightly to assist MTCNN pyramid scales
            detect_img = image if min(w, h) >= 256 else image.resize((256, 256), Image.Resampling.BILINEAR)
            dw, dh = detect_img.size
            scale_x, scale_y = w / dw, h / dh

            boxes, probs, landmarks = self.mtcnn.detect(detect_img, landmarks=True)
            if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                box = boxes[0]
                prob = float(probs[0])
                lms_raw = landmarks[0] if landmarks is not None else []
                lms = [[float(pt[0] * scale_x), float(pt[1] * scale_y)] for pt in lms_raw]
                
                # Bounding box clamped to image dimensions
                x1_c = max(0, int(box[0] * scale_x))
                y1_c = max(0, int(box[1] * scale_y))
                x2_c = min(w, int(box[2] * scale_x))
                y2_c = min(h, int(box[3] * scale_y))
                
                # Check if the detected face already occupies most of the frame (portrait mode)
                box_w = x2_c - x1_c
                box_h = y2_c - y1_c
                face_area_ratio = (box_w * box_h) / max(1, w * h)
                
                if face_area_ratio > 0.40 or box_w > 0.65 * w or box_h > 0.65 * h:
                    # Input is already a face portrait: use original frame resized directly
                    cropped_img = image.resize((256, 256), Image.Resampling.BILINEAR)
                else:
                    margin_x = box_w * 0.30
                    margin_y = box_h * 0.35
                    crop_x1 = max(0, int(x1_c - margin_x))
                    crop_y1 = max(0, int(y1_c - margin_y))
                    crop_x2 = min(w, int(x2_c + margin_x))
                    crop_y2 = min(h, int(y2_c + margin_y))
                    cropped_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).resize((256, 256), Image.Resampling.BILINEAR)
                    
                face_np = np.array(cropped_img)
                
                return {
                    'detected': True,
                    'confidence': prob,
                    'bbox': [x1_c, y1_c, x2_c, y2_c],
                    'landmarks': lms,
                    'face_img': cropped_img,
                    'face_np': face_np
                }
        except Exception:
            pass

        # Center crop fallback
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped_img = image.crop((left, top, left + min_dim, top + min_dim)).resize((256, 256), Image.Resampling.BILINEAR)
        return {
            'detected': False,
            'confidence': 0.50,
            'bbox': [left, top, left + min_dim, top + min_dim],
            'landmarks': [],
            'face_img': cropped_img,
            'face_np': np.array(cropped_img)
        }

    def extract_video_frames(self, video_path: str, max_frames: int = 16) -> List[Dict[str, Any]]:
        """
        Samples uniform keyframes from a video file, detects faces, and extracts alignment data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            return []

        # Determine frame indices to sample
        num_samples = min(max_frames, max(1, total_frames))
        frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        extracted_data = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            face_data = self.extract_face_and_landmarks(pil_img)
            
            timestamp = float(idx / fps)
            face_data['frame_idx'] = int(idx)
            face_data['timestamp_sec'] = round(timestamp, 2)
            extracted_data.append(face_data)

        cap.release()
        return extracted_data

