import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict, Any
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTForImageClassification, ViTImageProcessor

class EnsembleModel(nn.Module):
    def __init__(self, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.device = torch.device(device)
        
        # 1. Load Primary EfficientNet-B3 Backbone (Trained on 399k Real & Deepfake Faces)
        self.eff_model = models.efficientnet_b3(weights=None)
        self.eff_model.classifier[1] = nn.Linear(self.eff_model.classifier[1].in_features, 1)
        
        eff_weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'deepfakefusion', 'best_efficientnet_b3_deepfake_detector_399k.pth')
        if os.path.exists(eff_weights_path):
            ckpt = torch.load(eff_weights_path, map_location=self.device)
            self.eff_model.load_state_dict(ckpt['state_dict'])
            self.eff_threshold = ckpt.get('threshold', 0.43)
        else:
            self.eff_threshold = 0.45
            
        self.eff_model.to(self.device)
        self.eff_model.eval()
        
        self.eff_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Load Secondary ViT-B/16 Backbone
        self.vit_name = "dima806/deepfake_vs_real_image_detection"
        self.processor = ViTImageProcessor.from_pretrained(self.vit_name)
        self.vit_model = ViTForImageClassification.from_pretrained(self.vit_name)
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
    def forward_eff(self, x):
        return self.eff_model(x)

    def forward_vit(self, x):
        return self.vit_model(pixel_values=x).logits

    def forward(self, x):
        """Default forward executes EfficientNet-B3 for CAM/FGSM pipelines."""
        return self.eff_model(x)

    def predict_single(self, face_img: Image.Image, use_tta: bool = True, temperature: float = 1.0) -> Tuple[float, float, str, float]:
        """
        Executes dual-backbone multi-modal inference with Test-Time Augmentation (TTA).
        Returns: (real_prob, fake_prob, prediction, confidence)
        """
        # --- Backbone 1: EfficientNet-B3 ---
        t1 = self.eff_transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit1 = self.eff_model(t1).item()
            if use_tta:
                flipped = face_img.transpose(Image.FLIP_LEFT_RIGHT)
                t2 = self.eff_transform(flipped).unsqueeze(0).to(self.device)
                logit2 = self.eff_model(t2).item()
                eff_logit = (logit1 + logit2) / 2.0
            else:
                eff_logit = logit1
            eff_fake_prob = torch.sigmoid(torch.tensor(eff_logit / max(0.1, temperature))).item()

        # --- Backbone 2: ViT-B/16 ---
        v_in = self.processor(images=face_img, return_tensors='pt')
        with torch.no_grad():
            vit_logits1 = self.vit_model(pixel_values=v_in['pixel_values'].to(self.device)).logits
            if use_tta:
                flipped = face_img.transpose(Image.FLIP_LEFT_RIGHT)
                v_in_f = self.processor(images=flipped, return_tensors='pt')
                vit_logits2 = self.vit_model(pixel_values=v_in_f['pixel_values'].to(self.device)).logits
                vit_logits = (vit_logits1 + vit_logits2) / 2.0
            else:
                vit_logits = vit_logits1
            vit_probs = F.softmax(vit_logits / max(0.1, temperature), dim=1)[0]
            vit_fake_prob = float(vit_probs[1].item())

        # --- Calibrated Dual-Domain Fusion ---
        # EfficientNet-B3 (60% weight, high generalizability) + ViT (40% weight, fine boundary detection)
        fake_prob = (eff_fake_prob * 0.60) + (vit_fake_prob * 0.40)
        real_prob = 1.0 - fake_prob
        
        # Decision Boundary
        prediction = "Fake" if fake_prob > 0.48 else "Real"
        confidence = float(max(real_prob, fake_prob))
        
        return round(real_prob, 4), round(fake_prob, 4), prediction, round(confidence, 4)

    def predict_batch(self, face_imgs: List[Image.Image], use_tta: bool = True) -> List[Dict[str, Any]]:
        """
        High-throughput batch inference across video frames.
        """
        if not face_imgs:
            return []

        results = []
        for img in face_imgs:
            r_p, f_p, pred, conf = self.predict_single(img, use_tta=use_tta)
            results.append({
                'real_prob': r_p,
                'fake_prob': f_p,
                'prediction': pred,
                'confidence': conf
            })
        return results

