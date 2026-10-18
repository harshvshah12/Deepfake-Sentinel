import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTForImageClassification, ViTImageProcessor

EFF_MEAN = [0.485, 0.456, 0.406]
EFF_STD = [0.229, 0.224, 0.225]


def fake_class_index(id2label: Dict[str, Any]) -> int:
    """Map a Hugging Face id2label table to the index that means fake."""
    fake_names = {"fake", "deepfake", "ai", "synthetic", "manipulated"}
    for key, value in (id2label or {}).items():
        if str(value).strip().lower() in fake_names:
            return int(key)
    raise ValueError(f"No fake class in id2label: {id2label}")


class EnsembleModel(nn.Module):
    def __init__(self, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.device = torch.device(device)
        self.eff_status: Dict[str, Any] = {"available": False, "error": None}
        self.vit_status: Dict[str, Any] = {"available": False, "error": None}
        self.eff_checkpoint_meta: Dict[str, Any] = {}
        self.vit_id2label: Dict[str, Any] = {}
        self.fake_index = 1
        self.eff_model = None
        self.vit_model = None
        self.processor = None
        self.eff_threshold = None

        self.eff_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=EFF_MEAN, std=EFF_STD)
        ])

        eff_weights_path = os.path.join(
            os.path.dirname(__file__), 'weights', 'deepfakefusion',
            'best_efficientnet_b3_deepfake_detector_399k.pth'
        )
        try:
            if not os.path.exists(eff_weights_path):
                raise FileNotFoundError(eff_weights_path)
            ckpt = torch.load(eff_weights_path, map_location=self.device, weights_only=False)
            self.eff_model = models.efficientnet_b3(weights=None)
            self.eff_model.classifier[1] = nn.Linear(self.eff_model.classifier[1].in_features, 1)
            self.eff_model.load_state_dict(ckpt['state_dict'])
            self.eff_model.to(self.device)
            self.eff_model.eval()
            self.eff_threshold = float(ckpt.get('threshold', 0.43))
            self.eff_checkpoint_meta = {
                "model_name": ckpt.get("model_name"),
                "image_size": ckpt.get("image_size"),
                "label_map": ckpt.get("label_map"),
                "threshold": self.eff_threshold,
                "version": ckpt.get("version"),
                "final_test_metrics": ckpt.get("final_test_metrics"),
                "source": "checkpoint_embedded_training_run",
                "note": (
                    "These metrics were stored inside the checkpoint by its original training job. "
                    "They are not a measurement of the images in this repository's dataset folder."
                ),
            }
            self.eff_status = {
                "available": True,
                "checkpoint": eff_weights_path,
                "architecture": "efficientnet_b3",
                "output": "single_logit_sigmoid_is_p_fake",
                "label_map": ckpt.get("label_map") or {"real": 0, "fake": 1},
                "input_size": [256, 256],
                "normalization": {"mean": EFF_MEAN, "std": EFF_STD},
                "threshold": self.eff_threshold,
            }
        except Exception as exc:
            self.eff_model = None
            self.eff_status = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

        self.vit_name = "dima806/deepfake_vs_real_image_detection"
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.vit_name)
            self.vit_model = ViTForImageClassification.from_pretrained(self.vit_name)
            self.vit_model.to(self.device)
            self.vit_model.eval()
            self.vit_id2label = dict(self.vit_model.config.id2label)
            self.fake_index = fake_class_index(self.vit_id2label)
            self.vit_status = {
                "available": True,
                "checkpoint": self.vit_name,
                "architecture": "vit_b_16",
                "output": "softmax_probability",
                "id2label": self.vit_id2label,
                "fake_index": self.fake_index,
                "input_size": [int(self.vit_model.config.image_size)] * 2,
                "task": "deepfake_vs_real_image_classification",
                "limitation": (
                    "This checkpoint is a face deepfake-versus-real classifier. "
                    "A high fake score is not evidence that every diffusion generator was in its training set."
                ),
            }
        except Exception as exc:
            self.vit_model = None
            self.processor = None
            self.vit_status = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        
    def forward_eff(self, x):
        return self.eff_model(x)

    def forward_vit(self, x):
        return self.vit_model(pixel_values=x).logits

    def forward(self, x):
        """Default forward executes EfficientNet-B3 for CAM/FGSM pipelines."""
        return self.eff_model(x)

    def predict_backbones(self, face_img: Image.Image, use_tta: bool = True, temperature: float = 1.0) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Return (eff_fake_prob, vit_fake_prob, detail). None means that model did not run."""
        import time
        detail: Dict[str, Any] = {"efficientnet": None, "vit": None}
        eff_fake_prob = None
        vit_fake_prob = None
        temp = max(0.1, float(temperature))

        if self.eff_model is not None:
            started = time.perf_counter()
            try:
                t1 = self.eff_transform(face_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logit1 = float(self.eff_model(t1).view(-1)[0].item())
                    if use_tta:
                        flipped = face_img.transpose(Image.FLIP_LEFT_RIGHT)
                        t2 = self.eff_transform(flipped).unsqueeze(0).to(self.device)
                        logit2 = float(self.eff_model(t2).view(-1)[0].item())
                        eff_logit = (logit1 + logit2) / 2.0
                    else:
                        eff_logit = logit1
                eff_fake_prob = float(torch.sigmoid(torch.tensor(eff_logit / temp)).item())
                detail["efficientnet"] = {
                    "available": True,
                    "logit": round(eff_logit, 6),
                    "fake_probability": round(eff_fake_prob, 6),
                    "output_semantics": "sigmoid(mean logit). Label map real=0, fake=1. This is a model score, not a calibrated probability on this deployment.",
                    "checkpoint_threshold": self.eff_threshold,
                    "above_checkpoint_threshold": bool(eff_fake_prob >= (self.eff_threshold or 0.5)),
                    "tta": use_tta,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": None,
                }
            except Exception as exc:
                detail["efficientnet"] = {"available": False, "error": f"{type(exc).__name__}: {exc}", "fake_probability": None}
        else:
            detail["efficientnet"] = {"available": False, "error": self.eff_status.get("error"), "fake_probability": None}

        if self.vit_model is not None and self.processor is not None:
            started = time.perf_counter()
            try:
                v_in = self.processor(images=face_img, return_tensors='pt')
                with torch.no_grad():
                    logits1 = self.vit_model(pixel_values=v_in['pixel_values'].to(self.device)).logits
                    if use_tta:
                        flipped = face_img.transpose(Image.FLIP_LEFT_RIGHT)
                        v_in_f = self.processor(images=flipped, return_tensors='pt')
                        logits2 = self.vit_model(pixel_values=v_in_f['pixel_values'].to(self.device)).logits
                        logits = (logits1 + logits2) / 2.0
                    else:
                        logits = logits1
                    probs = F.softmax(logits / temp, dim=1)[0]
                    vit_fake_prob = float(probs[self.fake_index].item())
                detail["vit"] = {
                    "available": True,
                    "logits": [round(float(x), 6) for x in logits.view(-1).tolist()],
                    "fake_probability": round(vit_fake_prob, 6),
                    "fake_index": self.fake_index,
                    "id2label": self.vit_id2label,
                    "output_semantics": "softmax probability of the Fake class from id2label",
                    "tta": use_tta,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": None,
                }
            except Exception as exc:
                detail["vit"] = {"available": False, "error": f"{type(exc).__name__}: {exc}", "fake_probability": None}
        else:
            detail["vit"] = {"available": False, "error": self.vit_status.get("error"), "fake_probability": None}

        return eff_fake_prob, vit_fake_prob, detail

    def predict_single(self, face_img: Image.Image, use_tta: bool = True, temperature: float = 1.0) -> Tuple[float, float, str, float]:
        """
        Executes dual-backbone multi-modal inference with Test-Time Augmentation (TTA).
        Delegates decision logic to unified ForensicDecisionEngine.
        Returns: (real_prob, fake_prob, prediction, confidence)
        """
        from backend.fusion import ForensicDecisionEngine

        eff_fake_prob, vit_fake_prob, _detail = self.predict_backbones(face_img, use_tta=use_tta, temperature=temperature)
        fusion_result = ForensicDecisionEngine.fuse_evidence(
            eff_fake_prob=eff_fake_prob,
            vit_fake_prob=vit_fake_prob,
            face_detected=True,
            eff_available=eff_fake_prob is not None,
            vit_available=vit_fake_prob is not None,
        )

        def _r(value):
            return None if value is None else round(float(value), 4)

        return (
            _r(fusion_result["real_probability"]),
            _r(fusion_result["fake_probability"]),
            fusion_result["prediction"],
            _r(fusion_result["confidence"]),
        )

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

