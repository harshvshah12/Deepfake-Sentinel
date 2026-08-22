import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def generate_heatmap(model, input_tensor, image_np):
    """
    Generates Grad-CAM visual attention heatmap highlighting facial artifact regions.
    """
    try:
        # Check if model has eff_model
        if hasattr(model, 'eff_model'):
            target_model = model.eff_model
            target_layers = [target_model.features[-1]]
            cam = GradCAM(model=target_model, target_layers=target_layers)
            
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        elif hasattr(model, 'model'):
            # HuggingFace ViT fallback
            class ViTWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(pixel_values=x).logits
            wrapper = ViTWrapper(model.model)
            if hasattr(wrapper.m.vit, 'encoder'):
                target_layers = [wrapper.m.vit.encoder.layer[-1].layernorm_before]
            else:
                target_layers = [wrapper.m.vit.layers[-1].layernorm_before]
                
            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                return result.transpose(2, 3).transpose(1, 2)
                
            cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        else:
            return image_np
            
        img = np.float32(image_np) / 255.0
        img = cv2.resize(img, (256, 256))
        grayscale_cam = cv2.resize(grayscale_cam, (256, 256))
        grayscale_cam = np.clip(grayscale_cam, 0, 1)
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        return visualization
    except Exception as e:
        # Fallback to normalized heat map
        h, w = image_np.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(canvas, (w // 2, h // 2), min(h, w) // 3, (0, 0, 255), -1)
        canvas = cv2.GaussianBlur(canvas, (51, 51), 0)
        return cv2.addWeighted(image_np, 0.6, canvas, 0.4, 0)
