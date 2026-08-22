import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any

def analyze_chroma_and_boundary(image_np: np.ndarray, bbox: list = None) -> Dict[str, Any]:
    """
    Forensic Color Space and Boundary Discontinuity Analyzer.
    Exposes GAN/Diffusion color co-occurrence discrepancies and face-swap blending seams.
    """
    h, w = image_np.shape[:2]
    img_rgb = image_np.astype(np.float32)
    
    # 1. Cross-Channel Covariance Analysis (RGB)
    r = img_rgb[:, :, 0].flatten()
    g = img_rgb[:, :, 1].flatten()
    b = img_rgb[:, :, 2].flatten()
    
    cov_rg = np.corrcoef(r, g)[0, 1] if np.std(r) > 0 and np.std(g) > 0 else 1.0
    cov_gb = np.corrcoef(g, b)[0, 1] if np.std(g) > 0 and np.std(b) > 0 else 1.0
    cov_rb = np.corrcoef(r, b)[0, 1] if np.std(r) > 0 and np.std(b) > 0 else 1.0
    
    # In natural optical photos, channel correlations are typically > 0.92
    mean_channel_corr = float((cov_rg + cov_gb + cov_rb) / 3.0)
    chroma_anomaly = float(np.clip(1.0 - mean_channel_corr, 0.0, 1.0))
    
    # 2. YCbCr Chrominance Noise Distribution
    ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    cb = ycbcr[:, :, 1].astype(np.float32)
    cr = ycbcr[:, :, 2].astype(np.float32)
    
    cb_std = float(np.std(cb))
    cr_std = float(np.std(cr))
    chroma_variance_ratio = float(abs(cb_std - cr_std) / max(cb_std, cr_std, 1.0))
    
    # 3. High-Frequency Boundary Gradient Seam Detection (Laplacian)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())
    
    # Check boundary perimeter vs interior texture variance if bbox is given
    boundary_discontinuity = 0.15
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        interior = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if interior.size > 100:
            int_var = float(np.var(interior))
            ext_var = float(np.var(gray))
            boundary_discontinuity = float(np.clip(abs(int_var - ext_var) / max(int_var, ext_var, 1.0), 0.0, 1.0))
            
    # Composite Color & Boundary Forensic Index
    color_anomaly_index = float(np.clip(
        (chroma_anomaly * 0.40) + 
        (chroma_variance_ratio * 0.35) + 
        (boundary_discontinuity * 0.25),
        0.0, 1.0
    ))
    
    return {
        'channel_correlation': round(mean_channel_corr, 4),
        'chroma_anomaly_score': round(chroma_anomaly, 4),
        'chroma_variance_ratio': round(chroma_variance_ratio, 4),
        'boundary_discontinuity': round(boundary_discontinuity, 4),
        'color_anomaly_index': round(color_anomaly_index, 4)
    }
