import numpy as np
import cv2
from typing import Tuple, List, Dict, Any

def compute_fft_spectrum(image_np: np.ndarray) -> np.ndarray:
    """
    Computes 2D Fast Fourier Transform (FFT) log-magnitude power spectrum.
    Returns RGB heatmap ready for rendering.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    gray = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)
    
    norm_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    # Apply high-contrast inferno/viridis colormap
    heatmap = cv2.applyColorMap(norm_spectrum.astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def calculate_spectral_analysis(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Performs full 2D Fourier spectral decomposition:
    - High-frequency power ratio
    - Radial profile (azimuthal average)
    - GAN checkerboard anomaly indicator
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    gray = cv2.resize(gray, (256, 256)).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r_max = min(h, w) // 2
    
    # 1. High-frequency mask calculation
    r_cutoff = r_max // 2
    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r_cutoff, 0, -1)
    
    high_freq_energy = np.sum(magnitude * mask)
    total_energy = np.sum(magnitude) + 1e-7
    raw_high_freq_ratio = float(high_freq_energy / total_energy)
    
    # Scaled anomaly score (0.0 to 1.0)
    anomaly_score = float(min(1.0, max(0.0, raw_high_freq_ratio * 2.8)))
    
    # 2. Compute Radial Profile (Azimuthal Average in 8 radial bins)
    y, x = np.indices((h, w))
    radii = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    bin_size = max(1, r_max // 8)
    radial_profile = []
    for b in range(8):
        r_start = b * bin_size
        r_end = (b + 1) * bin_size
        ring_mask = (radii >= r_start) & (radii < r_end)
        if np.any(ring_mask):
            mean_val = float(np.mean(magnitude[ring_mask]))
            radial_profile.append(round(mean_val, 2))
        else:
            radial_profile.append(0.0)
            
    # Normalize radial profile to 0-100 for UI charts
    max_profile = max(radial_profile) if max(radial_profile) > 0 else 1.0
    normalized_profile = [round((val / max_profile) * 100, 1) for val in radial_profile]

    return {
        'anomaly_score': round(anomaly_score, 4),
        'high_freq_ratio': round(raw_high_freq_ratio, 4),
        'radial_profile': normalized_profile,
        'has_checkerboard_artifact': anomaly_score > 0.65
    }

def calculate_spectral_energy(image_np: np.ndarray) -> float:
    """Legacy helper for single float anomaly score."""
    return calculate_spectral_analysis(image_np)['anomaly_score']

