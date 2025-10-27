# src/metrics.py
import numpy as np
import cv2
from PIL import Image
from math import log10

def psnr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32); y = y.astype(np.float32)
    mse = np.mean((x - y) ** 2)
    if mse == 0: return float('inf')
    return 10.0 * log10((255.0**2) / mse)

def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    # quick gray SSIM using OpenCV (approx). For precise SSIM use skimage if allowed.
    xg = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim==3 else x
    yg = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY) if y.ndim==3 else y
    return cv2.quality.QualitySSIM_compute(xg, yg)[0][0]

def ber(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    n = min(len(bits_true), len(bits_pred))
    if n == 0: return 1.0
    return float(np.sum(bits_true[:n] != bits_pred[:n]) / n)

def ssim_rgb(x: np.ndarray, y: np.ndarray) -> float:
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    return cv2.quality.QualitySSIM_compute(x, y)[0].mean()
