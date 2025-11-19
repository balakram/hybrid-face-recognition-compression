"""
utils.py - helper functions for dataset loading, image IO and metrics
"""
import numpy as np
import cv2
from pathlib import Path
from math import log10, sqrt

def load_image_gray(path, size=(100,100)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to load image: {path}")
    img = cv2.resize(img, size)
    return img

def flatten_image(img):
    return img.flatten().astype('float32')

def load_dataset(data_dir, size=(100,100), max_per_class=None):
    \"\"\"Load dataset structured as data_dir/class_x/*.jpg\"\"\"
    data_dir = Path(data_dir)
    X = []
    y = []
    labels = []
    for idx, class_dir in enumerate(sorted([d for d in data_dir.iterdir() if d.is_dir()])):
        files = list(class_dir.glob("*"))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            img = load_image_gray(f, size=size)
            X.append(flatten_image(img))
            y.append(idx)
        labels.append(class_dir.name)
    import numpy as _np
    return _np.array(X), _np.array(y), labels

def compute_psnr(orig, recon):
    \"\"\"Compute PSNR between two grayscale images (numpy arrays).\"\"\"
    import numpy as _np
    mse = _np.mean((_np.float32(orig) - _np.float32(recon)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))
