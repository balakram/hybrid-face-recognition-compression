import numpy as np
def compute_psnr(orig, recon):
    orig = np.asarray(orig, dtype=np.float32)
    recon = np.asarray(recon, dtype=np.float32)
    if orig.max() <= 1.1:
        orig = orig * 255.0
        recon = recon * 255.0
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / (mse ** 0.5))

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()
