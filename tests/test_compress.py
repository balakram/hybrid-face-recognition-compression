import pickle
from src.full_compression import compress_image, decompress_image
import numpy as np

def test_compress_decompress_identity():
    # create a small synthetic image
    img = np.zeros((16,16), dtype=np.uint8)
    img[0:8,0:8] = 120
    payload = compress_image(img)
    recon = decompress_image(payload)
    # recon might be same shape as padded image
    assert recon.shape[0] >= img.shape[0]
    assert recon.shape[1] >= img.shape[1]
    # PSNR should be finite
    from src.utils import compute_psnr
    psnr = compute_psnr(img, recon[:img.shape[0], :img.shape[1]])
    assert psnr > 10  # very loose threshold
