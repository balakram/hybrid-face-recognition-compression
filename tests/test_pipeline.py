import numpy as np
from src.compression.full_compression import compress_image, decompress_image
import cv2

def test_full_pipeline():
    img = np.ones((200,200), dtype=np.uint8) * 127
    cv2.imwrite("tmp.jpg", img)

    encoded, meta = compress_image("tmp.jpg", quality=50)
    rec = decompress_image(encoded, meta)

    assert rec.shape == img.shape
