import numpy as np

JPEG_Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

def quantize(block, quality=50):
    """
    Quantize an 8x8 DCT block using scaled JPEG table.
    """
    if quality < 1: quality = 1
    if quality > 100: quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    Q_scaled = np.floor((JPEG_Q * scale + 50) / 100)
    Q_scaled[Q_scaled == 0] = 1

    return np.round(block / Q_scaled).astype(np.int32), Q_scaled


def dequantize(block, Q_scaled):
    """
    Reverse quantization.
    """
    return (block * Q_scaled).astype(np.float32)
