import numpy as np
from scipy.fftpack import dct, idct

def block_dct(block):
    """
    Apply 2D DCT to an 8x8 block.
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def block_idct(block):
    """
    Apply 2D IDCT to an 8x8 block.
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def image_to_blocks(img, block_size=8):
    """
    Convert image to (H/8 × W/8) blocks of size 8×8.
    """
    h, w = img.shape
    blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            blocks.append(block)

    return np.array(blocks)


def blocks_to_image(blocks, h, w, block_size=8):
    """
    Reconstruct image from blocks.
    """
    out = np.zeros((h, w))
    idx = 0

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            out[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1

    return np.clip(out, 0, 255).astype(np.uint8)
