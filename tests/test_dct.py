import numpy as np
from src.compression.dct import block_dct, block_idct

def test_dct_idct():
    block = np.random.randint(0, 255, (8,8)).astype(np.float32)
    d = block_dct(block)
    r = block_idct(d)
    assert np.allclose(block, r, atol=1e-1)
