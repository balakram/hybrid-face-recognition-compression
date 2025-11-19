"""
compression.py - DCT + Huffman implementation (with full Huffman coding)
"""
import numpy as np
from scipy.fftpack import dct, idct
import heapq
from collections import defaultdict, Counter
import pickle

def blockify(img, block_size=8):
    h, w = img.shape
    h_pad = (block_size - (h % block_size)) % block_size
    w_pad = (block_size - (w % block_size)) % block_size
    img_p = np.pad(img, ((0,h_pad),(0,w_pad)), mode='constant', constant_values=0)
    blocks = []
    for i in range(0, img_p.shape[0], block_size):
        for j in range(0, img_p.shape[1], block_size):
            blocks.append(img_p[i:i+block_size, j:j+block_size])
    return blocks, img_p.shape

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, qtable):
    return np.round(block / qtable).astype(np.int32)

def dequantize(block_q, qtable):
    return (block_q * qtable).astype(np.float32)


# ---------------- Huffman coding -----------------
class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        # for heap comparisons
        return self.freq < other.freq

def build_huffman_tree(freqs):
    """
    freqs: dict mapping symbol -> frequency
    returns root of Huffman tree
    """
    heap = []
    for sym, fr in freqs.items():
        heapq.heappush(heap, (fr, HuffmanNode(symbol=sym, freq=fr)))
    if len(heap) == 0:
        return None
    # Edge case: single symbol
    if len(heap) == 1:
        fr, node = heapq.heappop(heap)
        root = HuffmanNode(freq=fr, left=node)  # make a parent
        return root

    while len(heap) > 1:
        fr1, node1 = heapq.heappop(heap)
        fr2, node2 = heapq.heappop(heap)
        merged = HuffmanNode(symbol=None, freq=fr1+fr2, left=node1, right=node2)
        heapq.heappush(heap, (merged.freq, merged))
    return heapq.heappop(heap)[1]

def generate_codes(root):
    """
    root: HuffmanNode
    returns dict symbol->code (string of '0'/'1')
    """
    codes = {}
    def _gen(node, prefix):
        if node is None:
            return
        if node.symbol is not None:
            # leaf
            codes[node.symbol] = prefix if prefix != "" else "0"
            return
        _gen(node.left, prefix + "0")
        _gen(node.right, prefix + "1")
    _gen(root, "")
    return codes

def huffman_encode(symbols):
    """
    symbols: iterable of hashable symbols (e.g., ints)
    returns bytes of encoded bitstream, the codebook (dict), and padding length
    """
    freqs = Counter(symbols)
    root = build_huffman_tree(freqs)
    codes = generate_codes(root) if root is not None else {}
    # encode
    bitstr = "".join(codes[sym] for sym in symbols)
    # pad to byte
    padding = (8 - len(bitstr) % 8) % 8
    bitstr_padded = bitstr + ("0" * padding)
    b = bytearray()
    for i in range(0, len(bitstr_padded), 8):
        byte = bitstr_padded[i:i+8]
        b.append(int(byte, 2))
    return bytes(b), codes, padding

def huffman_decode(encoded_bytes, codes, padding):
    """
    encoded_bytes: bytes produced by huffman_encode
    codes: dict symbol->code (string)
    padding: number of padding bits added at end
    returns list of decoded symbols
    """
    if not codes:
        return []
    # build reverse map
    rev = {v: k for k, v in codes.items()}
    # build bitstring
    bitstr = "".join(f"{byte:08b}" for byte in encoded_bytes)
    if padding:
        bitstr = bitstr[:-padding]
    # decode by walking bits
    out = []
    cur = ""
    for bit in bitstr:
        cur += bit
        if cur in rev:
            out.append(rev[cur])
            cur = ""
    return out

# Utilities to serialize codebook
def serialize_codebook(codes):
    # codes: symbol->code; symbols may be ints (numpy ints) -> convert to int
    return {int(k): v for k, v in codes.items()}

def deserialize_codebook(cb):
    return {int(k): v for k, v in cb.items()}
