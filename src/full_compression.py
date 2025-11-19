"""
compression.py - Full DCT + Quantization + Zigzag + RLE + Huffman pipeline
"""
import numpy as np
from scipy.fftpack import dct, idct
from .compression import *  # this will import earlier Huffman functions if present
# However to avoid circular import, reimplement locally the required Huffman functions if not available.
# For safety, we will implement the needed helpers inline here.

from collections import Counter

# Standard 8x8 luminance quantization table (JPEG-like)
STANDARD_Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.float32)

# Zig-zag order for 8x8
ZIGZAG_ORDER = [(i,j) for s in range(0,16) for i in range(8) for j in range(8) if i+j==s]

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

def quantize_block(block, qtable=STANDARD_Q):
    return np.round(block / qtable).astype(np.int32)

def dequantize_block(block_q, qtable=STANDARD_Q):
    return (block_q * qtable).astype(np.float32)

def zigzag_scan(block):
    assert block.shape == (8,8)
    arr = [block[i,j] for (i,j) in ZIGZAG_ORDER]
    return arr

def inverse_zigzag(arr):
    block = np.zeros((8,8), dtype=np.int32)
    for (i,j), val in zip(ZIGZAG_ORDER, arr):
        block[i,j] = int(val)
    return block

def rle_encode(coeffs):
    \"\"\"Simple run-length encode zeros: produce list of (run,length) pairs for zeros and values for non-zero.\"\"\"
    out = []
    run = 0
    for c in coeffs:
        if c == 0:
            run += 1
        else:
            out.append((run, int(c)))
            run = 0
    # end marker
    if run>0:
        out.append((run, 0))
    return out

def rle_decode(rle_list, total_len=64):
    out = []
    for run, val in rle_list:
        if val == 0:
            out.extend([0]*run)
        else:
            out.extend([0]*run)
            out.append(int(val))
    # pad with zeros if short
    if len(out) < total_len:
        out.extend([0]*(total_len-len(out)))
    return out[:total_len]

# We will re-use Huffman functions from top-level compression if available
try:
    from src.compression import huffman_encode as huff_encode_top, huffman_decode as huff_decode_top
    def huffman_encode_symbols(symbols):
        return huff_encode_top(symbols)
    def huffman_decode_symbols(encoded_bytes, codes, padding):
        return huff_decode_top(encoded_bytes, codes, padding)
except Exception as e:
    # fallback to simple Python implementations if not found
    from collections import Counter, defaultdict
    import heapq
    class HuffmanNode:
        def __init__(self, sym=None, freq=0, left=None, right=None):
            self.sym, self.freq, self.left, self.right = sym, freq, left, right
        def __lt__(self, other):
            return self.freq < other.freq
    def build_tree(freqs):
        heap = [HuffmanNode(sym, fr) for sym, fr in freqs.items()]
        heapq.heapify(heap)
        if len(heap)==0: return None
        while len(heap)>1:
            a = heapq.heappop(heap); b = heapq.heappop(heap)
            parent = HuffmanNode(None, a.freq+b.freq, a, b)
            heapq.heappush(heap, parent)
        return heap[0]
    def gen_codes(node, prefix="", codes=None):
        if codes is None: codes={}
        if node is None: return codes
        if node.sym is not None:
            codes[node.sym] = prefix if prefix!="" else "0"
        else:
            gen_codes(node.left, prefix+"0", codes)
            gen_codes(node.right, prefix+"1", codes)
        return codes
    def huffman_encode_symbols(symbols):
        freqs = Counter(symbols)
        root = build_tree(freqs)
        codes = gen_codes(root)
        bitstr = "".join(codes[s] for s in symbols)
        padding = (8 - len(bitstr) % 8) % 8
        bitstr_p = bitstr + "0"*padding
        b = bytearray()
        for i in range(0, len(bitstr_p), 8):
            b.append(int(bitstr_p[i:i+8], 2))
        return bytes(b), codes, padding
    def huffman_decode_symbols(encoded_bytes, codes, padding):
        if not codes:
            return []
        rev = {v:k for k,v in codes.items()}
        bitstr = "".join(f"{byte:08b}" for byte in encoded_bytes)
        if padding: bitstr = bitstr[:-padding]
        out = []
        cur = ""
        for bit in bitstr:
            cur += bit
            if cur in rev:
                out.append(rev[cur])
                cur = ""
        return out

# High-level compress_image / decompress_image
import pickle
def compress_image(img, qtable=STANDARD_Q):
    \"\"\"img: 2D numpy uint8 grayscale image\n       returns dict containing compressed payload and metadata\"\"\"
    blocks, shape = blockify(img, 8)
    all_symbols = []
    meta = {'orig_shape': shape}
    for b in blocks:
        b = b.astype(np.float32) - 128.0  # level shift
        B = dct2(b)
        Bq = quantize_block(B, qtable)
        zz = zigzag_scan(Bq)
        rle = rle_encode(zz)
        # encode rle tuples into symbols, we will represent as pairs (run,val)
        for pair in rle:
            all_symbols.append(pair)
    # Flatten symbol representation to bytes-friendly tuple -> convert to string keys if needed
    # For Huffman, use tuple as symbol (works with pickle-like approach in our huffman implementation)
    encoded_bytes, codes, padding = huffman_encode_symbols(all_symbols)
    meta.update({'codes': serialize_codebook(codes) if codes else {}, 'padding': padding})
    # store also the number of blocks to reconstruct segmentation
    meta['num_blocks'] = len(blocks)
    # We will store quant table too
    meta['qtable'] = qtable.tolist()
    payload = {'encoded': encoded_bytes, 'meta': meta}
    return payload

def decompress_image(payload):
    encoded_bytes = payload['encoded']
    meta = payload['meta']
    codes = deserialize_codebook(meta['codes']) if meta.get('codes') else {}
    padding = meta.get('padding', 0)
    num_blocks = meta['num_blocks']
    qtable = np.array(meta['qtable'], dtype=np.float32)
    all_pairs = huffman_decode_symbols(encoded_bytes, codes, padding)
    # all_pairs is list of tuples (run,val)
    # reconstruct each block sequentially
    blocks = []
    idx = 0
    for _ in range(num_blocks):
        # gather entries until block filled (64 values)
        rle_list = []
        total = 0
        while total < 64 and idx < len(all_pairs):
            pair = all_pairs[idx]
            rle_list.append(pair)
            # estimate length contribution
            run, val = pair
            if val == 0:
                total += run
            else:
                total += run + 1
            idx += 1
            # safety break
            if idx >= len(all_pairs) and total < 64:
                # pad with zeros
                rle_list.append((64-total, 0))
                total = 64
        vals = rle_decode(rle_list, 64)
        block_q = inverse_zigzag(vals)
        block_f = dequantize_block(block_q, qtable)
        block_rec = idct2(block_f) + 128.0
        block_rec = np.clip(block_rec, 0, 255).astype(np.uint8)
        blocks.append(block_rec)
    # reconstruct image from blocks
    # we know original padded shape
    h, w = meta['orig_shape']
    bh = bw = 8
    rows = h // bh
    cols = w // bw
    img = np.zeros((h, w), dtype=np.uint8)
    k = 0
    for i in range(rows):
        for j in range(cols):
            img[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = blocks[k]
            k += 1
    return img
