import sys
from pathlib import Path
# ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
from src.compression import huffman_encode, huffman_decode
from collections import Counter

def test_huffman_basic():
    symbols = [1,2,3,3,3,2,1,4,4,4,4]
    encoded, codes, padding = huffman_encode(symbols)
    decoded = huffman_decode(encoded, codes, padding)
    assert decoded == symbols

def test_huffman_single_symbol():
    symbols = [7]*10
    encoded, codes, padding = huffman_encode(symbols)
    decoded = huffman_decode(encoded, codes, padding)
    assert decoded == symbols
    assert list(codes.keys()) == [7] or set(codes.keys())=={7}

def test_huffman_entropy_reduction():
    symbols = [0]*80 + [1]*15 + [2]*5
    encoded, codes, padding = huffman_encode(symbols)
    freqs = Counter(symbols)
    total = sum(freqs.values())
    avg_len = sum(len(codes[s]) * freqs[s] for s in freqs) / total
    assert avg_len <= 2.0 + 1e-6
