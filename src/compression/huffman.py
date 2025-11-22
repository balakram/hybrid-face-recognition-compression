from heapq import heappush, heappop
from collections import defaultdict

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency_table(arr):
    freq = defaultdict(int)
    for val in arr:
        freq[val] += 1
    return freq


def build_huffman_tree(freq_table):
    heap = []

    for sym, freq in freq_table.items():
        heappush(heap, Node(sym, freq))

    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = Node(None, n1.freq+n2.freq, n1, n2)
        heappush(heap, merged)

    return heap[0]


def build_code_table(root):
    codes = {}

    def traverse(node, path=""):
        if node.symbol is not None:
            codes[node.symbol] = path
            return
        traverse(node.left, path + "0")
        traverse(node.right, path + "1")

    traverse(root)
    return codes


def huffman_encode(arr):
    freq = build_frequency_table(arr)
    root = build_huffman_tree(freq)
    codes = build_code_table(root)

    encoded = ''.join([codes[v] for v in arr])
    return encoded, codes, root


def huffman_decode(encoded, root):
    result = []
    node = root
    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            node = root

    return result
