from src.compression.huffman import huffman_encode, huffman_decode

def test_huffman():
    arr = [1,2,3,1,2,1,1]
    enc, codes, tree = huffman_encode(arr)
    dec = huffman_decode(enc, tree)
    assert dec == arr
