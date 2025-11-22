import numpy as np

def zigzag(block):
    """
    Convert 8x8 block → zigzag ordered 1D array of length 64.
    """
    h, w = block.shape
    solution = []
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                x = j
                y = i - j
                if x < h and y < w:
                    solution.append(block[x][y])
        else:
            for j in range(i + 1):
                x = i - j
                y = j
                if x < h and y < w:
                    solution.append(block[x][y])
    return np.array(solution)


def inverse_zigzag(arr, h=8, w=8):
    """
    Convert zigzag array → 8×8 block.
    """
    block = np.zeros((h, w))
    idx = 0
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                x = j
                y = i - j
                if x < h and y < w:
                    block[x][y] = arr[idx]
                    idx += 1
        else:
            for j in range(i + 1):
                x = i - j
                y = j
                if x < h and y < w:
                    block[x][y] = arr[idx]
                    idx += 1
    return block
