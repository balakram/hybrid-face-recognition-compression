# import cv2
# import numpy as np
# from pathlib import Path

# from .dct import block_dct, block_idct, image_to_blocks, blocks_to_image
# from .quantization import quantize, dequantize
# from .zigzag import zigzag, inverse_zigzag
# from .huffman import huffman_encode, huffman_decode


# def compress_image(input_path, quality=50):
#     """
#     Complete JPEG-style compression:
#     - grayscale
#     - block split
#     - DCT
#     - quantization
#     - zigzag
#     - huffman encoding
#     Returns:
#         encoded_data, metadata (height, width, Q_table, huffman_tree)
#     """
#     img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (200, 200))
#     h, w = img.shape

#     blocks = image_to_blocks(img)
#     dct_blocks = [block_dct(b.astype(np.float32) - 128) for b in blocks]

#     q_blocks = []
#     zigzag_blocks = []
#     Q_used = None

#     for b in dct_blocks:
#         qb, Q_used = quantize(b, quality=quality)
#         q_blocks.append(qb)
#         zigzag_blocks.append(zigzag(qb))

#     zigzag_flat = np.concatenate(zigzag_blocks)

#     encoded, codes, tree = huffman_encode(zigzag_flat)

#     meta = {
#         "h": h,
#         "w": w,
#         "quality": quality,
#         "Q": Q_used,
#         "tree": tree,
#         "block_count": len(blocks),
#     }

#     return encoded, meta


# def decompress_image(encoded, meta):
#     """
#     Reverse steps:
#     - huffman decode
#     - inverse zigzag
#     - dequantization
#     - IDCT
#     - merge blocks
#     """
#     arr = np.array(huffman_decode(encoded, meta["tree"]), dtype=np.int32)

#     blocks = []
#     idx = 0

#     for _ in range(meta["block_count"]):
#         zz = arr[idx:idx+64]
#         idx += 64
#         block = inverse_zigzag(zz)
#         dq = dequantize(block, meta["Q"])
#         idct_block = block_idct(dq) + 128
#         blocks.append(idct_block)

#     recon = blocks_to_image(blocks, meta["h"], meta["w"])
#     return recon


import os
from pathlib import Path
import numpy as np
import cv2
from .dct import block_dct, block_idct, image_to_blocks, blocks_to_image
from .quantization import JPEG_Q
from .zigzag import zigzag, inverse_zigzag
from .huffman import huffman_encode, huffman_decode


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# SINGLE IMAGE COMPRESSION
# -------------------------------------------------------------------

def compress_image(input_path, output_folder, quality=75):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] Input image not found: {input_path}")

    # Load grayscale
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError("[ERROR] Could not read grayscale image.")

    img_gray = cv2.resize(img_gray, (200, 200))

    # Convert grayscale to Y channel (YCbCr dummy)
    Y = img_gray.astype(np.float32)

    # Adjust quantization
    Q = np.floor((100 - quality + 1) / 100 * JPEG_Q).astype(np.int32)
    Q[Q == 0] = 1

    blocks = image_to_blocks(Y)
    dct_blocks = [block_dct(b) for b in blocks]
    q_blocks = [np.round(dct / Q) for dct in dct_blocks]
    zigzag_blocks = [zigzag(b) for b in q_blocks]

    encoded, huff_table = huffman_encode(zigzag_blocks)

    # Save compressed output
    base = os.path.splitext(os.path.basename(input_path))[0]
    ensure_dir(output_folder)

    np.save(os.path.join(output_folder, f"{base}_compressed.npy"), encoded)
    np.save(os.path.join(output_folder, f"{base}_hufftable.npy"), huff_table)
    np.save(os.path.join(output_folder, f"{base}_shape.npy"), Y.shape)

    return True


# -------------------------------------------------------------------
# SINGLE IMAGE DECOMPRESSION
# -------------------------------------------------------------------

def decompress_image(input_base, compressed_folder, output_folder):
    encoded = np.load(os.path.join(compressed_folder, f"{input_base}_compressed.npy"), allow_pickle=True)
    huff_table = np.load(os.path.join(compressed_folder, f"{input_base}_hufftable.npy"), allow_pickle=True)
    shape = tuple(np.load(os.path.join(compressed_folder, f"{input_base}_shape.npy")))

    decoded_blocks = huffman_decode(encoded, huff_table)
    inv_zigzag = [inverse_zigzag(b) for b in decoded_blocks]

    # Default Q (quality 75)
    Q = JPEG_Q

    deq = [b * Q for b in inv_zigzag]
    idct_blocks = [block_idct(b) for b in deq]

    Y = blocks_to_image(idct_blocks, shape)

    # Construct fake color from Y only
    final_img = np.stack([Y, Y, Y], axis=2).astype(np.uint8)

    ensure_dir(output_folder)
    cv2.imwrite(os.path.join(output_folder, f"{input_base}_reconstructed.jpg"), final_img)

    return True


# -------------------------------------------------------------------
# BATCH COMPRESSION (ALL 20 IMAGES)
# -------------------------------------------------------------------

def compress_folder(gray_folder, output_folder, quality=75):
    ensure_dir(output_folder)

    imgs = sorted([f for f in os.listdir(gray_folder) if f.lower().endswith(".jpg")])

    for img in imgs:
        ipath = os.path.join(gray_folder, img)
        compress_image(ipath, output_folder, quality)

    print(f"[INFO] Compressed {len(imgs)} images.")
    return True


# -------------------------------------------------------------------
# BATCH DECOMPRESSION (ALL 20)
# -------------------------------------------------------------------

def decompress_folder(compressed_folder, output_folder):
    ensure_dir(output_folder)

    comp_files = sorted([f for f in os.listdir(compressed_folder) if f.endswith("_compressed.npy")])

    for file in comp_files:
        base = file.replace("_compressed.npy", "")
        decompress_image(base, compressed_folder, output_folder)

    print(f"[INFO] Decompressed {len(comp_files)} images.")
    return True
