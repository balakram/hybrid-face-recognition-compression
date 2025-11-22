# FaceRec_Compression — Finalized Project (Updated)

This archive is a refined version of your Face Recognition + Image Compression project.
It updates capture, compression, and main CLI with safe persistence and evaluation.

## Quick start

1. Install requirements (recommended):
```bash
pip install -r requirements.txt
# ensure scikit-image and scipy are installed
```

2. Capture images for a user (example):
```bash
python capture/multi_person_capture.py --username Alice --count 20
```

This will save both color and grayscale images under `data/Alice/color/` and `data/Alice/gray/`.

3. Train PCA (Eigenfaces):
```bash
python main.py --mode train --data_dir data --model_dir models --pca_components 50
```

4. Compress a grayscale image (from data):
```bash
python main.py --mode compress --input data/Alice/gray/001.jpg --output results/compressed/alice_001.npz
```

5. Decompress and evaluate:
```bash
python main.py --mode decompress --input results/compressed/alice_001.npz --output results/decompressed/alice_001.png
```

To compute PSNR and SSIM, call decompress_image with compute_metrics=True in your own script or use the provided function in `src/full_compression.py`.
