# Hybrid Face Recognition & Image Compression System

**Author:** Balakram Tudu

![Banner](assets/banner.png)

## Overview
This repository implements a hybrid system that performs face recognition using PCA (Eigenfaces) + LDA (Fisherfaces) and image compression using DCT + Huffman coding. The code is modular, well-documented, and meant for educational/demo purposes.

## Features
- PCA-based eigenface extraction
- LDA-based discriminant features (Fisherfaces)
- DCT (8x8) based image compression with quantization and Huffman coding
- Evaluation metrics: PSNR, MSE, Compression Ratio
- Jupyter notebooks for experiments and demos

## Repo Structure
```
hybrid-face-recognition-compression/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── architecture.png
│   ├── face_recognition_pipeline.png
│   ├── compression_pipeline.png
│   ├── combined_flow.png
│   ├── module_interaction.png
│   └── project_report.pdf
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   │
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── pca_eigenfaces.py
│   │   ├── lda_fisherfaces.py
│   │   └── classifier.py
│   │
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── dct.py
│   │   ├── quantization.py
│   │   ├── zigzag.py
│   │   └── huffman.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── helpers.py
│       └── fileio.py
│
├── notebooks/
│   ├── PCA_demo.ipynb
│   ├── LDA_demo.ipynb
│   └── DCT_Huffman_demo.ipynb
│
├── examples/
│   ├── run_recognition.py
│   ├── run_compression.py
│   └── demo_all_in_one.py
│
└── tests/
    ├── test_pca.py
    ├── test_lda.py
    ├── test_dct.py
    ├── test_huffman.py
    └── test_pipeline.py

```
## 
```

```
## Quickstart
1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the 🎬 Demo :
```bash
[![Run Demo](https://img.shields.io/badge/Run%20Demo-Open%20Notebook-brightgreen?style=for-the-badge)](https://github.com/balakram/hybrid-face-recognition-compression/blob/main/notebooks/demo.ipynb)
```

3. 🚀 Run in Google Colab:
```bash
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balakram/hybrid-face-recognition-compression/blob/main/notebooks/demo.ipynb)

```
## 🆕 Auto Dataset Creation (With Webcam)

To automatically capture multiple persons' images:

```bash

python src/main.py --mode capture --data_dir data

## Contributing
Contributions are welcome — please open issues or pull requests.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
