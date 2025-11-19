"""
main.py - Demo entrypoint for training recognition and compressing images
"""
import argparse, os, pickle
from pathlib import Path
import numpy as np
from src.pca_eigenfaces import Eigenfaces
from src.lda_fisherfaces import Fisherfaces
from src.utils import load_dataset, load_image_gray, flatten_image, compute_psnr
from src.full_compression import compress_image, decompress_image

def train_recognition(data_dir, model_dir, img_size=(100,100), n_pca=100):
    X, y, labels = load_dataset(data_dir, size=img_size)
    print(f"Loaded {len(X)} samples, {len(labels)} classes")
    ef = Eigenfaces(n_components=n_pca)
    ef.fit(X)
    X_pca = ef.transform(X)
    lda = Fisherfaces()
    lda.fit(X_pca, y)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    ef.save(os.path.join(model_dir, "eigenfaces.pkl"))
    lda.save(os.path.join(model_dir, "fisherfaces.pkl"))
    with open(os.path.join(model_dir, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)
    print("Saved models to", model_dir)

def recognize(image_path, model_dir, img_size=(100,100)):
    import pickle
    ef = Eigenfaces.load(os.path.join(model_dir, "eigenfaces.pkl"))
    lda = Fisherfaces.load(os.path.join(model_dir, "fisherfaces.pkl"))
    with open(os.path.join(model_dir, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    img = load_image_gray(image_path, size=img_size)
    x = flatten_image(img).reshape(1, -1)
    x_pca = ef.transform(x)
    x_lda = lda.transform(x_pca)
    pred = lda.predict(x_pca)  # using LDA predict on PCA features is fine in our wrapper
    print("Predicted label index:", pred[0], "label name:", labels[pred[0]])

def demo_compress(img_path, out_path):
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    payload = compress_image(img)
    # save using pickle
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print("Compressed saved to", out_path)
    # decompress to check PSNR
    payload2 = payload
    recon = decompress_image(payload2)
    psnr = compute_psnr(img, recon)
    print("PSNR:", psnr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "recognize", "compress", "decompress"], default="train")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--infile", type=str, default="")
    parser.add_argument("--outfile", type=str, default="out.bin")
    args = parser.parse_args()

    if args.mode == "train":
        train_recognition(args.data_dir, args.model_dir)
    elif args.mode == "recognize":
        if not args.image: raise SystemExit("Provide --image for recognition")
        recognize(args.image, args.model_dir)
    elif args.mode == "compress":
        if not args.image: raise SystemExit("Provide --image for compression")
        demo_compress(args.image, args.outfile)
    elif args.mode == "decompress":
        import pickle, cv2
        with open(args.infile, "rb") as f:
            payload = pickle.load(f)
        recon = decompress_image(payload)
        cv2.imwrite(args.outfile, recon)
        print("Decompressed image saved to", args.outfile)
