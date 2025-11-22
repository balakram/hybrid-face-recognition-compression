#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.capture.multi_person_capture import ask_username_popup, capture_user


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    required=True,
                    choices=["capture", "train", "recognize", "compress_gray", "decompress","demo"])
    ap.add_argument("--input", help="input image for compression")
    ap.add_argument("--output", help="output folder")
    ap.add_argument("--quality", type=int, default=50)
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--model_dir", default="modes")
    ap.add_argument("--camera", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.mode == "capture":
        username = ask_username_popup()
        capture_user(username, count=20, camera_idx=0, out_base=args.data_root)

    elif args.mode == "train":
        from examples.run_recognition import train_pipeline
        train_pipeline(args.data_root, args.model_dir)

    elif args.mode == "recognize":
        from examples.recognize_live import main as live_rec
        live_rec(model_dir=args.model_dir, camera_idx=args.camera)

    elif args.mode == "compress_gray":
        from src.compression.full_compression import compress_image
        import pickle, os
        from pathlib import Path

        # Validate input image
        if not os.path.exists(args.input):
            print(f"[ERROR] Input image not found: {args.input}")
            return

        img_name = os.path.splitext(os.path.basename(args.input))[0]

        out_dir = Path("data/result/compressed")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Compressing image: {args.input}")
        encoded, meta = compress_image(args.input, quality=args.quality)

        pickle.dump(encoded, open(out_dir / f"{img_name}_encoded.bin", "wb"))
        pickle.dump(meta, open(out_dir / f"{img_name}_meta.pkl", "wb"))

        print(f"[SUCCESS] Compressed data saved to:")
        print(f"   {out_dir}/{img_name}_encoded.bin")
        print(f"   {out_dir}/{img_name}_meta.pkl")


    elif args.mode == "decompress":
        from src.compression.full_compression import decompress_image
        import pickle, os
        from pathlib import Path

        name = args.input  # example: myface

        enc_path = f"data/result/compressed/{name}_encoded.bin"
        meta_path = f"data/result/compressed/{name}_meta.pkl"

        if not os.path.exists(enc_path):
            print(f"[ERROR] Compressed file not found: {enc_path}")
            return
        if not os.path.exists(meta_path):
            print(f"[ERROR] Meta file not found: {meta_path}")
            return

        encoded = pickle.load(open(enc_path, "rb"))
        meta = pickle.load(open(meta_path, "rb"))

        out_dir = Path("data/result/reconstructed")
        out_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Decompressing...")

        rec = decompress_image(encoded, meta)

        save_path = out_dir / f"{name}_reconstructed.jpg"
        cv2.imwrite(str(save_path), rec)

        print(f"[SUCCESS] Decompressed image saved to:")
        print(f"   {save_path}")


    elif args.mode == "demo":
        from examples.demo_all_in_one import main as demo_main
        demo_main(data_root=args.data_root, model_dir=args.model_dir, camera=args.camera)

if __name__ == "__main__":
    main()
