#!/usr/bin/env python3
"""
Demo All-In-One:
1) Capture user images (Tkinter popup)
2) Train PCA+LDA models
3) Launch real-time recognizer
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.capture.multi_person_capture import ask_username_popup, capture_user
from examples.run_recognition import train_pipeline
from examples.recognize_live import main as live_main


def main(data_root="data/raw", model_dir="modes", camera=0):
    # Step 1: capture
    username = ask_username_popup()
    if username is None or username.strip() == "":
        print("[ERROR] No username entered.")
        return

    print("[STEP 1] Capturing images for", username)
    capture_user(username.strip(), count=20, camera_idx=camera, out_base=data_root)

    # Step 2: train
    print("[STEP 2] Training models...")
    train_pipeline(data_root, model_dir)

    # Step 3: live recognition
    print("[STEP 3] Starting live recognition...")
    live_main(model_dir=model_dir, camera_idx=camera)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--model_dir", default="modes")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    main(data_root=args.data_root, model_dir=args.model_dir, camera=args.camera)
