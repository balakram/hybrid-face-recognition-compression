#!/usr/bin/env python3
"""
Auto Capture With Popup Dialog (Tkinter + OpenCV)
"""
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def capture_user(username, count=20, camera_idx=0, out_base="data"):
    user_color_dir = Path(out_base) / username / "color"
    user_gray_dir = Path(out_base) / username / "gray"

    ensure_dir(user_color_dir)
    ensure_dir(user_gray_dir)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    saved = 0
    print(f"[INFO] Auto-capturing {count} images for {username}...")

    cv2.namedWindow("Auto Capture", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Auto Capture", cv2.WND_PROP_TOPMOST, 1)

    try:
        while saved < count:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(display, f"Captured: {saved}/{count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Auto Capture", display)

            if len(faces) > 0:
                x, y, w, h = faces[0]

                full_color = cv2.resize(frame, (200, 200))
                fname = f"{saved + 1:03d}.jpg"
                cv2.imwrite(str(user_color_dir / fname), full_color)

                face_crop = frame[y:y + h, x:x + w]
                if face_crop.size > 0:
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (200, 200))
                    cv2.imwrite(str(user_gray_dir / fname), face_gray)
                    saved += 1

                cv2.waitKey(150)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Finished capturing {saved} images for '{username}'.")


def ask_username_popup():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    username = simpledialog.askstring("User Name", "Enter Username:", parent=root)
    root.destroy()
    return username

if __name__ == "__main__":
    print("=== Face Auto-Capture System ===")
    username = ask_username_popup()
    if username is None or username.strip() == "":
        print("[ERROR] No username entered.")
    else:
        capture_user(username.strip(), count=20, camera_idx=0, out_base="data/raw")
