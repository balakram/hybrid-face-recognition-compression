from pathlib import Path
import cv2
import numpy as np

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_image_gray(path, target_size=(200,200)):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {p}")
    if target_size is not None:
        img = cv2.resize(img, tuple(target_size))
    return img

def flatten_image(img):
    arr = img.astype('float32')
    if arr.max() > 1.1:
        arr = arr / 255.0
    return arr.flatten()

def load_dataset(dataset_root, image_size=(200,200), allowed_exts=('.jpg','.jpeg','.png','.bmp')):
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")
    X_list = []
    y_list = []
    names = []
    for idx, person_dir in enumerate(sorted(root.iterdir())):
        if not person_dir.is_dir():
            continue
        gray_dir = person_dir / "gray"
        if gray_dir.exists() and gray_dir.is_dir():
            search_dir = gray_dir
        else:
            search_dir = person_dir
        files = [p for p in sorted(search_dir.iterdir()) if p.suffix.lower() in allowed_exts]
        if len(files) == 0:
            continue
        names.append(person_dir.name)
        for p in files:
            img = load_image_gray(p, target_size=image_size)
            X_list.append(flatten_image(img))
            y_list.append(idx)
    if len(X_list) == 0:
        raise RuntimeError("No images found in dataset: " + str(root))
    X = np.stack(X_list, axis=0).astype('float32')
    y = np.array(y_list, dtype='int32')
    return X, y, names
