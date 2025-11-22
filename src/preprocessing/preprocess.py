import cv2
import numpy as np

def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize_image(img, size=(200,200)):
    return cv2.resize(img, size)

def normalize(img):
    img = img.astype('float32')
    if img.max() > 1.1:
        img = img / 255.0
    return img

def equalize_hist(img):
    gray = to_gray(img)
    eq = cv2.equalizeHist((gray).astype('uint8'))
    return eq

def preprocess_face(img, size=(200,200), apply_eq=True):
    gray = to_gray(img)
    resized = resize_image(gray, size=size)
    if apply_eq:
        resized = equalize_hist(resized)
    norm = normalize(resized)
    return norm

def preprocess_batch(images, size=(200,200), apply_eq=True):
    return [preprocess_face(img, size=size, apply_eq=apply_eq) for img in images]
