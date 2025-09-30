
import cv2
import numpy as np
import torch

def read_image(path, channels=1):
    if path.lower().endswith('.dcm'):
        try:
            import pydicom
        except Exception as e:
            raise RuntimeError("Install pydicom to read DICOM files, or convert to PNG/JPG.") from e
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = normalize_minmax(img)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if channels==1 else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = img.astype(np.float32) / 255.0
        if channels == 1 and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if channels == 3 and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def normalize_minmax(img):
    m, M = np.min(img), np.max(img)
    if M - m < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return (img - m) / (M - m)

def right_orient(img, side):
    """Flip left breasts to appear as right-oriented (paper uses horizontal flip)."""
    if side is not None and side.strip().upper().startswith('L'):
        img = np.ascontiguousarray(np.fliplr(img))
    return img

def resize_keep_ratio(img, target_hw):
    th, tw = target_hw
    # direct resize as in most pipelines
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

def to_tensor(img, channels=1):
    if channels == 1:
        t = torch.from_numpy(img).float().unsqueeze(0)
    else:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        t = torch.from_numpy(img).permute(2,0,1).float()
    return t

def normalize_tensor(t):
    # simple zero-mean unit-std per-image
    mean = t.mean(dim=(1,2), keepdim=True)
    std = t.std(dim=(1,2), keepdim=True) + 1e-6
    return (t - mean) / std
