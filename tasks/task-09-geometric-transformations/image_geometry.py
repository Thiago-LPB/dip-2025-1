import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape
    
    shift_y, shift_x = 20, 20  
    translated = np.zeros_like(img)
    translated[shift_y:, shift_x:] = img[:h-shift_y, :w-shift_x]
    
    rotated = np.transpose(img[::-1, :])
    
    new_w = int(w * 1.5)
    stretched = np.zeros((h, new_w), dtype=img.dtype)
    x_coords = (np.arange(new_w) / 1.5).astype(int)
    x_coords = np.clip(x_coords, 0, w-1)
    stretched[:, :] = img[:, x_coords]
    
    mirrored = img[:, ::-1]
    
    distorted = np.zeros_like(img)
    y, x = np.indices((h, w))
    x_norm = (2 * x / w) - 1
    y_norm = (2 * y / h) - 1
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    k = 0.3  
    r_distorted = r * (1 + k * (r**2))
    
    theta = np.arctan2(y_norm, x_norm)
    x_new = r_distorted * np.cos(theta)
    y_new = r_distorted * np.sin(theta)

    x_mapped = ((x_new + 1) * w / 2).astype(int)
    y_mapped = ((y_new + 1) * h / 2).astype(int)
    
    valid = (x_mapped >= 0) & (x_mapped < w) & (y_mapped >= 0) & (y_mapped < h)
    distorted[y[valid], x[valid]] = img[y_mapped[valid], x_mapped[valid]]
    
    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
