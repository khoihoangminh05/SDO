import cv2
import numpy as np


def evaluate_complexity(image_crop: np.ndarray) -> float:
    """
    Evaluates the visual complexity of an image region (crop) using:
    1. Canny edge density (high-frequency details).
    2. Normalized variance.
    3. Normalized entropy.

    Returns a complexity score between 0.0 and 1.0.
    """
    if image_crop is None or image_crop.size == 0:
        return 0.0

    # Ensure shape has at least height and width
    if (len(image_crop.shape) < 2 or
            image_crop.shape[0] == 0 or
            image_crop.shape[1] == 0):
        return 0.0

    # 1. Convert to grayscale if it is a BGR/RGB image
    if len(image_crop.shape) == 3:
        if image_crop.shape[2] == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        elif image_crop.shape[2] == 4:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image_crop[:, :, 0]
    else:
        gray = image_crop

    # Ensure gray is uint8 for OpenCV operations
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating):
            min_val = gray.min()
            max_val = gray.max()
            if min_val >= -0.1 and max_val <= 1.1:
                gray = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
            elif max_val > min_val:
                gray = ((gray - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)
        else:
            gray = gray.astype(np.uint8)

    # 2. Canny Edge Detection (high-frequency details)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.mean(edges > 0)

    # 3. Variance (normalized)
    # Max variance for uint8 [0, 255] is (255^2)/4 = 16256.25
    var = np.var(gray)
    var_norm = min(1.0, float(var) / 16256.25)

    # 4. Entropy (normalized)
    # Max entropy for 256 states is log2(256) = 8.0
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total_pixels = hist.sum()
    if total_pixels > 0:
        probs = hist / total_pixels
        non_zero = probs > 0
        entropy = -np.sum(probs[non_zero] * np.log2(probs[non_zero]))
        entropy_norm = min(1.0, float(entropy) / 8.0)
    else:
        entropy_norm = 0.0

    # Linear combination
    # High weight on edge ratio (0.9) to distinguish objects/edges from
    # flat/empty regions. Low weight on variance (0.05) and entropy (0.05)
    # to capture smooth gradients without falsely triggering the threshold.
    score = 0.9 * edge_ratio + 0.05 * var_norm + 0.05 * entropy_norm

    return float(score)
