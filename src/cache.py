import numpy as np
from src.presence import make_binary

def coverage(a: np.ndarray, b: np.ndarray) -> float:
    """
    Asymmetric coverage metric: max(|A∩B|/|A|, |A∩B|/|B|).
    """
    inter = int(np.sum((a > 0) & (b > 0)))
    pa    = int(np.sum(a > 0))
    pb    = int(np.sum(b > 0))
    if pa == 0 and pb == 0:
        return 1.0
    if pa == 0 or pb == 0:
        return 0.0
    return max(inter / pa, inter / pb)

def same_subtitle(
    frame1: np.ndarray,
    frame2: np.ndarray,
    edge_density_min: int = 300,
    edge_density_max: int = 15000,
    min_components: int = 1,
    same_subtitle_threshold: float = 0.70,
    orig_height: int | None = None,
) -> bool:
    """
    Returns True if frame1 and frame2 contain the same subtitle.
    """
    H, W = frame1.shape[:2]
    ref_area = 1280.0 * 72.0
    scale_factor = (W * H) / ref_area

    scaled_min = int(round(edge_density_min * scale_factor))
    scaled_max = int(round(edge_density_max * scale_factor))

    b1, _, px1, k1 = make_binary(frame1, scaled_min, scaled_max, min_components, orig_height)
    b2, _, px2, k2 = make_binary(frame2, scaled_min, scaled_max, min_components, orig_height)

    has1 = (px1 >= scaled_min and k1 >= min_components)
    has2 = (px2 >= scaled_min and k2 >= min_components)

    if not has1 or not has2:
        return False

    # Check centroid difference (within 10% of frame width)
    cols = np.arange(b1.shape[1], dtype=np.float32)
    cx1 = float(np.sum((b1 > 0) * cols)) / px1
    cx2 = float(np.sum((b2 > 0) * cols)) / px2
    centroid_diff = abs(cx1 - cx2) / b1.shape[1]

    if centroid_diff > 0.10:
        return False

    # Check asymmetric coverage
    cov = coverage(b1, b2)
    return cov >= same_subtitle_threshold

class CoverageCache:
    """
    Stateful cache that tracks the previous binary mask and emits cache
    hits when coverage exceeds the threshold.
    """
    def __init__(self, threshold: float = 0.55, patience: int = 1):
        self.threshold    = threshold
        self.patience     = patience
        self._prev_binary = None
        self._absent      = 0

    def check(self, binary: np.ndarray, has_text: bool) -> tuple[bool, float | None]:
        if not has_text:
            self._absent += 1
            if self._absent > self.patience:
                self._prev_binary = None
            return False, None

        self._absent = 0

        if self._prev_binary is None:
            self._prev_binary = binary
            return False, None

        cov = coverage(binary, self._prev_binary)
        self._prev_binary = binary
        if cov >= self.threshold:
            return True, cov
        return False, cov

    def reset(self):
        self._prev_binary = None
        self._absent      = 0
