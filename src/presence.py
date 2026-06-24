import numpy as np
import cv2

def make_binary(
    frame: np.ndarray,
    edge_density_min: int = 300,
    edge_density_max: int = 15000,
    min_components: int = 1,
    orig_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Process one RGB frame through the Morphological Top-Hat CCL pipeline.

    Returns:
        binary      H×W mask with kept text components (uint8, 0 or 255)
        mask_all    H×W Otsu binarization before CCL filter
        strong_px   pixel count of binary
        kept_count  number of CCL components that passed geometry filter
    """
    # Convert frame to Y (luma) channel of YCrCb
    luma = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    H, W = luma.shape
    if orig_height is None:
        orig_height = H * 10
    scale_y = orig_height / 720.0

    # White top-hat to isolate bright thin structures (text) from slowly-varying backgrounds
    th_size = int(round(19 * scale_y))
    if th_size % 2 == 0:
        th_size += 1
    th_size = max(3, th_size)
    kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (th_size, th_size))
    top_hat = cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, kernel_th)

    # Otsu thresholding on top-hat
    _, mask = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Enforce minimum brightness in top-hat to avoid binarizing background noise on empty frames
    _, min_bright_mask = cv2.threshold(top_hat, 50, 255, cv2.THRESH_BINARY)
    mask_all = cv2.bitwise_and(mask, min_bright_mask)

    # Morphological close to merge strokes/characters
    close_size = int(round(5 * scale_y))
    close_size = max(1, close_size)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    closed = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel_close)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    binary = np.zeros_like(luma)
    kept_count = 0

    min_h = max(2, int(round(8 * scale_y)))
    min_w = max(2, int(round(8 * scale_y)))

    for i, stat in enumerate(stats[1:], 1):
        area = stat[cv2.CC_STAT_AREA]
        x = stat[cv2.CC_STAT_LEFT]
        y = stat[cv2.CC_STAT_TOP]
        h = stat[cv2.CC_STAT_HEIGHT]
        w = stat[cv2.CC_STAT_WIDTH]
        if h == 0 or w == 0:
            continue
        # Reject border-touching components (except close to top/bottom since we close strokes)
        if x <= 1 or (x + w) >= W - 1:
            continue
        aspect = w / h
        extent = area / (w * h)

        # Geometry constraints tailored for text characters and character groups
        if h >= min_h and w >= min_w:
            if 0.15 < aspect < 20.0 and 0.2 < extent < 0.9:
                binary[(labels == i) & (mask_all > 0)] = 255
                kept_count += 1

    strong_px = int(np.sum(binary > 0))
    return binary, mask_all, strong_px, kept_count


def has_subtitle(
    frame: np.ndarray,
    edge_density_min: int = 300,
    edge_density_max: int = 15000,
    min_components: int = 1,
    orig_height: int | None = None,
) -> tuple[bool, np.ndarray, int, int]:
    """
    Convenience wrapper. Returns (has_text, binary, strong_px, kept_count).
    """
    binary, _, strong_px, kept_count = make_binary(
        frame, edge_density_min, edge_density_max, min_components, orig_height
    )
    H, W = frame.shape[:2]
    # Compute proportional density thresholds
    ref_area = 1280.0 * 72.0
    scale_factor = (W * H) / ref_area
    
    scaled_min = int(round(edge_density_min * scale_factor))
    scaled_max = int(round(edge_density_max * scale_factor))

    present = (scaled_min <= strong_px <= scaled_max
               and kept_count >= min_components)
    return present, binary, strong_px, kept_count
