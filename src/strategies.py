import numpy as np

class CropStrategy:
    def __init__(self, top: float, bottom: float, left: float, right: float):
        """
        top, bottom, left, right are floats between 0.0 and 1.0 representing
        proportions of the frame height and width.
        """
        self.top_prop = top
        self.bottom_prop = bottom
        self.left_prop = left
        self.right_prop = right

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Crops the input frame using relative proportions.
        """
        h, w = frame.shape[:2]
        t = int(round(self.top_prop * h))
        b = int(round(self.bottom_prop * h))
        l = int(round(self.left_prop * w))
        r = int(round(self.right_prop * w))
        
        # Clip to safe boundaries
        t = max(0, min(t, h))
        b = max(0, min(b, h))
        l = max(0, min(l, w))
        r = max(0, min(r, w))
        
        return frame[t:b, l:r]

class FrameStrategy:
    def __init__(self, fps: float):
        self.fps = fps
