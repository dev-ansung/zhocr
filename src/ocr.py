import numpy as np

class SubtitleOCR:
    def __init__(self, languages: list[str] | None = None):
        """
        Initializes EasyOCR reader. Loads weights into memory (usually takes ~1.5s).
        """
        import easyocr
        self._reader = easyocr.Reader(
            languages or ["ch_sim", "en"],
            gpu=True,  # Enable GPU if available, else falls back to CPU
            verbose=False,
        )

    def read(self, frame: np.ndarray) -> str:
        """
        Run OCR on an RGB frame. Returns joined text string, empty if none found.
        """
        results = self._reader.readtext(frame, detail=0)
        return " ".join(results).strip()
