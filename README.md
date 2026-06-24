# Subtitle Extractor & Timestamp Refiner

An automated, high-precision video subtitle extraction and refinement pipeline. The system uses a **Morphological Top-Hat Connected Component Labeling (CCL)** presence detector, **EasyOCR**, and a **stateful temporal cache** to extract subtitles from a low-frame-rate cropped video. It then refines the start and end timestamps at the native video frame rate using a **noise-robust bisection search**.

---

## Key Features
* **Resolution Independence**: Structuring elements and geometry filters scale proportionally to the video resolution, preventing detection failures across variable dimensions.
* **Stateful Cache Optimization**: High-performance temporal caching using asymmetric mask coverage prevents redundant OCR calls on identical subtitle frames.
* **Noise-Robust Refinement**: Timestamp refinement selects the reference frame closest to the temporal center of the subtitle, preventing boundary noise (like clothing seams or camera movement) from corrupting refinements.
* **Automatic Post-Processing**: Filters transient noise (duration `< 0.15s`) and automatically merges consecutive duplicates within a `2.0s` gap.

---

## Codebase Architecture

```
.
├── main.py                     # CLI Entry point orchestrating the pipeline
├── pyproject.toml              # Project dependencies and tool configurations (using uv)
├── src/
│   ├── strategies.py           # CropStrategy and FrameStrategy definitions
│   ├── video.py                # VideoProcessor (seeking, cropping, downsampling)
│   ├── presence.py             # Top-Hat, Otsu binarization, CCL, and presence check
│   ├── cache.py                # Asymmetric coverage, subtitle similarity, and CoverageCache
│   ├── ocr.py                  # SubtitleOCR wrapper around EasyOCR
│   └── subtitle.py             # SubtitleProcessor (coarse extraction and refinement)
└── test/
    └── test_processors.py      # Automated unit tests covering all components
```

For detailed explanations of core subsystems and performance metrics, please read:
* [Architecture & Flow Guide](file:///Users/an/Development/Subtitle/docs/architecture.md)
* [Presence Detection & Morphology Guide](file:///Users/an/Development/Subtitle/docs/presence_detection.md)
* [Caching & Refinement Guide](file:///Users/an/Development/Subtitle/docs/caching_and_refinement.md)
* [Benchmark & Comparisons Guide](file:///Users/an/Development/Subtitle/docs/benchmark.md)

---

## Developer Setup

### Prerequisites
* Python 3.10+
* [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Installation
Clone the repository and sync the virtual environment:
```bash
# Clone the repository
cd Subtitle

# Sync dependencies and create venv using uv
uv sync
```

---

## CLI Usage

Run the main pipeline using `main.py`. The outputs (coarse SRT, refined SRT, cropped video, and debug frames) are automatically stored in unique timestamped directories under `./output/`.

```bash
# Run the pipeline for the entire video
uv run main.py

# Process a specific segment
uv run main.py --start 60.0 --duration 20.0
```

### CLI Parameters
* `--start`: Start time offset in seconds (default: `0.0`).
* `--duration`: Number of seconds to process (default: entire video).

---

## Running Unit Tests

Automated unit tests cover cropping, resolution-independent presence detection, cache thresholds, subtitle identity comparisons, OCR, and noise-robust refinement.

```bash
# Run all unit tests
uv run pytest

# Run a specific test case
uv run pytest -k test_subtitle_refinement_noise
```
