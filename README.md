# zhocr — Rapid Chinese Subtitle OCR

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
│   ├── subtitle.py             # SubtitleProcessor (coarse extraction and refinement)
│   └── shift_timestamps.py     # CLI utility to offset SRT timestamps from a given entry
└── test/
    └── test_processors.py      # Automated unit tests covering all components
```

For detailed explanations of core subsystems and performance metrics, please read:
* [Architecture & Flow Guide](docs/architecture.md)
* [Presence Detection & Morphology Guide](docs/presence_detection.md)
* [Caching & Refinement Guide](docs/caching_and_refinement.md)
* [Benchmark & Comparisons Guide](docs/benchmark.md)

---

## Installation & Usage

### Run directly (no clone needed)

Requires [uv](https://github.com/astral-sh/uv):

```bash
uvx --from git+https://github.com/dev-ansung/zhocr zhocr video.mp4
uvx --from git+https://github.com/dev-ansung/zhocr zhocr video.mp4 --start 60.0 --duration 20.0
```

### Developer Setup

Clone the repository and sync the virtual environment:
```bash
git clone https://github.com/dev-ansung/zhocr.git
cd zhocr
uv sync
```

Then run via:
```bash
uv run zhocr video.mp4
uv run zhocr video.mp4 --start 60.0 --duration 20.0
```

## CLI Usage

The outputs (coarse SRT, refined SRT, cropped video, and debug frames) are automatically stored in unique timestamped directories under `./output/`.

```bash
# Run the pipeline for the entire video
zhocr video.mp4

# Process a specific segment
zhocr video.mp4 --start 60.0 --duration 20.0
```

### CLI Parameters
* `--start`: Start time offset in seconds (default: `0.0`).
* `--duration`: Number of seconds to process (default: entire video).

---

## Utilities

### Shift SRT Timestamps

`src/shift_timestamps.py` offsets all timestamps in an SRT file from a given entry index. Useful for correcting sync when the first few entries (e.g. title cards) should stay fixed.

```bash
# Shift all entries from index 3 onwards by +5 seconds (default), in place
python3 src/shift_timestamps.py output/my_file.srt

# Write to a new file with a custom offset and start index
python3 src/shift_timestamps.py input.srt output.srt --offset 3000 --from-index 5
```

**Options:**
* `--offset`: Offset in milliseconds (default: `5000`).
* `--from-index`: First entry index to shift (default: `3`).

---

## Running Unit Tests

Automated unit tests cover cropping, resolution-independent presence detection, cache thresholds, subtitle identity comparisons, OCR, and noise-robust refinement.

```bash
# Run all unit tests
uv run pytest

# Run a specific test case
uv run pytest -k test_subtitle_refinement_noise
```
