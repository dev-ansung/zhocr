# Stateful Caching, Refinement Search & Post-Processing

This document details how the pipeline avoids redundant OCR calls using temporal similarity caching, refines coarse timestamps using binary searches, and cleans up the final subtitle files.

---

## 1. Stateful Similarity Caching

Running OCR on every frame containing text is computationally expensive. Because subtitles persist on screen for several frames, the system implements a temporal cache, `CoverageCache`, in [src/cache.py](file:///Users/an/Development/Subtitle/src/cache.py).

### Asymmetric Coverage Metric
To evaluate if two binary subtitle masks, $A$ and $B$, represent the same sentence, the system calculates their asymmetric coverage:
\[\text{Coverage}(A, B) = \max\left(\frac{|A \cap B|}{|A|}, \frac{|A \cap B|}{|B|}\right)\]
This handles minor character changes or stroke fading gracefully:
* If the coverage is **$\ge 0.70$**, it is considered a cache hit. The system reuses the last OCR text and skips the EasyOCR query.
* A threshold of **`0.70`** is chosen because centered Chinese subtitles naturally share a high spatial overlap (up to `56%` for completely different sentences) due to identical font placement. A lower threshold (like `0.55`) results in different subtitles being falsely merged.

---

## 2. high-Precision Refinement (Bisection Search)

Coarse subtitles extracted at 1.0 fps or 2.0 fps have imprecise boundary times. High-precision refinement is done in `process_fine_subtitle` in [src/subtitle.py](file:///Users/an/Development/Subtitle/src/subtitle.py) at the native video frame rate.

### Step 1: Temporal Center Reference Frame Selection
A reference frame (`ref_frame`) is required to match boundaries against.
* **The Strategy**: Select the frame in the coarse subtitle window `[S, S + span_dur]` that has `present=True` and is **closest to the temporal center** `(S + E) / 2.0`.
* **Rationale**: Selecting the frame with the maximum pixel count is highly vulnerable to noise (e.g., selecting a frame containing bright background clothing boundaries or camera movement). The temporal center of a subtitle segment is the most stable and representative, avoiding border noise.

### Step 2: Binary Search (Bisection)
* **Start Refinement**: We fetch frames in `[S - 0.5s, S + 0.5s]` and perform a bisection search (`bisect_start`) to find the first frame that matches the `ref_frame` via `same_subtitle`.
* **End Refinement**: We fetch frames in `[E - 0.5s, E + 0.5s]` and perform a bisection search (`bisect_end`) to find the last frame that matches the `ref_frame`.

### Step 3: Subtitle Similarity Check (`same_subtitle`)
Two frames contain the same subtitle if:
1. Both frames are detected as containing text (`has_subtitle` is True).
2. The horizontal centroid difference between their binary masks is less than **$10\%$** of the frame width:
   \[\frac{|cx_1 - cx_2|}{W} < 0.10\]
   This prevents different, spatially distinct text boxes from matching.
3. The asymmetric coverage between their binary masks is **$\ge 0.70$**.

---

## 3. Timing Correction & Post-Processing

### Timing Inversion Correction
If a search anomaly causes the refined end time to fall before the start time (`refined_end < refined_start`), the system searches backward through the boundary frames to locate the exact frame where the subtitle fades, falling back to a minimum single-frame duration:
```python
refined_end = max(refined_end, refined_start + 1.0 / native_fps)
```

### Post-Refinement Merging and Deduplication
To clean up SRT outputs and avoid flickering, `post_process_entries` applies:
1. **Duration Filter**: Filters out transient noise entries with duration `< 0.15` seconds.
2. **Merge Consecutive Entries**: Merges sequential entries containing identical text (ignoring spaces/punctuation) if their gap is **$\le 2.0$** seconds.
