import cv2
import numpy as np
from pathlib import Path
from src.presence import make_binary, has_subtitle
from src.cache import CoverageCache, same_subtitle
from src.ocr import SubtitleOCR

# --- Helper Functions ---

def format_srt_time(secs: float) -> str:
    h  = int(secs // 3600)
    m  = int((secs % 3600) // 60)
    s  = int(secs % 60)
    ms = int(round(secs % 1, 3) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_srt_time(s: str) -> float:
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(sec) + int(ms)/1000

def parse_srt(path: Path) -> list[dict]:
    entries = []
    if not path.exists():
        return entries
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        return entries
    for block in content.split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        times = lines[1].split(" --> ")
        entries.append({
            "index": int(lines[0]),
            "start": parse_srt_time(times[0].strip()),
            "end":   parse_srt_time(times[1].strip()),
            "text":  "\n".join(lines[2:]),
        })
    return entries

def levenshtein(a: str, b: str) -> int:
    if a == b:   return 0
    if not a:    return len(b)
    if not b:    return len(a)
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (ca != cb), dp[j+1] + 1, ndp[j] + 1))
        dp = ndp
    return dp[-1]

def fetch_window_cv(
    video: str,
    t_start: float,
    t_end: float,
    native_fps: float,
    crop_strat = None,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Fetch all frames in [t_start, t_end] using OpenCV CAP_PROP_POS_MSEC.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return [], []
    
    t_start = max(0.0, t_start)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000.0)
    
    timestamps = []
    frames = []
    
    while True:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = pos_msec / 1000.0
        if t > t_end + 0.001:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        if crop_strat:
            frame = crop_strat.apply(frame)
            
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        timestamps.append(t)
        frames.append(frame_rgb)
        
    cap.release()
    return timestamps, frames

# --- Subtitle Accumulator for Dedup ---

class SubtitleAccumulator:
    def __init__(self, fuzzy_threshold: int = 2, min_duration_frames: int = 1):
        self.fuzzy_threshold = fuzzy_threshold
        self.min_duration_frames = min_duration_frames
        self.entries = []
        self._prev_text = None
        self._prev_start = None

    def update(self, frame_idx: int, text: str):
        if not text:
            if self._prev_text is not None:
                self.entries.append((self._prev_start, frame_idx - 1, self._prev_text))
                self._prev_text = None
                self._prev_start = None
        elif (self._prev_text is not None
              and levenshtein(text, self._prev_text) <= self.fuzzy_threshold):
            pass
        else:
            if self._prev_text is not None:
                self.entries.append((self._prev_start, frame_idx - 1, self._prev_text))
            self._prev_text = text
            self._prev_start = frame_idx

    def flush(self, last_frame_idx: int):
        if self._prev_text is not None:
            self.entries.append((self._prev_start, last_frame_idx, self._prev_text))
            self._prev_text = None
            self._prev_start = None

    def write_srt(self, path: Path, fps: float, start_time_offset: float = 0.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(self.entries, 1):
                end = max(end, start + self.min_duration_frames)
                t_start = format_srt_time(start_time_offset + start / fps)
                t_end   = format_srt_time(start_time_offset + end / fps)
                f.write(f"{i}\n{t_start} --> {t_end}\n{text}\n\n")

# --- Refinement Search functions ---

def bisect_start(timestamps: list[float], frames: list[np.ndarray], ref_frame: np.ndarray, orig_height: int | None = None) -> float:
    lo, hi = 0, len(frames) - 1
    result = timestamps[hi]
    while lo <= hi:
        mid = (lo + hi) // 2
        if same_subtitle(frames[mid], ref_frame, orig_height=orig_height):
            result = timestamps[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return result

def bisect_end(timestamps: list[float], frames: list[np.ndarray], ref_frame: np.ndarray, orig_height: int | None = None) -> float:
    lo, hi = 0, len(frames) - 1
    result = timestamps[lo]
    while lo <= hi:
        mid = (lo + hi) // 2
        if same_subtitle(frames[mid], ref_frame, orig_height=orig_height):
            result = timestamps[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return result

def post_process_entries(entries: list[dict], native_fps: float) -> list[dict]:
    if not entries:
        return []
    
    # 1. Filter out extremely short entries
    valid_entries = []
    for e in entries:
        dur = e["end"] - e["start"]
        if dur >= 0.15: # minimum duration 150ms
            valid_entries.append(e)
            
    if not valid_entries:
        return []
        
    # 2. Merge consecutive entries with same text and small gap (< 2.0 seconds)
    merged = []
    curr = dict(valid_entries[0])
    
    for next_entry in valid_entries[1:]:
        # Exact comparison after stripping spaces and common punctuation/symbols
        t1 = curr["text"].strip().replace(" ", "").replace("?", "").replace("？", "").replace(".", "").replace(",", "").replace("-", "")
        t2 = next_entry["text"].strip().replace(" ", "").replace("?", "").replace("？", "").replace(".", "").replace(",", "").replace("-", "")
        
        gap = next_entry["start"] - curr["end"]
        if t1 == t2 and gap <= 2.0:
            curr["end"] = next_entry["end"]
            if len(next_entry["text"]) > len(curr["text"]):
                curr["text"] = next_entry["text"]
        else:
            merged.append(curr)
            curr = dict(next_entry)
            
    merged.append(curr)
    
    # Re-index
    for i, e in enumerate(merged, 1):
        e["index"] = i
        
    return merged

# --- Main Subtitle Processor ---

class SubtitleProcessor:
    def __init__(self, ocr_input_dir: str = "ocr_input", crop_strat = None):
        self.ocr_input_dir = Path(ocr_input_dir)
        self.ocr_input_dir.mkdir(parents=True, exist_ok=True)
        (self.ocr_input_dir / "refinement").mkdir(parents=True, exist_ok=True)
        
        # Default crop strategy to bottom strip if not specified
        from src.strategies import CropStrategy
        self.crop_strat = crop_strat or CropStrategy(top=0.80, bottom=1.0, left=0.0, right=1.0)

    def process_coarse_subtitle(
        self,
        input_video: str,
        output_srt: str,
        start_time: float = 0.0,
        duration: float | None = None,
    ):
        """
        Scans input_video (already cropped and at 1fps).
        Detects subtitle presence, runs OCR on text, and saves debugging images:
          - ./ocr_input/hh_mm_ss_ms_frame.jpg
          - ./ocr_input/hh_mm_ss_ms_mask.jpg
        Generates coarse_output.srt.
        """
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 1.0

        H_cropped = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop_height_prop = self.crop_strat.bottom_prop - self.crop_strat.top_prop
        orig_height = int(round(H_cropped / crop_height_prop)) if crop_height_prop > 0 else H_cropped

        ocr = SubtitleOCR()
        cache = CoverageCache(threshold=0.70, patience=1)
        acc = SubtitleAccumulator(fuzzy_threshold=2, min_duration_frames=1)

        frame_idx = 0
        last_ocr_text = None

        while True:
            current_time = start_time + frame_idx / fps

            ret, frame = cap.read()
            if not ret:
                break

            if duration is not None and (current_time - start_time) > duration:
                break

            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            has_text, binary, strong_px, kept_count = has_subtitle(frame_rgb, orig_height=orig_height)

            cache_hit, cov = cache.check(binary, has_text)

            if not has_text:
                acc.update(frame_idx, "")
            elif cache_hit:
                acc.update(frame_idx, last_ocr_text or "")
            else:
                # Frame is sent to OCR
                text = ocr.read(frame_rgb)
                last_ocr_text = text
                acc.update(frame_idx, text)

                # Save debug images (using frame's timestamp in the video)
                t_str = format_srt_time(current_time).replace(",", "_").replace(":", "_")
                frame_name = self.ocr_input_dir / f"{t_str}_frame.jpg"
                mask_name = self.ocr_input_dir / f"{t_str}_mask.jpg"

                # Frame is saved in BGR
                cv2.imwrite(str(frame_name), frame)
                cv2.imwrite(str(mask_name), binary)

            frame_idx += 1

        cap.release()
        acc.flush(frame_idx - 1)
        acc.write_srt(Path(output_srt), fps, start_time_offset=start_time)
        print(f"Generated coarse SRT → {output_srt}")

    def process_fine_subtitle(
        self,
        input_video: str,
        input_srt: str,
        output_srt: str,
        start_time: float = 0.0,
        duration: float | None = None,
    ):
        """
        Refines coarse subtitles using raw_video at native FPS via binary search.
        Saves coarse boundary debugging frames to:
          - ./ocr_input/refinement/entry_{idx}_coarse_start_at.jpg
          - ...
        """
        # Parse coarse SRT entries
        entries = parse_srt(Path(input_srt))
        if not entries:
            print("No subtitle entries to refine.")
            return

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video {input_video}")
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Filter entries outside the start_time/duration window first
        filtered_entries = []
        for entry in entries:
            S = entry["start"]
            if start_time > 0 and S < start_time:
                continue
            if duration is not None and S > (start_time + duration):
                continue
            filtered_entries.append(entry)

        # Merge duplicate coarse entries before refinement to save processing steps
        filtered_entries = post_process_entries(filtered_entries, native_fps)

        refined_entries = []
        ref_dir = self.ocr_input_dir / "refinement"

        for idx, entry in enumerate(filtered_entries, 1):
            S = entry["start"]
            E = entry["end"]

            # 2. Reference Frame Detection
            span_dur = max(E - S, 1.0 / native_fps)
            ref_ts, ref_frames = fetch_window_cv(input_video, S, S + span_dur, native_fps, self.crop_strat)
            
            # Select the present frame closest to the temporal center of the coarse subtitle
            center_t = (S + E) / 2.0
            ref_frame = None
            best_dist = float('inf')
            
            for t, frame in zip(ref_ts, ref_frames):
                present, _, _, _ = has_subtitle(frame, orig_height=orig_height)
                if present:
                    dist = abs(t - center_t)
                    if dist < best_dist:
                        ref_frame = frame
                        best_dist = dist

            if ref_frame is None:
                # If no reference frame, fallback to coarse times
                refined_entries.append({
                    "index": entry["index"],
                    "start": S,
                    "end": E,
                    "text": entry["text"]
                })
                continue

            # 3. Binary search refinement
            # Start boundary: [S - 0.5, S + 0.5]
            s_ts, s_frames = fetch_window_cv(input_video, S - 0.5, S + 0.5, native_fps, self.crop_strat)
            refined_start = bisect_start(s_ts, s_frames, ref_frame, orig_height=orig_height) if s_frames else S

            # End boundary: [E - 0.5, E + 0.5]
            e_ts, e_frames = fetch_window_cv(input_video, E - 0.5, E + 0.5, native_fps, self.crop_strat)
            refined_end = bisect_end(e_ts, e_frames, ref_frame, orig_height=orig_height) if e_frames else E

            # Correct if end refinement defaulted to a time before start
            if refined_end < refined_start and s_ts and s_frames:
                idx_start = min(range(len(s_ts)), key=lambda i: abs(s_ts[i] - refined_start))
                idx_end = idx_start
                for i in range(idx_start, len(s_ts)):
                    if same_subtitle(s_frames[i], ref_frame, orig_height=orig_height):
                        idx_end = i
                    else:
                        break
                refined_end = s_ts[idx_end]

            refined_end = max(refined_end, refined_start + 1.0 / native_fps)

            # 4. Save refined boundary frames for visual verification
            if s_ts and s_frames:
                idx_at = min(range(len(s_ts)), key=lambda i: abs(s_ts[i] - refined_start))
                idx_before = max(0, idx_at - 1)
                t_at_str = format_srt_time(s_ts[idx_at]).replace(",", "_").replace(":", "_")
                t_before_str = format_srt_time(s_ts[idx_before]).replace(",", "_").replace(":", "_")
                cv2.imwrite(str(ref_dir / f"entry_{entry['index']}_refined_start_at_{t_at_str}.jpg"), cv2.cvtColor(s_frames[idx_at], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(ref_dir / f"entry_{entry['index']}_refined_start_before_{t_before_str}.jpg"), cv2.cvtColor(s_frames[idx_before], cv2.COLOR_RGB2BGR))

            if e_ts and e_frames:
                idx_at = min(range(len(e_ts)), key=lambda i: abs(e_ts[i] - refined_end))
                idx_after = min(len(e_ts) - 1, idx_at + 1)
                t_at_str = format_srt_time(e_ts[idx_at]).replace(",", "_").replace(":", "_")
                t_after_str = format_srt_time(e_ts[idx_after]).replace(",", "_").replace(":", "_")
                cv2.imwrite(str(ref_dir / f"entry_{entry['index']}_refined_end_at_{t_at_str}.jpg"), cv2.cvtColor(e_frames[idx_at], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(ref_dir / f"entry_{entry['index']}_refined_end_after_{t_after_str}.jpg"), cv2.cvtColor(e_frames[idx_after], cv2.COLOR_RGB2BGR))

            refined_entries.append({
                "index": entry["index"],
                "start": refined_start,
                "end": refined_end,
                "text": entry["text"]
            })

        # Post-process entries to merge duplicates and remove noise
        refined_entries = post_process_entries(refined_entries, native_fps)

        # Write refined SRT
        out_path = Path(output_srt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for e in refined_entries:
                f.write(f"{e['index']}\n")
                f.write(f"{format_srt_time(e['start'])} --> {format_srt_time(e['end'])}\n")
                f.write(f"{e['text']}\n\n")

        print(f"Generated refined SRT → {output_srt}")

    def process_iterative_subtitle(
        self,
        coarse_video: str,
        raw_video: str,
        output_srt: str,
        start_time: float = 0.0,
        duration: float | None = None,
    ):
        """
        Iteratively extracts coarse subtitles from coarse_video, refines them on the fly 
        using raw_video, and writes/prints them one-by-one as they are generated.
        """
        cap_coarse = cv2.VideoCapture(coarse_video)
        if not cap_coarse.isOpened():
            raise ValueError(f"Could not open coarse video {coarse_video}")

        fps_coarse = cap_coarse.get(cv2.CAP_PROP_FPS)
        if fps_coarse <= 0:
            fps_coarse = 1.0

        H_cropped = int(cap_coarse.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop_height_prop = self.crop_strat.bottom_prop - self.crop_strat.top_prop
        orig_height = int(round(H_cropped / crop_height_prop)) if crop_height_prop > 0 else H_cropped

        cap_raw = cv2.VideoCapture(raw_video)
        if not cap_raw.isOpened():
            cap_coarse.release()
            raise ValueError(f"Could not open raw video {raw_video}")
        native_fps = cap_raw.get(cv2.CAP_PROP_FPS)
        raw_height = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_raw.release()

        ocr = SubtitleOCR()
        cache = CoverageCache(threshold=0.70, patience=1)

        prev_text = None
        start_frame_idx = None
        
        out_path = Path(output_srt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

        pending_entry = None
        entry_idx = 1

        def refine_and_buffer(coarse_start_time: float, coarse_end_time: float, text: str):
            nonlocal pending_entry, entry_idx
            if not text:
                return

            span_dur = max(coarse_end_time - coarse_start_time, 1.0 / native_fps)
            ref_ts, ref_frames = fetch_window_cv(raw_video, coarse_start_time, coarse_start_time + span_dur, native_fps, self.crop_strat)
            
            center_t = (coarse_start_time + coarse_end_time) / 2.0
            ref_frame = None
            best_dist = float('inf')
            
            for t, frame in zip(ref_ts, ref_frames):
                present, _, _, _ = has_subtitle(frame, orig_height=raw_height)
                if present:
                    dist = abs(t - center_t)
                    if dist < best_dist:
                        ref_frame = frame
                        best_dist = dist

            if ref_frame is None:
                refined_start = coarse_start_time
                refined_end = coarse_end_time
            else:
                s_ts, s_frames = fetch_window_cv(raw_video, coarse_start_time - 0.5, coarse_start_time + 0.5, native_fps, self.crop_strat)
                refined_start = bisect_start(s_ts, s_frames, ref_frame, orig_height=raw_height) if s_frames else coarse_start_time

                e_ts, e_frames = fetch_window_cv(raw_video, coarse_end_time - 0.5, coarse_end_time + 0.5, native_fps, self.crop_strat)
                refined_end = bisect_end(e_ts, e_frames, ref_frame, orig_height=raw_height) if e_frames else coarse_end_time

                if refined_end < refined_start and s_ts and s_frames:
                    idx_start = min(range(len(s_ts)), key=lambda i: abs(s_ts[i] - refined_start))
                    idx_end = idx_start
                    for i in range(idx_start, len(s_ts)):
                        if same_subtitle(s_frames[i], ref_frame, orig_height=raw_height):
                            idx_end = i
                        else:
                            break
                    refined_end = s_ts[idx_end]

                refined_end = max(refined_end, refined_start + 1.0 / native_fps)

            if (refined_end - refined_start) < 0.15:
                return

            new_entry = {
                "start": refined_start,
                "end": refined_end,
                "text": text.strip()
            }

            if pending_entry is not None:
                t1 = pending_entry["text"].replace(" ", "").replace("?", "").replace("？", "").replace(".", "").replace(",", "").replace("-", "")
                t2 = new_entry["text"].replace(" ", "").replace("?", "").replace("？", "").replace(".", "").replace(",", "").replace("-", "")
                gap = new_entry["start"] - pending_entry["end"]
                if t1 == t2 and gap <= 2.0:
                    pending_entry["end"] = new_entry["end"]
                    if len(new_entry["text"]) > len(pending_entry["text"]):
                        pending_entry["text"] = new_entry["text"]
                    return

            flush_pending()
            pending_entry = new_entry

        def flush_pending():
            nonlocal pending_entry, entry_idx
            if pending_entry is not None:
                start_str = format_srt_time(pending_entry["start"])
                end_str = format_srt_time(pending_entry["end"])
                srt_block = f"{entry_idx}\n{start_str} --> {end_str}\n{pending_entry['text']}\n\n"
                
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(srt_block)
                
                print(f"[Refined Subtitle #{entry_idx}] {start_str} --> {end_str} | {pending_entry['text']}")
                
                entry_idx += 1
                pending_entry = None

        frame_idx = 0
        last_ocr_text = None
        absent_frames = 0
        patience = 1

        while True:
            current_time = start_time + frame_idx / fps_coarse

            ret, frame = cap_coarse.read()
            if not ret:
                break

            if duration is not None and (current_time - start_time) > duration:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            has_text, binary, strong_px, kept_count = has_subtitle(frame_rgb, orig_height=orig_height)
            cache_hit, cov = cache.check(binary, has_text)

            if not has_text:
                absent_frames += 1
                if absent_frames > patience:
                    if prev_text is not None:
                        coarse_start = start_time + start_frame_idx / fps_coarse
                        coarse_end = start_time + (frame_idx - absent_frames) / fps_coarse
                        refine_and_buffer(coarse_start, coarse_end, prev_text)
                        prev_text = None
                        start_frame_idx = None
            else:
                absent_frames = 0
                if cache_hit:
                    text = last_ocr_text or ""
                else:
                    text = ocr.read(frame_rgb)
                    last_ocr_text = text
                    
                    t_str = format_srt_time(current_time).replace(",", "_").replace(":", "_")
                    cv2.imwrite(str(self.ocr_input_dir / f"{t_str}_frame.jpg"), frame)
                    cv2.imwrite(str(self.ocr_input_dir / f"{t_str}_mask.jpg"), binary)

                if prev_text is None:
                    prev_text = text
                    start_frame_idx = frame_idx
                elif levenshtein(text, prev_text) > 2:
                    coarse_start = start_time + start_frame_idx / fps_coarse
                    coarse_end = start_time + (frame_idx - 1) / fps_coarse
                    refine_and_buffer(coarse_start, coarse_end, prev_text)
                    prev_text = text
                    start_frame_idx = frame_idx

            frame_idx += 1

        if prev_text is not None:
            coarse_start = start_time + start_frame_idx / fps_coarse
            coarse_end = start_time + (frame_idx - 1) / fps_coarse
            refine_and_buffer(coarse_start, coarse_end, prev_text)

        flush_pending()
        cap_coarse.release()
        print(f"Iterative refined SRT generated → {output_srt}")

