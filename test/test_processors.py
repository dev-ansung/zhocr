import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.strategies import CropStrategy, FrameStrategy
from src.video import VideoProcessor
from src.presence import make_binary, has_subtitle
from src.cache import same_subtitle, coverage
from src.ocr import SubtitleOCR
from src.subtitle import SubtitleProcessor, fetch_window_cv, parse_srt

VIDEO = "full.mp4"
ARTIFACTS_DIR = ROOT / "test" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Helper to load a frame at a specific timestamp
def get_frame_at(t: float) -> np.ndarray:
    crop_strat = CropStrategy(top=0.80, bottom=1.0, left=0.0, right=1.0)
    ts, fs = fetch_window_cv(VIDEO, t, t + 0.05, 29.97, crop_strat)
    assert fs, f"Failed to fetch frame at {t}s"
    return fs[0]

# --- 1. Test CropStrategy ---
def test_crop_strategy():
    # Create a simple color grid frame
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[30:70, 50:150] = [0, 255, 0]  # green box
    
    crop = CropStrategy(top=0.30, bottom=0.70, left=0.25, right=0.75)
    cropped = crop.apply(frame)
    
    assert cropped.shape == (40, 100, 3)
    assert np.all(cropped == [0, 255, 0])
    
    # Save visual verification
    vis = np.hstack([
        cv2.resize(frame, (200, 100)),
        cv2.resize(cropped, (200, 100))
    ])
    cv2.putText(vis, "Original Box", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(vis, "Cropped Output", (210, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(ARTIFACTS_DIR / "test_crop_strategy.jpg"), vis)

# --- 2. Test VideoProcessor ---
def test_video_processor(tmp_path):
    output_video = tmp_path / "test_proc.mp4"
    video_processor = VideoProcessor()
    
    crop_strat = CropStrategy(top=0.80, bottom=1.0, left=0.0, right=1.0)
    frame_strat = FrameStrategy(fps=1.0)
    
    # Process a 3-second segment starting at 45.0s
    video_processor.process_video(
        input_video=VIDEO,
        output_video=str(output_video),
        modifications=[crop_strat, frame_strat],
        start_time=45.0,
        duration=3.0
    )
    
    assert output_video.exists()
    
    # Read the processed video and verify properties
    cap = cv2.VideoCapture(str(output_video))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    assert abs(fps - 1.0) < 0.05
    assert w == 1280
    assert h == 144  # 720 * 0.2 = 144 pixels
    
    ret, frame = cap.read()
    assert ret
    cap.release()
    
    # Save the processed frame for visual inspection
    # Upscale 2x so it's not a thin line
    frame_upscaled = cv2.resize(frame, (w, h * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(frame_upscaled, "Processed Frame (1fps, Cropped)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(ARTIFACTS_DIR / "test_video_processor_output.jpg"), frame_upscaled)

# --- 3. Test Presence Detection ---
def test_presence_detection():
    # 46.1s is "谢谢" (present=True)
    frame_sub = get_frame_at(46.1)
    sub_present, binary_sub, px_sub, k_sub = has_subtitle(frame_sub, orig_height=720)
    
    # 52.0s is an empty frame (present=False)
    frame_empty = get_frame_at(52.0)
    empty_present, binary_empty, px_empty, k_empty = has_subtitle(frame_empty, orig_height=720)
    
    assert sub_present is True
    assert empty_present is False
    
    # Verify that long subtitles at 77.14s and 77.47s (previously missed due to aspect ratio > 6.0) are correctly detected as present
    frame_long1 = get_frame_at(77.14)
    frame_long2 = get_frame_at(77.47)
    long1_present, _, _, _ = has_subtitle(frame_long1, orig_height=720)
    long2_present, _, _, _ = has_subtitle(frame_long2, orig_height=720)
    assert long1_present is True, "Long subtitle at 77.14s should be detected as present"
    assert long2_present is True, "Long subtitle at 77.47s should be detected as present"
    
    # Save presence visual verification (crop horizontally to center 11% to 89% and stack vertically)
    W = frame_sub.shape[1]
    x_start = int(round(0.11 * W))
    x_end = int(round(0.89 * W))
    crop_sub = frame_sub[:, x_start:x_end]
    crop_bin_sub = binary_sub[:, x_start:x_end]
    crop_empty = frame_empty[:, x_start:x_end]
    crop_bin_empty = binary_empty[:, x_start:x_end]
    
    vis_sub = np.vstack([cv2.cvtColor(crop_sub, cv2.COLOR_RGB2BGR), cv2.cvtColor(crop_bin_sub, cv2.COLOR_GRAY2BGR)])
    vis_sub = cv2.resize(vis_sub, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis_sub, f"SUB PRESENT (46.1s) - px={px_sub} k={k_sub}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(ARTIFACTS_DIR / "test_presence_active.jpg"), vis_sub)
    
    vis_empty = np.vstack([cv2.cvtColor(crop_empty, cv2.COLOR_RGB2BGR), cv2.cvtColor(crop_bin_empty, cv2.COLOR_GRAY2BGR)])
    vis_empty = cv2.resize(vis_empty, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis_empty, f"EMPTY GAP (52.0s) - px={px_empty} k={k_empty}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(ARTIFACTS_DIR / "test_presence_empty.jpg"), vis_empty)

# --- 4. Test Subtitle Identity Comparison ---
def test_subtitle_identity():
    f_sub1 = get_frame_at(46.1)
    f_sub2 = get_frame_at(46.5)
    f_diff = get_frame_at(55.5)
    
    # 46.1s vs 46.5s should be SAME
    same_res = same_subtitle(f_sub1, f_sub2, orig_height=720)
    # 46.5s vs 55.5s should be DIFF
    diff_res = same_subtitle(f_sub1, f_diff, orig_height=720)
    
    assert same_res is True
    assert diff_res is False

    # Verify distinct centered subtitles at 77.5s ("那个放着就好吧") and 79.5s ("用这个吗  没问题吗")
    # are recognized as DIFFERENT
    f_long = get_frame_at(77.5)
    f_next = get_frame_at(79.5)
    assert same_subtitle(f_long, f_next, orig_height=720) is False, "Distinct overlapping subtitles should not be identical"

    # Verify stateful CoverageCache correctly rejects different overlapping subtitles under 0.70 threshold
    from src.cache import CoverageCache
    cache = CoverageCache(threshold=0.70, patience=1)
    b_long = make_binary(f_long, orig_height=720)[0]
    b_next = make_binary(f_next, orig_height=720)[0]
    
    hit1, _ = cache.check(b_long, has_text=True)
    assert hit1 is False, "First cache insert must be a miss"
    
    hit2, cov2 = cache.check(b_next, has_text=True)
    assert hit2 is False, f"Overlapping different text (cov={cov2}) must not trigger cache hit under threshold 0.70"

    
    # Save visual verification for SAME (crop horizontally to center 11% to 89% and stack vertically)
    W = f_sub1.shape[1]
    x_start = int(round(0.11 * W))
    x_end = int(round(0.89 * W))
    b1 = make_binary(f_sub1, orig_height=720)[0]
    b2 = make_binary(f_sub2, orig_height=720)[0]
    cov_same = coverage(b1, b2)
    vis_same = np.vstack([
        cv2.cvtColor(f_sub1[:, x_start:x_end], cv2.COLOR_RGB2BGR),
        cv2.cvtColor(b1[:, x_start:x_end], cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(f_sub2[:, x_start:x_end], cv2.COLOR_RGB2BGR),
        cv2.cvtColor(b2[:, x_start:x_end], cv2.COLOR_GRAY2BGR)
    ])
    vis_same = cv2.resize(vis_same, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis_same, f"IDENTITY SAME (46.1s vs 46.5s) - Same: {same_res} (Expected: True), cov={cov_same:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(ARTIFACTS_DIR / "test_identity_same.jpg"), vis_same)

# --- 5. Test EasyOCR ---
def test_ocr():
    ocr = SubtitleOCR()
    # 46.1s is "谢谢"
    frame = get_frame_at(46.1)
    text = ocr.read(frame)
    
    assert "谢谢" in text or "谢谢" in text.replace(" ", "")
    
    # Save visual verification (crop horizontally to center 11% to 89% and stack vertically)
    W = frame.shape[1]
    x_start = int(round(0.11 * W))
    x_end = int(round(0.89 * W))
    vis = cv2.cvtColor(frame[:, x_start:x_end], cv2.COLOR_RGB2BGR)
    vis = cv2.resize(vis, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis, f"OCR Text: {text!r}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(ARTIFACTS_DIR / "test_ocr_recognition.jpg"), vis)

# --- 6. Test SubtitleProcessor (Coarse & Fine) ---
def test_subtitle_processor(tmp_path):
    proc_video = tmp_path / "processed.mp4"
    coarse_srt = tmp_path / "coarse.srt"
    fine_srt = tmp_path / "fine.srt"
    ocr_dir = tmp_path / "ocr_input"
    
    # First crop and resample video to 1fps
    video_processor = VideoProcessor()
    crop_strat = CropStrategy(top=0.80, bottom=1.0, left=0.0, right=1.0)
    frame_strat = FrameStrategy(fps=1.0)
    
    # Process a snippet between 45s and 50s (includes "谢谢" at 46s)
    video_processor.process_video(
        input_video=VIDEO,
        output_video=str(proc_video),
        modifications=[crop_strat, frame_strat],
        start_time=45.0,
        duration=5.0
    )
    
    # Run SubtitleProcessor
    sub_processor = SubtitleProcessor(ocr_input_dir=str(ocr_dir))
    
    sub_processor.process_coarse_subtitle(
        input_video=str(proc_video),
        output_srt=str(coarse_srt),
        start_time=45.0,
        duration=5.0
    )
    
    assert coarse_srt.exists()
    coarse_entries = parse_srt(coarse_srt)
    assert len(coarse_entries) >= 1
    assert "谢谢" in coarse_entries[0]["text"]
    
    # Verify that ocr debug images exist
    ocr_images = list(ocr_dir.glob("*.jpg"))
    assert len(ocr_images) >= 2  # frame and mask
    
    # Refine timestamps
    sub_processor.process_fine_subtitle(
        input_video=VIDEO,
        input_srt=str(coarse_srt),
        output_srt=str(fine_srt),
        start_time=45.0,
        duration=5.0
    )
    
    assert fine_srt.exists()
    fine_entries = parse_srt(fine_srt)
    assert len(fine_entries) >= 1
    assert fine_entries[0]["start"] != coarse_entries[0]["start"] or fine_entries[0]["end"] != coarse_entries[0]["end"]
    
    # Verify refinement debug files exist
    ref_dir = ocr_dir / "refinement"
    ref_images = list(ref_dir.glob("*refined*.jpg"))
    assert len(ref_images) >= 4  # at least start_at, start_before, end_at, end_after

    # Programmatically test the correctness of the boundary frames!
    start_at_file = next(ref_dir.glob("entry_1_refined_start_at_*.jpg"))
    start_before_file = next(ref_dir.glob("entry_1_refined_start_before_*.jpg"))
    end_at_file = next(ref_dir.glob("entry_1_refined_end_at_*.jpg"))
    end_after_file = next(ref_dir.glob("entry_1_refined_end_after_*.jpg"))

    img_start_at = cv2.cvtColor(cv2.imread(str(start_at_file)), cv2.COLOR_BGR2RGB)
    img_start_before = cv2.cvtColor(cv2.imread(str(start_before_file)), cv2.COLOR_BGR2RGB)
    img_end_at = cv2.cvtColor(cv2.imread(str(end_at_file)), cv2.COLOR_BGR2RGB)
    img_end_after = cv2.cvtColor(cv2.imread(str(end_after_file)), cv2.COLOR_BGR2RGB)

    present_start_at = has_subtitle(img_start_at, orig_height=720)[0]
    present_start_before = has_subtitle(img_start_before, orig_height=720)[0]
    present_end_at = has_subtitle(img_end_at, orig_height=720)[0]
    present_end_after = has_subtitle(img_end_after, orig_height=720)[0]

    assert present_start_at is True, "Refined start frame should show subtitle"
    assert present_start_before is False, "Frame immediately before refined start should not show subtitle"
    assert present_end_at is True, "Refined end frame should show subtitle"
    assert present_end_after is False, "Frame immediately after refined end should not show subtitle"


# --- 7. Test Subtitle Refinement with Noise ---
def test_subtitle_refinement_noise(tmp_path):
    proc_video = tmp_path / "processed.mp4"
    coarse_srt = tmp_path / "coarse.srt"
    fine_srt = tmp_path / "fine.srt"
    
    # Process a segment between 80s and 90s (contains "进去吧" at 86s with late background noise)
    video_processor = VideoProcessor()
    crop_strat = CropStrategy(top=0.80, bottom=1.0, left=0.0, right=1.0)
    frame_strat = FrameStrategy(fps=1.0)
    
    video_processor.process_video(
        input_video=VIDEO,
        output_video=str(proc_video),
        modifications=[crop_strat, frame_strat],
        start_time=80.0,
        duration=10.0
    )
    
    sub_processor = SubtitleProcessor(ocr_input_dir=str(tmp_path / "ocr_input"), crop_strat=crop_strat)
    
    sub_processor.process_coarse_subtitle(
        input_video=str(proc_video),
        output_srt=str(coarse_srt),
        start_time=80.0,
        duration=10.0
    )
    
    coarse_entries = parse_srt(coarse_srt)
    assert any("进去" in e["text"] for e in coarse_entries), "Coarse subtitle should contain '进去吧'"
    
    # Refine timestamps
    sub_processor.process_fine_subtitle(
        input_video=VIDEO,
        input_srt=str(coarse_srt),
        output_srt=str(fine_srt),
        start_time=80.0,
        duration=10.0
    )
    
    fine_entries = parse_srt(fine_srt)
    assert any("进去" in e["text"] for e in fine_entries), "Fine subtitle should contain '进去吧' after refinement"
    
    # Verify refined duration is reasonable (> 0.5s)
    target_entry = next(e for e in fine_entries if "进去" in e["text"])
    dur = target_entry["end"] - target_entry["start"]
    assert dur >= 0.5, f"Refined duration of '进去吧' is too short: {dur:.3f}s"


# --- 8. Test Iterative Subtitle Processor ---
def test_iterative_subtitle_processor(tmp_path):
    proc_video = tmp_path / "processed.mp4"
    fine_srt = tmp_path / "fine_iterative.srt"
    
    # Crop and resample video to 1fps
    video_processor = VideoProcessor()
    crop_strat = CropStrategy(top=0.80, bottom=1.0, left=0.0, right=1.0)
    frame_strat = FrameStrategy(fps=1.0)
    
    # Segment between 45s and 50s (includes "谢谢" at 46s)
    video_processor.process_video(
        input_video=VIDEO,
        output_video=str(proc_video),
        modifications=[crop_strat, frame_strat],
        start_time=45.0,
        duration=5.0
    )
    
    sub_processor = SubtitleProcessor(ocr_input_dir=str(tmp_path / "ocr_input"), crop_strat=crop_strat)
    
    # Process iteratively
    sub_processor.process_iterative_subtitle(
        coarse_video=str(proc_video),
        raw_video=VIDEO,
        output_srt=str(fine_srt),
        start_time=45.0,
        duration=5.0
    )
    
    assert fine_srt.exists()
    fine_entries = parse_srt(fine_srt)
    assert len(fine_entries) >= 1
    assert "谢谢" in fine_entries[0]["text"]
    
    # Duration should be reasonable (e.g. around 1.5 seconds)
    dur = fine_entries[0]["end"] - fine_entries[0]["start"]
    assert 0.5 <= dur <= 2.5


