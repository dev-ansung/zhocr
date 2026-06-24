import argparse
import datetime
from src.strategies import CropStrategy, FrameStrategy, DEFAULT_CROP
from src.video import VideoProcessor
from src.subtitle import SubtitleProcessor

def main():
    parser = argparse.ArgumentParser(description="Rapid Chinese subtitle OCR extractor")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration to process in seconds")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"output/{timestamp}"

    raw_video = args.video
    processed_video = f"{run_dir}/processed.mp4"
    coarse_subtitle = f"{run_dir}/coarse_output.srt"
    fine_subtitle = f"{run_dir}/fine_output.srt"
    ocr_input_dir = f"{run_dir}/ocr_input"

    print(f"=== Output directory: {run_dir} ===")

    # 1. Process Video (resample and crop)
    video_processor = VideoProcessor()
    crop_strat = CropStrategy(**DEFAULT_CROP)
    frame_strat = FrameStrategy(fps=1.0)
    modifications = [crop_strat, frame_strat]

    print(f"=== Step 1: Processing Video (Start: {args.start}s, Duration: {args.duration}s) ===")
    video_processor.process_video(
        input_video=raw_video, 
        output_video=processed_video, 
        modifications=modifications,
        start_time=args.start,
        duration=args.duration
    )

    # 2. Extract & Refine Subtitles Iteratively (On-the-Fly)
    print(f"\n=== Step 2 & 3: Iterative Subtitle Extraction & Refinement ===")
    subtitle_processor = SubtitleProcessor(ocr_input_dir=ocr_input_dir)
    subtitle_processor.process_iterative_subtitle(
        coarse_video=processed_video,
        raw_video=raw_video,
        output_srt=fine_subtitle,
        start_time=args.start,
        duration=args.duration
    )
    print(f"\n=== Subtitle extraction and refinement completed successfully! ===")
    print(f"=== Outputs saved under: {run_dir} ===")

if __name__ == "__main__":
    main()
