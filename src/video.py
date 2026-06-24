import cv2
import numpy as np
from pathlib import Path
from src.strategies import CropStrategy, FrameStrategy

class VideoProcessor:
    def process_video(
        self,
        input_video: str,
        output_video: str,
        modifications: list,
        start_time: float = 0.0,
        duration: float | None = None,
    ):
        """
        Processes the input video: seeks to start_time, reads frames for duration,
        applies CropStrategy and FrameStrategy modifications, and writes the
        result to output_video.
        """
        # Find strategies
        crop_strat = next((m for m in modifications if isinstance(m, CropStrategy)), None)
        frame_strat = next((m for m in modifications if isinstance(m, FrameStrategy)), None)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video {input_video}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_fps = frame_strat.fps if frame_strat else orig_fps
        target_width = int(round((crop_strat.right_prop - crop_strat.left_prop) * orig_width)) if crop_strat else orig_width
        target_height = int(round((crop_strat.bottom_prop - crop_strat.top_prop) * orig_height)) if crop_strat else orig_height

        # Seek to start time
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

        # Get starting frame index
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Initialize VideoWriter
        # On macOS, "mp4v" codec is highly compatible with .mp4 extension
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # Ensure output directory exists
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = cv2.VideoWriter(str(out_path), fourcc, target_fps, (target_width, target_height))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not open output video writer for {output_video}")

        frame_interval = 1.0 / target_fps if frame_strat else 0.0
        next_target_time = start_time
        written_count = 0

        # Small tolerance threshold for frame timing comparisons (e.g., half a frame time)
        # to avoid missing frames due to float precision
        tolerance = 0.5 / orig_fps if orig_fps > 0 else 0.01

        while True:
            current_time = frame_idx / orig_fps if orig_fps > 0 else 0.0

            ret, frame = cap.read()
            if not ret:
                break

            if duration is not None and (current_time - start_time) > duration:
                break

            if frame_strat:
                # If we've reached or passed the target timestamp, process and write
                if current_time >= (next_target_time - tolerance):
                    if crop_strat:
                        frame = crop_strat.apply(frame)
                    writer.write(frame)
                    written_count += 1
                    # Advance to next target timestamp (accounting for any potential skipping)
                    next_target_time += frame_interval
                    while next_target_time <= current_time:
                        next_target_time += frame_interval
            else:
                if crop_strat:
                    frame = crop_strat.apply(frame)
                writer.write(frame)
                written_count += 1

            frame_idx += 1

        cap.release()
        writer.release()
        print(f"Processed video: {written_count} frames written to {output_video} at {target_fps} fps (Crop: {target_width}x{target_height})")
