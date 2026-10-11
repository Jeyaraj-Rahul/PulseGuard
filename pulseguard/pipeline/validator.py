import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class VideoValidator:
    """
    Validates uploaded video files for duration, frame rate, resolution,
    and visual quality (blurriness, illumination).
    """

    def __init__(
        self,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 60.0,
        min_fps: float = 15.0,
        max_fps: float = 120.0,
        min_resolution: Tuple[int, int] = (240, 240),
        min_brightness: float = 30.0,
        max_brightness: float = 230.0,
        min_laplacian_var: float = 20.0  # blur threshold
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.min_resolution = min_resolution
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_laplacian_var = min_laplacian_var

    def validate(self, video_path: str) -> Dict[str, Any]:
        """
        Inspect video metadata and sample frames to check quality standards.
        """
        if not os.path.exists(video_path):
            return {
                "valid": False,
                "error": f"File does not exist: {video_path}"
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "valid": False,
                "error": "Failed to open video file. Invalid or unsupported format."
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle abnormal or zero FPS reporting
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # fallback estimation
            estimated_fps = False
        else:
            estimated_fps = True

        duration = total_frames / fps if fps > 0 else 0.0

        errors = []
        warnings = []

        if duration < self.min_duration_sec:
            errors.append(f"Video duration ({duration:.1f}s) is shorter than minimum required ({self.min_duration_sec:.1f}s). rPPG requires at least a few cardiac cycles.")
        elif duration > self.max_duration_sec:
            warnings.append(f"Video duration ({duration:.1f}s) exceeds recommended maximum ({self.max_duration_sec:.1f}s). Processing will be limited to first {self.max_duration_sec}s.")

        if fps < self.min_fps:
            errors.append(f"Frame rate ({fps:.1f} FPS) is too low for reliable pulse frequency sampling (Nyquist limit: ≥ 15 FPS).")

        if width < self.min_resolution[0] or height < self.min_resolution[1]:
            warnings.append(f"Resolution ({width}x{height}) is lower than standard ({self.min_resolution[0]}x{self.min_resolution[1]}). Signal-to-noise ratio might be degraded.")

        # Inspect quality by sampling up to 10 frames
        sample_indices = np.linspace(0, max(0, total_frames - 1), num=min(10, max(1, total_frames)), dtype=int)
        brightness_vals = []
        laplacian_vars = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_vals.append(float(np.mean(gray)))
            laplacian_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        cap.release()

        avg_brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0
        avg_blur = float(np.mean(laplacian_vars)) if laplacian_vars else 0.0

        if brightness_vals:
            if avg_brightness < self.min_brightness:
                warnings.append(f"Video appears underexposed (mean brightness {avg_brightness:.1f}/255). Pulse modulation may be masked.")
            elif avg_brightness > self.max_brightness:
                warnings.append(f"Video appears overexposed/washed out (mean brightness {avg_brightness:.1f}/255). Skin highlights may clip.")

            if avg_blur < self.min_laplacian_var:
                warnings.append(f"Video has high motion blur or soft focus (sharpness variance {avg_blur:.1f} < {self.min_laplacian_var}).")

        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "metadata": {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_sec": round(duration, 2),
                "width": width,
                "height": height,
                "avg_brightness": round(avg_brightness, 1),
                "avg_sharpness": round(avg_blur, 1),
                "fps_estimated": not estimated_fps
            }
        }
