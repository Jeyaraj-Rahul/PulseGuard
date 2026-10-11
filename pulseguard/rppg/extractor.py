import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canonical MediaPipe landmark indices for ROIs
# Forehead: middle of upper forehead
FOREHEAD_LANDMARKS = [10, 67, 109, 338, 297, 151, 9]
# Left Cheek (anatomical left / camera right)
LEFT_CHEEK_LANDMARKS = [330, 347, 348, 349, 350, 423, 427, 436]
# Right Cheek (anatomical right / camera left)
RIGHT_CHEEK_LANDMARKS = [101, 118, 119, 120, 121, 203, 207, 216]
# Nose bridge / Central face (additional reference region)
NOSE_LANDMARKS = [1, 2, 98, 327]

class FaceROIExtractor:
    """
    Extracts mean RGB time series signals from facial skin patches using MediaPipe FaceLandmarker.
    ROIs defined: Forehead, Left Cheek, Right Cheek, Central Nose.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default to bundled model in models dir
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.abspath(os.path.join(curr_dir, "..", "models", "face_landmarker.task"))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe Face Landmarker model not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _get_roi_bounding_box(self, landmarks, indices: List[int], img_w: int, img_h: int, margin: float = 0.08) -> Optional[Tuple[int, int, int, int]]:
        """Compute pixel bounding box for a set of landmarks with safety margin."""
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)
                pts.append((x, y))

        if not pts:
            return None

        pts = np.array(pts)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        # Pad slightly while staying inside frame
        w_pad = int((x_max - x_min) * margin)
        h_pad = int((y_max - y_min) * margin)

        x1 = max(0, x_min - w_pad)
        y1 = max(0, y_min - h_pad)
        x2 = min(img_w, x_max + w_pad)
        y2 = min(img_h, y_max + h_pad)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def extract_from_video(
        self,
        video_path: str,
        max_duration_sec: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process frames from video and return RGB time-series for each ROI.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_duration_sec is not None:
            max_frames = int(fps * max_duration_sec)
            frames_to_process = min(total_frames, max_frames)
        else:
            frames_to_process = total_frames

        roi_keys = ["forehead", "left_cheek", "right_cheek", "nose"]
        signals = {k: {"r": [], "g": [], "b": []} for k in roi_keys}
        timestamps = []
        annotated_sample_frames = []
        frames_tracked = 0
        current_frame_idx = 0

        # Cache last valid bounding boxes for brief tracking dropouts
        last_boxes = {}

        while cap.isOpened() and current_frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = self.detector.detect(mp_image)
            frame_boxes = {}

            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                landmarks = detection_result.face_landmarks[0]
                frames_tracked += 1

                frame_boxes["forehead"] = self._get_roi_bounding_box(landmarks, FOREHEAD_LANDMARKS, w, h)
                frame_boxes["left_cheek"] = self._get_roi_bounding_box(landmarks, LEFT_CHEEK_LANDMARKS, w, h)
                frame_boxes["right_cheek"] = self._get_roi_bounding_box(landmarks, RIGHT_CHEEK_LANDMARKS, w, h)
                frame_boxes["nose"] = self._get_roi_bounding_box(landmarks, NOSE_LANDMARKS, w, h)
                last_boxes = frame_boxes
            else:
                # Use fallback from last valid frame
                frame_boxes = last_boxes

            # Extract mean RGB for each ROI
            for roi in roi_keys:
                box = frame_boxes.get(roi)
                if box is not None:
                    x1, y1, x2, y2 = box
                    patch = rgb_frame[y1:y2, x1:x2]
                    if patch.size > 0:
                        r_mean = float(np.mean(patch[:, :, 0]))
                        g_mean = float(np.mean(patch[:, :, 1]))
                        b_mean = float(np.mean(patch[:, :, 2]))
                    else:
                        r_mean, g_mean, b_mean = (0.0, 0.0, 0.0)
                else:
                    # Missing patch (interpolation done later if needed)
                    r_mean, g_mean, b_mean = (np.nan, np.nan, np.nan)

                signals[roi]["r"].append(r_mean)
                signals[roi]["g"].append(g_mean)
                signals[roi]["b"].append(b_mean)

            timestamps.append(current_frame_idx / fps)

            # Keep a sample frame with drawn bounding boxes for UI preview
            if current_frame_idx == 0 or (len(annotated_sample_frames) < 3 and current_frame_idx % max(1, (frames_to_process // 3)) == 0):
                preview_frame = rgb_frame.copy()
                colors = {
                    "forehead": (255, 69, 0),      # Orange-red
                    "left_cheek": (30, 144, 255),   # Dodger blue
                    "right_cheek": (50, 205, 50),   # Lime green
                    "nose": (255, 215, 0)          # Gold
                }
                for r_name, box in frame_boxes.items():
                    if box:
                        bx1, by1, bx2, by2 = box
                        c = colors.get(r_name, (255, 255, 255))
                        cv2.rectangle(preview_frame, (bx1, by1), (bx2, by2), c, 2)
                        cv2.putText(preview_frame, r_name, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                annotated_sample_frames.append(preview_frame)

            current_frame_idx += 1
            if progress_callback and current_frame_idx % 15 == 0:
                progress_callback(min(1.0, current_frame_idx / frames_to_process))

        cap.release()

        # Handle NaNs via linear interpolation
        for roi in roi_keys:
            for ch in ["r", "g", "b"]:
                arr = np.array(signals[roi][ch], dtype=float)
                nans = np.isnan(arr)
                if np.all(nans):
                    arr[:] = 128.0
                elif np.any(nans):
                    arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
                signals[roi][ch] = arr

        track_ratio = (frames_tracked / max(1, current_frame_idx)) if current_frame_idx > 0 else 0.0

        return {
            "fps": fps,
            "total_frames_extracted": current_frame_idx,
            "face_tracking_ratio": track_ratio,
            "timestamps": np.array(timestamps),
            "signals": signals,
            "preview_frames": annotated_sample_frames
        }
