import os
import cv2
import numpy as np

def generate_sample_face_video(output_path: str = "sample_test_video.mp4", duration_sec: int = 4, fps: int = 30):
    """
    Creates an animated synthetic human face video with injected rPPG hemoglobin pulsation.
    Skin patches (forehead, cheeks, nose) pulsate subtly at 75 BPM (1.25 Hz).
    """
    width, height = 480, 480
    n_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    heart_rate_hz = 75.0 / 60.0  # 1.25 Hz

    for f_idx in range(n_frames):
        t = f_idx / fps
        frame = np.full((height, width, 3), (30, 30, 35), dtype=np.uint8)

        # Subtle cardiac color modulation in green channel (amplitude ~ 2.5 counts)
        pulse_mod = 2.5 * np.sin(2 * np.pi * heart_rate_hz * t)
        skin_r = int(195)
        skin_g = int(np.clip(155 + pulse_mod, 0, 255))
        skin_b = int(135)
        skin_color = (skin_b, skin_g, skin_r)  # BGR

        # Draw face base
        center = (width // 2, height // 2 + 10)
        cv2.ellipse(frame, center, (130, 165), 0, 0, 360, skin_color, -1)

        # Draw eyes
        eye_y = height // 2 - 25
        cv2.circle(frame, (width // 2 - 45, eye_y), 14, (240, 240, 240), -1)
        cv2.circle(frame, (width // 2 + 45, eye_y), 14, (240, 240, 240), -1)
        cv2.circle(frame, (width // 2 - 45, eye_y), 6, (40, 40, 40), -1)
        cv2.circle(frame, (width // 2 + 45, eye_y), 6, (40, 40, 40), -1)

        # Draw nose bridge
        cv2.line(frame, (width // 2, eye_y + 10), (width // 2, eye_y + 55), (110, 130, 170), 3)

        # Draw mouth
        mouth_y = height // 2 + 85
        cv2.ellipse(frame, (width // 2, mouth_y), (35, 12), 0, 0, 180, (70, 70, 160), -1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic test video: {output_path} ({n_frames} frames, ~75 BPM)")

if __name__ == "__main__":
    generate_sample_face_video()
