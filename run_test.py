import os
import sys
# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

import cv2
import numpy as np
import tempfile
from pulseguard.rppg.extractor import FaceROIExtractor
from pulseguard.rppg.processor import SignalProcessor
from pulseguard.rppg.features import ConsistencyFeatureExtractor

# Create a short synthetic video (2 seconds, 30 fps) with uniform gray frames
fps = 30
duration = 2  # seconds
frame_count = int(fps * duration)
# Use a temporary file for the video
temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
video_path = temp_video.name
temp_video.close()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_path, fourcc, fps, (320, 240))
gray_frame = np.full((240, 320, 3), 120, dtype=np.uint8)
for _ in range(frame_count):
    writer.write(gray_frame)
writer.release()

print('Generated video path:', video_path)

# Run extraction pipeline
extractor = FaceROIExtractor()
extraction = extractor.extract_from_video(video_path, max_duration_sec=None)
print('Face tracking ratio:', extraction['face_tracking_ratio'])
processor = SignalProcessor(low_cut_hz=0.7, high_cut_hz=3.0)
roi_signals = {}
roi_spectra = {}
for roi in ['forehead', 'left_cheek', 'right_cheek', 'nose']:
    sig = processor.extract_green_signal(extraction['signals'][roi]['g'], extraction['fps'])
    roi_signals[roi] = sig
    roi_spectra[roi] = processor.compute_spectrum(sig, extraction['fps'])

feat_extractor = ConsistencyFeatureExtractor(fps=extraction['fps'])
features = feat_extractor.extract_features(roi_signals, roi_spectra)
quality = feat_extractor.assess_signal_quality(features, extraction['face_tracking_ratio'], len(extraction['timestamps'])/extraction['fps'])
print('Quality assessment:', quality)

# Clean up temporary video file
os.remove(video_path)
