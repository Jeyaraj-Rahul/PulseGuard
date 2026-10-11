import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
import pandas as pd
import tempfile
import cv2

from pulseguard.pipeline.validator import VideoValidator
from pulseguard.rppg.processor import SignalProcessor
from pulseguard.rppg.features import ConsistencyFeatureExtractor
from pulseguard.models.classifier import PulseGuardClassifier

class TestPulseGuardPipeline(unittest.TestCase):

    def setUp(self):
        self.validator = VideoValidator(min_duration_sec=1.0, min_fps=10.0)
        self.processor = SignalProcessor(low_cut_hz=0.7, high_cut_hz=3.0)
        self.feat_extractor = ConsistencyFeatureExtractor(fps=30.0)

    def test_synthetic_pulse_frequency_estimation(self):
        """Test that FFT correctly recovers an injected 72 BPM (1.2 Hz) signal."""
        fps = 30.0
        duration_sec = 10.0
        n_samples = int(fps * duration_sec)
        t = np.linspace(0, duration_sec, n_samples)

        # Injected pure 72 BPM pulse (1.2 Hz)
        ground_truth_bpm = 72.0
        freq_hz = ground_truth_bpm / 60.0
        clean_sine = np.sin(2 * np.pi * freq_hz * t)

        # Add modest white noise
        noisy_signal = clean_sine + np.random.normal(0, 0.2, n_samples)
        filtered = self.processor.bandpass_filter(noisy_signal, fps)
        spectrum = self.processor.compute_spectrum(filtered, fps)

        estimated_bpm = spectrum["bpm"]
        self.assertAlmostEqual(estimated_bpm, ground_truth_bpm, delta=3.0,
                               msg=f"Estimated BPM {estimated_bpm} deviated from truth {ground_truth_bpm}")
        self.assertGreater(spectrum["snr_db"], 5.0, "Expected positive SNR for clean synthetic pulse")

    def test_detrend_and_filter_stability(self):
        """Verify that detrend removes strong linear drift."""
        x = np.linspace(0, 100, 300) + np.sin(np.linspace(0, 20, 300))
        detrended = self.processor.detrend(x)
        slope = np.polyfit(np.arange(len(detrended)), detrended, 1)[0]
        self.assertAlmostEqual(slope, 0.0, places=4)

    def test_cross_region_coherence(self):
        """Test that perfectly synchronized signals have high coherence while random noise has low."""
        fps = 30.0
        t = np.linspace(0, 10, 300)
        s1 = np.sin(2 * np.pi * 1.2 * t)
        s2 = np.sin(2 * np.pi * 1.2 * t + 0.1)  # small phase offset
        s_random = np.random.normal(0, 1, 300)

        # Synchronous pair
        sync_metrics = self.feat_extractor.compute_pairwise_metrics(s1, s2)
        self.assertGreater(sync_metrics["mean_coherence"], 0.70)
        self.assertGreater(sync_metrics["zero_lag_corr"], 0.85)

        # Asynchronous/Noise pair
        async_metrics = self.feat_extractor.compute_pairwise_metrics(s1, s_random)
        self.assertLess(async_metrics["zero_lag_corr"], 0.30)

    def test_classifier_training_and_serialization(self):
        """Test training pipeline, GroupKFold, and joblib save/load."""
        from pulseguard.pipeline.trainer import generate_synthetic_benchmark_dataset
        df = generate_synthetic_benchmark_dataset(num_subjects=6, samples_per_subject=4)

        clf = PulseGuardClassifier(model_type="svm")
        eval_res = clf.train_and_evaluate(df, df["label"].values, groups=df["subject_id"].values, n_splits=3)
        self.assertTrue(clf.is_trained)
        self.assertIn("accuracy_mean", eval_res)

        # Test prediction on a sample
        sample_dict = df.iloc[0].to_dict()
        pred = clf.predict_sample(sample_dict)
        self.assertTrue(pred["is_model_trained"])
        self.assertIn(pred["label"], [0, 1])

        # Test save and load
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            clf.save(tmp_path)
            loaded_clf = PulseGuardClassifier.load(tmp_path)
            self.assertTrue(loaded_clf.is_trained)
            loaded_pred = loaded_clf.predict_sample(sample_dict)
            self.assertEqual(pred["label"], loaded_pred["label"])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_video_validator_with_synthetic_video(self):
        """Generate a short test video and confirm validator functionality."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            test_video_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(test_video_path, fourcc, 30.0, (320, 240))
            for _ in range(90):  # 3 seconds at 30 fps
                frame = np.full((240, 320, 3), 120, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            val_res = self.validator.validate(test_video_path)
            self.assertTrue(val_res["valid"])
            self.assertEqual(val_res["metadata"]["width"], 320)
            self.assertEqual(val_res["metadata"]["height"], 240)
            self.assertAlmostEqual(val_res["metadata"]["duration_sec"], 3.0, delta=0.2)
        finally:
            if os.path.exists(test_video_path):
                os.unlink(test_video_path)

if __name__ == "__main__":
    unittest.main()
