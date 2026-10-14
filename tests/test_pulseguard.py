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

    def test_flat_signal_spectrum_and_coherence(self):
        """Verify that zero/flat signals do not report 42.2 BPM or 20.0 dB SNR, and yield 0.0 coherence without NaNs."""
        flat_signal = np.zeros(150)
        spec = self.processor.compute_spectrum(flat_signal, 30.0)
        self.assertEqual(spec["bpm"], 0.0, "Flat signal should report 0.0 BPM, not default 42.2")
        self.assertEqual(spec["snr_db"], -20.0, "Flat signal should report negative SNR floor, not 20.0 dB")

        # Coherence between two flat signals
        coh_res = self.feat_extractor.compute_pairwise_metrics(flat_signal, flat_signal)
        self.assertFalse(np.isnan(coh_res["mean_coherence"]), "Coherence of flat signals should not be NaN")
        self.assertEqual(coh_res["mean_coherence"], 0.0)
        self.assertEqual(coh_res["zero_lag_corr"], 0.0)

    def test_nan_and_infinite_feature_validation(self):
        """Verify that validate_features detects NaN and infinite feature values correctly."""
        clf = PulseGuardClassifier(model_type="svm")
        clean_features = {k: 1.0 for k in clf.FEATURE_NAMES}
        is_valid, invalid_keys, _ = clf.validate_features(clean_features)
        self.assertTrue(is_valid)
        self.assertEqual(len(invalid_keys), 0)

        # Inject NaN and Infs
        corrupted = clean_features.copy()
        corrupted["forehead_snr"] = np.nan
        corrupted["mean_cross_coherence"] = float("inf")
        corrupted["consistency_score"] = float("-inf")

        is_valid_bad, invalid_keys_bad, invalid_dict = clf.validate_features(corrupted)
        self.assertFalse(is_valid_bad)
        self.assertIn("forehead_snr", invalid_keys_bad)
        self.assertIn("mean_cross_coherence", invalid_keys_bad)
        self.assertIn("consistency_score", invalid_keys_bad)
        self.assertEqual(len(invalid_keys_bad), 3)

    def test_classifier_handles_nan_via_fitted_imputer(self):
        """Verify that the retrained pipeline with SimpleImputer handles missing/NaN features without crashing."""
        from pulseguard.pipeline.trainer import generate_synthetic_benchmark_dataset
        df = generate_synthetic_benchmark_dataset(num_subjects=6, samples_per_subject=4)

        clf = PulseGuardClassifier(model_type="svm")
        clf.train_and_evaluate(df, df["label"].values, groups=df["subject_id"].values, n_splits=3)

        # Create a sample with some missing/NaN features
        test_sample = df.iloc[0].to_dict()
        test_sample["forehead_snr"] = np.nan
        test_sample["coh_forehead_left_cheek"] = np.nan

        pred_res = clf.predict_sample(test_sample)
        self.assertTrue(pred_res["is_model_trained"])
        self.assertFalse(pred_res["is_valid_sample"], "Should flag that sample had invalid/NaN features")
        self.assertIn(pred_res["label"], [0, 1], "Imputer should allow pipeline to predict without throwing ValueError")

    def test_feature_count_and_order_alignment(self):
        """Verify that the feature vector exactly matches the model's trained feature schema."""
        from pulseguard.pipeline.trainer import generate_synthetic_benchmark_dataset
        df = generate_synthetic_benchmark_dataset(num_subjects=6, samples_per_subject=4)
        clf = PulseGuardClassifier(model_type="svm")
        clf.train_and_evaluate(df, df["label"].values, groups=df["subject_id"].values, n_splits=3)

        self.assertEqual(clf.pipeline.named_steps["imputer"].n_features_in_, len(clf.FEATURE_NAMES))
        self.assertEqual(clf.pipeline.named_steps["scaler"].n_features_in_, len(clf.FEATURE_NAMES))
        self.assertEqual(clf.pipeline.named_steps["classifier"].n_features_in_, len(clf.FEATURE_NAMES))

    def test_synthetic_signal_processing_suite(self):
        """
        Verify signal processor with:
        1. Clean pulse at known frequency (55 BPM, 0.917 Hz)
        2. Noisy pulse (recovering fundamental pulse under noise)
        3. Flat/zero signal (guarding against crashes, NaNs, and artificial default BPMs)
        4. Mixed-frequency signal (verifying dominant component resolution)
        """
        fps = 30.0
        dur = 10.0
        n = int(fps * dur)
        t = np.linspace(0, dur, n)

        # 1. Clean 55 BPM pulse
        pulse_55 = np.sin(2 * np.pi * (55.0 / 60.0) * t)
        filt_clean = self.processor.bandpass_filter(pulse_55, fps)
        spec_clean = self.processor.compute_spectrum(filt_clean, fps)
        self.assertAlmostEqual(spec_clean["bpm"], 55.0, delta=2.0)
        self.assertGreater(spec_clean["snr_db"], 10.0)

        # 2. Noisy pulse (additive Gaussian noise std=1.5)
        np.random.seed(42)
        noisy_55 = pulse_55 + np.random.normal(0, 1.5, n)
        filt_noisy = self.processor.bandpass_filter(noisy_55, fps)
        spec_noisy = self.processor.compute_spectrum(filt_noisy, fps)
        self.assertAlmostEqual(spec_noisy["bpm"], 55.0, delta=3.0)

        # 3. Flat signal (zeros)
        flat = np.zeros(n)
        spec_flat = self.processor.compute_spectrum(flat, fps)
        self.assertEqual(spec_flat["bpm"], 0.0)
        self.assertEqual(spec_flat["snr_db"], -20.0)

        # 4. Mixed-frequency signal: 55 BPM (amplitude 1.0) + 150 BPM (amplitude 0.4)
        mixed = pulse_55 + 0.4 * np.sin(2 * np.pi * (150.0 / 60.0) * t)
        filt_mixed = self.processor.bandpass_filter(mixed, fps)
        spec_mixed = self.processor.compute_spectrum(filt_mixed, fps)
        self.assertAlmostEqual(spec_mixed["bpm"], 55.0, delta=2.5)

    def test_signal_quality_gate_divergent_regions(self):
        """
        Verify that when facial patches exhibit severe BPM divergence (e.g., forehead noise vs cheeks/nose),
        the quality gate flags the sample as Inconclusive rather than issuing a false deepfake determination.
        """
        user_scenario_features = {
            "forehead_bpm": 153.3,
            "forehead_snr": -4.3,
            "left_cheek_bpm": 57.9,
            "left_cheek_snr": -0.2,
            "right_cheek_bpm": 49.7,
            "right_cheek_snr": 2.6,
            "nose_bpm": 52.1,
            "nose_snr": 1.4,
            "bpm_std_dev": 43.43,
            "bpm_max_diff": 103.6,
            "mean_bpm": 78.25,
            "mean_snr": -0.13,
            "mean_cross_coherence": 0.35,
            "consistency_score": 0.38
        }

        gate_res = self.feat_extractor.assess_signal_quality(
            user_scenario_features,
            face_tracking_ratio=1.0,
            duration_sec=10.0
        )
        self.assertFalse(gate_res["is_reliable"])
        self.assertEqual(gate_res["status"], "Inconclusive")
        self.assertTrue(any("divergence" in r.lower() for r in gate_res["reasons"]))

    def test_signal_quality_gate_sub_zero_mean_snr(self):
        """Verify that mean SNR below 0 dB flags the sample as Inconclusive."""
        features = {
            "forehead_bpm": 52.0, "forehead_snr": 3.1,
            "left_cheek_bpm": 53.0, "left_cheek_snr": -0.6,
            "right_cheek_bpm": 64.0, "right_cheek_snr": -1.6,
            "nose_bpm": 56.0, "nose_snr": -3.9,
            "bpm_std_dev": 4.76, "bpm_max_diff": 12.0,
            "mean_bpm": 56.25, "mean_snr": -0.74,
            "mean_cross_coherence": 0.473, "consistency_score": 0.397
        }
        gate_res = self.feat_extractor.assess_signal_quality(features, face_tracking_ratio=1.0, duration_sec=19.8)
        self.assertFalse(gate_res["is_reliable"])
        self.assertEqual(gate_res["status"], "Inconclusive")
        self.assertTrue(any("sub-zero mean" in r.lower() for r in gate_res["reasons"]))

    def test_signal_quality_gate_three_regions_negative_snr(self):
        """Verify that when 3 of 4 regions have negative SNR (even if mean SNR is slightly positive due to 1 outlier), it is Inconclusive."""
        features = {
            "forehead_bpm": 60.0, "forehead_snr": 5.0,     # positive
            "left_cheek_bpm": 61.0, "left_cheek_snr": -0.5, # negative
            "right_cheek_bpm": 62.0, "right_cheek_snr": -1.0, # negative
            "nose_bpm": 59.0, "nose_snr": -2.0,            # negative
            "bpm_std_dev": 1.1, "bpm_max_diff": 3.0,
            "mean_bpm": 60.5, "mean_snr": 0.375,          # mean > 0, but 3 regions negative
            "mean_cross_coherence": 0.55, "consistency_score": 0.52
        }
        gate_res = self.feat_extractor.assess_signal_quality(features, face_tracking_ratio=1.0, duration_sec=10.0)
        self.assertFalse(gate_res["is_reliable"])
        self.assertEqual(gate_res["status"], "Inconclusive")
        self.assertTrue(any("3 of 4 facial regions" in r.lower() for r in gate_res["reasons"]))

    def test_signal_quality_gate_adequate_snr_passes(self):
        """Verify that samples with adequate SNR, duration, tracking, and consistency pass the quality gate."""
        features = {
            "forehead_bpm": 70.0, "forehead_snr": 4.5,
            "left_cheek_bpm": 71.0, "left_cheek_snr": 3.8,
            "right_cheek_bpm": 69.5, "right_cheek_snr": 4.0,
            "nose_bpm": 70.5, "nose_snr": 2.5,
            "bpm_std_dev": 0.7, "bpm_max_diff": 1.5,
            "mean_bpm": 70.25, "mean_snr": 3.7,
            "mean_cross_coherence": 0.68, "consistency_score": 0.65
        }
        gate_res = self.feat_extractor.assess_signal_quality(features, face_tracking_ratio=0.95, duration_sec=12.0)
        self.assertTrue(gate_res["is_reliable"])
        self.assertEqual(gate_res["status"], "Reliable")
        self.assertEqual(len(gate_res["reasons"]), 0)

if __name__ == "__main__":
    unittest.main()

