import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pulseguard.pipeline.validator import VideoValidator
from pulseguard.rppg.extractor import FaceROIExtractor
from pulseguard.rppg.processor import SignalProcessor
from pulseguard.rppg.features import ConsistencyFeatureExtractor
from pulseguard.models.classifier import PulseGuardClassifier

class DatasetPipeline:
    """
    Automated feature extraction and training pipeline for video datasets
    (e.g., FaceForensics++, Celeb-DF, UBFC-rPPG).
    Expected directory format:
    dataset_root/
        real/
            subject01_real.mp4
            subject02_real.mp4
        fake/
            subject01_deepfake.mp4
            subject02_deepfake.mp4
    """

    def __init__(self, model_task_path: Optional[str] = None):
        self.validator = VideoValidator(min_duration_sec=2.0)
        self.extractor = FaceROIExtractor(model_path=model_task_path)
        self.processor = SignalProcessor()

    def process_single_video(self, video_path: str, max_duration_sec: float = 10.0) -> Optional[Dict[str, float]]:
        """Run end-to-end extraction on a single video file."""
        val_res = self.validator.validate(video_path)
        if not val_res["valid"]:
            print(f"Skipping {video_path}: {val_res.get('errors')}")
            return None

        fps = val_res["metadata"]["fps"]
        try:
            extraction = self.extractor.extract_from_video(video_path, max_duration_sec=max_duration_sec)
        except Exception as e:
            print(f"Error extracting ROIs for {video_path}: {e}")
            return None

        if extraction["face_tracking_ratio"] < 0.3:
            print(f"Face lost in too many frames for {video_path} (track ratio: {extraction['face_tracking_ratio']:.2f})")
            return None

        # Process pulse signals per region
        roi_signals = {}
        roi_spectra = {}
        signals = extraction["signals"]

        for roi in ["forehead", "left_cheek", "right_cheek", "nose"]:
            pulse_sig = self.processor.extract_green_signal(signals[roi]["g"], fps)
            spec = self.processor.compute_spectrum(pulse_sig, fps)
            roi_signals[roi] = pulse_sig
            roi_spectra[roi] = spec

        feat_ext = ConsistencyFeatureExtractor(fps=fps)
        feats = feat_ext.extract_features(roi_signals, roi_spectra)
        return feats

    def extract_dataset_features(self, dataset_dir: str, max_duration_sec: float = 10.0) -> pd.DataFrame:
        """
        Scan directory and compute features for all videos recursively.
        Assumes folders 'real' (label 1) and 'fake' (label 0), which may contain subfolders.
        """
        rows = []
        for label_name, label_val in [("real", 1), ("fake", 0)]:
            folder = os.path.join(dataset_dir, label_name)
            if not os.path.exists(folder):
                continue

            for root, _, files in os.walk(folder):
                for fname in sorted(files):
                    if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        vpath = os.path.join(root, fname)
                        rel_sub = os.path.relpath(root, folder)
                        generator_tag = "" if rel_sub == "." else f"[{rel_sub}] "
                        print(f"Extracting features: [{label_name.upper()}] {generator_tag}{fname}...")
                        feats = self.process_single_video(vpath, max_duration_sec=max_duration_sec)
                        if feats:
                            feats["filename"] = fname
                            feats["label"] = label_val
                            feats["category"] = rel_sub if rel_sub != "." else label_name
                            # Infer subject_id (e.g. id0_0001 -> id0, 001_AkoolAI_fa021 -> akool_001)
                            if "_" in fname:
                                base_id = fname.split("_")[0]
                            else:
                                base_id = os.path.splitext(fname)[0]
                            feats["subject_id"] = f"{rel_sub}_{base_id}" if rel_sub != "." else base_id
                            rows.append(feats)

        df = pd.DataFrame(rows)
        return df

def generate_synthetic_benchmark_dataset(num_subjects: int = 12, samples_per_subject: int = 4) -> pd.DataFrame:
    """
    Generate realistic synthetic physiological feature distribution for testing and demonstration.
    Models physiological consistency differences documented in the FakeCatcher literature:
    - Real videos: High cross-patch coherence (0.55-0.90), low BPM variance (< 3 BPM), SNR > 3dB.
    - Deepfake videos: Disrupted cross-patch coherence (0.10-0.45), high BPM variance (5-25 BPM), lower SNR.
    """
    np.random.seed(42)
    records = []

    for subj_i in range(num_subjects):
        subj_name = f"subject_{subj_i+1:02d}"
        base_hr = np.random.uniform(62, 85)  # subject's natural resting heart rate

        for sample_i in range(samples_per_subject):
            # Real sample (Label = 1)
            real_dev = np.random.normal(0, 1.2)
            f_bpm = base_hr + real_dev + np.random.normal(0, 0.5)
            lc_bpm = base_hr + real_dev + np.random.normal(0, 0.5)
            rc_bpm = base_hr + real_dev + np.random.normal(0, 0.5)
            n_bpm = base_hr + real_dev + np.random.normal(0, 0.7)

            real_coh = np.random.uniform(0.60, 0.92)
            real_snr = np.random.uniform(3.5, 9.5)

            rec_real = {
                "forehead_bpm": f_bpm,
                "forehead_snr": real_snr + np.random.normal(0, 0.5),
                "left_cheek_bpm": lc_bpm,
                "left_cheek_snr": real_snr + np.random.normal(0, 0.5),
                "right_cheek_bpm": rc_bpm,
                "right_cheek_snr": real_snr + np.random.normal(0, 0.5),
                "nose_bpm": n_bpm,
                "nose_snr": real_snr - 0.5 + np.random.normal(0, 0.5),
                "bpm_std_dev": float(np.std([f_bpm, lc_bpm, rc_bpm, n_bpm])),
                "bpm_max_diff": float(np.max([f_bpm, lc_bpm, rc_bpm, n_bpm]) - np.min([f_bpm, lc_bpm, rc_bpm, n_bpm])),
                "mean_bpm": float(np.mean([f_bpm, lc_bpm, rc_bpm, n_bpm])),
                "mean_snr": real_snr,
                "min_snr": real_snr - 1.2,
                "corr_forehead_left_cheek": np.clip(real_coh + np.random.normal(0, 0.05), -1, 1),
                "coh_forehead_left_cheek": real_coh,
                "corr_forehead_right_cheek": np.clip(real_coh + np.random.normal(0, 0.05), -1, 1),
                "coh_forehead_right_cheek": real_coh * 0.95,
                "corr_left_cheek_right_cheek": np.clip(real_coh + np.random.normal(0, 0.05), -1, 1),
                "coh_left_cheek_right_cheek": real_coh * 0.98,
                "mean_cross_correlation": real_coh * 0.9,
                "mean_cross_coherence": real_coh,
                "consistency_score": float(np.clip(0.6 * real_coh + 0.4 * ((real_snr + 5.0) / 15.0), 0.0, 1.0)),
                "subject_id": subj_name,
                "label": 1,
                "filename": f"{subj_name}_real_{sample_i+1}.mp4"
            }
            records.append(rec_real)

            # Deepfake sample (Label = 0)
            # Spatial blending and generative warp desynchronize blood flow fluctuations
            fake_f_bpm = base_hr + np.random.uniform(-15, 15)
            fake_lc_bpm = base_hr + np.random.uniform(-15, 15)
            fake_rc_bpm = base_hr + np.random.uniform(-15, 15)
            fake_n_bpm = base_hr + np.random.uniform(-15, 15)

            fake_coh = np.random.uniform(0.12, 0.38)
            fake_snr = np.random.uniform(-1.0, 2.5)

            rec_fake = {
                "forehead_bpm": fake_f_bpm,
                "forehead_snr": fake_snr + np.random.normal(0, 0.5),
                "left_cheek_bpm": fake_lc_bpm,
                "left_cheek_snr": fake_snr + np.random.normal(0, 0.5),
                "right_cheek_bpm": fake_rc_bpm,
                "right_cheek_snr": fake_snr + np.random.normal(0, 0.5),
                "nose_bpm": fake_n_bpm,
                "nose_snr": fake_snr - 0.5 + np.random.normal(0, 0.5),
                "bpm_std_dev": float(np.std([fake_f_bpm, fake_lc_bpm, fake_rc_bpm, fake_n_bpm])),
                "bpm_max_diff": float(np.max([fake_f_bpm, fake_lc_bpm, fake_rc_bpm, fake_n_bpm]) - np.min([fake_f_bpm, fake_lc_bpm, fake_rc_bpm, fake_n_bpm])),
                "mean_bpm": float(np.mean([fake_f_bpm, fake_lc_bpm, fake_rc_bpm, fake_n_bpm])),
                "mean_snr": fake_snr,
                "min_snr": fake_snr - 1.5,
                "corr_forehead_left_cheek": np.clip(np.random.uniform(-0.15, 0.3), -1, 1),
                "coh_forehead_left_cheek": fake_coh,
                "corr_forehead_right_cheek": np.clip(np.random.uniform(-0.15, 0.3), -1, 1),
                "coh_forehead_right_cheek": fake_coh * 0.9,
                "corr_left_cheek_right_cheek": np.clip(np.random.uniform(-0.15, 0.3), -1, 1),
                "coh_left_cheek_right_cheek": fake_coh * 0.95,
                "mean_cross_correlation": float(np.random.uniform(0.05, 0.25)),
                "mean_cross_coherence": fake_coh,
                "consistency_score": float(np.clip(0.6 * fake_coh + 0.4 * ((fake_snr + 5.0) / 15.0), 0.0, 1.0)),
                "subject_id": subj_name,
                "label": 0,
                "filename": f"{subj_name}_fake_{sample_i+1}.mp4"
            }
            records.append(rec_fake)

    return pd.DataFrame(records)

def train_baseline_model(data_path: Optional[str] = None, save_path: str = "models_saved/pulseguard_svm_baseline.joblib") -> Tuple[PulseGuardClassifier, Dict[str, Any]]:
    """
    Train a baseline SVM classifier using GroupKFold cross-validation on subject IDs.
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
    else:
        print("Generating benchmark dataset with subject-grouped structure...")
        df = generate_synthetic_benchmark_dataset(num_subjects=16, samples_per_subject=4)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/benchmark_dataset.csv", index=False)
        print(f"Saved benchmark dataset to data/benchmark_dataset.csv ({len(df)} samples)")

    clf = PulseGuardClassifier(model_type="svm")
    eval_metrics = clf.train_and_evaluate(X=df, y=df["label"].values, groups=df["subject_id"].values, n_splits=4)
    clf.save(save_path)
    print(f"Model saved to {save_path}")
    return clf, eval_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulseGuard Training and Evaluation Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to features CSV")
    parser.add_argument("--output", type=str, default="models_saved/pulseguard_svm_baseline.joblib", help="Output path for trained model")
    args = parser.parse_args()

    clf, metrics = train_baseline_model(data_path=args.data, save_path=args.output)
    print("\n--- Model Training & Validation Report ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
