import numpy as np
from scipy import signal
from typing import Dict, List, Any
import itertools

class ConsistencyFeatureExtractor:
    """
    Extracts cross-region physiological consistency and signal quality metrics:
    - Cross-correlation peak and lag between face patches
    - Spectral coherence (scipy.signal.coherence)
    - Phase synchronization / agreement
    - Region-wise SNR and BPM variance
    These features distinguish organic blood perfusion from synthetic rendering artifacts.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def compute_pairwise_metrics(self, sig1: np.ndarray, sig2: np.ndarray) -> Dict[str, float]:
        """
        Compute cross-correlation and spectral coherence between two ROI pulse signals.
        """
        n = min(len(sig1), len(sig2))
        if n < 16:
            return {
                "max_corr": 0.0,
                "zero_lag_corr": 0.0,
                "mean_coherence": 0.0,
                "peak_coherence": 0.0
            }

        s1 = sig1[:n]
        s2 = sig2[:n]

        # Pearson correlation at zero lag
        std1 = np.std(s1)
        std2 = np.std(s2)
        if std1 > 1e-8 and std2 > 1e-8:
            zero_corr = float(np.corrcoef(s1, s2)[0, 1])
        else:
            zero_corr = 0.0

        # Full normalized cross-correlation
        norm1 = s1 - np.mean(s1)
        norm2 = s2 - np.mean(s2)
        denom = (np.linalg.norm(norm1) * np.linalg.norm(norm2))
        if denom > 1e-8:
            xcorr = signal.correlate(norm1, norm2, mode='full') / denom
            max_corr = float(np.max(np.abs(xcorr)))
        else:
            max_corr = 0.0

        # Magnitude-squared coherence in cardiac band [0.7, 3.0 Hz]
        # Guard against zero or flat variance signals which cause 0/0 -> NaN in coherence
        if std1 <= 1e-8 or std2 <= 1e-8:
            mean_coh = 0.0
            peak_coh = 0.0
        else:
            nperseg = min(128, max(16, n // 2))
            try:
                freqs, coh = signal.coherence(s1, s2, fs=self.fps, nperseg=nperseg)
                coh = np.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0)
                cardiac_mask = (freqs >= 0.7) & (freqs <= 3.0)

                if np.any(cardiac_mask):
                    mean_coh = float(np.mean(coh[cardiac_mask]))
                    peak_coh = float(np.max(coh[cardiac_mask]))
                else:
                    mean_coh = float(np.mean(coh)) if len(coh) > 0 else 0.0
                    peak_coh = float(np.max(coh)) if len(coh) > 0 else 0.0
            except Exception:
                mean_coh = 0.0
                peak_coh = 0.0

        mean_coh = 0.0 if np.isnan(mean_coh) or np.isinf(mean_coh) else mean_coh
        peak_coh = 0.0 if np.isnan(peak_coh) or np.isinf(peak_coh) else peak_coh

        return {
            "max_corr": round(max_corr, 4),
            "zero_lag_corr": round(zero_corr, 4),
            "mean_coherence": round(mean_coh, 4),
            "peak_coherence": round(peak_coh, 4)
        }

    def extract_features(
        self,
        roi_signals: Dict[str, np.ndarray],
        roi_spectra: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregate features across all available facial ROIs into a feature dictionary.
        ROIs: forehead, left_cheek, right_cheek, nose
        """
        features = {}
        regions = list(roi_signals.keys())

        # 1. Per-region SNR and BPM
        bpms = []
        snrs = []
        for r in regions:
            spec = roi_spectra.get(r, {})
            bpm = spec.get("bpm", 0.0)
            snr = spec.get("snr_db", 0.0)
            bpms.append(bpm)
            snrs.append(snr)
            features[f"{r}_bpm"] = float(bpm)
            features[f"{r}_snr"] = float(snr)

        # Variance and spread of BPM across face regions
        # Real biological pulse is uniform across face patches; fakes often have erratic patches
        valid_bpms = [b for b in bpms if b > 0]
        if len(valid_bpms) > 1:
            features["bpm_std_dev"] = float(np.std(valid_bpms))
            features["bpm_max_diff"] = float(np.max(valid_bpms) - np.min(valid_bpms))
            features["mean_bpm"] = float(np.mean(valid_bpms))
        else:
            features["bpm_std_dev"] = 0.0
            features["bpm_max_diff"] = 0.0
            features["mean_bpm"] = float(valid_bpms[0]) if valid_bpms else 0.0

        features["mean_snr"] = float(np.mean(snrs)) if snrs else 0.0
        features["min_snr"] = float(np.min(snrs)) if snrs else 0.0

        # 2. Pairwise consistency metrics
        corr_list = []
        coherence_list = []

        pairs = list(itertools.combinations(regions, 2))
        for r1, r2 in pairs:
            sig1 = roi_signals[r1]
            sig2 = roi_signals[r2]
            metrics = self.compute_pairwise_metrics(sig1, sig2)

            pair_name = f"{r1}_{r2}"
            features[f"corr_{pair_name}"] = metrics["zero_lag_corr"]
            features[f"max_corr_{pair_name}"] = metrics["max_corr"]
            features[f"coh_{pair_name}"] = metrics["mean_coherence"]

            corr_list.append(metrics["zero_lag_corr"])
            coherence_list.append(metrics["mean_coherence"])

        # 3. Overall cross-region agreement indices
        valid_corr = [c for c in corr_list if not (np.isnan(c) or np.isinf(c))]
        valid_coh = [c for c in coherence_list if not (np.isnan(c) or np.isinf(c))]

        features["mean_cross_correlation"] = float(np.mean(valid_corr)) if valid_corr else 0.0
        features["mean_cross_coherence"] = float(np.mean(valid_coh)) if valid_coh else 0.0
        features["min_cross_coherence"] = float(np.min(valid_coh)) if valid_coh else 0.0

        # Multi-region physiological agreement index [0.0 - 1.0]
        # Normalized purely from cross-patch coherence and SNR without arbitrary fake penalties
        norm_coh = max(0.0, min(1.0, features["mean_cross_coherence"]))
        norm_snr = max(0.0, min(1.0, (features["mean_snr"] + 5.0) / 15.0))
        features["consistency_score"] = float(np.clip(0.6 * norm_coh + 0.4 * norm_snr, 0.0, 1.0))

        # Final sanitization: ensure no NaN/inf persists in returned features
        for k, v in features.items():
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

        return features

    def assess_signal_quality(
        self,
        features: Dict[str, float],
        face_tracking_ratio: float,
        duration_sec: float
    ) -> Dict[str, Any]:
        """
        Signal-Quality Gate:
        Evaluates whether extracted rPPG signals contain sufficient physiological evidence.
        If signals are noisy, duration is too short, or regions exhibit extreme divergence,
        flags the analysis as inconclusive rather than jumping to a deepfake verdict.
        """
        reasons = []

        # Gate 1: Face tracking coverage
        if face_tracking_ratio < 0.60:
            reasons.append(f"Low face tracking coverage ({face_tracking_ratio*100:.1f}% < 60%).")

        # Gate 2: Signal duration for frequency resolution
        if duration_sec < 4.0:
            reasons.append(f"Video duration ({duration_sec:.1f}s) is too short for fine spectral resolution (minimum 4s required).")

        # Gate 3: Regional SNR floor
        mean_snr = features.get("mean_snr", -20.0)
        regional_snrs = [
            features.get("forehead_snr", -20.0),
            features.get("left_cheek_snr", -20.0),
            features.get("right_cheek_snr", -20.0),
            features.get("nose_snr", -20.0)
        ]
        sub_zero_count = sum(1 for s in regional_snrs if s < 0.0)

        if mean_snr < 0.0:
            reasons.append(f"Sub-zero mean blood perfusion SNR ({mean_snr:.2f} dB < 0.0 dB). Signals are noise-dominated across the face.")
        elif sub_zero_count >= 3:
            reasons.append(f"Negative SNR on {sub_zero_count} of 4 facial regions (sub-zero perfusion). Only {4 - sub_zero_count} region(s) exhibit signal power above noise floor.")

        # Gate 4: Extreme cross-region heart rate divergence
        bpm_std = features.get("bpm_std_dev", 0.0)
        bpm_diff = features.get("bpm_max_diff", 0.0)
        if bpm_diff > 35.0 or bpm_std > 25.0:
            reasons.append(f"Substantial heart rate divergence across facial patches (std={bpm_std:.1f} BPM, max diff={bpm_diff:.1f} BPM). May indicate localized motion, lighting shadows, or boundary artifact.")

        # Gate 5: Cross-patch coherence floor
        mean_coh = features.get("mean_cross_coherence", 0.0)
        if mean_coh < 0.20 and mean_snr < 1.0:
            reasons.append(f"Negligible cross-patch coherence ({mean_coh:.3f} < 0.20) combined with low perfusion SNR ({mean_snr:.2f} dB).")

        is_reliable = len(reasons) == 0

        return {
            "is_reliable": is_reliable,
            "status": "Reliable" if is_reliable else "Inconclusive",
            "reasons": reasons,
            "summary": (
                "Sufficient physiological signal quality detected across facial ROIs."
                if is_reliable else
                "Inconclusive — insufficient physiological evidence to produce a verified determination."
            )
        }
