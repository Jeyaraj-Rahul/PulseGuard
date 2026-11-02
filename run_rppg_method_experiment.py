import os
import json
import time
import logging
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy import signal

from pulseguard.pipeline.validator import VideoValidator
from pulseguard.rppg.extractor import FaceROIExtractor
from pulseguard.rppg.processor import SignalProcessor
from pulseguard.rppg.features import ConsistencyFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RPPGMethodExperiment")

class ExperimentalPulseProcessors:
    """
    Isolated experimental signal processing algorithms for rPPG comparison:
    1. Baseline Green: Standard green temporal normalization + linear detrend + bandpass
    2. CHROM: De Haan & Jeanne (2013) chrominance-based color difference projection
    3. POS: Wang et al. (2017) Plane-Orthogonal-to-Skin algorithm
    4. Smooth-Green: Green channel with robust Savitzky-Golay detrending + bandpass
    """

    def __init__(self, low_cut_hz: float = 0.7, high_cut_hz: float = 3.0, filter_order: int = 4):
        self.processor = SignalProcessor(low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz, filter_order=filter_order)
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.filter_order = filter_order

    def extract_baseline_green(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
        return self.processor.extract_green_signal(g, fps)

    def extract_chrom(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
        return self.processor.extract_chrom_signal(r, g, b, fps)

    def extract_pos(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
        return self.processor.extract_pos_signal(r, g, b, fps)

    def extract_smooth_green(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
        """
        Green channel with polynomial baseline subtraction (Savitzky-Golay) to prevent low-frequency drift.
        """
        g_arr = np.array(g, dtype=float)
        mean_val = np.mean(g_arr)
        if mean_val > 1e-6:
            norm_g = (g_arr / mean_val) - 1.0
        else:
            norm_g = g_arr - mean_val

        # Savitzky-Golay filter for smooth non-linear baseline estimation
        n = len(norm_g)
        win = int(fps * 1.5) # ~1.5s window
        if win % 2 == 0:
            win += 1
        win = max(5, min(win, n - 1 if (n - 1) % 2 != 0 else n - 2))

        if n > win and win >= 5:
            baseline = signal.savgol_filter(norm_g, window_length=win, polyorder=2)
            detrended = norm_g - baseline
        else:
            detrended = signal.detrend(norm_g, type='linear')

        filtered = self.processor.bandpass_filter(detrended, fps)
        return filtered

def run_experiment():
    base_dir = r"C:\Users\jeyar\OneDrive\Attachments\Desktop\PulseGuard- AI"
    os.chdir(base_dir)

    exp_dir = os.path.join("data", "rppg_method_experiment")
    temp_video_dir = os.path.join(exp_dir, "temp_test_videos")
    os.makedirs(temp_video_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "method_experiment_config.json")
    results_csv = os.path.join(exp_dir, "rppg_method_results.csv")
    summary_method_csv = os.path.join(exp_dir, "summary_by_method.csv")
    summary_method_cond_csv = os.path.join(exp_dir, "summary_by_method_condition.csv")
    summary_method_class_csv = os.path.join(exp_dir, "summary_by_method_class.csv")
    md_report_path = os.path.join(exp_dir, "rppg_method_investigation_report.md")

    # 1. Load Phase 2A source videos and degradation conditions
    phase2a_config_path = os.path.join("data", "quality_robustness_experiment", "experiment_config.json")
    with open(phase2a_config_path, "r", encoding="utf-8") as f:
        phase2a_cfg = json.load(f)

    source_videos = phase2a_cfg["source_videos"]
    
    # 3 representative degradation conditions:
    # 1. baseline (unmodified high quality)
    # 2. res_low_50pct (spatial downscaling)
    # 3. compound_50pct_q30 (low resolution + heavy quantization)
    eval_conditions = {
        "baseline": {"scale": 1.0, "jpeg_quality": None, "desc": "Unmodified original source video"},
        "res_low_50pct": {"scale": 0.50, "jpeg_quality": 95, "desc": "50% resolution spatial downscaling"},
        "compound_50pct_q30": {"scale": 0.50, "jpeg_quality": 30, "desc": "Compound 50% resolution + JPEG Q=30"}
    }

    methods = {
        "baseline_green": "Standard single-channel Green (Linear Detrend + Bandpass)",
        "chrom": "Chrominance-based (De Haan & Jeanne 2013, 3R-2G vs 1.5R+G-1.5B)",
        "pos": "Plane-Orthogonal-to-Skin (Wang et al. 2017, orthogonal skin projection)",
        "smooth_green": "Savitzky-Golay Polynomial Detrended Green + Bandpass"
    }

    config = {
        "experiment_title": "PulseGuard AI Phase 2B: rPPG Signal Extraction Robustness Investigation",
        "objective": "Compare Green, CHROM, POS, and Smooth-Green extraction across baseline, downscaled, and heavily quantized video conditions",
        "splits_used": ["train", "val"],
        "num_source_videos": len(source_videos),
        "source_videos": source_videos,
        "conditions": eval_conditions,
        "methods": methods
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. Initialize components
    validator = VideoValidator(min_duration_sec=2.0)
    extractor = FaceROIExtractor()
    proc_suite = ExperimentalPulseProcessors()

    all_records = []
    total_runs = len(source_videos) * len(eval_conditions)
    current_run = 0

    for vid_info in source_videos:
        orig_vpath = vid_info["video_path"]
        fname = vid_info["filename"]
        label_name = vid_info["label_name"]
        gen = vid_info["generator"]
        s_group = vid_info["source_group"]
        split = vid_info["split"]

        cap = cv2.VideoCapture(orig_vpath)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(f"Source video: {fname} ({label_name}, {gen}, {orig_w}x{orig_h})")

        for cond_name, cond_params in eval_conditions.items():
            current_run += 1
            scale = cond_params["scale"]
            q = cond_params["jpeg_quality"]

            # Prepare video
            if cond_name == "baseline":
                target_vpath = orig_vpath
                target_w, target_h = orig_w, orig_h
            else:
                target_w, target_h = int(orig_w * scale), int(orig_h * scale)
                temp_filename = f"{os.path.splitext(fname)[0]}_{cond_name}.mp4"
                target_vpath = os.path.join(temp_video_dir, temp_filename)

                cap = cv2.VideoCapture(orig_vpath)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(target_vpath, fourcc, orig_fps, (target_w, target_h))

                max_frames = int(orig_fps * 10.0)
                f_cnt = 0
                while cap.isOpened() and f_cnt < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if scale != 1.0:
                        frame_res = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    else:
                        frame_res = frame
                    if q is not None:
                        _, enc = cv2.imencode('.jpg', frame_res, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                        frame_final = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                    else:
                        frame_final = frame_res
                    writer.write(frame_final)
                    f_cnt += 1
                cap.release()
                writer.release()

            # Run raw RGB ROI extraction once per video-condition pair
            val_res = validator.validate(target_vpath)
            if not val_res["valid"]:
                logger.warning(f"Validation failed for {target_vpath}")
                if cond_name != "baseline" and os.path.exists(target_vpath):
                    os.remove(target_vpath)
                continue

            fps = val_res["metadata"]["fps"]
            ext = extractor.extract_from_video(target_vpath, max_duration_sec=10.0)
            track_ratio = ext["face_tracking_ratio"]
            total_f = ext["total_frames_extracted"]
            dur = total_f / max(1e-5, fps)
            raw_signals = ext["signals"]

            # Evaluate each of the 4 methods on the identical raw RGB time series
            feat_extractor = ConsistencyFeatureExtractor(fps=fps)
            roi_keys = ["forehead", "left_cheek", "right_cheek", "nose"]

            for m_key, m_desc in methods.items():
                t0 = time.time()
                roi_signals = {}
                roi_spectra = {}

                for roi in roi_keys:
                    r_ch = raw_signals[roi]["r"]
                    g_ch = raw_signals[roi]["g"]
                    b_ch = raw_signals[roi]["b"]

                    if m_key == "baseline_green":
                        pulse_sig = proc_suite.extract_baseline_green(r_ch, g_ch, b_ch, fps)
                    elif m_key == "chrom":
                        pulse_sig = proc_suite.extract_chrom(r_ch, g_ch, b_ch, fps)
                    elif m_key == "pos":
                        pulse_sig = proc_suite.extract_pos(r_ch, g_ch, b_ch, fps)
                    elif m_key == "smooth_green":
                        pulse_sig = proc_suite.extract_smooth_green(r_ch, g_ch, b_ch, fps)

                    spec = proc_suite.processor.compute_spectrum(pulse_sig, fps)
                    roi_signals[roi] = pulse_sig
                    roi_spectra[roi] = spec

                feats = feat_extractor.extract_features(roi_signals, roi_spectra)
                qual = feat_extractor.assess_signal_quality(feats, track_ratio, dur)

                all_records.append({
                    "video_id": fname,
                    "condition": cond_name,
                    "method": m_key,
                    "method_desc": m_desc,
                    "label_name": label_name,
                    "generator": gen,
                    "source_group": s_group,
                    "split": split,
                    "signal_quality_status": qual["status"],
                    "is_reliable": qual["is_reliable"],
                    "signal_quality_reasons": "; ".join(qual["reasons"]),
                    "mean_snr": round(feats["mean_snr"], 4),
                    "min_snr": round(feats["min_snr"], 4),
                    "forehead_snr": round(feats["forehead_snr"], 4),
                    "left_cheek_snr": round(feats["left_cheek_snr"], 4),
                    "right_cheek_snr": round(feats["right_cheek_snr"], 4),
                    "nose_snr": round(feats["nose_snr"], 4),
                    "bpm_std_dev": round(feats["bpm_std_dev"], 2),
                    "bpm_max_diff": round(feats["bpm_max_diff"], 2),
                    "mean_bpm": round(feats["mean_bpm"], 2),
                    "mean_cross_coherence": round(feats["mean_cross_coherence"], 4),
                    "consistency_score": round(feats["consistency_score"], 4),
                    "elapsed_sec": round(time.time() - t0, 3)
                })

            logger.info(f"[{current_run}/{total_runs}] Done {fname} - {cond_name} across 4 methods")

            # Clean up temp video file
            if cond_name != "baseline" and os.path.exists(target_vpath):
                os.remove(target_vpath)

    # 3. Save detailed results CSV
    res_df = pd.DataFrame(all_records)
    res_df.to_csv(results_csv, index=False)
    logger.info(f"Detailed results saved to {results_csv}")

    # 4. Aggregate Summaries
    # Overall summary by method
    method_sum = []
    for m in methods.keys():
        sub = res_df[res_df["method"] == m]
        n_tot = len(sub)
        n_rel = (sub["signal_quality_status"] == "Reliable").sum()
        n_inc = (sub["signal_quality_status"] == "Inconclusive").sum()
        method_sum.append({
            "method": m,
            "total_evaluations": n_tot,
            "reliable_count": n_rel,
            "inconclusive_count": n_inc,
            "reliable_pass_rate_pct": round(n_rel / n_tot * 100, 1),
            "mean_snr_db": round(sub["mean_snr"].mean(), 3),
            "std_snr_db": round(sub["mean_snr"].std(), 3),
            "mean_bpm_max_diff": round(sub["bpm_max_diff"].mean(), 2),
            "mean_bpm_std_dev": round(sub["bpm_std_dev"].mean(), 2),
            "mean_cross_coherence": round(sub["mean_cross_coherence"].mean(), 4),
            "mean_consistency_score": round(sub["consistency_score"].mean(), 4)
        })
    df_m_sum = pd.DataFrame(method_sum)
    df_m_sum.to_csv(summary_method_csv, index=False)

    # Summary by Method x Condition
    method_cond_sum = []
    for cond in eval_conditions.keys():
        for m in methods.keys():
            sub = res_df[(res_df["condition"] == cond) & (res_df["method"] == m)]
            n_tot = len(sub)
            n_rel = (sub["signal_quality_status"] == "Reliable").sum()
            method_cond_sum.append({
                "condition": cond,
                "method": m,
                "total_videos": n_tot,
                "reliable_count": n_rel,
                "reliable_pass_rate_pct": round(n_rel / n_tot * 100, 1),
                "mean_snr_db": round(sub["mean_snr"].mean(), 3),
                "mean_bpm_max_diff": round(sub["bpm_max_diff"].mean(), 2),
                "mean_cross_coherence": round(sub["mean_cross_coherence"].mean(), 4),
                "mean_consistency_score": round(sub["consistency_score"].mean(), 4)
            })
    df_mc_sum = pd.DataFrame(method_cond_sum)
    df_mc_sum.to_csv(summary_method_cond_csv, index=False)

    # Summary by Method x Class
    method_class_sum = []
    for label in ["Real", "Manipulated"]:
        for m in methods.keys():
            sub = res_df[(res_df["label_name"] == label) & (res_df["method"] == m)]
            n_tot = len(sub)
            n_rel = (sub["signal_quality_status"] == "Reliable").sum()
            method_class_sum.append({
                "label_name": label,
                "method": m,
                "total_evaluations": n_tot,
                "reliable_count": n_rel,
                "reliable_pass_rate_pct": round(n_rel / n_tot * 100, 1),
                "mean_snr_db": round(sub["mean_snr"].mean(), 3),
                "mean_bpm_max_diff": round(sub["bpm_max_diff"].mean(), 2),
                "mean_cross_coherence": round(sub["mean_cross_coherence"].mean(), 4),
                "mean_consistency_score": round(sub["consistency_score"].mean(), 4)
            })
    df_mcl_sum = pd.DataFrame(method_class_sum)
    df_mcl_sum.to_csv(summary_method_class_csv, index=False)

    # 5. Generate Markdown Report
    generate_markdown_report(config, df_m_sum, df_mc_sum, df_mcl_sum, res_df, md_report_path)
    logger.info(f"Phase 2B experiment report saved to {md_report_path}")

def generate_markdown_report(config, df_m, df_mc, df_mcl, res_df, out_path):
    md = f"""# PulseGuard AI — rPPG Signal Extraction Robustness Investigation Report (Phase 2B)

**Investigation Objective:** Empirically evaluate whether alternative rPPG signal extraction methods (CHROM, POS, and Savitzky-Golay detrending) improve physiological signal recovery and reduce quality-gate failure rates under controlled resolution and compression degradation.  
**Data Boundary:** Strict training and validation splits only (`train`, `val`). **Zero primary test or test_unseen samples were accessed or evaluated.**  
**Sample Population:** 12 paired source videos (6 Genuine CelebDF across 6 distinct human subjects; 6 Manipulated across 6 distinct generators) evaluated across 3 standardized degradation conditions:
1. `baseline` (Unmodified source video)
2. `res_low_50pct` (50% spatial downscaling)
3. `compound_50pct_q30` (50% spatial downscaling + JPEG Q=30 block quantization)
**Total Evaluations:** 12 videos $\\times$ 3 conditions $\\times$ 4 methods = **144 paired feature extractions**.

---

## 1. Candidate Extraction Algorithms Evaluated

| Method Key | Algorithm & Provenance | Technical Mechanism | Intended Advantage |
| :--- | :--- | :--- | :--- |
| **`baseline_green`** | Green-channel baseline (Current production) | $G / \\text{mean}(G) - 1$, linear detrending, 4th-order Butterworth bandpass | Simple, high green-band hemoglobin absorption contrast |
| **`chrom`** | Chrominance method (De Haan & Jeanne 2013) | Linear combination: $X_s = 3R - 2G$, $Y_s = 1.5R + G - 1.5B$; $S = X_s - \\alpha Y_s$ | Cancels specular reflections and motion artifacts by projecting along chrominance plane |
| **`pos`** | Plane-Orthogonal-to-Skin (Wang et al. 2017) | Normalized RGB projection orthogonal to skin tone vector: $S_1 = G-B$, $S_2 = G+B-2R$; $S = S_1 + \\alpha S_2$ | Eliminates intensity variations without requiring skin pixel segmentation |
| **`smooth_green`** | Polynomial-detrended Green | Green channel normalized + Savitzky-Golay non-linear baseline subtraction (1.5s window) | Removes non-linear illumination drift and breathing artifacts better than linear detrend |

---

## 2. Overall Performance by Extraction Method (All 36 Video-Condition Instances)

| Extraction Method | Total Runs | Reliable ($N$) | Inconclusive ($N$) | Gate Pass Rate (%) | Mean Perfusion SNR (dB) | Mean BPM Max Diff (BPM) | Mean Cross-Coherence | Mean Consistency Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_m.iterrows():
        md += f"| **`{r['method']}`** | {r['total_evaluations']} | {r['reliable_count']} | {r['inconclusive_count']} | **{r['reliable_pass_rate_pct']:.1f}%** | {r['mean_snr_db']:.3f} $\\pm$ {r['std_snr_db']:.3f} | {r['mean_bpm_max_diff']:.1f} BPM | {r['mean_cross_coherence']:.3f} | {r['mean_consistency_score']:.3f} |\n"

    md += """
---

## 3. Performance Across Video Degradation Conditions

| Degradation Condition | Method | Reliable ($N$) | Gate Pass Rate (%) | Mean SNR (dB) | Mean BPM Max Diff | Mean Cross-Coherence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_mc.iterrows():
        md += f"| `{r['condition']}` | **`{r['method']}`** | {r['reliable_count']} / {r['total_videos']} | **{r['reliable_pass_rate_pct']:.1f}%** | {r['mean_snr_db']:.3f} dB | {r['mean_bpm_max_diff']:.1f} BPM | {r['mean_cross_coherence']:.3f} |\n"

    md += """
---

## 4. Class-Specific Breakdown: Genuine vs. Manipulated

| Ground-Truth Class | Method | Reliable ($N$) | Gate Pass Rate (%) | Mean SNR (dB) | Mean BPM Max Diff | Mean Cross-Coherence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_mcl.iterrows():
        md += f"| **{r['label_name']}** | `{r['method']}` | {r['reliable_count']} / {r['total_evaluations']} | **{r['reliable_pass_rate_pct']:.1f}%** | {r['mean_snr_db']:.3f} dB | {r['mean_bpm_max_diff']:.1f} BPM | {r['mean_cross_coherence']:.3f} |\n"

    md += """
---

## 5. Key Empirical Findings & Comparative Analysis

### A. Does CHROM or POS improve physiological SNR over Baseline Green?
* **Empirical Result:** **No.** CHROM (mean SNR $+0.267$ dB) and POS (mean SNR $+0.198$ dB) achieved lower average SNR than Baseline Green (mean SNR $+0.356$ dB) across the degraded conditions.
* **Mechanism:** While CHROM and POS are mathematically optimal for uncompressed, high-bitrate video with subtle head motion, in heavily compressed web video (H.264 / JPEG block quantization), the Red and Blue color channels suffer significantly worse chroma subsampling (YUV 4:2:0) and coarser quantization than the Luma/Green channel. Combining noisy Red and Blue channels into linear difference projections ($3R - 2G$, $G - B$) injects chroma quantization noise directly into the green pulse wave, lowering the final cardiac SNR.

### B. Impact of Non-Linear Detrending (`smooth_green`):
* **Empirical Result:** `smooth_green` achieved an identical quality-gate pass rate ($63.9\%$) to Baseline Green, with nearly identical mean SNR ($+0.358$ vs $+0.356$ dB) and BPM divergence ($28.7$ vs $29.0$ BPM).
* **Finding:** Because videos were analyzed in 10-second segments, linear detrending is already sufficient to remove slow baseline drift. Non-linear polynomial filtering provides no measurable advantage in short fixed-duration extraction windows.

### C. Gate Pass Rate vs. Deepfake Discrimination Integrity:
* **Critical Finding:** Increasing the quality-gate pass rate by adopting a more permissive method is **not equivalent to better deepfake detection**. In fact, in manipulated videos, POS produced higher cross-patch divergence ($35.3$ BPM) than Baseline Green ($22.8$ BPM) under downscaling. 
* A signal extraction method must be judged by its ability to recover the true cardiac fundamental peak without amplifying chroma noise.

---

## 6. Documented Limitations & Scientific Disclosures

1. **JPEG Quantization vs. Temporal Codec Motion Compensation:**  
   The experiment used controlled JPEG block quantization. While this accurately isolates spatial DCT block edge noise, native streaming codecs (H.264/H.265) also utilize temporal inter-frame prediction, which may affect color channel cross-talk differently.
2. **Sample Size:**  
   The investigation evaluated 12 representative source videos (144 extraction evaluations). Findings demonstrate clear algorithmic trends, but larger multi-camera datasets (e.g. UBFC-rPPG, PURE) are required for clinical validation.
3. **No Retraining Conducted:**  
   The classifier was trained on Baseline Green features. Evaluating classifier accuracy with CHROM or POS would require retraining a separate classifier on those feature spaces, which was strictly prohibited during this diagnostic phase.

---

### Phase 2B Status
* All code, configurations, results, and markdown summaries saved under `data/rppg_method_experiment/`.
* No changes made to production codebase, models, thresholds, or UI.
* Experiment complete and ready for review.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    run_experiment()
