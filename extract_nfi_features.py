import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from pulseguard.models.classifier import PulseGuardClassifier
from pulseguard.rppg.extractor import FaceROIExtractor
from pulseguard.pipeline.validator import VideoValidator
from pulseguard.rppg.processor import SignalProcessor
from pulseguard.rppg.features import ConsistencyFeatureExtractor

def extract_features_from_manifest(
    manifest_path: str = "data/manifest_nfi.csv",
    output_features_path: str = "data/features_nfi.csv",
    log_file_path: str = "data/extraction_failures.log",
    max_duration_sec: float = 10.0,
    resume: bool = True
):
    print("=" * 70)
    print("PULSEGUARD AI - NFI DATASET SEQUENTIAL FEATURE EXTRACTOR")
    print("=" * 70)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    total_videos = len(manifest_df)
    print(f"Loaded manifest with {total_videos} videos.")

    # Initialize components once
    validator = VideoValidator(min_duration_sec=2.0)
    extractor = FaceROIExtractor()
    processor = SignalProcessor()

    # Logging setup
    logger = logging.getLogger("NFI_Extractor")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path, mode="a" if resume else "w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Check for already processed videos if resuming
    processed_files = set()
    if resume and os.path.exists(output_features_path):
        try:
            existing_df = pd.read_csv(output_features_path)
            if "filename" in existing_df.columns:
                processed_files = set(existing_df["filename"].dropna())
                print(f"Resuming: found {len(processed_files)} already processed records in {output_features_path}")
        except Exception as e:
            print(f"Warning reading existing features file: {e}")

    # Column schema
    # Metadata columns + 22 classifier features + quality metrics
    metadata_cols = [
        "video_path", "filename", "label", "label_name",
        "generator", "manipulation_type", "source_group", "split",
        "extraction_success", "face_tracking_ratio", "duration_sec",
        "signal_quality_status", "signal_quality_reasons", "failure_reason"
    ]
    all_cols = metadata_cols + PulseGuardClassifier.FEATURE_NAMES

    # Ensure output CSV header exists
    if not (resume and os.path.exists(output_features_path) and len(processed_files) > 0):
        pd.DataFrame(columns=all_cols).to_csv(output_features_path, index=False)

    start_time = time.time()
    success_count = 0
    failure_count = 0
    inconclusive_count = 0

    for idx, row in manifest_df.iterrows():
        vpath = row["video_path"]
        fname = row["filename"]
        label = row["label"]
        label_name = row["label_name"]
        gen = row["generator"]
        m_type = row["manipulation_type"]
        s_group = row["source_group"]
        split = row["split"]

        if fname in processed_files:
            continue

        item_res = {
            "video_path": vpath,
            "filename": fname,
            "label": label,
            "label_name": label_name,
            "generator": gen,
            "manipulation_type": m_type,
            "source_group": s_group,
            "split": split,
            "extraction_success": False,
            "face_tracking_ratio": 0.0,
            "duration_sec": 0.0,
            "signal_quality_status": "Failed",
            "signal_quality_reasons": "",
            "failure_reason": ""
        }
        # Initialize 22 features to NaN
        for f in PulseGuardClassifier.FEATURE_NAMES:
            item_res[f] = np.nan

        # 1. Validate video container
        val_res = validator.validate(vpath)
        if not val_res["valid"]:
            err = str(val_res.get("errors", val_res.get("error", "Unknown validation error")))
            item_res["failure_reason"] = f"Validation failed: {err}"
            logger.warning(f"[{idx+1}/{total_videos}] Validation failed for {vpath}: {err}")
            failure_count += 1
            # Append failed row
            pd.DataFrame([item_res])[all_cols].to_csv(output_features_path, mode="a", header=False, index=False)
            continue

        fps = val_res["metadata"]["fps"]

        # 2. Extract facial ROIs
        try:
            extraction = extractor.extract_from_video(vpath, max_duration_sec=max_duration_sec)
        except Exception as e:
            err = f"ROI extraction error: {str(e)}"
            item_res["failure_reason"] = err
            logger.error(f"[{idx+1}/{total_videos}] Error extracting ROIs for {vpath}: {e}")
            failure_count += 1
            pd.DataFrame([item_res])[all_cols].to_csv(output_features_path, mode="a", header=False, index=False)
            continue

        track_ratio = extraction["face_tracking_ratio"]
        total_frames = extraction["total_frames_extracted"]
        duration = total_frames / max(1e-5, fps)
        item_res["face_tracking_ratio"] = round(float(track_ratio), 4)
        item_res["duration_sec"] = round(float(duration), 2)

        # Tracking check gate (minimum 0.30 for basic extraction)
        if track_ratio < 0.30:
            err = f"Face tracking ratio too low ({track_ratio:.2f} < 0.30)"
            item_res["failure_reason"] = err
            logger.warning(f"[{idx+1}/{total_videos}] Low tracking for {vpath}: {track_ratio:.2f}")
            failure_count += 1
            pd.DataFrame([item_res])[all_cols].to_csv(output_features_path, mode="a", header=False, index=False)
            continue

        # 3. Process signals & compute spectrum
        try:
            roi_signals = {}
            roi_spectra = {}
            signals = extraction["signals"]
            for roi in ["forehead", "left_cheek", "right_cheek", "nose"]:
                pulse_sig = processor.extract_green_signal(signals[roi]["g"], fps)
                spec = processor.compute_spectrum(pulse_sig, fps)
                roi_signals[roi] = pulse_sig
                roi_spectra[roi] = spec

            feat_ext = ConsistencyFeatureExtractor(fps=fps)
            feats = feat_ext.extract_features(roi_signals, roi_spectra)

            # Check signal quality gate
            quality = feat_ext.assess_signal_quality(feats, track_ratio, duration)
            item_res["signal_quality_status"] = quality["status"]
            item_res["signal_quality_reasons"] = "; ".join(quality["reasons"])
            if not quality["is_reliable"]:
                inconclusive_count += 1

            # Populate 22 features
            for f in PulseGuardClassifier.FEATURE_NAMES:
                val = feats.get(f, np.nan)
                item_res[f] = float(val) if not np.isnan(val) else 0.0

            item_res["extraction_success"] = True
            success_count += 1
            logger.info(f"[{idx+1}/{total_videos}] Success: {fname} (tracking={track_ratio:.2f}, status={quality['status']})")

        except Exception as e:
            err = f"Signal processing error: {str(e)}"
            item_res["failure_reason"] = err
            logger.error(f"[{idx+1}/{total_videos}] Signal processing error for {vpath}: {e}")
            failure_count += 1

        # Append row to CSV sequentially
        pd.DataFrame([item_res])[all_cols].to_csv(output_features_path, mode="a", header=False, index=False)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_videos:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(1.0, elapsed)
            print(f"Progress: [{idx+1}/{total_videos}] ({(idx+1)/total_videos*100:.1f}%) | "
                  f"Success: {success_count}, Inconclusive: {inconclusive_count}, Failures: {failure_count} | "
                  f"{rate:.2f} vids/s")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"EXTRACTION COMPLETE in {elapsed_total/60:.2f} minutes.")
    print(f"Total processed: {total_videos}")
    print(f"Successful extractions: {success_count}")
    print(f"Inconclusive signals: {inconclusive_count}")
    print(f"Failures: {failure_count}")
    print(f"Output saved to: {output_features_path}")
    print(f"Failure logs written to: {log_file_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rPPG features from NFI manifest.")
    parser.add_argument("--manifest", type=str, default="data/manifest_nfi.csv")
    parser.add_argument("--output", type=str, default="data/features_nfi.csv")
    parser.add_argument("--log", type=str, default="data/extraction_failures.log")
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    extract_features_from_manifest(
        manifest_path=args.manifest,
        output_features_path=args.output,
        log_file_path=args.log,
        max_duration_sec=args.max_duration,
        resume=not args.no_resume
    )
