import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pulseguard.pipeline.validator import VideoValidator
from pulseguard.rppg.extractor import FaceROIExtractor
from pulseguard.rppg.processor import SignalProcessor
from pulseguard.rppg.features import ConsistencyFeatureExtractor
from pulseguard.models.classifier import PulseGuardClassifier
from pulseguard.utils.plots import create_confusion_matrix_figure, create_waveform_and_spectrum_figure

st.set_page_config(
    page_title="PulseGuard AI - rPPG Deepfake Detection",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2D333F;
        margin-bottom: 12px;
    }
    .badge-real {
        background-color: #1b5e20;
        color: #a5d6a7;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 14px;
    }
    .badge-fake {
        background-color: #b71c1c;
        color: #ffcdd2;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 14px;
    }
    .badge-unvalidated {
        background-color: #e65100;
        color: #ffe0b2;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system_components():
    """Cache loaded models and extractors across app reruns."""
    validator = VideoValidator(min_duration_sec=2.5, max_duration_sec=60.0)
    extractor = FaceROIExtractor()
    processor = SignalProcessor(low_cut_hz=0.7, high_cut_hz=3.0)
    
    # Check for trained baseline model
    model_path = os.path.join("models_saved", "pulseguard_svm_baseline.joblib")
    if os.path.exists(model_path):
        classifier = PulseGuardClassifier.load(model_path)
    else:
        classifier = PulseGuardClassifier(model_type="svm")
        
    return validator, extractor, processor, classifier

validator, extractor, processor, classifier = load_system_components()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
st.sidebar.title("PulseGuard AI")
st.sidebar.caption("Biophysiological rPPG Deepfake Forensic Engine")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    ["Video Analysis Dashboard", "Model Evaluation & Benchmarking", "Methodology & Limitations"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("rPPG Signal Settings")
rppg_method = st.sidebar.selectbox("Signal Extraction Algorithm", ["Green Channel (Fast)", "POS (Plane-Orthogonal)", "CHROM (Chrominance)"])
max_analysis_seconds = st.sidebar.slider("Max Duration to Analyze (s)", min_value=3, max_value=30, value=12)

# --- VIEW 1: VIDEO ANALYSIS DASHBOARD ---
if app_mode == "Video Analysis Dashboard":
    st.title("Remote PPG Video Deepfake Analysis")
    st.markdown(
        "Upload a video of a human subject. PulseGuard AI detects facial ROIs (Forehead, Cheeks, Nose), "
        "recovers blood volume pulse fluctuations via subtle chrominance changes, and assesses biological cross-patch consistency."
    )

    uploaded_file = st.file_uploader(
        "Choose a video file (.mp4, .mov, .avi)",
        type=["mp4", "mov", "avi"],
        help="Well-lit videos with steady face orientation yield the highest rPPG signal quality."
    )

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_temp_path = tfile.name

        col_vid, col_val = st.columns([1, 1])
        with col_vid:
            st.video(video_temp_path)

        with col_val:
            st.subheader("Step 1: Video Quality & Duration Validation")
            with st.spinner("Validating video parameters..."):
                val_result = validator.validate(video_temp_path)

            meta = val_result.get("metadata", {})
            st.markdown(f"**Duration:** `{meta.get('duration_sec', 0)}s` | **FPS:** `{meta.get('fps', 0)}` | **Resolution:** `{meta.get('width', 0)}x{meta.get('height', 0)}`")
            st.markdown(f"**Avg Brightness:** `{meta.get('avg_brightness', 0)}/255` | **Sharpness Variance:** `{meta.get('avg_sharpness', 0)}`")

            if val_result.get("warnings"):
                for w in val_result["warnings"]:
                    st.warning(f"⚠️ {w}")

            if not val_result["valid"]:
                for err in val_result.get("errors", []):
                    st.error(f"❌ {err}")
                st.stop()
            else:
                st.success("✅ Video parameters validated. Ready for pulse extraction.")

        # Analysis Trigger
        st.markdown("---")
        if st.button("🚀 Run Physiological Pulse Extraction & Analysis", type="primary"):
            progress_bar = st.progress(0, text="Initializing face landmark tracker...")
            
            try:
                extraction = extractor.extract_from_video(
                    video_temp_path,
                    max_duration_sec=float(max_analysis_seconds),
                    progress_callback=lambda p: progress_bar.progress(int(p * 80), text=f"Tracking facial skin patches: {int(p*100)}%")
                )
            except Exception as e:
                st.error(f"Extraction error: {e}")
                st.stop()

            progress_bar.progress(85, text="Applying zero-phase bandpass filtering and FFT...")

            fps = extraction["fps"]
            timestamps = extraction["timestamps"]
            signals = extraction["signals"]

            roi_signals = {}
            roi_spectra = {}

            for roi in ["forehead", "left_cheek", "right_cheek", "nose"]:
                if rppg_method == "Green Channel (Fast)":
                    sig = processor.extract_green_signal(signals[roi]["g"], fps)
                elif rppg_method == "POS (Plane-Orthogonal)":
                    sig = processor.extract_pos_signal(signals[roi]["r"], signals[roi]["g"], signals[roi]["b"], fps)
                else:
                    sig = processor.extract_chrom_signal(signals[roi]["r"], signals[roi]["g"], signals[roi]["b"], fps)

                spec = processor.compute_spectrum(sig, fps)
                roi_signals[roi] = sig
                roi_spectra[roi] = spec

            progress_bar.progress(95, text="Computing cross-region physiological coherence...")
            feat_extractor = ConsistencyFeatureExtractor(fps=fps)
            features = feat_extractor.extract_features(roi_signals, roi_spectra)

            progress_bar.progress(100, text="Analysis complete!")

            # --- DISPLAY RESULTS ---
            st.subheader("Step 2: Facial ROI Tracking Preview")
            if extraction.get("preview_frames"):
                prev_cols = st.columns(len(extraction["preview_frames"]))
                for idx, p_frame in enumerate(extraction["preview_frames"]):
                    with prev_cols[idx]:
                        st.image(p_frame, caption=f"ROI Tracking Sample #{idx+1}", use_container_width=True)

            track_ratio = extraction["face_tracking_ratio"]
            if track_ratio == 0.0:
                st.error("❌ Face tracking failed completely (0.0% of frames locked). No human face detected in the video.")
            elif track_ratio < 0.35:
                st.warning(f"⚠️ Low face tracking stability ({track_ratio*100:.1f}%). Pulse signals and cross-region coherence may be unreliable.")
            else:
                st.success(f"✅ Face tracking locked on {track_ratio*100:.1f}% of video frames.")

            # --- METRICS PANEL ---
            st.markdown("---")
            st.subheader("Step 3: Multi-Region Heart Rate & Signal Quality")

            if track_ratio == 0.0:
                st.info("ℹ️ No facial skin regions detected. Pulse metrics are unavailable.")
            else:
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Forehead Pulse", f"{features['forehead_bpm']:.1f} BPM" if features['forehead_bpm'] > 0 else "No Pulse", f"SNR: {features['forehead_snr']:.1f} dB")
                with m_col2:
                    st.metric("Left Cheek Pulse", f"{features['left_cheek_bpm']:.1f} BPM" if features['left_cheek_bpm'] > 0 else "No Pulse", f"SNR: {features['left_cheek_snr']:.1f} dB")
                with m_col3:
                    st.metric("Right Cheek Pulse", f"{features['right_cheek_bpm']:.1f} BPM" if features['right_cheek_bpm'] > 0 else "No Pulse", f"SNR: {features['right_cheek_snr']:.1f} dB")
                with m_col4:
                    st.metric("Nose Bridge Pulse", f"{features['nose_bpm']:.1f} BPM" if features['nose_bpm'] > 0 else "No Pulse", f"SNR: {features['nose_snr']:.1f} dB")

            # --- VISUALIZATION: WAVEFORMS AND PSD ---
            st.markdown("---")
            st.subheader("Step 4: Filtered rPPG Pulse Waveforms & FFT Spectra")
            if track_ratio == 0.0:
                st.warning("⚠️ Cannot render pulse waveforms: face was not detected in video frames.")
            else:
                fig = create_waveform_and_spectrum_figure(timestamps, roi_signals, roi_spectra)
                st.pyplot(fig)

            # --- STEP 5: PREDICTION & EVALUATION DISTINCTION ---
            st.markdown("---")
            st.subheader("Step 5: Forensic Classification & Physiological Consistency")

            col_pred, col_details = st.columns([1, 1])

            with col_pred:
                st.markdown("### Forensic Determination")
                if track_ratio == 0.0:
                    st.markdown("<div class='badge-unvalidated' style='font-size:18px; display:inline-block; padding:8px 16px;'>⚠️ SAMPLE REJECTED: NO FACE DETECTED</div>", unsafe_allow_html=True)
                    st.warning("Prediction skipped: No face was found to extract rPPG physiological signals. Upload a video with a visible, unobstructed face.")
                else:
                    # Validate features before inference
                    is_valid_feat, invalid_keys, invalid_dict = classifier.validate_features(features)
                    if not is_valid_feat:
                        st.warning(f"⚠️ Detected {len(invalid_keys)} degraded/invalid features: {', '.join(invalid_keys)}. Features will be safely imputed via median training statistics.")

                    pred_result = classifier.predict_sample(features)

                    if pred_result["is_model_trained"] and pred_result.get("prediction") not in ["Unknown", "Error"]:
                        if pred_result["label"] == 1:
                            st.markdown(f"<div class='badge-real' style='font-size:20px; display:inline-block; padding:10px 20px; margin-bottom:10px;'>🟢 PREDICTION: {pred_result['prediction'].upper()}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='badge-fake' style='font-size:20px; display:inline-block; padding:10px 20px; margin-bottom:10px;'>🔴 PREDICTION: {pred_result['prediction'].upper()}</div>", unsafe_allow_html=True)
                        
                        st.write(f"**Confidence:** `{pred_result['confidence']*100:.1f}%`")
                        st.caption("⚠️ Notice: Prediction is provided by the calibrated subject-independent baseline model. See 'Methodology & Limitations' for deepfake spoofing caveats.")
                    else:
                        st.markdown("<div class='badge-unvalidated' style='font-size:18px; display:inline-block; padding:8px 16px;'>⚠️ UNVALIDATED PRELIMINARY ANALYSIS</div>", unsafe_allow_html=True)
                        st.info(pred_result.get("explanation", "Showing direct signal metrics without empirical classification."))

            with col_details:
                st.markdown("### Biophysiological Indicators")
                if track_ratio == 0.0:
                    st.write("Biophysiological indicators unavailable (face not detected).")
                else:
                    st.write(f"- **Mean Cross-Patch Coherence:** `{features['mean_cross_coherence']:.3f}` *(Typical Real: > 0.50, Fake: < 0.35)*")
                    st.write(f"- **BPM Standard Deviation Across Face:** `{features['bpm_std_dev']:.2f} BPM` *(Typical Real: < 3.0, Fake: > 6.0)*")
                    st.write(f"- **Mean Blood Perfusion SNR:** `{features['mean_snr']:.2f} dB` *(Typical Real: > 3.0 dB)*")
                    st.write(f"- **Cross-Region Consistency Score:** `{features['consistency_score']:.3f} / 1.0`")

        # Cleanup temp file on end
        try:
            os.unlink(video_temp_path)
        except Exception:
            pass

# --- VIEW 2: MODEL EVALUATION & BENCHMARKING ---
elif app_mode == "Model Evaluation & Benchmarking":
    st.title("Machine Learning Pipeline & Evaluation")
    st.markdown(
        "PulseGuard AI evaluates classifiers using **Subject-Independent GroupKFold Cross-Validation**, "
        "guaranteeing that video clips and synthetic counterparts of the same individual never overlap between training and testing folds."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Status & Parameters")
        st.write(f"**Loaded Model Type:** `{classifier.model_type.upper()}`")
        st.write(f"**Model Status:** `{'Trained & Calibrated' if classifier.is_trained else 'Untrained'}`")
        
        eval_metrics = classifier.evaluation_results
        if eval_metrics:
            st.markdown("### Cross-Validation Metrics")
            st.write(f"- **Validation Scheme:** `{eval_metrics.get('cv_method')}`")
            st.write(f"- **Folds (Splits):** `{eval_metrics.get('splits')}`")
            st.write(f"- **Mean Subject-Independent Accuracy:** `{eval_metrics.get('accuracy_mean', 0.0)*100:.2f}%` (± `{eval_metrics.get('accuracy_std', 0.0)*100:.2f}%`)")
            st.write(f"- **Mean F1-Score:** `{eval_metrics.get('f1_mean', 0.0)*100:.2f}%`")
            st.write(f"- **Mean ROC-AUC:** `{eval_metrics.get('auc_mean', 0.0):.4f}`")

    with col2:
        if classifier.evaluation_results and "confusion_matrix" in classifier.evaluation_results:
            st.subheader("Cross-Validation Confusion Matrix")
            cm = classifier.evaluation_results["confusion_matrix"]
            fig_cm = create_confusion_matrix_figure(cm)
            st.pyplot(fig_cm)
            st.caption("Class 0: Manipulated / Deepfake | Class 1: Real")

    st.markdown("---")
    st.subheader("Retrain Baseline Classifier")
    st.write("You can retrain the baseline model on a custom features CSV or generate fresh benchmark samples.")

    custom_csv = st.file_uploader("Upload Features CSV (optional)", type=["csv"])
    if st.button("🔄 Retrain Baseline SVM Model"):
        with st.spinner("Executing GroupKFold training pipeline..."):
            from pulseguard.pipeline.trainer import train_baseline_model
            data_p = None
            if custom_csv:
                tmp_csv = "data/custom_uploaded_features.csv"
                with open(tmp_csv, "wb") as f:
                    f.write(custom_csv.read())
                data_p = tmp_csv

            clf_new, results = train_baseline_model(data_path=data_p)
            st.success("Model successfully retrained and saved to disk!")
            st.json(results)

# --- VIEW 3: METHODOLOGY & LIMITATIONS ---
elif app_mode == "Methodology & Limitations":
    st.title("Forensic Methodology & Scientific Limitations")

    st.markdown("""
    ### 🔬 Remote Photoplethysmography (rPPG) in Deepfake Forensics

    PulseGuard AI operates on the physiological reality of the human cardiovascular system:
    1. **Hemodynamic Perfusion:** With every ventricular contraction, oxygenated blood enters the microvascular bed of the facial dermis. Hemoglobin absorbs green wavelength photons (approx. 520–580 nm) significantly more than red or blue photons.
    2. **Multi-Region Coherence:** On an authentic human face, periodic pulse signals across the forehead, left cheek, right cheek, and nose exhibit phase synchrony and uniform fundamental heart rate frequencies.
    3. **Synthetic Disruption in Deepfakes:** Conventional deepfake architectures (autoencoders, GANs, diffusion-based face swappers) operate frame-by-frame or on short spatial latents, ignoring long-range temporal blood flow consistency. This introduces:
       - Spatial phase cancellation across facial patches.
       - Desynchronized heart rate estimations between facial quadrants.
       - Disrupted spectral power distributions and reduced cardiac SNR.

    ---

    ### ⚠️ Crucial Scientific Limitations (Honest Reporting)

    As emphasized in contemporary biometrics literature (*Ciftci et al., FakeCatcher, and 2025 deepfake studies*):
    - **No Fabricated Certainty:** A clean rPPG pulse alone is **not** definitive proof of authenticity. Advanced 2024–2026 generative models that synthesize audio-driven avatars or high-fidelity face reenactments can inherit or mimic pulse fluctuations from source donor videos.
    - **Compression Artifacts:** High compression (H.264/H.265 with CRF > 28, social media re-encoding) degrades high-frequency chrominance changes, inducing false alarms on authentic videos.
    - **Subject Motion & Uneven Lighting:** Head rotations and flickering ambient light create motion noise orders of magnitude larger than cardiac micro-color changes.
    - **Demographic & Melanin Considerations:** Melanin absorbs light broadly across the visible spectrum. Standard green-channel rPPG exhibits lower signal-to-noise ratios on darker skin tones (Fitzpatrick types V and VI). POS and CHROM algorithms partially mitigate this but require adequate ambient lux.
    - **Dataset Generalization:** A model trained on a specific dataset (e.g., FaceForensics++) typically experiences an accuracy drop of 5–15% when tested on unconstrained 'in-the-wild' videos.
    """)

st.markdown("---")
st.caption("PulseGuard AI Research Prototype • Developed for Biophysiological Deepfake Forensics")
